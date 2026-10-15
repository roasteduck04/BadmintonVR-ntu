"""pose24 monocular fit (Stage B) pieces."""
import json

import numpy as np
import pytest
from scipy.optimize import approx_fprime

import pose24_synth as syn
from analysis import pose_v2 as pv
from analysis.pose24 import confidence, fit, kpset


def _inputs(T=4):
    cams = syn.rig(["a"])
    X = kpset.from133(syn.motion(T, z0=-3.0)) * np.array([1.0, -1.0, 1.0])
    b = syn.bundle({"P1": syn.motion(T, z0=-3.0)}, cams, ["a"])
    uv, w = kpset.player_obs(b, "P1")
    P = pv.proj_mats(cams, ["a"])
    edges = fit.edge_arrays({"edges": {"thigh": 0.4}, "soft": {}})
    return P, uv, w, X, edges


def _window(depth=None, T=4, **kw):
    P, uv, w, X, edges = _inputs(T)
    free = kw.pop("free", np.ones(T, bool))
    anchor = np.full_like(X, np.nan)
    contact = np.zeros((T, 6), bool)
    if kw.get("nodepth"):
        return fit.WindowProblem24(P, uv, w, X + 0.01, free, edges, anchor, contact, 1.0), X
    return fit.WindowProblem24(P, uv, w, X + 0.01, free, edges, anchor, contact, 1.0, depth=depth), X


def test_depth_none_is_unchanged():
    p0, X = _window()
    x = p0.X0[p0.free].ravel()
    r0 = p0.residuals(x)
    assert r0.shape[0] == p0.jacobian(x).shape[0]


def test_depth_none_identical_to_no_kwarg():
    free = np.array([False, True, True, True])
    p0, _ = _window(free=free, nodepth=True)
    p1, _ = _window(None, free=free)
    x = p0.X0[p0.free].ravel() + 0.003
    assert np.array_equal(p0.residuals(x), p1.residuals(x))
    J0, J1 = p0.jacobian(x), p1.jacobian(x)
    assert J0.shape == J1.shape and np.array_equal(J0.toarray(), J1.toarray())


def test_fit_segment_depth_none_identical():
    P, uv, w, X, edges = _inputs(6)
    anchor, contact = X.copy(), np.zeros((6, 6), bool)
    a = fit._fit_segment(P, uv, w, X + 0.02, edges, anchor, contact, 1.0, 60.0, True)
    b = fit._fit_segment(P, uv, w, X + 0.02, edges, anchor, contact, 1.0, 60.0, True, depth=None)
    assert np.array_equal(a, b)


def test_depth_term_residual_and_jacobian_match_numeric():
    axis = np.array([0.0, 0.0, 1.0])
    depth = {"axis": axis, "t": np.array([0, 1, 2]), "k": np.array([9, 10, 0]), "d": np.array([0.1, -0.2, 0.0]),
             "sigma": 0.05}
    p, _ = _window(depth)
    p0, _ = _window()
    x = p.X0[p.free].ravel()
    assert len(p.residuals(x)) == len(p0.residuals(x)) + 3
    J = p.jacobian(x).toarray()[-3:]
    num = np.stack([approx_fprime(x, lambda y, i=i: p.residuals(y)[-3 + i], 1e-6) for i in range(3)])
    np.testing.assert_allclose(J, num, atol=1e-4)


def test_depth_rows_residual_value_and_filtering():
    axis = np.array([0.6, 0.0, 0.8])
    depth = {"axis": axis, "t": np.array([0, 1, 2, 3]), "k": np.array([9, 11, 12, 0]),
             "d": np.array([0.1, 0.0, 0.0, -0.3]), "sigma": 0.05}
    free = np.array([False, True, True, True])
    p, _ = _window(depth, free=free)
    assert p.dt.tolist() == [3] and p.dk.tolist() == [0]      # frame 0 not free, hips dropped
    x = p.X0[p.free].ravel()
    X = p.unpack(x)
    want = ((X[3, 0] - 0.5 * (X[3, 11] + X[3, 12])) @ axis + 0.3) / 0.05
    assert p.residuals(x)[-1] == pytest.approx(want)


def test_fit_segment_slices_depth_per_window(monkeypatch):
    seen = []
    orig = fit.WindowProblem24

    class Spy(orig):
        def __init__(self, *a, depth=None, **kw):
            seen.append(None if depth is None else (depth["t"].copy(), depth["k"].copy()))
            super().__init__(*a, depth=depth, **kw)

        def solve(self, max_nfev=40):
            return self.X0.copy()

    monkeypatch.setattr(fit, "WindowProblem24", Spy)
    monkeypatch.setattr(fit, "WINDOW", 4)
    monkeypatch.setattr(fit, "OVERLAP", 1)
    P, uv, w, X, edges = _inputs(7)
    depth = {"axis": np.array([0.0, 0.0, 1.0]), "t": np.array([0, 3, 5, 6]), "k": np.array([0, 1, 2, 3]),
             "d": np.zeros(4), "sigma": 0.1}
    fit._fit_segment(P, uv, w, X, edges, X.copy(), np.zeros((7, 6), bool), 1.0, 60.0, False, depth=depth)
    # windows: start 0 [lo 0, stop 4); start 3 [lo 1, stop 7)
    assert seen[0][0].tolist() == [0, 3] and seen[0][1].tolist() == [0, 1]
    assert seen[1][0].tolist() == [2, 4, 5] and seen[1][1].tolist() == [1, 2, 3]


def test_one_view_sigma_is_finite_with_depth_prior_and_capped_without():
    cams = syn.rig(["a"])
    P = pv.proj_mats(cams, ["a"])
    F, K = 5, 35
    X = np.tile(np.array([0.0, -1.0, 0.0]), (F, K, 1))
    fr = {"X": X, "w": np.ones((F, K, 1)), "resid": np.zeros((F, K, 1)), "inlier": np.ones((F, K, 1), bool)}
    s0, _, m0 = confidence.keypoint_sigma(fr, P)
    assert np.all(s0 == confidence.CAP_MM) and not m0.any()
    R = cams["a"][1]
    s1, _, m1 = confidence.keypoint_sigma(fr, P, depth_axis=R[2], depth_sigma_m=0.1)
    assert np.all(np.isfinite(s1)) and np.all(s1 <= 101.0) and np.all(s1 >= 99.0) and m1.all()


def test_keypoint_sigma_default_unchanged_multi_view():
    cams = syn.rig(["a", "b"])
    P = pv.proj_mats(cams, ["a", "b"])
    rng = np.random.default_rng(0)
    F, K = 6, 35
    X = np.array([0.0, -1.0, 0.0]) + rng.normal(0, 0.3, (F, K, 3))
    inl = rng.random((F, K, 2)) > 0.3
    fr = {"X": X, "w": rng.random((F, K, 2)), "resid": rng.random((F, K, 2)) * 10, "inlier": inl}
    a = confidence.keypoint_sigma(fr, P)
    b = confidence.keypoint_sigma(fr, P, depth_axis=None, depth_sigma_m=None)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


def test_depth_rows_drop_nonfinite_targets_and_normalise_axis():
    depth = {"axis": np.array([0.0, 0.0, 2.0]), "t": np.array([0, 1, 2]), "k": np.array([9, 10, 0]),
             "d": np.array([0.1, np.nan, np.inf]), "sigma": 0.05}
    p, _ = _window(depth)
    assert p.dk.tolist() == [9] and np.isfinite(p.residuals(p.X0[p.free].ravel())).all()
    np.testing.assert_allclose(p.daxis, [0.0, 0.0, 1.0])


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_depth_sigma_must_be_positive(sigma):
    depth = {"axis": np.array([0.0, 0.0, 1.0]), "t": np.array([0]), "k": np.array([9]), "d": np.array([0.1]),
             "sigma": sigma}
    with pytest.raises(ValueError, match="sigma"):
        _window(depth)


def test_keypoint_sigma_half_given_prior_raises():
    cams = syn.rig(["a"])
    P = pv.proj_mats(cams, ["a"])
    fr = {"X": np.zeros((2, 35, 3)), "w": np.ones((2, 35, 1)), "resid": np.zeros((2, 35, 1)),
          "inlier": np.ones((2, 35, 1), bool)}
    with pytest.raises(ValueError, match="depth_axis"):
        confidence.keypoint_sigma(fr, P, depth_axis=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="depth_axis"):
        confidence.keypoint_sigma(fr, P, depth_sigma_m=0.1)


from analysis.pose24 import mono


def test_ray_plane_hits_the_floor():
    cams = syn.rig(["a"])
    K, R, t = cams["a"]
    Xd = np.array([[0.5, -0.02, -2.0]])                 # 2 cm above the floor (Y-down)
    uv, _ = pv.project(pv.proj_mats(cams, ["a"]), Xd)
    p = mono.ray_plane(uv[:, 0], cams["a"], 0.02)
    np.testing.assert_allclose(p, Xd, atol=1e-6)


def test_ray_plane_nan_behind_camera_and_for_nan_pixels():
    cams = syn.rig(["a"])
    p = mono.ray_plane(np.array([[1080.0, 0.0], [np.nan, 5.0]]), cams["a"], 0.0)   # top row: ray points up
    assert np.isnan(p).all()


def test_world_sign_one_camera():
    cams = syn.rig(["a", "b"])
    assert kpset.world_sign({"a": cams["a"]}) == kpset.world_sign(cams)


def test_contact_from_2d_marks_the_planted_foot():
    F = 60
    X = syn.motion(F, z0=-3.0, walk=False, arm=False, plant=(0, F))
    b = syn.bundle({"P1": X}, syn.rig(["a"]), ["a"])
    uv, w = kpset.player_obs(b, "P1")
    c = mono.contact_from_2d(uv[:, :, 0], w[:, :, 0], 60.0)
    assert c.mean() > 0.8


def test_contact_from_2d_not_while_the_body_moves_fast():
    F = 60
    X = syn.motion(F, z0=-3.0, walk=False, arm=False)
    X = X + np.array([2.0, 0.0, 0.0]) * (np.arange(F) / 60.0)[:, None, None]    # 2 m/s sideways
    b = syn.bundle({"P1": X}, syn.rig(["a"]), ["a"])
    uv, w = kpset.player_obs(b, "P1")
    assert not mono.contact_from_2d(uv[:, :, 0], w[:, :, 0], 60.0).any()


def test_lengths_for_prior_and_oracle():
    Lp = mono.lengths_for("prior", None, "P1", stature_m=1.70)
    assert Lp["stature"]["measured_m"] == pytest.approx(1.70)
    assert set(Lp["edges"]) == {n for n, _, _ in fit.EDGES} and set(Lp["soft"]) == set(fit.SOFT)
    Lq = mono.lengths_for("prior", None, "P1", stature_m=1.50)
    assert Lq["edges"]["thigh"] == pytest.approx(Lp["edges"]["thigh"] * 1.5 / 1.7)
    Xup = kpset.from133(syn.motion(40, z0=-3.0))
    cache = {"extra": {"P1": {"X": Xup.astype(np.float32)}}}
    Lo = mono.lengths_for("oracle", cache, "P1")
    assert Lo["edges"]["thigh"] == pytest.approx(syn.LIMB_TRUE["thigh"], abs=2e-3)
    stored = {"P1": {"edges": {"thigh": 0.4, "shank": 0.41}, "soft": {"shoulder_width": 0.36},
                     "stature": {"prior_m": [1.7, 0.1], "measured_m": 1.62, "posterior_m": 1.63, "implied_m": {}}}}
    Ls = mono.lengths_for("oracle", dict(cache, players_lengths=stored), "P1")
    assert Ls["edges"]["thigh"] == 0.4 and Ls["soft"]["shoulder_width"] == 0.36
    assert Ls["stature"]["measured_m"] == 1.62
    Lx = mono.lengths_for("oracle", dict(cache, players_lengths={"P2": stored["P1"]}), "P1")   # other pid: fallback
    assert Lx["edges"]["thigh"] == pytest.approx(Lo["edges"]["thigh"])
    with pytest.raises(ValueError):
        mono.lengths_for("wham", cache, "P1")


def test_load_prior_roundtrip(tmp_path):
    F = 5
    path = tmp_path / "a_wham.npz"
    meta = {"method": "wham", "clip": "c", "view": "a", "timeline_of": "c", "fps": 60.0, "frame_count": F}
    np.savez(path, meta=json.dumps(meta), P1_kp=np.arange(3), P1_rel_cam=np.zeros((F, 3, 3)), P1_conf=np.ones(F))
    pr = mono.load_prior(str(path), "P1")
    assert pr["kp"].tolist() == [0, 1, 2] and pr["rel_cam"].shape == (F, 3, 3) and pr["contact"] is None
    assert pr["meta"]["method"] == "wham"
    assert mono.load_prior(str(path), "P2") is None


def test_init_mono_places_the_body_near_the_truth():
    F = 30
    Xup = syn.motion(F, z0=-3.0, arm=False)
    cams = syn.rig(["a"])
    b = syn.bundle({"P1": Xup}, cams, ["a"])
    uv, w = kpset.player_obs(b, "P1")
    true35 = kpset.from133(Xup) * np.array([1.0, -1.0, 1.0])
    L = mono.lengths_from_points(true35)
    X = mono.init_mono(uv[:, :, 0], w[:, :, 0], cams["a"], L, kpset.world_sign(cams), np.ones((F, 6), bool),
                       ankle_h=0.08)
    hip, hip_t = X[:, [11, 12]].mean(1), true35[:, [11, 12]].mean(1)
    assert np.median(np.linalg.norm((hip - hip_t)[:, [0, 2]], axis=-1)) < 0.1
    left, left_t = X[:, 11] - X[:, 12], true35[:, 11] - true35[:, 12]
    assert (np.sum(left * left_t, axis=-1) > 0).all()                       # facing the right way


def _mono_bundle(F, Xup, cams):
    b = syn.bundle({"P1": Xup}, cams, ["a"], noise_px=0.5)
    b.update({"det_index": {"P1": np.zeros((1, F), int)}, "lr_swapped": {"P1": np.zeros((1, F), bool)}})   # (V, F)
    return b


def test_fit_mono_puts_root_on_the_court_and_limbs_the_right_way(tmp_path):
    """Single synthetic view, known lengths, no learned prior: pelvis XZ within 15 cm, feet on the floor."""
    F = 90
    Xup = syn.motion(F, z0=-3.0)
    cams = syn.rig(["a"])
    b = _mono_bundle(F, Xup, cams)
    true35 = kpset.from133(Xup)
    lengths = mono.lengths_from_points(true35 * np.array([1.0, -1.0, 1.0]))
    fr = mono.fit_mono(b, "P1", "a", lengths=lengths, passes=2, log=lambda *a: None)
    Xf = fr["X"] * np.array([1.0, -1.0, 1.0])
    hip, hip_t = Xf[:, [11, 12]].mean(1), true35[:, [11, 12]].mean(1)
    assert np.nanmedian(np.linalg.norm((hip - hip_t)[:, [0, 2]], axis=-1)) < 0.15
    assert np.nanmedian(Xf[:, [19, 22], 1]) < 0.08
    assert fr["depth_axis"].shape == (3,) and fr["depth_sigma_m"] == mono.NO_PRIOR_DEPTH_SIGMA
    for k in ("X", "uv", "w", "resid", "inlier", "raw", "contact", "lengths", "sgn", "segments", "peak_wrist_mps"):
        assert k in fr
    assert fr["w"].shape == (F, 35, 1) and fr["inlier"].shape == (F, 35, 1)
    assert fr["lengths"]["stature"]["measured_m"] is not None
    from analysis.pose24 import build as pb
    r = pb.regress_player(b, fr, "smpl-28", depth_axis=fr["depth_axis"], depth_sigma_m=fr["depth_sigma_m"])
    assert np.isfinite(r["J"]).all() and r["measured35"].any()


def test_depth_prior_fixes_the_arm_depth(tmp_path):
    """With a perfect synthetic depth prior the right wrist's camera depth relative to the hips is within 5 cm."""
    F = 60
    Xup = syn.motion(F, z0=-3.0)
    cams = syn.rig(["a"])
    b = _mono_bundle(F, Xup, cams)
    true35 = kpset.from133(Xup) * np.array([1.0, -1.0, 1.0])
    K, R, t = cams["a"]
    rel = (true35 - true35[:, [11, 12]].mean(1, keepdims=True)) @ R.T
    kp = np.arange(35)
    prior = {"kp": kp, "rel_cam": rel, "conf": np.ones(F), "contact": None}
    lengths = mono.lengths_from_points(true35)
    fr = mono.fit_mono(b, "P1", "a", lengths=lengths, prior=prior, log=lambda *a: None)
    d_fit = ((fr["X"][:, 10] - fr["X"][:, [11, 12]].mean(1)) @ R[2])
    d_true = rel[:, 10, 2]
    assert np.nanmedian(np.abs(d_fit - d_true)) < 0.05
    assert fr["depth_sigma_m"] == mono.PRIOR_DEPTH_SIGMA


def test_fit_mono_prior_lengths_end_to_end():
    F = 60
    Xup = syn.motion(F, z0=-3.0)
    b = _mono_bundle(F, Xup, syn.rig(["a"]))
    fr = mono.fit_mono(b, "P1", "a", lengths="prior", passes=1, log=lambda *a: None)
    assert fr["lengths"]["stature"]["measured_m"] == pytest.approx(1.70)
    from analysis.pose24 import build as pb
    r = pb.regress_player(b, fr, "smpl-28", depth_axis=fr["depth_axis"], depth_sigma_m=fr["depth_sigma_m"])
    ok = np.isfinite(r["J"]).all(-1).all(-1)
    assert ok.mean() > 0.9 and np.isfinite(r["J"][ok]).all()


def test_fit_mono_prior_contact_is_ored_in():
    """Prior contact > 0.5 marks a heel planted even where 2D contact (a fast sideways body) never does."""
    F = 40
    Xup = syn.motion(F, z0=-3.0, walk=False, arm=False)
    Xup = Xup + np.array([2.0, 0.0, 0.0]) * (np.arange(F) / 60.0)[:, None, None]
    b = _mono_bundle(F, Xup, syn.rig(["a"]))
    L = mono.lengths_from_points(kpset.from133(Xup) * np.array([1.0, -1.0, 1.0]))
    pc = np.zeros((F, 6))
    pc[10:30, 0] = 0.9
    prior = {"kp": np.arange(35), "rel_cam": np.zeros((F, 35, 3)), "conf": np.zeros(F), "contact": pc}
    fr = mono.fit_mono(b, "P1", "a", lengths=L, prior=prior, passes=1, log=lambda *a: None)
    assert fr["contact"][10:30, 0].all()
    fr0 = mono.fit_mono(b, "P1", "a", lengths=L, passes=1, log=lambda *a: None)
    assert not fr0["contact"][10:30, 0].all()


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_fit_mono_depth_sigma_must_be_positive(sigma):
    F = 20
    b = _mono_bundle(F, syn.motion(F, z0=-3.0), syn.rig(["a"]))
    with pytest.raises(ValueError):
        mono.fit_mono(b, "P1", "a", lengths="prior", passes=1, depth_sigma_m=sigma, log=lambda *a: None)


def test_fit_mono_explicit_depth_sigma_is_kept():
    F = 20
    b = _mono_bundle(F, syn.motion(F, z0=-3.0), syn.rig(["a"]))
    fr = mono.fit_mono(b, "P1", "a", lengths="prior", passes=1, depth_sigma_m=0.5, log=lambda *a: None)
    assert fr["depth_sigma_m"] == 0.5


def test_depth_rows_log_the_scale_fallback():
    F = 5
    prior = {"kp": np.arange(3), "rel_cam": np.zeros((F, 3, 3)), "conf": np.ones(F)}
    lines = []
    d = mono._depth_rows(prior, {"edges": {"thigh": 0.4, "shank": 0.4}}, np.array([0.0, 0.0, 1.0]), F,
                         log=lines.append)
    assert d["scale"] == 1.0 and any("scale 1.0" in ln for ln in lines)


def test_ankle_h_reads_the_stored_cache_key():
    assert mono._ankle_h({"ankle_h_m": 0.15}) == 0.15
    assert mono._ankle_h({}) == mono.DEFAULT_ANKLE_H and mono._ankle_h({"ankle_h_m": None}) == mono.DEFAULT_ANKLE_H


# --- pose24 build --mono (Task 13) ---
import os                                                            # noqa: E402

from analysis import schema24                                        # noqa: E402
from analysis.people import persons as personsmod                    # noqa: E402
from analysis.pose24 import build as pb                              # noqa: E402
from analysis.pose24 import config                                   # noqa: E402
from analysis.pose24 import doc as docmod                            # noqa: E402
from analysis.pose24 import track                                    # noqa: E402

FM = 90


@pytest.fixture(scope="module")
def mono_built(tmp_path_factory):
    root = tmp_path_factory.mktemp("p24mono")
    old = os.getcwd()
    os.chdir(root)
    try:
        views = ("a", "b", "c", "d")
        players = {"P1": syn.motion(FM, x0=-0.5, z0=-3.0), "P2": syn.motion(FM, x0=0.5, z0=3.0)}
        entry = syn.write_clip(str(root), "synclip", players, syn.rig(views), views, shifts={"c": 2})
        reg = {"synclip": entry,
               "synclip_1cam_b": {"base": "synclip", "views": ["b"], "timeline_of": "synclip", "mono": True,
                                  "mono_lengths": "prior"}}
        adir = str(root / "analysis")
        os.makedirs(adir)
        kw = dict(analysis_dir=adir, cache_dir=str(root / "cache"), data_dir=str(root / "data"), registry=reg,
                  log=lambda *a: None)
        full = pb.build("synclip", fit_opts={"passes": 1}, **kw)
        for p in [os.path.join(adir, "synclip_1cam_a_pose24.racket_rigid.json"),
                  personsmod.path(adir, "synclip_1cam_a")]:
            with open(p, "w") as fh:                               # stale outputs of an earlier build
                fh.write("{}")
        mono_doc = pb.build("synclip", mono="a", out_clip="synclip_1cam_a", mono_opts={"passes": 1}, **kw)
        yield {"root": root, "full": full, "mono": mono_doc, "players": players, "adir": adir, "kw": kw}
    finally:
        os.chdir(old)


def _jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_build_mono_writes_a_valid_doc_and_cache(mono_built):
    d = _jload(mono_built["mono"])
    assert schema24.validate_analysis24(d) == []
    c = docmod.load_debug_cache(track.cache_path("synclip_1cam_a", mono_built["kw"]["cache_dir"]))
    assert c["build_id"] == d["meta"]["pose24"]["build_id"] and c["format"] == docmod.CACHE_FORMAT
    full = _jload(mono_built["full"])
    assert c["build_id"] != full["meta"]["pose24"]["build_id"]
    m = d["meta"]["pose24"]["mono"]
    assert m == {"view": "a", "lengths": "oracle", "prior": None, "identity": "oracle (full-rig tracking)"}
    assert d["meta"]["pose24"]["players"]["P1"]["coverage"] > 0.9
    assert "monocular" in d["meta"]["label"] and d["meta"]["label"] == "SMPL-28 · monocular (no prior) · 1 cam a"
    pvm = d["meta"]["pose_variant"]
    assert pvm["views"] == ["a"] and pvm["timeline_of"] == "synclip" and pvm["time_ref_view"] == "a"
    assert pvm["offsets"] == {"a": full["meta"]["pose_variant"]["offsets"]["a"]}
    assert [cm["name"] for cm in d["meta"]["cameras"]] == ["a"]
    assert all(fr["racket"] is None for p in d["players"] for fr in p["frames"])
    assert c["views"] == ["a"] and c["timeline_of"] == "synclip" and c["extra"]["P1"]["X"].shape == (FM, 35, 3)
    assert c["extra"]["P1"]["det_index"].shape == (1, FM) and c["extra"]["P1"]["lr_swapped"].shape == (1, FM)
    assert set(c["players_lengths"]["P1"]) == {"edges", "soft", "stature"}


def test_build_mono_camera_has_the_full_build_image_size(mono_built):
    d, full = _jload(mono_built["mono"]), _jload(mono_built["full"])
    (cm,) = d["meta"]["cameras"]
    fm = next(c for c in full["meta"]["cameras"] if c["name"] == "a")
    assert isinstance(cm["width"], int) and isinstance(cm["height"], int)
    assert (cm["width"], cm["height"]) == (fm["width"], fm["height"])


def test_build_mono_image_size_from_the_full_doc_without_cache_sizes(mono_built, tmp_path, monkeypatch):
    """A full cache written before image_size was stored: the size comes from the full doc's meta.cameras."""
    load = docmod.load_debug_cache
    monkeypatch.setattr(docmod, "load_debug_cache", lambda p: {k: v for k, v in load(p).items() if k != "image_size"})
    kw = dict(mono_built["kw"], cache_dir=mono_built["kw"]["cache_dir"])
    out = mono.build_mono("synclip", "b", "synclip_1cam_bx", passes=1, **{k: kw[k] for k in ("analysis_dir",
                                                                                             "cache_dir", "log")})
    try:
        (cm,) = _jload(out)["meta"]["cameras"]
        fm = next(c for c in _jload(mono_built["full"])["meta"]["cameras"] if c["name"] == "b")
        assert (cm["width"], cm["height"]) == (fm["width"], fm["height"]) and isinstance(cm["width"], int)
    finally:
        os.remove(out)
        os.remove(track.cache_path("synclip_1cam_bx", kw["cache_dir"]))


def test_build_mono_pose_is_near_the_full_build(mono_built):
    d, full = _jload(mono_built["mono"]), _jload(mono_built["full"])
    pm = next(p for p in d["players"] if p["id"] == "P1")
    pf = next(p for p in full["players"] if p["id"] == "P1")
    dx = [np.hypot(a["position"][0] - b["position"][0], a["position"][1] - b["position"][1])
          for a, b in zip(pm["frames"], pf["frames"]) if a["position"] and b["position"]]
    assert np.median(dx) < 0.15


def test_build_mono_writes_no_persons_or_racket_sidecars(mono_built):
    adir = mono_built["adir"]
    assert not os.path.exists(os.path.join(adir, "synclip_1cam_a_pose24.racket_rigid.json"))
    assert not os.path.exists(personsmod.path(adir, "synclip_1cam_a"))


def test_full_build_cache_stores_players_lengths(mono_built):
    c = docmod.load_debug_cache(track.cache_path("synclip", mono_built["kw"]["cache_dir"]))
    full = _jload(mono_built["full"])
    st = c["players_lengths"]["P1"]
    assert set(st) == {"edges", "soft", "stature"}
    assert st["edges"] == pytest.approx(full["meta"]["pose24"]["players"]["P1"]["limb_lengths_m"])
    L = mono.lengths_for("oracle", c, "P1")
    assert L["edges"] == pytest.approx(st["edges"]) and L["stature"]["measured_m"] == st["stature"]["measured_m"]


def test_build_mono_needs_the_full_cache(mono_built, tmp_path):
    kw = dict(mono_built["kw"], cache_dir=str(tmp_path / "nocache"), analysis_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="synclip"):
        pb.build("synclip", mono="a", out_clip="synclip_1cam_a", **kw)
    with pytest.raises(ValueError):
        pb.build("synclip", mono="a", **mono_built["kw"])                 # out_clip required


def _mono_prior_file(root, cache, build_id, view="a"):
    from analysis.pose24.priors import common
    F = int(cache["frame_count"])
    meta = {"method": "motionbert-lite", "clip": "synclip", "view": view, "timeline_of": "synclip",
            "fps": float(cache["fps"]), "frame_count": F, "build_id": build_id}
    path = os.path.join(str(root), "prior", f"{view}_motionbert-lite_{build_id}.npz")
    common.write_prior(path, meta, {"P1": {"kp": np.arange(3), "rel_cam": np.zeros((F, 3, 3)), "conf": np.zeros(F)}})
    return path


def test_prior_build_id_mismatch_refused(mono_built, tmp_path):
    """A prior made from another build of the clip (its meta build_id != the cache's) is refused before any fit."""
    kw = dict(mono_built["kw"], analysis_dir=str(tmp_path / "an"), cache_dir=mono_built["kw"]["cache_dir"])
    cache = docmod.load_debug_cache(track.cache_path("synclip", kw["cache_dir"]))
    p = _mono_prior_file(tmp_path, cache, "notthisbuild")
    with pytest.raises(ValueError, match="build_id"):
        mono.build_mono("synclip", "a", "synclip_1cam_a_mb", prior_path=p, analysis_dir=kw["analysis_dir"],
                        cache_dir=kw["cache_dir"], passes=1, log=lambda *a: None)
    assert not os.path.exists(pb.output_path(kw["analysis_dir"], "synclip_1cam_a_mb"))


def test_check_prior_meta_accepts_matching_and_warns_without_build_id():
    mono.check_prior_meta({"build_id": "b1", "view": "a"}, "b1", "a", "p.npz")
    with pytest.warns(UserWarning, match="build_id"):
        mono.check_prior_meta({"view": "a"}, "b1", "a", "p.npz")
    with pytest.raises(ValueError, match="build_id"):
        mono.check_prior_meta({"build_id": "b2", "view": "a"}, "b1", "a", "p.npz")
    with pytest.raises(ValueError, match="view"):
        mono.check_prior_meta({"build_id": "b1", "view": "b"}, "b1", "a", "p.npz")


def test_read_prior_meta(tmp_path):
    cache = {"frame_count": 4, "fps": 60.0}
    p = _mono_prior_file(tmp_path, cache, "bX")
    assert mono.read_prior_meta(p)["build_id"] == "bX"


def test_build_mono_doc_written_last(mono_built, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(docmod, "save_debug_cache", boom)
    with pytest.raises(OSError):
        pb.build("synclip", mono="a", out_clip="synclip_1cam_x", mono_opts={"passes": 1},
                 **dict(mono_built["kw"], analysis_dir=str(tmp_path)))
    assert not os.path.exists(pb.output_path(str(tmp_path), "synclip_1cam_x"))


def test_registry_mono_entry_builds_via_the_mono_path(mono_built, tmp_path, monkeypatch, capsys):
    """`pose24 build --all` / --clip on a registry entry with "mono": true -> build_mono (base clip, its 1 view)."""
    kw = dict(mono_built["kw"], analysis_dir=str(tmp_path))
    out = pb.build("synclip_1cam_b", mono_opts={"passes": 1}, **kw)
    d = _jload(out)
    assert schema24.validate_analysis24(d) == []
    assert d["clip_id"] == "synclip_1cam_b" and d["meta"]["pose24"]["mono"]["view"] == "b"
    assert d["meta"]["pose24"]["mono"]["lengths"] == "prior"
    with pytest.raises(ValueError, match="mono"):
        config.resolve("synclip_1cam_b", registry=kw["registry"])
    from analysis.pose24 import __main__ as cli
    monkeypatch.setattr(config, "load_registry", lambda *a, **k: kw["registry"])
    calls = []
    monkeypatch.setattr(pb, "build", lambda c, **k: calls.append(c))
    assert cli.main(["build", "--all"]) == 0
    assert calls == ["synclip", "synclip_1cam_b"]
    assert cli.main(["list", "--analysis-dir", str(tmp_path)]) == 0
    row = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("synclip_1cam_b"))
    assert "1 views" in row and "mono" in row


def test_doc_label_mono():
    assert docmod.doc_label("smpl-28", ["a"], {}, method="wham") == "SMPL-28 · monocular (wham prior) · 1 cam a"
    assert docmod.doc_label("smpl-28", ["a"], {"e": 1.0}) == "SMPL-28 · monocular (no prior) · 1 cam a"
    assert docmod.doc_label("smpl-28", ["a", "b"], {}) == "SMPL-28 · triangulated · 2 cams (a, b)"


def test_cli_build_mono_arguments(monkeypatch):
    from analysis.pose24 import __main__ as cli
    monkeypatch.setattr(config, "load_registry", lambda *a, **k: {"c1": {}, "c2": {}})
    calls = []
    monkeypatch.setattr(pb, "build", lambda c, **k: calls.append((c, k)))
    for bad in (["build", "--clip", "c1", "--mono", "a"],
                ["build", "--clip", "c1", "--clip", "c2", "--mono", "a", "--out-clip", "x"],
                ["build", "--all", "--mono", "a", "--out-clip", "x"],
                ["build", "--clip", "c1", "--mono", "a", "--out-clip", "x", "--views", "a,b"],
                ["build", "--clip", "c1", "--mono", "a", "--out-clip", "x", "--mono-lengths", "wham"]):
        with pytest.raises(SystemExit):
            cli.main(bad)
    assert not calls
    assert cli.main(["build", "--clip", "c1", "--mono", "a", "--out-clip", "c1_1cam_a", "--mono-lengths", "prior",
                     "--mono-prior", "p.npz"]) == 0
    c, k = calls[0]
    assert c == "c1" and k["mono"] == "a" and k["out_clip"] == "c1_1cam_a"
    assert k["mono_opts"] == {"lengths": "prior", "prior_path": "p.npz"}


def test_export_mono_is_a_real_mono_build(mono_built, tmp_path):
    from analysis.pose24.camsubsets import export as ex
    kw = mono_built["kw"]
    regp = tmp_path / "clips.json"
    regp.write_text(json.dumps({"clips": {"synclip": kw["registry"]["synclip"]}}, indent=2) + "\n", encoding="utf-8")
    out = ex.export("synclip", ("d",), registry_path=str(regp), method="mono-len-prior", mono_opts={"passes": 1},
                    name="synclip_1cam_d_prior",
                    log=lambda *a: None, analysis_dir=str(tmp_path), cache_dir=kw["cache_dir"])
    reg = config.load_registry(str(regp))
    assert reg["synclip_1cam_d_prior"] == {"base": "synclip", "views": ["d"], "timeline_of": "synclip", "mono": True,
                                           "mono_lengths": "prior"}
    d = _jload(out)
    assert out == pb.output_path(str(tmp_path), "synclip_1cam_d_prior") and schema24.validate_analysis24(d) == []
    assert d["meta"]["pose24"]["mono"]["lengths"] == "prior" and d["meta"]["pose24"]["mono"]["view"] == "d"
    assert "synclip_1cam_d_prior" in config.all_clips(reg)


def test_cvdebug_sidecar_of_the_mono_build(mono_built):
    """The exported 1-camera build is viewable: cvdebug builds a JsonUtility-safe sidecar, the other cameras of the
    base clip are held out, and every frame is 'prior' (no second view)."""
    from analysis import cvdebug
    kw = mono_built["kw"]
    reg = dict(kw["registry"], synclip_1cam_a={"base": "synclip", "views": ["a"], "timeline_of": "synclip",
                                                "mono": True, "mono_lengths": "oracle"})
    out, sc = cvdebug.build("synclip_1cam_a", "pose24", analysis_dir=kw["analysis_dir"], cache_dir=kw["cache_dir"],
                            data_dir=kw["data_dir"], registry=reg, write=True, log=lambda *a: None)
    assert os.path.exists(out) and cvdebug.validate_sidecar(sc) == []
    cams = {c["name"]: c for c in sc["cameras"]}
    assert set(cams) == {"a", "b", "c", "d"} and not cams["a"]["held_out"]
    assert all(cams[v]["held_out"] for v in "bcd")
    full = docmod.load_debug_cache(track.cache_path("synclip", kw["cache_dir"]))
    assert cams["c"]["offset"] == pytest.approx(full["offsets_all"]["c"], abs=1e-3) and sc["proxy_clip"] == "synclip"
    p1 = next(p for p in sc["players"] if p["id"] == "P1")
    assert len(p1["det_index"]) == 4 * FM and cvdebug.STATE_PRIOR in p1["frame_state"]
    assert cvdebug.STATE_MEASURED not in p1["frame_state"]            # measured = hips seen by >= 2 views
    det = np.array(p1["det_index"]).reshape(4, FM)
    assert (det[1:] == -1).all()                                        # only camera a carries detections


def test_eval_mono_build_against_its_full_clip(mono_built):
    from analysis.pose24 import eval as ev
    kw = mono_built["kw"]
    res = ev.evaluate("synclip_1cam_a", reference="synclip", posefit=False, analysis_dir=kw["analysis_dir"],
                      cache_dir=kw["cache_dir"], out_dir=str(mono_built["root"] / "eval"), log=lambda *a: None)
    assert set(res["heldout"]) == {"b", "c", "d"} and res["views"] == ["a"]
    assert np.isfinite(res["vs_reference"]["pelvis_xz_cm"]["median"])


def test_image_size_with_none_entries_falls_back_to_the_doc(tmp_path):
    doc = tmp_path / "full.json"
    doc.write_text(json.dumps({"meta": {"cameras": [{"name": "a", "width": 1920, "height": 1080}]}}), encoding="utf-8")
    assert mono._image_size({"image_size": {"a": [None, None]}}, "a", str(doc)) == [1920, 1080]
    assert mono._image_size({"image_size": {"a": [640, None]}}, "a", str(doc)) == [1920, 1080]
    assert mono._image_size({"image_size": {"a": [640, 480]}}, "a", str(doc)) == [640, 480]
    assert mono._image_size({"image_size": {"a": [None, None]}}, "a", str(tmp_path / "none.json")) is None


def test_mono_settings_record_the_mono_constants():
    st = mono.settings()
    assert st["depth_fit_sigma_m"] == mono.DEPTH_FIT_SIGMA and st["default_ankle_h_m"] == mono.DEFAULT_ANKLE_H
    assert st["no_prior_depth_sigma_m"] == mono.NO_PRIOR_DEPTH_SIGMA
    assert st["prior_depth_sigma_m"] == mono.PRIOR_DEPTH_SIGMA
    assert st["foot_h_m"] and st["contact"] == {"v_frac": mono.CONTACT_V_FRAC, "lower_foot_px": mono.LOWER_FOOT_PX}
    json.dumps(st, allow_nan=False)
