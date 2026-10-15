import json
import os

import numpy as np
import pytest

from analysis import cvdebug as cd
from analysis import racket_rigidfit as rf
from analysis.pose24 import doc as docmod
import cvdebug_synth as cs
import pose24_synth as syn


def _cam():
    K, R, t = syn.rig(("b",))["b"]
    return K, R, t


def test_unity_camera_reproduces_the_pinhole_projection():
    K, R, t = _cam()
    pos, quat, vfov = cd.unity_camera(K, R, t)
    X = np.array([[0.0, 1.0, -3.0], [1.2, 0.2, 2.0], [-2.0, 1.7, 0.5]])      # Y-up analysis world
    uv_ref = rf.Camera(K, R, t).project(X)[0]
    assert np.allclose(cd.unity_project(pos, quat, K, X), uv_ref, atol=1e-6)
    assert vfov == pytest.approx(np.degrees(2 * np.arctan(0.5 * K[1, 2] * 2 / K[1, 1])))
    assert abs(np.linalg.norm(quat) - 1) < 1e-9


def test_project_behind_camera_is_nan():
    K, R, t = _cam()
    cam = rf.Camera(K, R, t)
    centre = (-R.T @ t) * np.array([1, -1, 1])                 # camera centre in the Y-up world
    fwd = R[2] * np.array([1, -1, 1])
    X = np.array([centre + 3 * fwd, centre - 3 * fwd, [np.nan] * 3])
    uv = cd.project(cam, X)
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all() and np.isnan(uv[2]).all()


def test_bits_round_trip():
    m = np.random.default_rng(0).random((7, 5, 6)) > 0.5
    bits = cd.pack_bits(m)
    assert bits.shape == (7, 5) and bits.max() < 2 ** 6
    assert (cd.unpack_bits(bits, 6) == m).all()
    with pytest.raises(ValueError):
        cd.pack_bits(np.zeros((1, 1, cd.MAX_VIEWS + 1), bool))


def test_quantise_uses_minus_one_for_unknown():
    out = cd.quantise(np.array([1.24, np.nan, 0.0, 12.35]), 10)
    assert out.tolist() == [12, -1, 0, 124]


def test_variant_joint_map_by_name():
    names = ["pelvis", "left_hip", "left_shoulder", "right_wrist", "left_big_toe", "spine1", "left_thumb1"]
    m = cd.variant_joint_keypoints(names)
    assert m == [[], [], [5], [10], [17], [], []]


def test_pose24_sources_trunk_joints_have_no_error_source():
    from analysis import skeleton24 as sk
    mask_src, err_src = cd.pose24_joint_sources(sk.SMPL28_NAMES)
    i = sk.SMPL28_NAMES.index
    assert err_src[i("spine2")] == [] and mask_src[i("spine2")] == [5, 6, 11, 12]
    assert err_src[i("left_wrist")] == [9] and err_src[i("left_hip")] == [11, 12]


def test_doc_world_joints_adds_position_back():
    doc = {"skeleton": {"joint_names": ["pelvis", "a"]}, "players": [{"id": "P1", "frames": [
        {"joints_flat": [0, 0.9, 0, 0.8, 0.1, 1.0, 0.2, 0.0], "position": [1.0, 2.0]}, {"joints_flat": None}]}]}
    J, C = cd.doc_world_joints(doc)["P1"]
    assert np.allclose(J[0, 0], [1.0, 0.9, 2.0]) and np.isnan(J[0, 1]).all() and np.isnan(J[1]).all()
    assert C[0].tolist() == [0.8, 0.0]


@pytest.fixture(scope="module")
def syn_built(tmp_path_factory):
    return cs.built_clip(tmp_path_factory.mktemp("cvd"))


def _build(b, clip="syn", **kw):
    return cd.build(clip, variant="pose24", analysis_dir=b["adir"], cache_dir=b["cache_dir"],
                    data_dir=b["data_dir"], registry=b["reg"], log=lambda *a: None, **kw)


def test_pose24_sidecar_shapes_and_json_safety(syn_built):
    path, sc = _build(syn_built)
    assert path.endswith("syn_pose24.cvdebug.json") and os.path.exists(path)
    raw = open(path, encoding="utf-8").read()
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"{c} in JSON"))
    assert raw == json.dumps(json.loads(raw), separators=(",", ":"))     # compact: no ", " / ": " separators
    assert not os.path.exists(path + ".tmp")
    assert cd.validate_sidecar(sc) == [] and cd.validate_sidecar(json.loads(raw)) == []
    F, J, V = sc["frame_count"], len(sc["joint_names"]), len(sc["cameras"])
    doc_F = json.load(open(syn_built["full"], encoding="utf-8"))["meta"]["frame_count"]
    assert (F, J, V) == (doc_F, 28, 4) and sc["format"] == "cvdebug/1"
    assert syn_built["F"] - 3 <= F <= syn_built["F"]        # the tracker trims the largest sync shift
    for p in sc["players"]:
        for k in ("seen_mask", "view_mask", "err_dpx", "sigma_mm", "filled"):
            assert len(p[k]) == F * J, k
        assert len(p["det_index"]) == V * F and len(p["frame_state"]) == F and len(p["frame_err_px"]) == F
        view = np.array(p["view_mask"]).reshape(F, J)
        seen = np.array(p["seen_mask"]).reshape(F, J)
        assert ((view & ~seen) == 0).all()                  # an inlier view was always seen
        assert np.median(np.array(p["err_dpx"])[np.array(p["err_dpx"]) >= 0]) < 50   # < 5 px on 0.5 px noise
        assert (np.array(p["frame_state"]) == cd.STATE_MEASURED).mean() > 0.9


def test_pose24_trunk_joints_have_unknown_error(syn_built):
    _, sc = _build(syn_built, write=False)
    J = len(sc["joint_names"])
    e = np.array(sc["players"][0]["err_dpx"]).reshape(-1, J)
    assert (e[:, sc["joint_names"].index("spine2")] == -1).all()
    assert (e[:, sc["joint_names"].index("left_knee")] >= 0).mean() > 0.9


def test_pose24_det_index_and_offsets(syn_built):
    _, sc = _build(syn_built, write=False)
    cams = {c["name"]: c for c in sc["cameras"]}
    assert cams["c"]["offset"] == pytest.approx(2.0, abs=0.6) and cams["c"]["offset_source"] == "stored"
    F = sc["frame_count"]
    di = np.array(sc["players"][0]["det_index"]).reshape(len(sc["cameras"]), F)
    assert (di[:, 5:F - 5] >= 1).mean() > 0.9               # detection 0 is the bystander


def test_pose24_det_shift_is_the_trackers_integer_shift(syn_built):
    """det_index is a detection in raw frame f + cache shifts[v] (np.round, half to even), which can differ from
    the video frame floor(f + offset + 0.5): offset exactly 0.5 -> shift 0, video frame f + 1."""
    from analysis.pose24 import track
    cache = docmod.load_debug_cache(track.cache_path("syn", syn_built["cache_dir"]))
    with open(cd.doc_path(syn_built["adir"], "syn", "pose24"), "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    calib = os.path.join(syn_built["data_dir"], "calib")
    sc = cd.build_pose24(doc, cache, ["a", "b", "c", "d"], calib, {})
    cams = {c["name"]: c for c in sc["cameras"]}
    assert {v: cams[v]["det_shift"] for v in "abcd"} == {v: int(cache["shifts"][v]) for v in "abcd"}
    assert cams["c"]["det_shift"] == 2
    cache["offsets_all"] = dict(cache.get("offsets_all") or cache["offsets"], b=0.5)
    cache["shifts"] = dict(cache["shifts"], b=0)
    cams = {c["name"]: c for c in cd.build_pose24(doc, cache, ["a", "b", "c", "d"], calib, {})["cameras"]}
    assert cams["b"]["offset"] == 0.5 and cams["b"]["det_shift"] == 0


def test_pose24_contact_flags_come_from_the_cache(syn_built):
    from analysis.pose24 import track
    cache = docmod.load_debug_cache(track.cache_path("syn", syn_built["cache_dir"]))
    _, sc = _build(syn_built, write=False)
    F = sc["frame_count"]
    for p in sc["players"]:
        want = np.asarray(cache["extra"][p["id"]]["contact"])[:F].astype(np.int64)
        assert p["contact"] == want.ravel().tolist() and len(p["contact"]) == F * 6
    assert cd.validate_sidecar(sc) == []

def test_subset_build_marks_held_out_views_and_takes_their_offsets_from_the_timeline(syn_built):
    _, sc = _build(syn_built, clip="syn_2cam", write=False)
    cams = {c["name"]: c for c in sc["cameras"]}
    assert [c["name"] for c in sc["cameras"]] == ["a", "b", "c", "d"]
    assert cams["a"]["held_out"] and cams["d"]["held_out"] and not cams["b"]["held_out"]
    assert cams["a"]["offset_source"] == "timeline"
    F, J = sc["frame_count"], len(sc["joint_names"])
    bits = cd.unpack_bits(np.array(sc["players"][0]["seen_mask"]).reshape(F, J), 4)
    assert not bits[..., 0].any() and not bits[..., 3].any() and bits[..., 1].any()
    assert (cams["a"]["width"], cams["a"]["height"]) == (2160, 3840)     # held-out: size from its calib file
    assert (cams["d"]["width"], cams["d"]["height"]) == (2160, 3840)


def test_pose24_refuses_mismatched_cache(syn_built, tmp_path):
    d = json.load(open(syn_built["full"], encoding="utf-8"))
    d["meta"]["pose24"]["build_id"] = "0" * 32
    adir = tmp_path / "an"
    adir.mkdir()
    docmod.write_json(str(adir / "syn_pose24.analysis.json"), d)
    with pytest.raises(cd.PairingError, match="build_id"):
        cd.build("syn", variant="pose24", analysis_dir=str(adir), cache_dir=syn_built["cache_dir"],
                 data_dir=syn_built["data_dir"], registry=syn_built["reg"], log=lambda *a: None)


@pytest.mark.parametrize("p24", [{"build_id": ""}, {}, None])
def test_pose24_doc_without_build_id_is_a_pairing_error(syn_built, tmp_path, p24):
    d = json.load(open(syn_built["full"], encoding="utf-8"))
    if p24 is None:
        d["meta"].pop("pose24")
    else:
        d["meta"]["pose24"] = p24
    adir = tmp_path / "an"
    adir.mkdir()
    docmod.write_json(str(adir / "syn_pose24.analysis.json"), d)
    with pytest.raises(cd.PairingError, match="build_id"):
        cd.build("syn", variant="pose24", analysis_dir=str(adir), cache_dir=syn_built["cache_dir"],
                 data_dir=syn_built["data_dir"], registry=syn_built["reg"], write=False, log=lambda *a: None)


def test_excluded_view_has_no_bits_and_banner_data(syn_built, tmp_path, monkeypatch):
    """Fake a view-health drop of view d (as frontright_21): masks carry no bit for it, used_frac 0."""
    cache = docmod.load_debug_cache(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"))
    doc = json.load(open(syn_built["full"], encoding="utf-8"))
    k = cache["views"].index("d")
    for pid, e in cache["extra"].items():
        e["inlier_mask"][..., k] = False
    cache["obs"] = {pid: {v: (o if v != "d" else np.full_like(o, np.nan)) for v, o in ob.items()}
                    for pid, ob in cache["obs"].items()}
    cache["excluded"] = {"d": 927.0}
    sc = cd.build_pose24(doc, cache, ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                         {v: None for v in "abcd"})
    cams = {c["name"]: c for c in sc["cameras"]}
    assert cams["d"]["excluded"] and cams["d"]["health_px"] == 927.0
    F, J = sc["frame_count"], len(sc["joint_names"])
    assert not cd.unpack_bits(np.array(sc["players"][0]["view_mask"]).reshape(F, J), 4)[..., 3].any()
    assert [v["used_frac"] for v in sc["stats"]["views"] if v["name"] == "d"] == [0.0]


def test_body_only_drops_face_joints(syn_built):
    _, sc = _build(syn_built, write=False, body_only=True)
    assert len(sc["joint_names"]) == cd.BODY_ONLY_J and sc["joint_names"][-1] == "right_wrist"
    assert len(sc["players"][0]["err_dpx"]) == sc["frame_count"] * cd.BODY_ONLY_J


def test_validate_sidecar_catches_unity_hazards():
    ok = {"format": "cvdebug/1", "cameras": [{"K": [1.0] * 9}], "players": [], "stats": {"views": []}}
    assert cd.validate_sidecar(ok) == []
    assert cd.validate_sidecar(dict(ok, x=[[1, 2]]))                 # nested list
    assert cd.validate_sidecar(dict(ok, x=[1, None]))                # null in an array
    assert cd.validate_sidecar(dict(ok, x=float("nan")))             # NaN
    assert cd.validate_sidecar(dict(ok, offsets={"back_051": 0.5}))  # free-key dict (not a schema field)


# --- variant path (base / posev2 / posefit): measured by reprojecting the doc's 3D into every camera -------------

def _variant_doc_from_pose24(path, drop_variant_meta=False):
    """A 'posefit-like' doc: the pose24 doc with its pose24 meta removed (so the variant path runs)."""
    d = json.load(open(path, encoding="utf-8"))
    d["meta"].pop("pose24")
    if drop_variant_meta:
        d["meta"].pop("pose_variant")
    return d


def test_measure_reprojection_on_exact_data():
    K, R, t = _cam()
    cam = rf.Camera(K, R, t)
    Jw = np.array([[[0.0, 1.0, -3.0], [0.3, 1.2, -3.0]]])               # (1,2,3)
    uv = np.full((1, 133, 1, 2), np.nan)
    w = np.zeros((1, 133, 1))
    uv[0, 5, 0] = cd.project(cam, Jw[0, 0]) + [3.0, 4.0]                  # 5 px off
    w[0, 5, 0] = 0.9
    uv[0, 6, 0] = cd.project(cam, Jw[0, 1]) + [300.0, 0.0]                # seen but rejected (> 25 px)
    w[0, 6, 0] = 0.9
    seen, view, err, ferr = cd.measure_reprojection(Jw, uv, w, [cam], [[5], [6]])
    assert seen[0, :, 0].tolist() == [True, True] and view[0, :, 0].tolist() == [True, False]
    assert err[0, 0] == pytest.approx(5.0) and np.isnan(err[0, 1]) and ferr[0] == pytest.approx(5.0)


def test_variant_path_with_stored_offsets_uses_the_pose_v2_style_cache(syn_built, tmp_path):
    # the pose24 debug cache is a superset of the pose_v2 cache format: save it under the pose_v2 name
    import shutil
    cdir = tmp_path / "cache"
    cdir.mkdir()
    shutil.copy(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), cdir / "syn.npz")
    d = _variant_doc_from_pose24(syn_built["full"])
    sc = cd.build_variant(d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(cdir),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"))
    assert cd.validate_sidecar(sc) == []
    J = len(sc["joint_names"])
    e = np.array(sc["players"][0]["err_dpx"]).reshape(-1, J)
    li = sc["joint_names"].index("left_knee")
    assert (e[:, li] >= 0).mean() > 0.9 and np.median(e[e[:, li] >= 0, li]) < 60     # < 6 px
    assert (e[:, sc["joint_names"].index("pelvis")] == -1).all()
    assert {c["offset_source"] for c in sc["cameras"]} == {"stored"}
    assert all(v == -1 for v in sc["players"][0]["sigma_mm"])
    # no stored time_ref_view: the view whose stored offset is 0 (never just the first camera)
    zero = [v for v, o in d["meta"]["pose_variant"]["offsets"].items() if abs(o) < 1e-6]
    d["meta"]["pose_variant"].pop("time_ref_view")
    d["meta"]["pose_variant"]["offsets"] = {v: o for v, o in d["meta"]["pose_variant"]["offsets"].items() if v != "a"}
    sc = cd.build_variant(d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(cdir),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    assert sc["time_ref_view"] == ([v for v in zero if v != "a"] or [""])[0]


def _with_extra_joints(d, n_extra):
    """A >28-joint doc (as posefit's SMPL-X 58): n_extra joints appended at the pelvis."""
    nj = len(d["skeleton"]["joint_names"])
    d["skeleton"]["joint_names"] = list(d["skeleton"]["joint_names"]) + [f"extra{k}" for k in range(n_extra)]
    d["skeleton"].pop("parents", None)
    for p in d["players"]:
        for fr in p["frames"]:
            if fr.get("joints_flat") is not None:
                a = np.asarray(fr["joints_flat"], float).reshape(nj, 4)
                fr["joints_flat"] = np.vstack([a, np.repeat(a[:1], n_extra, axis=0)]).ravel().tolist()
    return d


def test_variant_doc_with_more_than_28_joints_defaults_to_body_only(syn_built, tmp_path):
    import shutil
    cdir = tmp_path / "cache"
    cdir.mkdir()
    shutil.copy(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), cdir / "syn.npz")
    d = _with_extra_joints(_variant_doc_from_pose24(syn_built["full"]), 30)
    kw = dict(cache_dir=str(cdir), pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    args = (d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"), {v: "" for v in "abcd"})
    sc = cd.build_variant(*args, **kw)
    assert len(sc["joint_names"]) == cd.BODY_ONLY_J
    assert len(sc["players"][0]["err_dpx"]) == sc["frame_count"] * cd.BODY_ONLY_J
    full = cd.build_variant(*args, all_joints=True, **kw)
    assert len(full["joint_names"]) == 58 and cd.validate_sidecar(full) == []
    small = cd.build_variant(_variant_doc_from_pose24(syn_built["full"]), *args[1:], **kw)
    assert len(small["joint_names"]) == 28                     # <= 28 joints: every joint by default


def test_cli_all_joints_flag_reaches_build(monkeypatch):
    monkeypatch.setattr(cd.config, "load_registry", lambda *a, **k: {"x": {"views": ["a"]}})
    seen = []
    monkeypatch.setattr(cd, "build", lambda clip, variant, **kw: seen.append(kw["all_joints"]) or ("p", {}))
    cd.main(["build", "--clip", "x", "--variant", "posefit"])
    cd.main(["build", "--clip", "x", "--variant", "posefit", "--all-joints"])
    assert seen == [False, True]


def test_variant_excluded_view_takes_its_offset_from_the_pose24_timeline(syn_built, tmp_path):
    """The variant doc stores no offset for an excluded view: the same clip's pose24 cache (offsets_all) gives it."""
    from analysis.pose24 import track
    import shutil
    cdir = tmp_path / "cache"
    cdir.mkdir()
    d = _variant_doc_from_pose24(syn_built["full"])
    pvm = d["meta"]["pose_variant"]
    pvm["offsets"].pop("c")
    pvm["excluded_views"] = {"c": 900.0}
    args = (d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"), {v: "" for v in "abcd"})
    kw = dict(cache_dir=str(cdir), pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    cams = {c["name"]: c for c in cd.build_variant(*args, **kw)["cameras"]}
    assert cams["c"]["offset_source"] == "none" and cams["c"]["offset"] == 0.0      # nothing to take it from
    shutil.copy(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), cdir / "syn_pose24.npz")
    want = track.load_timeline("syn", syn_built["cache_dir"])["offsets"]["c"]
    cams = {c["name"]: c for c in cd.build_variant(*args, **kw)["cameras"]}
    assert cams["c"]["offset_source"] == "timeline" and cams["c"]["offset"] == pytest.approx(want, abs=1e-3)
    assert cams["c"]["excluded"] and cams["a"]["offset_source"] == "stored"


def test_variant_path_without_offsets_estimates_them_from_raw_kp2d(syn_built, tmp_path):
    d = _variant_doc_from_pose24(syn_built["full"], drop_variant_meta=True)
    sc = cd.build_variant(d, "base", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(tmp_path),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    cams = {c["name"]: c for c in sc["cameras"]}
    assert cams["c"]["offset_source"] == "estimated" and cams["c"]["offset"] == 2
    assert sc["time_ref_view"] == ""                          # no stored offsets: no reference view to name
    assert all(c["det_shift"] == int(np.floor(c["offset"] + 0.5)) for c in sc["cameras"])   # what assign_raw used
    F = sc["frame_count"]
    assert sc["players"][0]["contact"] == [0] * (F * 6)
    di = np.array(sc["players"][0]["det_index"]).reshape(4, F)
    assert (di[:, 5:F - 5] >= 1).mean() > 0.9                 # the bystander (detection 0) is never assigned


def test_variant_cache_with_other_offsets_falls_back_to_raw(syn_built, tmp_path):
    import shutil
    cdir = tmp_path / "cache"
    cdir.mkdir()
    shutil.copy(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), cdir / "syn.npz")
    d = _variant_doc_from_pose24(syn_built["full"])
    d["meta"]["pose_variant"]["offsets"]["c"] = 5.0             # doc on another timeline than the cache
    logs = []
    sc = cd.build_variant(d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(cdir),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=logs.append)
    assert any("offsets differ" in s for s in logs)
    assert {c["name"]: c for c in sc["cameras"]}["c"]["offset"] == 5.0
    F, J = sc["frame_count"], len(sc["joint_names"])
    di = np.array(sc["players"][0]["det_index"]).reshape(4, F)
    assert (di[0, 5:F - 5] >= 1).mean() > 0.9                  # the raw path really measured view a
    seen = cd.unpack_bits(np.array(sc["players"][0]["seen_mask"]).reshape(F, J), 4)
    assert seen[..., 0].any() and (np.array(sc["players"][0]["err_dpx"]) >= 0).any()


def test_variant_cache_path_fills_det_index_from_the_raw_detections(syn_built, tmp_path):
    import shutil
    cdir = tmp_path / "cache"
    cdir.mkdir()
    shutil.copy(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), cdir / "syn.npz")
    d = _variant_doc_from_pose24(syn_built["full"])
    sc = cd.build_variant(d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(cdir),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    F = sc["frame_count"]
    for p in sc["players"]:
        di = np.array(p["det_index"]).reshape(4, F)
        assert (di[:, 5:F - 5] >= 1).mean() > 0.9             # the bystander (detection 0) is never the player
    d0 = np.array(sc["players"][0]["det_index"]).reshape(4, F)[:, 5:F - 5]
    d1 = np.array(sc["players"][1]["det_index"]).reshape(4, F)[:, 5:F - 5]
    assert ((d0 != d1) | (d0 < 0)).mean() > 0.95               # the two players get different detections


def _cache_without_view(src, dst, drop):
    """Rewrite a debug cache without one view (as a 2-cam pose_v2 build would have)."""
    with np.load(src) as z:
        meta = json.loads(str(z["meta"]))
        keep = [k for k, v in enumerate(meta["views"]) if v != drop]
        arrs = {}
        for key in z.files:
            if key.endswith("_uv") or key.endswith("_w"):
                arrs[key] = z[key][:, :, keep]
            elif key != "meta":
                arrs[key] = z[key]
    meta["views"] = [v for v in meta["views"] if v != drop]
    for k in ("offsets", "cams"):
        meta[k].pop(drop, None)
    np.savez(dst, meta=json.dumps(meta), **arrs)


def test_variant_view_outside_the_cache_is_measured_from_raw_kp2d(syn_built, tmp_path):
    cdir = tmp_path / "cache"
    cdir.mkdir()
    _cache_without_view(os.path.join(syn_built["cache_dir"], "syn_pose24.npz"), str(cdir / "syn.npz"), "d")
    d = _variant_doc_from_pose24(syn_built["full"])
    pvm = d["meta"]["pose_variant"]
    pvm["offsets"].pop("d")
    pvm["views"] = ["a", "b", "c"]
    sc = cd.build_variant(d, "posefit", ["a", "b", "c", "d"], os.path.join(syn_built["data_dir"], "calib"),
                          {v: "" for v in "abcd"}, cache_dir=str(cdir),
                          pose2d_dir=os.path.join(syn_built["data_dir"], "pose2d"), log=lambda *a: None)
    cams = {c["name"]: c for c in sc["cameras"]}
    assert cams["d"]["offset_source"] == "estimated" and cams["d"]["offset"] == 0 and cams["d"]["held_out"]
    assert cams["a"]["offset_source"] == "stored"
    F, J = sc["frame_count"], len(sc["joint_names"])
    view = cd.unpack_bits(np.array(sc["players"][0]["view_mask"]).reshape(F, J), 4)
    assert view[..., 3].any(axis=1).mean() > 0.9               # view d is measured (inliers) on most frames
    di = np.array(sc["players"][0]["det_index"]).reshape(4, F)
    assert (di[3, 5:F - 5] >= 1).mean() > 0.9


def test_cli_all_skips_missing_docs_and_isolates_failures(monkeypatch, capsys):
    reg = {"x": {"views": ["a"]}, "y": {"views": ["a"]}, "z": {"views": ["a"]}}
    monkeypatch.setattr(cd.config, "load_registry", lambda *a, **k: reg)
    calls = []

    def fake_build(clip, variant, **kw):
        calls.append((clip, variant))
        if clip == "y":
            raise FileNotFoundError("no analysis doc y")
        if clip == "z":
            raise RuntimeError("boom")
        return "p", {}
    monkeypatch.setattr(cd, "build", fake_build)
    rc = cd.main(["build", "--all", "--variant", "pose24"])
    out = capsys.readouterr().out
    assert rc == 1 and calls == [("x", "pose24"), ("y", "pose24"), ("z", "pose24")]
    assert "skip y" in out and "FAILED z" in out and "boom" in out


def test_eye_centre_seen_follows_the_fits_ring_minimum(syn_built):
    """review 1 L6: the fit (kpset.obs35) scores an eye-ring centre by the ring's minimum; cvdebug used the mean, so
    a ring with one unseen point was drawn "seen" in a view the fit treated as unobserved."""
    from analysis.pose24 import doc as docmod
    from analysis.pose24 import kpset, track
    b = syn_built
    doc = json.load(open(b["full"], encoding="utf-8"))
    cache = docmod.load_debug_cache(track.cache_path("syn", b["cache_dir"]))
    v0 = cache["views"][0]
    pid = cache["pids"][0]
    cache["obs"][pid][v0][:, kpset.LEFT_EYE_RING[0], 2] = 0.0        # one ring point never scored in view 0
    views_all = ["a", "b", "c", "d"]
    sc = cd.build_pose24(doc, cache, views_all, os.path.join(b["data_dir"], "calib"), {v: None for v in views_all})
    F, J = sc["frame_count"], len(sc["joint_names"])
    p = next(p for p in sc["players"] if p["id"] == pid)
    bits = cd.unpack_bits(np.array(p["seen_mask"]).reshape(F, J), len(views_all))
    k = views_all.index(v0)
    le, re = sc["joint_names"].index("left_eye"), sc["joint_names"].index("right_eye")
    assert bits[:, re, k].any() and not bits[:, le, k].any()
