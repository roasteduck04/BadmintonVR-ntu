"""Pose v2 / pose fit / SMPL-X fuse / appearance / eval — synthetic multi-camera checks."""
import numpy as np
import pytest

from analysis import appearance, eval_pose_mv, pose_fit, pose_v2 as pv, schema, smplx_fuse
from analysis import multiview as mv


def _look_at(center, target=(0.0, -1.0, 0.0)):
    """Court world is Y-down (floor Y=0, up = -Y); OpenCV camera: x right, y down, z forward."""
    c = np.asarray(center, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    down = np.array([0.0, 1.0, 0.0])
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right)
    d = np.cross(fwd, right)
    R = np.vstack([right, d, fwd])
    return R, -R @ c


@pytest.fixture
def rig():
    K = mv.intrinsics_from_fov(2160, 3840, 70.0)
    cams = {}
    for name, c in (("a", [0.0, -2.5, -10.0]), ("b", [-5.0, -2.2, -3.0]), ("c", [5.0, -2.2, -3.0]),
                    ("d", [-4.5, -2.2, 3.0])):
        R, t = _look_at(c)
        cams[name] = (K, R, t)
    return cams


def _proj(cams, views, X):
    uv, depth = pv.project(pv.proj_mats(cams, views), X)
    return uv, depth


def test_norm_score_calibration():
    assert pv.norm_score(pv.SCORE_MID) == pytest.approx(0.5)
    assert pv.norm_score(2.0) < 0.1 and pv.norm_score(3.5) > 0.9
    assert np.all(np.diff(pv.norm_score(np.linspace(0, 8, 50))) > 0)


def test_triangulate_batch_recovers_points_and_rejects_outlier_view(rig):
    views = sorted(rig)
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(-2, 2, 200), rng.uniform(-2, 0, 200), rng.uniform(-4, 0, 200)]
    uv, _ = _proj(rig, views, X)
    uv = uv + rng.normal(0, 0.5, uv.shape)
    uv[:100, 0] += 200.0                       # view "a" badly wrong for half the points
    w = np.full((200, len(views)), 0.95)
    Xh, conf, err, inl = pv.triangulate_batch(rig, views, uv, w)
    assert np.isfinite(Xh).all()
    assert np.median(np.linalg.norm(Xh - X, axis=1)) < 0.01
    assert not inl[:100, 0].any() and inl[100:, 0].mean() > 0.95
    assert (conf > 0.5).all()


def test_triangulate_batch_needs_two_confident_views(rig):
    views = sorted(rig)
    X = np.array([[0.5, -1.0, -2.0]])
    uv, _ = _proj(rig, views, X)
    w = np.array([[0.95, 0.1, 0.1, 0.1]])
    Xh, conf, _, _ = pv.triangulate_batch(rig, views, uv, w)
    assert np.isnan(Xh).all() and conf[0] == 0.0


def test_court_pos_uses_hip_plane_without_ankles(rig):
    K, R, t = rig["a"]
    hip = np.array([1.0, -pv.HIP_HEIGHT, 2.0])
    uv, _ = _proj(rig, ["a"], hip[None])
    kps = np.zeros((133, 3))
    kps[[11, 12], :2] = uv[0, 0]
    kps[[11, 12], 2] = 0.9
    assert pv.court_pos(kps, K, R, t) == pytest.approx((1.0, 2.0), abs=1e-6)


def test_estimate_offsets_recovers_shift(rig):
    views = ["a", "b"]
    F, shift = 200, 3
    t = np.arange(F + 10) / 60.0
    base = np.c_[1.5 * np.sin(2.1 * t), -1.0 + 0.3 * np.sin(5 * t), -2 + 1.2 * np.cos(1.7 * t)]
    offs = np.linspace(-0.4, 0.4, len(pv.SYNC_JOINTS))
    D = {v: np.full((F, 133, 3), np.nan, np.float32) for v in views}
    for k, j in enumerate(pv.SYNC_JOINTS):
        X = base + np.array([offs[k], 0.1 * k, -offs[k]])
        uv, _ = _proj(rig, views, X)
        D["a"][:, j, :2] = uv[:F, 0]
        D["b"][:, j, :2] = np.r_[np.full((shift, 2), np.nan), uv[:F - shift, 1]]
        D["a"][:, j, 2] = D["b"][:, j, 2] = 0.95
    off, _ = pv.estimate_offsets(rig, {"P1": D}, views, "a", max_offset=6, step=2, log=lambda *a: None)
    assert off["b"] == pytest.approx(shift, abs=0.15)


def _rest_motion(F=240, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(F) / 60.0
    trans = np.c_[1.2 * np.sin(2 * np.pi * 1.0 * t), np.zeros(F), -2 + 0.5 * np.cos(2 * np.pi * 0.7 * t)]
    rest = schema.REST_POSE - schema.REST_POSE[0] + np.array([0, 0.95, 0])
    J = rest[None] + trans[:, None, :] + rng.normal(0, noise, (F, schema.NUM_JOINTS, 3))
    return J, np.full((F, schema.NUM_JOINTS), 0.9)


def test_postprocess_is_zero_lag():
    J, C = _rest_motion(noise=0.01)
    Jp, Cp, interp = pv.postprocess(J, C, 60.0)
    clean, _ = _rest_motion(noise=0.0)
    x, y = np.diff(clean[:, 0, 0]), np.diff(Jp[:, 0, 0])
    lags = range(-4, 5)
    corr = [np.dot(x[max(0, -L):len(x) - max(0, L)], y[max(0, L):len(y) - max(0, -L)]) for L in lags]
    assert list(lags)[int(np.argmax(corr))] == 0
    assert np.abs(Jp[:, 0] - clean[:, 0]).mean() < 0.01
    assert not interp.any()


def test_fill_gaps_marks_interpolated_and_respects_max_gap():
    J, C = _rest_motion(F=120)
    J[20:25, 5] = np.nan
    J[50:50 + pv.MAX_GAP + 5, 6] = np.nan
    Jf, Cf, filled = pv.fill_gaps(J, C)
    assert filled[20:25, 5].all() and np.isfinite(Jf[20:25, 5]).all()
    assert np.isnan(Jf[50:50 + pv.MAX_GAP + 5, 6]).all()
    assert (Cf[20:25, 5] < C[20:25, 5]).all()


def test_enforce_bones_gives_constant_lengths():
    J, C = _rest_motion(F=60, noise=0.02)
    L = pv.median_bone_lengths(J, C, schema.PARENTS)
    Je = pv.enforce_bones(J, L, schema.PARENTS)
    for j, target in L.items():
        d = np.linalg.norm(Je[:, j] - Je[:, schema.PARENTS[j]], axis=1)
        assert np.allclose(d, target, atol=1e-9)


def test_frames_from_world_flags_and_body_local():
    J, C = _rest_motion(F=10)
    interp = np.zeros(10, bool)
    interp[3] = True
    J[7] = np.nan
    fr = pv.frames_from_world(J, C, interp)
    assert fr[3]["interpolated"] and not fr[2]["interpolated"]
    assert fr[7]["joints_flat"] is None
    a = np.array(fr[0]["joints_flat"]).reshape(-1, 4)
    assert a[0, 0] == pytest.approx(0) and a[0, 2] == pytest.approx(0)
    assert fr[0]["position"] == pytest.approx([J[0, 0, 0], J[0, 0, 2]], abs=1e-4)


def _fit_problem(rig, T=6, noise_init=0.05, seed=2):
    rng = np.random.default_rng(seed)
    views = sorted(rig)
    K = len(pose_fit.K_IDX)
    base = np.c_[rng.uniform(-0.3, 0.3, K), -rng.uniform(0.2, 1.7, K), rng.uniform(-0.3, 0.3, K)]
    base[pose_fit._FEET, 1] = -0.05
    Xt = np.stack([base + np.array([0.02 * t, 0, -2.0]) for t in range(T)])
    uv, _ = _proj(rig, views, Xt)
    w = np.full(uv.shape[:-1], 0.95)
    ok = np.ones((T, K), bool)
    L = pose_fit.bone_lengths(Xt, ok)
    P = pv.proj_mats(rig, views)
    X0 = Xt + rng.normal(0, noise_init, Xt.shape)
    X0[..., 1] = np.minimum(X0[..., 1], -0.01)
    return P, uv, w, Xt, X0, L


def test_pose_fit_jacobian_matches_finite_differences(rig):
    P, uv, w, Xt, X0, L = _fit_problem(rig, T=4)
    prob = pose_fit.WindowProblem(P, uv, w, X0, np.ones(4, bool), L)
    x = X0.ravel()
    J = prob.jacobian(x).toarray()
    eps = 1e-6
    rng = np.random.default_rng(3)
    for i in rng.choice(len(x), 25, replace=False):
        dx = np.zeros_like(x)
        dx[i] = eps
        num = (prob.residuals(x + dx) - prob.residuals(x - dx)) / (2 * eps)
        assert np.allclose(J[:, i], num, atol=1e-3 * max(1.0, np.abs(num).max()))


def test_pose_fit_recovers_skeleton_from_noisy_init(rig):
    P, uv, w, Xt, X0, L = _fit_problem(rig, T=6, noise_init=0.05)
    prob = pose_fit.WindowProblem(P, uv, w, X0, np.ones(6, bool), L)
    Xs = prob.solve()
    assert np.linalg.norm(Xs - Xt, axis=-1).mean() < 0.01 < np.linalg.norm(X0 - Xt, axis=-1).mean()


def test_smplx_cam_to_court_rel_convention(rig):
    K, R, t = rig["b"]
    rest = schema.REST_POSE - schema.REST_POSE[0]          # Y-up, pelvis-relative
    down = rest * np.array([1.0, -1.0, 1.0])                # court frame is Y-down
    j_cam = down @ R.T                                      # rotate into the camera
    back = smplx_fuse.cam_to_court_rel(j_cam, R)
    assert np.allclose(back, rest, atol=1e-9)


def test_smplx_fuse_views_drops_outlier():
    rest = schema.REST_POSE - schema.REST_POSE[0]
    c = np.full(schema.NUM_JOINTS, 0.9)
    wrong = rest.copy()
    wrong[:, 0] *= -1.0                                     # mirrored body
    fused, conf = smplx_fuse.fuse_views([(rest, c, 1.0), (rest + 0.01, c, 1.0), (wrong, c, 1.0)])
    assert np.abs(fused - rest).max() < 0.02
    assert smplx_fuse._iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_appearance_histograms():
    white = np.tile(np.array([[0, 10, 240]], np.uint8), (100, 1))
    red = np.tile(np.array([[2, 200, 200]], np.uint8), (100, 1))
    hw, hr = appearance.region_hist(white), appearance.region_hist(red)
    assert hw.sum() == pytest.approx(1) and hw[appearance.N_HUE] == pytest.approx(1)
    a, b = np.r_[hw, hw], np.r_[hr, hw]
    assert appearance.hellinger(a, a) == pytest.approx(0, abs=1e-9)
    assert appearance.hellinger(a, b) > 0.4


def test_eval_motion_stats_rigid_bones():
    J, _ = _rest_motion(F=120)
    idx = [schema.JOINT_NAMES.index(n) for n in eval_pose_mv.LIMBS]
    st = eval_pose_mv.motion_stats(J[:, idx], 60.0)
    assert st["bone_cv"] == pytest.approx(0, abs=1e-6)
    assert st["jerk"] is not None and st["skate"] is not None


def test_reject_outliers_never_blames_a_derived_joint():
    """Review 2 L13: pelvis = hip midpoint (conf = min of the hips); a bad, better-scored hip NaN'd the pelvis."""
    N = {n: i for i, n in enumerate(schema.JOINT_NAMES)}
    F = 60
    J = np.repeat(schema.REST_POSE[None].astype(float), F, axis=0)
    C = np.full((F, 58), 0.9)
    C[:, N["right_hip"]] = 0.8
    C[:, N["pelvis"]] = 0.8
    J[30, N["left_hip"], 0] += 0.25                                  # hip width blown up in one frame
    J[30, N["pelvis"]] = 0.5 * (J[30, N["left_hip"]] + J[30, N["right_hip"]])
    groups = [pv.joint_group(n) for n in schema.JOINT_NAMES]
    out = pv.reject_outliers(J, C, schema.PARENTS, groups)
    assert np.isfinite(out[30, N["pelvis"]]).all()
    assert not np.isfinite(out[30, N["left_hip"]]).all()


def test_eval_lovo_base_row_names_the_producer_it_measures():
    """Review 2 L3: the base LOVO rebuild runs the current build_real (schema 5.0, 58 joints); a frozen schema 4.0
    base doc beside it is another producer, so its held-out numbers go to their own row."""
    cur = {"schema_version": schema.SCHEMA_VERSION, "skeleton": {"joint_names": list(schema.JOINT_NAMES)}}
    old = {"schema_version": "4.0", "skeleton": {"joint_names": list(schema.JOINT_NAMES[:28])}}
    assert eval_pose_mv.lovo_base_row(cur) == "base"
    assert eval_pose_mv.lovo_base_row(old) == "base_rebuilt"


def test_eval_pv_clip_prefers_exact_then_longest_prefix():
    assert eval_pose_mv.pv_clip("doubles_2sides_4angle") == "doubles_2sides_4angle"
    assert eval_pose_mv.pv_clip("doubles_2sides_4angle_2cam") == "doubles_2sides_4angle"
    assert eval_pose_mv.pv_clip("doubles_2sides") == "doubles_2sides"


def test_track_and_sync_keeps_the_reference_timeline_when_the_reference_view_is_dropped(monkeypatch, tmp_path):
    """Review 2 L4: dropping views[0] (the sync reference) must not move the posev2 timeline by that view's offset:
    the base racket / shuttle / rallies are copied by frame index."""
    truth = {"a": 0.0, "b": 3.0, "c": 5.0, "d": -2.0}         # video frame = reference frame + offset

    class FakeTracker:
        def __init__(self, cams, kp2d, *a, **kw):
            self.frame_count, self.reacquired, self.sides = 50, 0, {}

        def run(self):
            return {}

    def fake_offsets(cams, dense0, views, ref, log=print):
        return {v: truth[v] - truth[ref] for v in views}, {v: 1.0 for v in views}

    calls = []

    def fake_health(cams, views, obs):
        calls.append(list(views))
        return {v: (99.0 if (v == "a" and len(calls) == 1) else 5.0) for v in views}

    monkeypatch.setattr(pv, "TrackerV2", FakeTracker)
    monkeypatch.setattr(pv, "estimate_offsets", fake_offsets)
    monkeypatch.setattr(pv, "view_health", fake_health)
    monkeypatch.setattr(pv, "load_cams", lambda paths: {v: None for v in paths})
    monkeypatch.setattr(pv, "load_kp2d", lambda path: ([], 60.0, None))
    monkeypatch.setattr(appearance, "load_clip", lambda clip: {})
    seeds = tmp_path / "seeds.json"
    seeds.write_text('{"views": {}}')
    cfg = {"clip": "x", "views": ["a", "b", "c", "d"], "mode": "doubles", "frame_count": 50,
           "cam_paths": {v: v for v in "abcd"}, "kp2d_paths": {v: v for v in "abcd"}, "seeds_path": str(seeds)}
    b = pv.track_and_sync(cfg, log=lambda *a: None)
    assert b["views"] == ["b", "c", "d"] and b["excluded"] == {"a": 99.0}
    assert b["time_ref_view"] == "a"
    assert b["offsets"] == pytest.approx({"b": 3.0, "c": 5.0, "d": -2.0})


def test_clip_config_time_ref_is_the_base_docs_reference(tmp_path):
    import json
    doc = {"meta": {"frame_count": 10, "cameras": [{"name": "frontleft_41"}, {"name": "backright_41"}]}}
    (tmp_path / "doubles_2sides_4angle.analysis.json").write_text(json.dumps(doc))
    cfg = pv.clip_config("doubles_2sides_4angle", analysis_dir=str(tmp_path))
    assert cfg["views"] == ["backright_41", "frontleft_41"] and cfg["time_ref"] == "backright_41"


def test_smplx_archive_keeps_every_previous_output(tmp_path):
    """Review 2 L16: the previous <clip>_smplx output was archived only if no file of that name was archived
    yet, so every later rebuild overwrote its predecessor."""
    import os
    out = tmp_path / "c_smplx.analysis.json"
    arch = tmp_path / "_archive"
    for i, t in enumerate((1_700_000_000, 1_700_000_100, 1_700_000_100)):
        out.write_text(f"v{i}")
        os.utime(out, (t, t))
        dst = smplx_fuse.archive_previous(str(out), str(arch), log=lambda *a: None)
        assert dst and not out.exists()
    kept = sorted(p.read_text() for p in arch.iterdir())
    assert kept == ["v0", "v1", "v2"]
    assert all(p.name.startswith("c_smplx.") and p.name.endswith(".analysis.json") for p in arch.iterdir())
    assert smplx_fuse.archive_previous(str(out), str(arch), log=lambda *a: None) is None   # nothing to archive
