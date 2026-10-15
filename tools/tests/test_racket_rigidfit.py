import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis import racket_rigidfit as rf


def _look_at_camera(center, target, f=1600.0, w=2160, h=3840):
    """Camera at `center` looking at `target` (both analysis world, Y-up)."""
    C = np.asarray(center, float) * rf._Y_FLIP
    T = np.asarray(target, float) * rf._Y_FLIP
    z = (T - C) / np.linalg.norm(T - C)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return rf.Camera(K, R, -R @ C)


@pytest.fixture
def cams():
    tgt = [0.0, 1.3, -3.0]
    return {
        "back": _look_at_camera([0.0, 1.6, -11.0], tgt),
        "backleft": _look_at_camera([-4.5, 1.6, -9.0], tgt),
        "backright": _look_at_camera([4.5, 1.6, -9.0], tgt),
        "frontleft": _look_at_camera([-4.5, 1.6, 4.0], tgt),
    }


TRUE_GRIP = np.array([0.3, 1.25, -3.1])
TRUE_R = rf.rotation_from_shaft([0.5, 0.7, 0.3], roll=0.6)


def _obs(cam, tpl, grip, R, noise=0.0, rng=None, score=0.9):
    P = grip + tpl.keypoints() @ R.T
    uv, _ = cam.project(P)
    if noise:
        uv = uv + rng.normal(0, noise, uv.shape)
    return np.column_stack([uv, np.full(5, score)])


def _shaft_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip(Ra[:, 2] @ Rb[:, 2], -1, 1)))


def _normal_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip(abs(Ra[:, 1] @ Rb[:, 1]), -1, 1)))


def test_template_keypoints_match_dimensions():
    tpl = rf.RacketTemplate(length=0.67, head_length=0.24, head_width=0.2)
    kp = tpl.keypoints()
    assert np.isclose(np.linalg.norm(kp[rf.TOP] - kp[rf.HANDLE]), 0.67)
    assert np.isclose(np.linalg.norm(kp[rf.TOP] - kp[rf.BOTTOM]), 0.24)
    assert np.isclose(np.linalg.norm(kp[rf.LEFT] - kp[rf.RIGHT]), 0.2)
    assert np.allclose(kp[rf.HANDLE], [0, 0, -tpl.grip_z])


def test_camera_projection_roundtrip(cams):
    uv, z = cams["back"].project(np.array([0.0, 1.3, -3.0]))
    assert z > 0
    assert np.allclose(uv, [1080, 1920], atol=1e-6)


def test_fit_recovers_pose_multiview(cams):
    rng = np.random.default_rng(0)
    tpl = rf.RacketTemplate()
    cand = {v: [_obs(c, tpl, TRUE_GRIP, TRUE_R, 2.0, rng)] for v, c in cams.items()}
    res = rf.RacketFitter(cams, tpl).fit(cand, hand=TRUE_GRIP + [0.02, 0.0, 0.01])
    assert res is not None
    assert len(res.views) == 4
    assert np.linalg.norm(res.grip - TRUE_GRIP) < 0.03
    assert _shaft_err_deg(res.R, TRUE_R) < 3.0
    assert _normal_err_deg(res.R, TRUE_R) < 5.0


def test_left_right_label_swap_is_tolerated(cams):
    tpl = rf.RacketTemplate()
    cand = {}
    for i, (v, c) in enumerate(cams.items()):
        o = _obs(c, tpl, TRUE_GRIP, TRUE_R)
        cand[v] = [o[rf._SWAP] if i % 2 else o]
    res = rf.RacketFitter(cams, tpl).fit(cand, hand=TRUE_GRIP)
    assert len(res.views) == 4
    assert _normal_err_deg(res.R, TRUE_R) < 2.0


def test_single_view_with_hand_and_prior(cams):
    tpl = rf.RacketTemplate()
    cand = {"backleft": [_obs(cams["backleft"], tpl, TRUE_GRIP, TRUE_R)]}
    near = Rotation.from_rotvec([0.15, -0.1, 0.1]) * Rotation.from_matrix(TRUE_R)
    prior = np.concatenate([TRUE_GRIP + 0.05, near.as_rotvec()])
    res = rf.RacketFitter(cams, tpl).fit(cand, hand=TRUE_GRIP, prior=prior)
    assert res is not None and res.views == ["backleft"]
    assert _shaft_err_deg(res.R, TRUE_R) < 6.0


def test_single_view_without_hand_is_refused(cams):
    tpl = rf.RacketTemplate()
    cand = {"back": [_obs(cams["back"], tpl, TRUE_GRIP, TRUE_R)]}
    assert rf.RacketFitter(cams, tpl).fit(cand, hand=None) is None


def test_outlier_view_is_dropped(cams):
    tpl = rf.RacketTemplate()
    cand = {v: [_obs(c, tpl, TRUE_GRIP, TRUE_R)] for v, c in cams.items()}
    bad = _obs(cams["frontleft"], tpl, TRUE_GRIP + [1.2, -0.5, 0.4],
               rf.rotation_from_shaft([-1, 0.1, 0]))
    cand["frontleft"] = [bad]
    res = rf.RacketFitter(cams, tpl).fit(cand, hand=TRUE_GRIP)
    assert "frontleft" not in res.views
    assert _shaft_err_deg(res.R, TRUE_R) < 2.0


def test_wrong_first_candidate_is_reassociated(cams):
    tpl = rf.RacketTemplate()
    cand = {v: [_obs(c, tpl, TRUE_GRIP, TRUE_R)] for v, c in cams.items()}
    other = _obs(cams["backright"], tpl, TRUE_GRIP + [0.6, 0.2, 0.3],
                 rf.rotation_from_shaft([0, -1, 0.2]))
    cand["backright"] = [other, cand["backright"][0]]
    res = rf.RacketFitter(cams, tpl).fit(cand, hand=TRUE_GRIP)
    assert res.choice.get("backright") == 1


def _result(grip, R, n_views=3):
    x = np.concatenate([grip, Rotation.from_matrix(R).as_rotvec()])
    return rf.FitResult(x=x, cost=0.0, err=0.02, per_view={}, views=["a", "b", "c"][:n_views],
                        choice={}, score=0.8)


def test_finalize_removes_spike_and_fills_gap():
    n = 30
    results = []
    for i in range(n):
        R = rf.rotation_from_shaft([np.cos(i * 0.05), 0.5, np.sin(i * 0.05)])
        results.append(_result(np.array([0.01 * i, 1.2, 0.0]), R))
    results[10] = _result(np.array([0.10, 1.2, 0.0]), rf.rotation_from_shaft([0, -1, 0.1]))
    results[20] = results[21] = results[22] = None
    tr = rf.finalize_track(results, fps=60.0)
    assert tr["status"][10] == "interpolated"
    assert list(tr["status"][20:23]) == ["interpolated"] * 3
    assert tr["status"][5] == "detected"
    expected = rf.rotation_from_shaft([np.cos(0.5), 0.5, np.sin(0.5)])[:, 2]
    assert np.degrees(np.arccos(np.clip(tr["R"][10][:, 2] @ expected, -1, 1))) < 5.0


def test_finalize_keeps_symmetric_orientation_continuous():
    results = []
    for i in range(20):
        R = rf.rotation_from_shaft([1, 0.3, 0], roll=0.02 * i)
        if i % 3 == 0:
            R = R @ rf._RZ_PI
        results.append(_result(np.zeros(3), R))
    tr = rf.finalize_track(results, fps=60.0)
    normals = tr["R"][:, :, 1]
    assert (np.einsum("ij,ij->i", normals[1:], normals[:-1]) > 0.9).all()


def test_estimate_template_recovers_dimensions(cams):
    true = rf.RacketTemplate(length=0.68, head_length=0.26, head_width=0.22, width_z=0.55)
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(20):
        R = rf.rotation_from_shaft(rng.normal(size=3), roll=rng.uniform(0, np.pi))
        g = TRUE_GRIP + rng.normal(0, 0.2, 3)
        samples.append({v: _obs(c, true, g, R) for v, c in cams.items()})
    tpl, n = rf.estimate_template(cams, samples)
    assert n >= 10
    assert abs(tpl.length - 0.68) < 0.005
    assert abs(tpl.head_width - 0.22) < 0.005
    assert abs(tpl.head_length - 0.26) < 0.005
