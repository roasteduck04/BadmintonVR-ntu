"""pose24 fit keypoint set + the synthetic harness itself."""
import numpy as np
import pytest

import pose24_synth as syn
from analysis import pose_v2 as pv
from analysis.pose24 import kpset


def test_sources_and_names_line_up():
    assert kpset.K == len(kpset.SOURCES) == len(kpset.NAMES) == 35
    assert kpset.SOURCES[23] == [96] and kpset.SOURCES[34] == [53] and len(kpset.SOURCES[32]) == 6
    assert kpset.NAMES[19] == "left_heel" and kpset.NAMES[kpset.CONTACT[0]] == "left_big_toe"


def test_obs35_means_rings_and_takes_min_score():
    F, V = 2, 3
    uv = np.random.default_rng(0).uniform(0, 100, (F, 133, V, 2))
    w = np.full((F, 133, V), 0.9)
    w[:, 66, 1] = 0.2
    u35, w35 = kpset.obs35(uv, w)
    assert u35.shape == (F, 35, V, 2) and w35.shape == (F, 35, V)
    assert np.allclose(u35[:, 32], uv[:, 65:71].mean(axis=1)) and np.allclose(u35[:, 23], uv[:, 96])
    assert w35[0, 32, 1] == pytest.approx(0.2) and w35[0, 32, 0] == pytest.approx(0.9)


def test_forward_sign_follows_the_feet_and_flips_under_a_mirror():
    X = kpset.from133(syn.motion(F=5))
    assert X.shape == (5, 35, 3)
    assert kpset.forward_sign(X) == -1.0                       # real-data handedness in the Y-up analysis world
    assert kpset.forward_sign(X * np.array([1.0, -1.0, 1.0])) == 1.0
    assert kpset.forward_sign(kpset.from133(syn.motion(F=5, mirror=False))) == 1.0


@pytest.mark.parametrize("s", [1.0, -1.0])
def test_rest_offsets_follow_the_requested_handedness(s):
    assert kpset.forward_sign(kpset.rest_offsets_down(s)[None]) == s


def test_synthetic_projection_round_trips():
    cams = syn.rig(("a", "b"))
    X = syn.motion(F=4, z0=-3.0)
    obs = syn.project(X, cams, ("a", "b"))
    Xd = X * np.array([1.0, -1.0, 1.0])
    Xt, _, _, _ = pv.triangulate_batch(cams, ["a", "b"],
                                       np.stack([obs["a"][..., :2], obs["b"][..., :2]], -2).reshape(-1, 2, 2),
                                       pv.norm_score(np.stack([obs["a"][..., 2], obs["b"][..., 2]], -1)).reshape(-1, 2))
    assert np.nanmax(np.abs(Xt - Xd.reshape(-1, 3))) < 1e-6


def test_synthetic_limb_lengths_and_jump():
    X = syn.motion(F=90, jump=(30, 60))
    assert np.linalg.norm(X[:, 13] - X[:, 11], axis=-1) == pytest.approx(syn.LIMB_TRUE["thigh"])
    assert X[45, 0, 1] > 2.0 and np.abs(X[10, 15, 1] - 0.08) < 1e-9


def test_forward_sign_default_when_nothing_tells_the_handedness():
    X = np.zeros((1, kpset.K, 3))
    X[:, [0, 1, 2, 3, 4] + list(range(17, 23))] = np.nan          # no face, no feet
    X[:, 5], X[:, 6], X[:, 11], X[:, 12] = [0.2, 1.4, 0], [-0.2, 1.4, 0], [0.1, 0.9, 0], [-0.1, 0.9, 0]
    assert kpset.forward_sign(X) == 1.0
    assert kpset.forward_sign(X, default=-1.0) == -1.0


def test_world_sign_from_the_calibration():
    cams = syn.rig()
    assert kpset.world_sign(cams) == 1.0                           # proper rotations: right-handed court world
    flip = np.diag([-1.0, 1.0, 1.0])                               # the same rig in a mirrored world
    assert kpset.world_sign({v: (K, R @ flip, t) for v, (K, R, t) in cams.items()}) == -1.0


def test_rest_offsets_reject_an_unknown_sign():
    with pytest.raises(ValueError):
        kpset.rest_offsets_down(float("nan"))


def test_handedness_anchored_in_pixels():
    """Pixel-level anchor, independent of the synthetic template: a camera (OpenCV: x right, y down) sees a
    person facing it; the person's left hip is on the image's right, the toes are nearer than the heels.
    Back-projected into that calibration's court world, forward_sign must equal world_sign (+1 here) — the
    Y-down court world of a det R = +1 calibration is right-handed."""
    from analysis import multiview as mv
    K = mv.intrinsics_from_fov(syn.W, syn.H, 60.0)
    R, t = syn.look_at([0.0, -1.0, -10.0], (0.0, -1.0, 0.0))
    cx, cy = K[0, 2], K[1, 2]

    def back(u, v, depth):
        return R.T @ (depth * np.linalg.inv(K) @ np.array([u, v, 1.0]) - t)
    X = np.full((1, kpset.K, 3), np.nan)
    X[0, 11], X[0, 12] = back(cx + 60, cy + 80, 10.0), back(cx - 60, cy + 80, 10.0)     # left hip image right
    X[0, 5], X[0, 6] = back(cx + 80, cy - 150, 10.0), back(cx - 80, cy - 150, 10.0)     # shoulders above
    X[0, 17], X[0, 19] = back(cx + 50, cy + 400, 9.85), back(cx + 50, cy + 390, 10.05)   # left toe / heel
    X[0, 20], X[0, 22] = back(cx - 50, cy + 400, 9.85), back(cx - 50, cy + 390, 10.05)
    assert kpset.world_sign({"a": (K, R, t)}) == 1.0
    assert kpset.forward_sign(X) == 1.0
