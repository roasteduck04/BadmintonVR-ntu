"""pose24 3D uncertainty and confidence."""
import numpy as np
import pytest

import pose24_synth as syn
from analysis import multiview as mv
from analysis import pose_v2 as pv
from analysis.pose24 import confidence


def _pair(c1, c2):
    K = mv.intrinsics_from_fov(2160, 3840, 70.0)
    cams = {"p": (K,) + syn.look_at(c1, (0.0, -1.0, 0.0)), "q": (K,) + syn.look_at(c2, (0.0, -1.0, 0.0))}
    return pv.proj_mats(cams, ["p", "q"])


def _sig(P, inl=(True, True)):
    return confidence.point_sigma_mm(P, np.array([[0.0, -1.0, 0.0]]), np.full((1, 2), 0.9), np.array([inl]))[0]


def test_sigma_larger_for_near_antiparallel_pair_than_orthogonal():
    ortho = _pair([0, -1, -10], [10, -1, 0])
    anti = _pair([0, -1, -10], [0.5, -1, 10])
    # ortho worst direction is analytically 8.89 px / 2742 px x 10 m = 32.4 mm (one view per lateral axis)
    assert _sig(anti) > 3 * _sig(ortho) and _sig(ortho) < 40


def test_one_or_zero_views_are_capped():
    P = _pair([0, -1, -10], [10, -1, 0])
    assert _sig(P, (True, False)) == confidence.CAP_MM and _sig(P, (False, False)) == confidence.CAP_MM


def test_fill_decay_grows_with_distance_from_measured():
    sig = np.full((10, 1), 1000.0)
    sig[:3] = 5.0
    meas = np.zeros((10, 1), bool)
    meas[:3] = True
    out = confidence.fill_decay(sig, meas)
    assert out[3, 0] == 15.0 and out[9, 0] == 75.0 and (np.diff(out[3:, 0]) > 0).all()
    assert (confidence.fill_decay(sig, np.zeros((10, 1), bool)) == confidence.CAP_MM).all()


def test_residual_inflates_sigma():
    P = _pair([0, -1, -10], [10, -1, 0])
    X = np.full((1, 1, 3), [0.0, -1.0, 0.0])
    base = {"X": X, "w": np.full((1, 1, 2), 0.9), "inlier": np.ones((1, 1, 2), bool)}
    s0, _, m = confidence.keypoint_sigma(dict(base, resid=np.zeros((1, 1, 2))), P)
    s1, _, _ = confidence.keypoint_sigma(dict(base, resid=np.full((1, 1, 2), 2 * confidence.SIGMA_PX)), P)
    assert m.all() and s1[0, 0] == pytest.approx(2 * s0[0, 0])


def test_joint_sigma_takes_the_max_of_its_sources_and_confidence_maps():
    sig = np.array([[5.0, 20.0, 40.0]])
    assert np.allclose(confidence.joint_sigma(sig, [[0], [0, 2], [1]]), [[5.0, 40.0, 20.0]])
    c = confidence.to_conf(np.array([0.0, 40.0, np.nan, 400.0]))
    assert c[0] == 1.0 and c[1] == pytest.approx(np.exp(-1)) and c[2] == 0.0 and c[3] < 1e-4


def test_fill_decay_takes_the_better_of_both_measured_neighbours():
    sig = np.full((10, 1), 1000.0)
    sig[0], sig[9] = 100.0, 5.0
    meas = np.zeros((10, 1), bool)
    meas[[0, 9]] = True
    out = confidence.fill_decay(sig, meas)
    assert out[4, 0] == 55.0                  # from frame 9 (5 + 5 x 10), not the nearer frame 0 (100 + 4 x 10)
    assert out[1, 0] == 85.0 and out[8, 0] == 15.0      # frame 1: min(100 + 10, 5 + 80)


def test_nan_score_does_not_hide_the_residual_inflation():
    P = _pair([0, -1, -10], [10, -1, 0])
    X = np.full((1, 1, 3), [0.0, -1.0, 0.0])
    base = {"X": X, "w": np.array([[[np.nan, 0.9]]]), "inlier": np.ones((1, 1, 2), bool)}
    s0, _, _ = confidence.keypoint_sigma(dict(base, resid=np.zeros((1, 1, 2))), P)
    s1, _, _ = confidence.keypoint_sigma(dict(base, resid=np.full((1, 1, 2), 1.5 * confidence.SIGMA_PX)), P)
    assert s0[0, 0] < 1.5 * s0[0, 0] < confidence.CAP_MM          # the NaN view counts at the score floor
    assert s1[0, 0] == pytest.approx(1.5 * s0[0, 0])
