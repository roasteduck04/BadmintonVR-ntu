"""pose24 anatomical angles on the 35-keypoint set."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pose24_synth as syn
from analysis.pose24 import angles, kpset

UP = np.array([0.0, 1.0, 0.0])


def _rest(F=3):
    return kpset.from133(syn.motion(F=F, arm=False, walk=False))


def test_rest_angles():
    X = _rest()
    s = kpset.forward_sign(X)
    for side in ("left", "right"):
        assert np.all(np.abs(angles.knee_flexion(X, s, UP, side)) < 10)
        assert np.all(angles.elbow_flexion(X, side) < 15)
        assert np.all(np.abs(angles.ankle_dorsiflexion(X, side)) < 20)
    assert np.all(np.abs(angles.trunk_flexion(X, s, UP)) < 3)
    assert np.all(np.abs(angles.trunk_lateral_flexion(X)) < 1e-6)
    assert np.all(angles.trunk_axial(X) < 1e-6)


def test_knee_flexion_sign():
    X = _rest(1)
    fwd, back = X.copy(), X.copy()
    fwd[:, 13, 2] += 0.15
    back[:, 13, 2] -= 0.15
    s = kpset.forward_sign(X)
    assert angles.knee_flexion(fwd, s, UP, "left")[0] > 20 > -20 > angles.knee_flexion(back, s, UP, "left")[0]


def test_lateral_flexion_follows_the_side_bend():
    X = kpset.from133(syn.motion(F=150, arm=False, walk=False, side_bend=20.0))
    lat = angles.trunk_lateral_flexion(X)
    assert lat.max() - lat.min() == pytest.approx(40.0, abs=1.5)


def test_axial_twist():
    X = _rest(1)
    s_mid = 0.5 * (X[:, 5] + X[:, 6])
    R = Rotation.from_rotvec(np.radians(30) * UP).as_matrix()
    X[:, [5, 6]] = s_mid[:, None] + (X[:, [5, 6]] - s_mid[:, None]) @ R.T
    assert angles.trunk_axial(X)[0] == pytest.approx(30.0, abs=0.5)


def test_angles_are_mirror_invariant_given_the_sign():
    X = kpset.from133(syn.motion(F=20, side_bend=15.0))
    M = X * np.array([1.0, -1.0, 1.0])
    s, sm = kpset.forward_sign(X), kpset.forward_sign(M)
    assert np.allclose(angles.knee_flexion(X, s, UP, "right"), angles.knee_flexion(M, sm, -UP, "right"))
    assert np.allclose(angles.trunk_flexion(X, s, UP), angles.trunk_flexion(M, sm, -UP))
    assert np.allclose(angles.trunk_lateral_flexion(X), angles.trunk_lateral_flexion(M))


def _bent_left_leg(inner_deg, rot_deg):
    """Rest pose with the left leg bent by `inner_deg` in the sagittal plane (+ = knee towards the toes,
    - = hyperextended), then knee + shank + foot rotated `rot_deg` about the vertical through the hip."""
    X = _rest(1)
    H = X[:, 11]
    fwd = 0.5 * (X[:, 17] + X[:, 18]) - X[:, 19]
    fwd[:, 1] = 0.0
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
    down = -UP
    Lt = np.linalg.norm(X[:, 13] - H, axis=-1, keepdims=True)
    Ls = np.linalg.norm(X[:, 15] - X[:, 13], axis=-1, keepdims=True)
    h = np.radians(inner_deg / 2)
    foot = X[:, [17, 18, 19]] - X[:, 15:16]
    X[:, 13] = H + Lt * (np.cos(h) * down + np.sin(h) * fwd)
    X[:, 15] = X[:, 13] + Ls * (np.cos(h) * down - np.sin(h) * fwd)
    X[:, [17, 18, 19]] = X[:, 15:16] + foot
    R = Rotation.from_rotvec(np.radians(rot_deg) * UP).as_matrix()
    idx = [13, 15, 17, 18, 19]
    X[:, idx] = H[:, None] + (X[:, idx] - H[:, None]) @ R.T
    return X


@pytest.mark.parametrize("rot", [0.0, 100.0, 135.0])
def test_knee_flexion_sign_follows_the_foot_when_the_leg_is_rotated(rot):
    s = kpset.forward_sign(_rest(1))
    assert angles.knee_flexion(_bent_left_leg(60.0, rot), s, UP, "left")[0] > 50
    assert angles.knee_flexion(_bent_left_leg(-20.0, rot), s, UP, "left")[0] < -10


def test_knee_flexion_is_nan_without_a_foot_direction():
    X = _rest()
    X[:, [17, 18, 19]] = np.nan                          # left foot never seen
    assert np.isnan(angles.knee_flexion(X, 1.0, None, "left")).all()
    assert np.isfinite(angles.knee_flexion(X, 1.0, None, "right")).all()
