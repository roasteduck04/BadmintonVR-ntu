"""pose24 eval helpers."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis import skeleton24 as sk
from analysis.pose24 import eval as ev


def test_procrustes_removes_a_similarity():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 3))
    R = Rotation.from_euler("xyz", [20, 50, -30], degrees=True).as_matrix()
    B = 1.3 * A @ R.T + [1.0, 2.0, 3.0]
    assert ev.procrustes_error(A, B) < 1e-9
    assert ev.procrustes_error(A, B + rng.normal(0, 0.01, B.shape)) < 0.02


def test_joint_angles_on_the_rest_pose():
    X = sk.REST28[None].copy()
    a = ev.joint_angles(X, sk.SMPL28_NAMES)
    # REST28 hip x 0.08 vs knee/ankle x 0.09 -> atan(0.01/0.40) = 1.43 deg
    assert a["left_knee_flexion"][0] == pytest.approx(1.43, abs=0.1)
    assert a["left_elbow_flexion"][0] < 1
    assert a["left_shoulder_abduction"][0] == pytest.approx(90.0, abs=5)       # T-pose
    assert abs(a["trunk_forward_lean"][0]) < 1 and abs(a["trunk_lateral_lean"][0]) < 1


def test_reliability_detects_a_monotone_relation():
    s = np.linspace(1, 100, 500)
    r = ev.reliability(s, s * 0.3 + np.random.default_rng(1).normal(0, 1, 500))
    assert r["spearman"] > 0.9 and len(r["bins"]) == 5
    assert ev.reliability(s[:10], s[:10]) is None


def test_compare_joints_zero_on_identical_and_root_aligned_on_shift():
    names = sk.SMPL28_NAMES
    rng = np.random.default_rng(2)
    A = sk.REST28[None].repeat(20, 0) + rng.normal(0, 0.01, (20, 28, 3))
    r = ev.compare_joints(A, A.copy(), names)
    assert r["w_mpjpe_cm"]["median"] < 1e-9 and r["pa_mpjpe_cm"]["median"] < 1e-6
    B = A + np.array([0.3, 0.0, 0.0])                   # whole body 30 cm to the side
    r = ev.compare_joints(A, B, names)
    assert r["w_mpjpe_cm"]["median"] == pytest.approx(30.0, abs=1e-6)
    assert r["root_mpjpe_cm"]["median"] < 1e-6 and r["pelvis_xz_cm"]["median"] == pytest.approx(30.0, abs=1e-6)
    assert r["pelvis_h_cm"]["median"] < 1e-9


def test_compare_joints_mask_and_all_nan():
    names = sk.SMPL28_NAMES
    A = sk.REST28[None].repeat(10, 0).copy()
    B = A.copy()
    B[5:] += 0.1
    m = np.zeros(10, bool)
    m[:5] = True
    assert ev.compare_joints(A, B, names, mask=m)["w_mpjpe_cm"]["median"] < 1e-9
    r = ev.compare_joints(np.full_like(A, np.nan), B, names)
    assert r["w_mpjpe_cm"] is None and r["pa_mpjpe_cm"] is None


def test_joint_angles_on_constructed_poses():
    """Spec § 8: knee 90 deg, lateral lean 20 deg (shoulder abduction 90 is the existing T-pose test)."""
    names = sk.SMPL28_NAMES
    i = {n: k for k, n in enumerate(names)}
    X = sk.REST28[None].copy()
    knee = X[0, i["left_knee"]]
    shank = np.linalg.norm(X[0, i["left_ankle"]] - knee)
    X[0, i["left_hip"]] = knee + np.array([0.0, 0.40, 0.0])              # thigh vertical
    X[0, i["left_ankle"]] = knee + np.array([0.0, 0.0, -shank])          # shank horizontal
    assert ev.joint_angles(X, names)["left_knee_flexion"][0] == pytest.approx(90.0, abs=0.5)
    Y = sk.REST28[None].copy()
    pel = Y[0, i["pelvis"]].copy()
    R = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    upper = [k for k, n in enumerate(names) if n not in ("pelvis", "left_hip", "right_hip") and "knee" not in n
             and "ankle" not in n and "foot" not in n]
    Y[0, upper] = pel + (Y[0, upper] - pel) @ R.T
    assert abs(ev.joint_angles(Y, names)["trunk_lateral_lean"][0]) == pytest.approx(20.0, abs=1.0)
