"""pose24 anatomical regression: SMPL template reproduction, rules, equivariance, fallbacks."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pose24_synth as syn
from analysis import skeleton24 as sk
from analysis.pose24 import kpset, regress

# SMPL neutral template (m, +x left, +y up, +z forward): J_regressor @ v_template, and COCO nose / eyes /
# ears from the Pose2Mesh vertex regressor (computed 2026-09-28 from models/smpl + models/regressors).
SMPL_J = np.array([
    [-0.0018, -0.2233, 0.0282], [0.0677, -0.3147, 0.0214], [-0.0695, -0.3139, 0.0239], [-0.0043, -0.1144, 0.0015],
    [0.1020, -0.6899, 0.0169], [-0.1078, -0.6964, 0.0150], [0.0012, 0.0208, 0.0026], [0.0884, -1.0879, -0.0268],
    [-0.0920, -1.0948, -0.0273], [0.0026, 0.0737, 0.0280], [0.1148, -1.1437, 0.0925], [-0.1174, -1.1430, 0.0961],
    [-0.0002, 0.2876, -0.0148], [0.0815, 0.1955, -0.0060], [-0.0791, 0.1926, -0.0106], [0.0050, 0.3526, 0.0365],
    [0.1724, 0.2260, -0.0149], [-0.1752, 0.2251, -0.0197], [0.4321, 0.2132, -0.0424], [-0.4289, 0.2118, -0.0411],
    [0.6813, 0.2222, -0.0435], [-0.6842, 0.2196, -0.0467], [0.7653, 0.2140, -0.0585], [-0.7688, 0.2134, -0.0570]])
SMPL_FACE = np.array([[0.0012, 0.4209, 0.1390], [0.0345, 0.4507, 0.0997], [-0.0325, 0.4503, 0.0980],
                      [0.0756, 0.4233, 0.0052], [-0.0762, 0.4240, 0.0049]])      # nose, l/r eye, l/r ear


def _smpl_keypoints():
    """35 fit keypoints on the SMPL template: COCO body = SMPL joints (F0), face from the vertex regressor,
    toes placed so the ball-of-foot rule lands on SMPL's foot joint, knuckles centred on SMPL's hand."""
    K = np.full((35, 3), np.nan)
    K[0:5] = SMPL_FACE
    for k, j in ((5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21), (11, 1), (12, 2), (13, 4), (14, 5),
                 (15, 7), (16, 8)):
        K[k] = SMPL_J[j]
    for a, bt, st, heel, foot in ((15, 17, 18, 19, 10), (16, 20, 21, 22, 11)):
        toe = K[a] + (SMPL_J[foot] - K[a]) / 0.65
        K[bt], K[st], K[heel] = toe + [0.015, 0, 0], toe - [0.015, 0, 0], K[a] + [0, -0.06, -0.06]
    for base, hand in ((23, 22), (27, 23)):
        K[base:base + 4] = SMPL_J[hand] + np.array([[0, 0, 0.03], [0, 0, 0.01], [0, 0, -0.01], [0, 0, -0.03]])
    K[31], K[32], K[33], K[34] = SMPL_FACE[0] + [0, -0.1, -0.02], SMPL_FACE[1], SMPL_FACE[2], SMPL_FACE[0]
    return K[None]


def test_rules_reproduce_the_smpl_template_within_1cm():
    """With the COCO hips taken as hip joint centres (F0: the vertex regressor can't tell them apart),
    every SMPL-derived rule lands within 1 cm of the SMPL neutral joint."""
    J, _, info = regress.regress(_smpl_keypoints(), const={"hjc_medial": 1.0, "hjc_inferior": 0.0})
    for j, name in enumerate(sk.SMPL24_NAMES):
        assert np.linalg.norm(J[0, j] - SMPL_J[j]) < 0.01, name
    assert not info["head_fallback"].any() and not info["mirrored"]


def test_head_from_nose_fallback_is_close_on_the_template():
    kp = _smpl_keypoints()
    kp[:, 3:5] = np.nan
    J, _, info = regress.regress(kp, const={"hjc_medial": 1.0, "hjc_inferior": 0.0})
    assert info["head_fallback"][0] and np.linalg.norm(J[0, 15] - SMPL_J[15]) < 0.01


def test_hip_joint_centres_move_medially_and_down():
    kp = _smpl_keypoints()
    J, _, _ = regress.regress(kp)
    wh = np.linalg.norm(kp[0, 11] - kp[0, 12])
    assert np.linalg.norm(J[0, 1] - J[0, 2]) == pytest.approx(0.70 * wh, rel=1e-6)
    drop = 0.5 * (J[0, 1] + J[0, 2]) - 0.5 * (kp[0, 11] + kp[0, 12])
    assert drop[1] == pytest.approx(-0.12 * wh, abs=1e-5)       # pelvis y = up orthogonalised to the hip line


def test_rigid_motion_and_mirror_equivariance():
    kp = kpset.from133(syn.motion(F=4, mirror=False))
    R = Rotation.from_euler("y", 70, degrees=True).as_matrix()
    t = np.array([1.0, 0.0, -2.0])
    J0, a0, _ = regress.regress(kp)
    J1, a1, _ = regress.regress(kp @ R.T + t)
    assert np.allclose(J1, J0 @ R.T + t, atol=1e-9) and np.allclose(a1, a0 @ R.T + t)
    M = np.array([-1.0, 1.0, 1.0])
    J2, _, info = regress.regress(kp * M)
    assert info["mirrored"] and np.allclose(J2, J0 * M, atol=1e-9)


def test_spine_twists_with_the_thorax():
    kp = kpset.from133(syn.motion(F=1, arm=False, walk=False))
    tw = kp.copy()
    s = 0.5 * (kp[:, 5] + kp[:, 6])
    R = Rotation.from_rotvec(np.radians(30.0) * np.array([0, 1.0, 0])).as_matrix()
    upper = list(range(0, 11)) + list(range(23, 35))
    tw[:, upper] = s[:, None] + (kp[:, upper] - s[:, None]) @ R.T

    def heading(J, name, t):
        n = sk.SMPL28_NAMES.index
        o = J[0, n(name)] - (J[0, 0] + t * (J[0, n("neck")] - J[0, 0]))
        return np.degrees(np.arctan2(o[0], o[2]))
    J0, _, _ = regress.regress(kp)
    J1, _, _ = regress.regress(tw)
    d3 = heading(J1, "spine3", 0.577) - heading(J0, "spine3", 0.577)
    assert d3 == pytest.approx(0.577 * 30.0, abs=2.0)


def test_hand_fallback_and_aux():
    kp = kpset.from133(syn.motion(F=2))
    kp[:, 23:27] = np.nan
    J, aux, info = regress.regress(kp)
    assert info["hand_fallback"][:, 0].all() and not info["hand_fallback"][:, 1].any()
    assert np.linalg.norm(J[0, 22] - kp[0, 9]) == pytest.approx(0.08)
    assert np.allclose(aux[:, 0], kp[:, 19]) and np.allclose(aux[:, 10], kp[:, 27])


def test_observed_mask_hides_never_seen_keypoints():
    kp = kpset.from133(syn.motion(F=2))
    obs = np.ones(35, bool)
    obs[27:31] = False
    _, _, info = regress.regress(kp, observed=obs)
    assert info["hand_fallback"][:, 1].all()


def test_sources_cover_every_joint():
    assert len(regress.SOURCE_LIST) == 28 and all(regress.SOURCE_LIST)
    assert len(regress.AUX_FIT) == 14 and [kpset.NAMES[i] for i in regress.AUX_FIT] == sk.AUX_NAMES


def test_mirrored_player_without_feet_or_face_uses_the_world_handedness():
    """A mirror-handed player (real data, Y-up analysis world) whose feet and face were never seen: the
    regression can't read the handedness from the body, so it takes the caller's (from the calibration)."""
    kp = _smpl_keypoints() * np.array([-1.0, 1.0, 1.0])            # mirror-handed, as the analysis world
    J_ref, _, info = regress.regress(kp)
    assert info["mirrored"]
    seen = np.ones(35, bool)
    seen[[0, 1, 2, 3, 4] + list(range(15, 23)) + [31, 32, 33, 34]] = False     # no face, no ankles/feet
    J, _, info = regress.regress(kp, observed=seen, default_sign=-1.0)
    assert info["mirrored"]
    for name in ("pelvis", "spine1", "spine2", "spine3", "neck", "left_collar", "right_collar"):
        j = sk.SMPL28_NAMES.index(name)
        assert np.linalg.norm(J[0, j] - J_ref[0, j]) < 1e-9, name
