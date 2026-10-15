import numpy as np
from analysis import wholebody, skeleton


def _synthetic_wholebody133():
    """133 keypoints: body 0-16, foot 17-22, face 23-90, hands 91-132."""
    j = np.zeros((133, 3), dtype=float)
    c = np.ones(133, dtype=float)
    # Body (matches COCO-17 layout)
    j[0] = [0.0, 1.65, 0.08]     # nose
    j[1] = [0.03, 1.58, 0.06]    # left_eye
    j[2] = [-0.03, 1.58, 0.06]   # right_eye
    j[5] = [0.18, 1.4, 0.0]      # left_shoulder
    j[6] = [-0.18, 1.4, 0.0]     # right_shoulder
    j[7] = [0.42, 1.4, 0.0]      # left_elbow
    j[8] = [-0.42, 1.4, 0.0]     # right_elbow
    j[9] = [0.65, 1.4, 0.0]      # left_wrist
    j[10] = [-0.65, 1.4, 0.0]    # right_wrist
    j[11] = [0.08, 0.9, 0.0]     # left_hip
    j[12] = [-0.08, 0.9, 0.0]    # right_hip
    j[13] = [0.09, 0.5, 0.0]     # left_knee
    j[14] = [-0.09, 0.5, 0.0]    # right_knee
    j[15] = [0.09, 0.08, 0.0]    # left_ankle
    j[16] = [-0.09, 0.08, 0.0]   # right_ankle
    # Foot
    j[17] = [0.09, 0.02, 0.12]   # left_big_toe
    j[19] = [0.09, 0.02, -0.06]  # left_heel
    j[20] = [-0.09, 0.02, 0.12]  # right_big_toe
    j[22] = [-0.09, 0.02, -0.06] # right_heel
    # Face: jaw at 23+8=31, nose_tip at 23+30=53
    j[31] = [0.0, 1.52, 0.04]    # jaw
    j[53] = [0.0, 1.56, 0.08]    # nose_tip
    for i in range(65, 71):       # left eye (6 pts)
        j[i] = [0.03, 1.58, 0.06]
    for i in range(59, 65):       # right eye (6 pts)
        j[i] = [-0.03, 1.58, 0.06]
    # Left hand (91-111): 0=wrist, 1=thumb_cmc, 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip,
    #   5=index_mcp, 6=index_pip, 7=index_dip, 8=index_tip, ...
    j[91] = [0.65, 1.4, 0.0]     # left wrist (same as body wrist)
    j[92] = [0.70, 1.40, -0.02]  # left_thumb CMC -> thumb1
    j[93] = [0.74, 1.40, -0.03]  # left_thumb MCP -> thumb2
    j[94] = [0.78, 1.40, -0.04]  # left_thumb IP -> thumb3
    j[96] = [0.70, 1.40, 0.00]   # left_index MCP -> index1
    j[97] = [0.75, 1.40, 0.00]   # left_index PIP -> index2
    j[98] = [0.80, 1.40, 0.00]   # left_index DIP -> index3
    j[100] = [0.70, 1.40, 0.02]  # left_middle MCP -> middle1
    j[101] = [0.75, 1.40, 0.02]  # left_middle PIP -> middle2
    j[102] = [0.80, 1.40, 0.02]  # left_middle DIP -> middle3
    j[104] = [0.70, 1.40, 0.04]  # left_ring MCP -> ring1
    j[105] = [0.75, 1.40, 0.04]  # left_ring PIP -> ring2
    j[106] = [0.80, 1.40, 0.04]  # left_ring DIP -> ring3
    j[108] = [0.70, 1.40, 0.06]  # left_pinky MCP -> pinky1
    j[109] = [0.75, 1.40, 0.06]  # left_pinky PIP -> pinky2
    j[110] = [0.80, 1.40, 0.06]  # left_pinky DIP -> pinky3
    # Right hand (112-132): same layout, mirrored
    j[112] = [-0.65, 1.4, 0.0]
    j[113] = [-0.70, 1.40, -0.02]
    j[114] = [-0.74, 1.40, -0.03]
    j[115] = [-0.78, 1.40, -0.04]
    j[117] = [-0.70, 1.40, 0.00]
    j[118] = [-0.75, 1.40, 0.00]
    j[119] = [-0.80, 1.40, 0.00]
    j[121] = [-0.70, 1.40, 0.02]
    j[122] = [-0.75, 1.40, 0.02]
    j[123] = [-0.80, 1.40, 0.02]
    j[125] = [-0.70, 1.40, 0.04]
    j[126] = [-0.75, 1.40, 0.04]
    j[127] = [-0.80, 1.40, 0.04]
    j[129] = [-0.70, 1.40, 0.06]
    j[130] = [-0.75, 1.40, 0.06]
    j[131] = [-0.80, 1.40, 0.06]
    return j, c


def test_output_shape():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    assert j58.shape == (58, 3)
    assert c58.shape == (58,)


def test_body_joints_mapped():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["left_shoulder"]], [0.18, 1.4, 0.0])
    assert np.allclose(j58[idx["left_ankle"]], [0.09, 0.08, 0.0])


def test_big_toe_mapped():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["left_big_toe"]], [0.09, 0.02, 0.12])
    assert np.allclose(j58[idx["right_big_toe"]], [-0.09, 0.02, 0.12])


def test_heel_mapped():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["left_heel"]], [0.09, 0.02, -0.06])
    assert np.allclose(j58[idx["right_heel"]], [-0.09, 0.02, -0.06])


def test_left_hand_finger_mapping():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["left_thumb1"]], [0.70, 1.40, -0.02])
    assert np.allclose(j58[idx["left_thumb2"]], [0.74, 1.40, -0.03])
    assert np.allclose(j58[idx["left_thumb3"]], [0.78, 1.40, -0.04])
    assert np.allclose(j58[idx["left_index1"]], [0.70, 1.40, 0.00])
    assert np.allclose(j58[idx["left_middle1"]], [0.70, 1.40, 0.02])
    assert np.allclose(j58[idx["left_ring1"]], [0.70, 1.40, 0.04])
    assert np.allclose(j58[idx["left_pinky1"]], [0.70, 1.40, 0.06])
    assert c58[idx["left_thumb1"]] == 1.0


def test_right_hand_finger_mapping():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["right_thumb1"]], [-0.70, 1.40, -0.02])
    assert np.allclose(j58[idx["right_index3"]], [-0.80, 1.40, 0.00])
    assert np.allclose(j58[idx["right_pinky3"]], [-0.80, 1.40, 0.06])


def test_face_joints_mapped():
    j58, c58 = wholebody.wholebody133_to_smpl58(*_synthetic_wholebody133())
    idx = {n: i for i, n in enumerate(skeleton.JOINT_NAMES)}
    assert np.allclose(j58[idx["jaw"]], [0.0, 1.52, 0.04])
    assert np.allclose(j58[idx["nose_tip"]], [0.0, 1.56, 0.08])
    assert c58[idx["jaw"]] == 1.0


def test_hand_zero_confidence_when_detector_missing():
    j, c = _synthetic_wholebody133()
    c[91:133] = 0.0
    j58, c58 = wholebody.wholebody133_to_smpl58(j, c)
    for ji in range(22, 52):
        assert c58[ji] == 0.0
