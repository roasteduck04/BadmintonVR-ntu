import numpy as np
from analysis import skeleton


def test_joint_count():
    assert skeleton.NUM_JOINTS == 58
    assert len(skeleton.JOINT_NAMES) == 58
    assert len(skeleton.PARENTS) == 58
    assert skeleton._REST.shape == (58, 3)


def test_body_joints_0_to_21():
    """First 22 joints are SMPL-X canonical body."""
    expected_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_big_toe",
        "right_big_toe", "neck", "left_collar", "right_collar", "head",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]
    assert skeleton.JOINT_NAMES[:22] == expected_names
    expected_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                        16, 17, 18, 19]
    assert skeleton.PARENTS[:22] == expected_parents


def test_left_hand_joints_22_to_36():
    """15 left hand finger joints, all rooted at left_wrist (20)."""
    expected = [
        "left_thumb1", "left_thumb2", "left_thumb3",
        "left_index1", "left_index2", "left_index3",
        "left_middle1", "left_middle2", "left_middle3",
        "left_ring1", "left_ring2", "left_ring3",
        "left_pinky1", "left_pinky2", "left_pinky3",
    ]
    assert skeleton.JOINT_NAMES[22:37] == expected
    for j in range(22, 37):
        if (j - 22) % 3 == 0:
            assert skeleton.PARENTS[j] == 20, f"finger root {j} must parent to left_wrist"
        else:
            assert skeleton.PARENTS[j] == j - 1, f"finger chain {j} must parent to {j-1}"


def test_right_hand_joints_37_to_51():
    """15 right hand finger joints, all rooted at right_wrist (21)."""
    expected = [
        "right_thumb1", "right_thumb2", "right_thumb3",
        "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3",
        "right_ring1", "right_ring2", "right_ring3",
        "right_pinky1", "right_pinky2", "right_pinky3",
    ]
    assert skeleton.JOINT_NAMES[37:52] == expected
    for j in range(37, 52):
        if (j - 37) % 3 == 0:
            assert skeleton.PARENTS[j] == 21, f"finger root {j} must parent to right_wrist"
        else:
            assert skeleton.PARENTS[j] == j - 1, f"finger chain {j} must parent to {j-1}"


def test_face_joints_52_to_55():
    assert skeleton.JOINT_NAMES[52] == "jaw"
    assert skeleton.JOINT_NAMES[53] == "left_eye"
    assert skeleton.JOINT_NAMES[54] == "right_eye"
    assert skeleton.JOINT_NAMES[55] == "nose_tip"
    for j in range(52, 56):
        assert skeleton.PARENTS[j] == 15, "face joints parent to head"


def test_heel_joints_56_57():
    assert skeleton.JOINT_NAMES[56] == "left_heel"
    assert skeleton.JOINT_NAMES[57] == "right_heel"
    assert skeleton.PARENTS[56] == 7   # left_ankle
    assert skeleton.PARENTS[57] == 8   # right_ankle


def test_rest_pose_face_above_neck():
    neck_y = skeleton._REST[12, 1]
    for j in range(52, 56):
        assert skeleton._REST[j, 1] > neck_y


def test_rest_pose_hands_near_wrists():
    """Hand joint rest positions should be near their parent wrist."""
    for j in range(22, 37):
        dist = np.linalg.norm(skeleton._REST[j] - skeleton._REST[20])
        assert dist < 0.25, f"left hand joint {j} too far from wrist"
    for j in range(37, 52):
        dist = np.linalg.norm(skeleton._REST[j] - skeleton._REST[21])
        assert dist < 0.25, f"right hand joint {j} too far from wrist"


def test_no_parent_cycles():
    visited = set()
    for j in range(58):
        chain = set()
        cur = j
        while cur != -1:
            assert cur not in chain, f"cycle at joint {cur}"
            chain.add(cur)
            cur = skeleton.PARENTS[cur]
