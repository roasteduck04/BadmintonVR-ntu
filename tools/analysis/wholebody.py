"""COCO-WholeBody 133 keypoints -> 58-joint skeleton mapping.

COCO-WholeBody has 17 body + 6 foot + 68 face + 42 hand keypoints.
The 6 foot keypoints (indices 17-22) give us real big_toe/heel positions
that COCO-17 lacks — much better foot placement than the ankle proxy.
The 68 face keypoints (indices 23-90) give us real jaw, nose_tip, and
eye positions for the skeleton's 4 face joints. The 42 hand keypoints
(indices 91-132, 21 per hand) give us real finger joint positions for
the skeleton's 30 hand joints. See analysis.skeleton.
"""
from __future__ import annotations

import numpy as np

from analysis import skeleton

N_KEYPOINTS = 133

# Face landmark indices within 133 COCO-WholeBody keypoints.
# Body 0-16, foot 17-22, face 23-90 (68 landmarks in iBUG order), hands 91-132.
_FACE_BASE = 23
_JAW_IDX = _FACE_BASE + 8         # chin bottom (iBUG face index 8)
_NOSE_TIP_IDX = _FACE_BASE + 30   # nose tip (iBUG face index 30)
# iBUG: RIGHT eye = face indices 36-41, LEFT eye = face indices 42-47
# (anatomical right/left — viewer's opposite)
_RIGHT_EYE = list(range(_FACE_BASE + 36, _FACE_BASE + 42))  # globals 59-64
_LEFT_EYE = list(range(_FACE_BASE + 42, _FACE_BASE + 48))   # globals 65-70

# Hand keypoint block base indices (21 keypoints per hand).
_LEFT_HAND_BASE = 91
_RIGHT_HAND_BASE = 112

# Per-finger offsets within a 21-keypoint hand block, mapped to the
# 15 finger-joint offsets (0-14) within a hand's SMPL-X joint range.
# 0=wrist (discarded), 4/8/12/16/20=fingertips (discarded).
_HAND_MAP = [
    (1, 0), (2, 1), (3, 2),       # thumb CMC/MCP/IP -> thumb1/2/3
    (5, 3), (6, 4), (7, 5),       # index MCP/PIP/DIP -> index1/2/3
    (9, 6), (10, 7), (11, 8),     # middle MCP/PIP/DIP -> middle1/2/3
    (13, 9), (14, 10), (15, 11),  # ring MCP/PIP/DIP -> ring1/2/3
    (17, 12), (18, 13), (19, 14), # pinky MCP/PIP/DIP -> pinky1/2/3
]

_BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_FOOT_NAMES = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

_B = {n: i for i, n in enumerate(_BODY_NAMES)}
_F = {n: 17 + i for i, n in enumerate(_FOOT_NAMES)}

_DIRECT = {
    "left_hip": _B["left_hip"], "right_hip": _B["right_hip"],
    "left_knee": _B["left_knee"], "right_knee": _B["right_knee"],
    "left_ankle": _B["left_ankle"], "right_ankle": _B["right_ankle"],
    "left_shoulder": _B["left_shoulder"], "right_shoulder": _B["right_shoulder"],
    "left_elbow": _B["left_elbow"], "right_elbow": _B["right_elbow"],
    "left_wrist": _B["left_wrist"], "right_wrist": _B["right_wrist"],
}


def wholebody133_to_smpl58(joints133, conf133):
    """Convert 133-keypoint wholebody to the 58-joint skeleton.

    Uses real big_toe/heel foot keypoints instead of ankle proxies, and
    real finger keypoints for the 30 hand joints instead of extrapolation.
    """
    joints133 = np.asarray(joints133, dtype=np.float64)
    conf133 = np.asarray(conf133, dtype=np.float64)
    names = skeleton.JOINT_NAMES
    out = np.zeros((58, 3), dtype=np.float64)
    oc = np.zeros(58, dtype=np.float64)
    idx = {n: i for i, n in enumerate(names)}

    for name, c in _DIRECT.items():
        out[idx[name]] = joints133[c]
        oc[idx[name]] = conf133[c]

    def mid(a, b):
        return (joints133[a] + joints133[b]) / 2.0, min(conf133[a], conf133[b])

    out[idx["pelvis"]], oc[idx["pelvis"]] = mid(_B["left_hip"], _B["right_hip"])
    out[idx["neck"]], oc[idx["neck"]] = mid(
        _B["left_shoulder"], _B["right_shoulder"])
    out[idx["head"]] = joints133[_B["nose"]]
    oc[idx["head"]] = conf133[_B["nose"]]

    # Big toes: direct from RTMPose foot keypoints.
    out[idx["left_big_toe"]] = joints133[_F["left_big_toe"]]
    oc[idx["left_big_toe"]] = conf133[_F["left_big_toe"]]
    out[idx["right_big_toe"]] = joints133[_F["right_big_toe"]]
    oc[idx["right_big_toe"]] = conf133[_F["right_big_toe"]]

    # Heels: direct from RTMPose foot keypoints.
    out[idx["left_heel"]] = joints133[_F["left_heel"]]
    oc[idx["left_heel"]] = conf133[_F["left_heel"]]
    out[idx["right_heel"]] = joints133[_F["right_heel"]]
    oc[idx["right_heel"]] = conf133[_F["right_heel"]]

    pelvis_pos = out[idx["pelvis"]]
    neck_pos = out[idx["neck"]]
    pelvis_conf = oc[idx["pelvis"]]
    neck_conf = oc[idx["neck"]]
    spine_conf = min(pelvis_conf, neck_conf)
    for name, u in (("spine1", 0.25), ("spine2", 0.5), ("spine3", 0.75)):
        out[idx[name]] = pelvis_pos + (neck_pos - pelvis_pos) * u
        oc[idx[name]] = spine_conf

    out[idx["left_collar"]] = (neck_pos + joints133[_B["left_shoulder"]]) / 2.0
    oc[idx["left_collar"]] = min(neck_conf, conf133[_B["left_shoulder"]])
    out[idx["right_collar"]] = (neck_pos + joints133[_B["right_shoulder"]]) / 2.0
    oc[idx["right_collar"]] = min(neck_conf, conf133[_B["right_shoulder"]])

    # Hands: real finger keypoints from RTMPose Wholebody hand blocks.
    for kp, fj in _HAND_MAP:
        src = _LEFT_HAND_BASE + kp
        out[22 + fj] = joints133[src]
        oc[22 + fj] = conf133[src]
    for kp, fj in _HAND_MAP:
        src = _RIGHT_HAND_BASE + kp
        out[37 + fj] = joints133[src]
        oc[37 + fj] = conf133[src]

    # Face joints from COCO-WholeBody 68 face keypoints
    out[idx["jaw"]] = joints133[_JAW_IDX]
    oc[idx["jaw"]] = conf133[_JAW_IDX]

    out[idx["nose_tip"]] = joints133[_NOSE_TIP_IDX]
    oc[idx["nose_tip"]] = conf133[_NOSE_TIP_IDX]

    # Eyes: midpoint of 6 keypoints each
    le_pts = joints133[_LEFT_EYE]
    le_confs = conf133[_LEFT_EYE]
    out[idx["left_eye"]] = le_pts.mean(axis=0)
    oc[idx["left_eye"]] = float(le_confs.min())

    re_pts = joints133[_RIGHT_EYE]
    re_confs = conf133[_RIGHT_EYE]
    out[idx["right_eye"]] = re_pts.mean(axis=0)
    oc[idx["right_eye"]] = float(re_confs.min())

    return out, oc
