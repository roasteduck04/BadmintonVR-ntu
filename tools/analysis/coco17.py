"""COCO-17 (rtmlib Body default) -> 58-joint skeleton mapping.

COCO-17 lacks pelvis, neck, head-top, and feet (toes/heels) that Halpe-26 has.
We synthesize them from midpoints:
  pelvis = (left_hip + right_hip) / 2
  neck   = (left_shoulder + right_shoulder) / 2
  head   = nose (best available proxy)
  big_toe = ankles (no toe data)
COCO-17 has no hand or heel keypoints, so those 32 joints (hands 22-51,
heels 56-57) remain zero with zero confidence.
"""
from __future__ import annotations

import numpy as np

from analysis import skeleton

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_C = {n: i for i, n in enumerate(COCO17_NAMES)}

_DIRECT = {
    "left_hip": _C["left_hip"], "right_hip": _C["right_hip"],
    "left_knee": _C["left_knee"], "right_knee": _C["right_knee"],
    "left_ankle": _C["left_ankle"], "right_ankle": _C["right_ankle"],
    "left_shoulder": _C["left_shoulder"], "right_shoulder": _C["right_shoulder"],
    "left_elbow": _C["left_elbow"], "right_elbow": _C["right_elbow"],
    "left_wrist": _C["left_wrist"], "right_wrist": _C["right_wrist"],
}


def coco17_to_smpl58(joints17, conf17):
    joints17 = np.asarray(joints17, dtype=np.float64)
    conf17 = np.asarray(conf17, dtype=np.float64)
    names = skeleton.JOINT_NAMES
    out = np.zeros((58, 3), dtype=np.float64)
    oc = np.zeros(58, dtype=np.float64)
    idx = {n: i for i, n in enumerate(names)}

    for name, c in _DIRECT.items():
        out[idx[name]] = joints17[c]
        oc[idx[name]] = conf17[c]

    def mid(a, b):
        return (joints17[a] + joints17[b]) / 2.0, min(conf17[a], conf17[b])

    out[idx["pelvis"]], oc[idx["pelvis"]] = mid(_C["left_hip"], _C["right_hip"])
    out[idx["neck"]], oc[idx["neck"]] = mid(_C["left_shoulder"], _C["right_shoulder"])
    out[idx["head"]] = joints17[_C["nose"]]
    oc[idx["head"]] = conf17[_C["nose"]]

    out[idx["left_big_toe"]] = joints17[_C["left_ankle"]]
    oc[idx["left_big_toe"]] = conf17[_C["left_ankle"]] * 0.5
    out[idx["right_big_toe"]] = joints17[_C["right_ankle"]]
    oc[idx["right_big_toe"]] = conf17[_C["right_ankle"]] * 0.5

    def lerp(a_idx, b_idx, u):
        a, b = joints17[a_idx], joints17[b_idx]
        c = min(conf17[a_idx], conf17[b_idx])
        return a + (b - a) * u, c

    pelvis_pos = out[idx["pelvis"]]
    neck_pos = out[idx["neck"]]
    pelvis_conf = oc[idx["pelvis"]]
    neck_conf = oc[idx["neck"]]
    spine_conf = min(pelvis_conf, neck_conf)
    for name, u in (("spine1", 0.25), ("spine2", 0.5), ("spine3", 0.75)):
        out[idx[name]] = pelvis_pos + (neck_pos - pelvis_pos) * u
        oc[idx[name]] = spine_conf

    out[idx["left_collar"]] = (neck_pos + joints17[_C["left_shoulder"]]) / 2.0
    oc[idx["left_collar"]] = min(neck_conf, conf17[_C["left_shoulder"]])
    out[idx["right_collar"]] = (neck_pos + joints17[_C["right_shoulder"]]) / 2.0
    oc[idx["right_collar"]] = min(neck_conf, conf17[_C["right_shoulder"]])

    # Hands (22-51) and heels (56-57): COCO-17 has no hand or heel data,
    # so those joints remain zero with zero confidence.

    # Face: COCO-17 has nose (0), left_eye (1), right_eye (2)
    out[idx["nose_tip"]] = joints17[_C["nose"]]
    oc[idx["nose_tip"]] = conf17[_C["nose"]]
    out[idx["left_eye"]] = joints17[_C["left_eye"]]
    oc[idx["left_eye"]] = conf17[_C["left_eye"]]
    out[idx["right_eye"]] = joints17[_C["right_eye"]]
    oc[idx["right_eye"]] = conf17[_C["right_eye"]]
    # jaw: no COCO-17 source -> remains zero conf

    return out, oc
