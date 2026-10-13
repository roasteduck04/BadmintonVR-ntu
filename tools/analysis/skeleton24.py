"""SMPL-24 / SMPL-28 / AUX-14 definitions for the pose24 route (analysis.json schema 4.1).

SMPL-24 are the SMPL body joints; SMPL-28 appends jaw, left_eye, right_eye and nose_tip (children of
head) for head orientation; AUX-14 are extra keypoints kept in a separate per-frame list (heels, toes,
finger knuckles) for foot contact, ankle angles and the racket grip. Copied from _archive/smpl24.py and
_archive/skeleton_smpl28.py (both kept there). The SMPL-X 58 skeleton stays in analysis.skeleton.
"""
from __future__ import annotations

import numpy as np

SMPL24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                  16, 17, 18, 19, 20, 21]
FACE4_NAMES = ["jaw", "left_eye", "right_eye", "nose_tip"]
SMPL28_NAMES = SMPL24_NAMES + FACE4_NAMES
SMPL28_PARENTS = SMPL24_PARENTS + [15, 15, 15, 15]

# Approximate T-pose rest (Y-up metres, +x = subject's left), index-aligned to SMPL28_NAMES.
REST28 = np.array([
    [0.00, 0.95, 0.00], [0.08, 0.90, 0.00], [-0.08, 0.90, 0.00], [0.00, 1.05, 0.00],
    [0.09, 0.50, 0.00], [-0.09, 0.50, 0.00], [0.00, 1.15, 0.00], [0.09, 0.08, 0.00],
    [-0.09, 0.08, 0.00], [0.00, 1.25, 0.00], [0.09, 0.02, 0.12], [-0.09, 0.02, 0.12],
    [0.00, 1.45, 0.00], [0.06, 1.38, 0.00], [-0.06, 1.38, 0.00], [0.00, 1.60, 0.00],
    [0.18, 1.40, 0.00], [-0.18, 1.40, 0.00], [0.42, 1.40, 0.00], [-0.42, 1.40, 0.00],
    [0.65, 1.40, 0.00], [-0.65, 1.40, 0.00], [0.72, 1.40, 0.00], [-0.72, 1.40, 0.00],
    [0.00, 1.52, 0.04], [0.03, 1.58, 0.06], [-0.03, 1.58, 0.06], [0.00, 1.56, 0.08],
])

AUX_NAMES = [
    "left_heel", "right_heel", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_index_mcp", "left_middle_mcp", "left_ring_mcp", "left_pinky_mcp",
    "right_index_mcp", "right_middle_mcp", "right_ring_mcp", "right_pinky_mcp",
]
AUX_WB_IDX = [19, 22, 17, 20, 18, 21, 96, 100, 104, 108, 117, 121, 125, 129]   # COCO-WholeBody 133

TYPES = {
    "smpl-24": (SMPL24_NAMES, SMPL24_PARENTS, "SMPL-24"),
    "smpl-28": (SMPL28_NAMES, SMPL28_PARENTS, "SMPL-28 (SMPL-24 + 4 face)"),
}


def skeleton_block(kind: str = "smpl-28") -> dict:
    """The analysis doc's `skeleton` object for `kind` ("smpl-24" | "smpl-28")."""
    names, parents, label = TYPES[kind]
    return {"type": kind, "family": "SMPL", "label": label, "joint_names": list(names),
            "parents": list(parents), "aux_names": list(AUX_NAMES)}
