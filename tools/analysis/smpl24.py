"""Canonical SMPL-24 joint layout — the analysis contract's body definition.

The `analysis.json v3` pose is SMPL-24, body-local, Y-up meters. These constants
are the single source of truth for that layout (joint count, names, kinematic
parents, and an approximate T-pose rest). They live in the analysis module so the
live contract owns its own body definition and does not depend on any pose-engine
tool. Previously these were reused from `smpl_to_skeleton.py` (the monocular
WHAM/SMPL lane, now archived under `_archive/old-pipeline/`).
"""
from __future__ import annotations

import numpy as np

NUM_SMPL_JOINTS = 24

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]

# Approximate SMPL rest pose (Y-up meters), a T-pose. Index-aligned to
# SMPL_JOINT_NAMES. Used to exercise the contract/report with no GPU.
_REST = np.array([
    [0.00, 0.95, 0.00], [0.08, 0.90, 0.00], [-0.08, 0.90, 0.00], [0.00, 1.05, 0.00],
    [0.09, 0.50, 0.00], [-0.09, 0.50, 0.00], [0.00, 1.15, 0.00], [0.09, 0.08, 0.00],
    [-0.09, 0.08, 0.00], [0.00, 1.25, 0.00], [0.09, 0.02, 0.12], [-0.09, 0.02, 0.12],
    [0.00, 1.45, 0.00], [0.06, 1.38, 0.00], [-0.06, 1.38, 0.00], [0.00, 1.60, 0.00],
    [0.18, 1.40, 0.00], [-0.18, 1.40, 0.00], [0.42, 1.40, 0.00], [-0.42, 1.40, 0.00],
    [0.65, 1.40, 0.00], [-0.65, 1.40, 0.00], [0.72, 1.40, 0.00], [-0.72, 1.40, 0.00],
])
