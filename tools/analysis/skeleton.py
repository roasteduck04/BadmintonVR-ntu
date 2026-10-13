"""Canonical 58-joint skeleton — SMPL-X body + hands + face + heels.

The analysis.json v5 pose uses this 58-joint layout, body-local, Y-up meters.
Joints 0-21: SMPL-X canonical body (10/11 = big_toe, no left_hand/right_hand).
Joints 22-36: left hand (5 fingers × 3 joints, rooted at left_wrist=20).
Joints 37-51: right hand (mirror, rooted at right_wrist=21).
Joints 52-55: face (jaw, left_eye, right_eye, nose_tip, all parented to head=15).
Joints 56-57: heels (left_heel parented to left_ankle=7, right_heel to right_ankle=8).

Replaces the archived skeleton_smpl28.py as the single source of truth.
"""
from __future__ import annotations

import numpy as np

NUM_JOINTS = 58

JOINT_NAMES = [
    # Body (0-21)
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_big_toe",
    "right_big_toe", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    # Left hand (22-36)
    "left_thumb1", "left_thumb2", "left_thumb3",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    # Right hand (37-51)
    "right_thumb1", "right_thumb2", "right_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    # Face (52-55)
    "jaw", "left_eye", "right_eye", "nose_tip",
    # Heels (56-57)
    "left_heel", "right_heel",
]

PARENTS = [
    -1,  # 0  pelvis
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_big_toe
     8,  # 11 right_big_toe
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_thumb1
    22,  # 23 left_thumb2
    23,  # 24 left_thumb3
    20,  # 25 left_index1
    25,  # 26 left_index2
    26,  # 27 left_index3
    20,  # 28 left_middle1
    28,  # 29 left_middle2
    29,  # 30 left_middle3
    20,  # 31 left_ring1
    31,  # 32 left_ring2
    32,  # 33 left_ring3
    20,  # 34 left_pinky1
    34,  # 35 left_pinky2
    35,  # 36 left_pinky3
    21,  # 37 right_thumb1
    37,  # 38 right_thumb2
    38,  # 39 right_thumb3
    21,  # 40 right_index1
    40,  # 41 right_index2
    41,  # 42 right_index3
    21,  # 43 right_middle1
    43,  # 44 right_middle2
    44,  # 45 right_middle3
    21,  # 46 right_ring1
    46,  # 47 right_ring2
    47,  # 48 right_ring3
    21,  # 49 right_pinky1
    49,  # 50 right_pinky2
    50,  # 51 right_pinky3
    15,  # 52 jaw
    15,  # 53 left_eye
    15,  # 54 right_eye
    15,  # 55 nose_tip
     7,  # 56 left_heel
     8,  # 57 right_heel
]

_REST = np.array([
    # Body (0-21)
    [0.00, 0.95, 0.00],   # 0  pelvis
    [0.08, 0.90, 0.00],   # 1  left_hip
    [-0.08, 0.90, 0.00],  # 2  right_hip
    [0.00, 1.05, 0.00],   # 3  spine1
    [0.09, 0.50, 0.00],   # 4  left_knee
    [-0.09, 0.50, 0.00],  # 5  right_knee
    [0.00, 1.15, 0.00],   # 6  spine2
    [0.09, 0.08, 0.00],   # 7  left_ankle
    [-0.09, 0.08, 0.00],  # 8  right_ankle
    [0.00, 1.25, 0.00],   # 9  spine3
    [0.09, 0.02, 0.12],   # 10 left_big_toe
    [-0.09, 0.02, 0.12],  # 11 right_big_toe
    [0.00, 1.45, 0.00],   # 12 neck
    [0.06, 1.38, 0.00],   # 13 left_collar
    [-0.06, 1.38, 0.00],  # 14 right_collar
    [0.00, 1.60, 0.00],   # 15 head
    [0.18, 1.40, 0.00],   # 16 left_shoulder
    [-0.18, 1.40, 0.00],  # 17 right_shoulder
    [0.42, 1.40, 0.00],   # 18 left_elbow
    [-0.42, 1.40, 0.00],  # 19 right_elbow
    [0.65, 1.40, 0.00],   # 20 left_wrist
    [-0.65, 1.40, 0.00],  # 21 right_wrist
    # Left hand (22-36) — fingers spread in T-pose, each ~2cm apart
    [0.70, 1.40, -0.02],  # 22 left_thumb1
    [0.74, 1.40, -0.03],  # 23 left_thumb2
    [0.78, 1.40, -0.04],  # 24 left_thumb3
    [0.70, 1.40, 0.00],   # 25 left_index1
    [0.75, 1.40, 0.00],   # 26 left_index2
    [0.80, 1.40, 0.00],   # 27 left_index3
    [0.70, 1.40, 0.02],   # 28 left_middle1
    [0.75, 1.40, 0.02],   # 29 left_middle2
    [0.80, 1.40, 0.02],   # 30 left_middle3
    [0.70, 1.40, 0.04],   # 31 left_ring1
    [0.75, 1.40, 0.04],   # 32 left_ring2
    [0.80, 1.40, 0.04],   # 33 left_ring3
    [0.70, 1.40, 0.06],   # 34 left_pinky1
    [0.75, 1.40, 0.06],   # 35 left_pinky2
    [0.80, 1.40, 0.06],   # 36 left_pinky3
    # Right hand (37-51) — mirror of left
    [-0.70, 1.40, -0.02], # 37 right_thumb1
    [-0.74, 1.40, -0.03], # 38 right_thumb2
    [-0.78, 1.40, -0.04], # 39 right_thumb3
    [-0.70, 1.40, 0.00],  # 40 right_index1
    [-0.75, 1.40, 0.00],  # 41 right_index2
    [-0.80, 1.40, 0.00],  # 42 right_index3
    [-0.70, 1.40, 0.02],  # 43 right_middle1
    [-0.75, 1.40, 0.02],  # 44 right_middle2
    [-0.80, 1.40, 0.02],  # 45 right_middle3
    [-0.70, 1.40, 0.04],  # 46 right_ring1
    [-0.75, 1.40, 0.04],  # 47 right_ring2
    [-0.80, 1.40, 0.04],  # 48 right_ring3
    [-0.70, 1.40, 0.06],  # 49 right_pinky1
    [-0.75, 1.40, 0.06],  # 50 right_pinky2
    [-0.80, 1.40, 0.06],  # 51 right_pinky3
    # Face (52-55)
    [0.00, 1.52, 0.04],   # 52 jaw
    [0.03, 1.58, 0.06],   # 53 left_eye
    [-0.03, 1.58, 0.06],  # 54 right_eye
    [0.00, 1.56, 0.08],   # 55 nose_tip
    # Heels (56-57)
    [0.09, 0.02, -0.06],  # 56 left_heel
    [-0.09, 0.02, -0.06], # 57 right_heel
])
