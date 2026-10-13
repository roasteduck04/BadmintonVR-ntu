"""The 35 keypoints pose24 fits (spec § 4.1): COCO-WholeBody body 17 + feet 6 (same indices as the 133
layout), 8 finger knuckles (MCPs), jaw, the two eye-ring centres and the nose tip."""
from __future__ import annotations

import warnings

import numpy as np

from analysis import conventions
from analysis import pose_v2 as pv

MCP_WB = [96, 100, 104, 108, 117, 121, 125, 129]
LEFT_EYE_RING = list(range(65, 71))     # iBUG 42-47: the subject's left eye
RIGHT_EYE_RING = list(range(59, 65))    # iBUG 36-41
SOURCES = [[i] for i in range(23)] + [[i] for i in MCP_WB] + [[31], LEFT_EYE_RING, RIGHT_EYE_RING, [53]]
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle",
         "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
         "left_index_mcp", "left_middle_mcp", "left_ring_mcp", "left_pinky_mcp",
         "right_index_mcp", "right_middle_mcp", "right_ring_mcp", "right_pinky_mcp",
         "jaw", "left_eye_centre", "right_eye_centre", "nose_tip"]
K = len(NAMES)
CONTACT = [17, 18, 19, 20, 21, 22]      # toes + heels: planted-foot detection

# Rest pose (Y-up body-local metres, facing +z, +x = the subject's left: the right-handed SMPL convention; real
# analysis-world data is mirror-handed, so use rest_offsets_down(sgn), not REST35 directly). Only used to place a keypoint
# that no camera ever saw in a segment (it is then flagged unobserved and gets ~0 confidence).
REST35 = np.array([
    [0.00, 1.62, 0.10], [0.035, 1.66, 0.08], [-0.035, 1.66, 0.08], [0.076, 1.63, 0.0], [-0.076, 1.63, 0.0],
    [0.19, 1.42, 0.0], [-0.19, 1.42, 0.0], [0.22, 1.12, 0.0], [-0.22, 1.12, 0.0],
    [0.24, 0.86, 0.03], [-0.24, 0.86, 0.03], [0.12, 0.92, 0.0], [-0.12, 0.92, 0.0],
    [0.12, 0.50, 0.03], [-0.12, 0.50, 0.03], [0.12, 0.08, 0.0], [-0.12, 0.08, 0.0],
    [0.13, 0.02, 0.17], [0.17, 0.02, 0.14], [0.12, 0.025, -0.06],
    [-0.13, 0.02, 0.17], [-0.17, 0.02, 0.14], [-0.12, 0.025, -0.06],
    [0.25, 0.78, 0.07], [0.25, 0.78, 0.05], [0.25, 0.78, 0.03], [0.25, 0.78, 0.01],
    [-0.25, 0.78, 0.07], [-0.25, 0.78, 0.05], [-0.25, 0.78, 0.03], [-0.25, 0.78, 0.01],
    [0.00, 1.52, 0.08], [0.035, 1.66, 0.085], [-0.035, 1.66, 0.085], [0.00, 1.61, 0.115],
])


def rest_offsets_down(sgn: float = 1.0) -> np.ndarray:
    """REST35 relative to the hip centre in the court world's Y-down convention, with the handedness
    `forward_sign(...) == sgn`. The plain Y flip of the (right-handed, SMPL-convention) REST35 gives -1;
    real players in the Y-down court world are +1, so x is negated whenever the flip gives the other sign.
    Pass the player's own sign."""
    if not np.isfinite(sgn):
        raise ValueError(f"rest_offsets_down needs a handedness sign of +1 / -1, got {sgn!r}")
    D = (REST35 - 0.5 * (REST35[11] + REST35[12])) * np.array([1.0, -1.0, 1.0])
    if forward_sign(D[None]) != (1.0 if sgn >= 0 else -1.0):
        D = D * np.array([-1.0, 1.0, 1.0])
    return D


def scores35(w133):
    """(F,133,...) calibrated scores -> (F,35,...): a ring centre's score is its ring's minimum (NaN = 0). The one
    rule the fit (obs35) and the cvdebug seen bits share (review 1 L6)."""
    w133 = np.nan_to_num(np.asarray(w133, dtype=np.float64))
    return np.stack([w133[:, s].min(axis=1) for s in SOURCES], axis=1)


def obs35(uv133, w133):
    """(F,133,V,2) pixels + (F,133,V) calibrated scores -> (F,35,V,2), (F,35,V). Ring centres average the
    ring's pixels (NaN-safe); their score is the ring's minimum."""
    F, _, V, _ = uv133.shape
    uv = np.full((F, K, V, 2), np.nan)
    for k, src in enumerate(SOURCES):
        if len(src) == 1:
            uv[:, k] = uv133[:, src[0]]
        else:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                uv[:, k] = np.nanmean(uv133[:, src], axis=1)
    return uv, scores35(w133)


def player_obs(bundle, pid):
    """The 35-keypoint observations of one player: uv (F,35,V,2), w (F,35,V)."""
    uv133, w133 = pv.obs_arrays(bundle, pid, idx=list(range(133)))
    return obs35(uv133, w133)


def from133(X):
    """(..., 133, 3) 3D keypoints -> (..., 35, 3) (ring centres = mean)."""
    X = np.asarray(X, dtype=np.float64)
    return np.stack([X[..., s, :].mean(axis=-2) for s in SOURCES], axis=-2)


def forward_sign(X, default: float = 1.0) -> float:
    """+1.0 when cross(left hip - right hip, trunk up) points the way the feet (else the face) point,
    -1.0 when the coordinates are mirrored (e.g. Y flipped). Used so frame rules work in any world.
    `default` when neither feet nor face were seen — pass the world's sign (world_sign) there."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, K, 3)
    left = X[:, 11] - X[:, 12]
    trunk = 0.5 * (X[:, 5] + X[:, 6]) - 0.5 * (X[:, 11] + X[:, 12])
    fwd = np.cross(left, trunk)
    feet = 0.5 * ((X[:, 17] - X[:, 19]) + (X[:, 20] - X[:, 22]))
    face = X[:, 0] - 0.5 * (X[:, 3] + X[:, 4])
    for ref in (feet, face):
        d = np.sum(fwd * ref, axis=-1)
        d = d[np.isfinite(d)]
        if len(d):
            return 1.0 if np.median(np.sign(d)) >= 0 else -1.0
    return float(default)


def world_sign(cams) -> float:
    """forward_sign a real person has in the Y-down court world of these cameras (conventions.world_sign):
    +1.0 for proper rotations (det R = +1, every real calibration); the Y-up analysis world is then -1."""
    return conventions.world_sign(cams)
