"""Coordinate conventions shared by the pose route: the Y flip, handedness signs and the body heading.

Worlds
  court world     Y down, right-handed: every real calibration has det R = +1 (OpenCV x right / y down / z forward).
  analysis world  Y up, metres: the court world with only Y negated (`Y_FLIP`), so it is MIRROR-handed relative to
                  the real scene. Every analysis.json joint is in this world (CLAUDE.md "Handedness").

Handedness sign (`forward_sign`)
  +1.0 when cross(left - right, up) points the way the body faces (feet / face): the right-handed SMPL rest
  convention (+x = the subject's left, +y up, +z forward). -1.0 when the coordinates are mirrored.
  `world_sign(cams)` = the sign a real person has in the Y-down court world of these cameras (+1 for every real
  calibration); `analysis_sign(cams)` = the sign in the Y-up analysis world (= -world_sign, the Y flip mirrors).
  Never hard-code it: read it per clip / player (pose24.kpset.forward_sign on the 35 fit keypoints,
  `forward_sign_smpl` on SMPL-layout doc joints) with the world's sign as the fallback.

Heading
  `heading_of(left, right, up, sign)` = atan2(forward.x, forward.z), forward = sign * cross(left - right, up):
  0 = facing +Z, +pi/2 = facing +X (same in the Y-up and Y-down worlds: x and z are not flipped).
"""
from __future__ import annotations

import numpy as np

Y_FLIP = np.array([1.0, -1.0, 1.0])      # court world (Y down) <-> analysis world (Y up)
UP_Y = np.array([0.0, 1.0, 0.0])          # world up in the analysis world
UP_COURT = np.array([0.0, -1.0, 0.0])     # world up in the court world
# Fallback when a doc carries neither cameras nor a foot / face cue: the analysis-world sign of every real
# calibration so far (det R = +1 -> world_sign +1 -> analysis -1). Only used when nothing better is known.
REAL_ANALYSIS_SIGN = -1.0

# SMPL body indices (identical in SMPL-24, SMPL-28 and the first 22 joints of SMPL-X 58)
SMPL_L_HIP, SMPL_R_HIP = 1, 2
SMPL_L_ANKLE, SMPL_R_ANKLE = 7, 8
SMPL_L_FOOT, SMPL_R_FOOT = 10, 11
SMPL_L_SHOULDER, SMPL_R_SHOULDER = 16, 17


def world_sign(cams) -> float:
    """forward_sign a real person has in the Y-down court world of these cameras: +1.0 when their rotations
    are proper (det R = +1 -> a right-handed world), -1.0 when the world is mirrored. `cams` = {view: (K, R,
    ...)}. Every real calibration has det R = +1 (2026-09-28); the Y-up analysis world is then -1."""
    d = [np.linalg.det(np.asarray(c[1], dtype=np.float64)) for c in cams.values()]
    return 1.0 if np.median(np.sign(d)) >= 0 else -1.0


def analysis_sign(cams) -> float:
    """forward_sign a real person has in the Y-up analysis world of these cameras (the Y flip mirrors)."""
    return -world_sign(cams)


def _sign_value(sign) -> float:
    s = float(sign)
    if not np.isfinite(s) or s == 0.0:
        raise ValueError(f"handedness sign must be +1 / -1, got {sign!r}")
    return 1.0 if s > 0 else -1.0


def forward_of(left, right, up, sign):
    """Body forward (unnormalised) from a left / right joint pair: sign * cross(left - right, up). Broadcasts."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return _sign_value(sign) * np.cross(left - right, np.broadcast_to(np.asarray(up, float), left.shape))


def heading_of(left, right, up, sign):
    """Heading (rad) of the body whose left / right joints are given: 0 = facing +Z, +pi/2 = facing +X.
    `sign` = the handedness of the coordinates (+1 right-handed, -1 mirror-handed: real analysis docs).
    Scalar inputs -> float (None when degenerate); arrays (..., 3) -> array (NaN when degenerate)."""
    f = forward_of(left, right, up, sign)
    h = np.hypot(f[..., 0], f[..., 2])
    with np.errstate(invalid="ignore"):
        out = np.where(h > 1e-8, np.arctan2(f[..., 0], f[..., 2]), np.nan)
    if np.ndim(out) == 0:
        return None if not np.isfinite(out) else float(out)
    return out


def forward_sign_smpl(J, default: float) -> float:
    """forward_sign of SMPL-layout joints J (F, >= 22, 3), any world: +1 when cross(left hip - right hip, trunk up)
    points the way the feet (ankle -> foot) point, -1 when mirrored; `default` when no frame has both. The trunk
    (hip centre -> shoulder centre) is used as "up", so it works in the Y-up and the Y-down world alike."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 3 or J.shape[1] < 22:
        return _sign_value(default)
    left = J[:, SMPL_L_HIP] - J[:, SMPL_R_HIP]
    trunk = 0.5 * (J[:, SMPL_L_SHOULDER] + J[:, SMPL_R_SHOULDER]) - 0.5 * (J[:, SMPL_L_HIP] + J[:, SMPL_R_HIP])
    feet = 0.5 * ((J[:, SMPL_L_FOOT] - J[:, SMPL_L_ANKLE]) + (J[:, SMPL_R_FOOT] - J[:, SMPL_R_ANKLE]))
    d = np.sum(np.cross(left, trunk) * feet, axis=-1)
    d = d[np.isfinite(d) & (np.abs(d) > 1e-12)]
    if not len(d):
        return _sign_value(default)
    return 1.0 if np.median(np.sign(d)) >= 0 else -1.0
