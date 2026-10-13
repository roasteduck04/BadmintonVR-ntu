"""Facing direction estimation from 3D joints.

Computes body orientation (hip/shoulder headings) and head orientation
(face-joint heading) per frame. Results are baked into analysis.json so
downstream consumers don't need to recompute.

Handedness: the hip / shoulder heading needs the coordinates' handedness `sign` (analysis.conventions):
real analysis docs are mirror-handed (sign -1). Callers pass it per clip / player: pose24 the sign its
regressor read from the fit keypoints, the legacy routes `frames_sign(frames, default=analysis_sign(cams))`.
The head heading (eye centre -> nose) is purely geometric and needs no sign.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter

from analysis import conventions

# --- Parameters (spec §Parameters) ---
HIP_CONF_MIN = 0.1       # min hip joint confidence for facing
FACING_SG_WINDOW = 7     # Savitzky-Golay smoothing window
FACING_SG_ORDER = 2      # Savitzky-Golay polynomial order

# SMPL-X 58 joint indices
_L_HIP = 1
_R_HIP = 2
_L_SHOULDER = 16
_R_SHOULDER = 17
_LEFT_EYE = 53
_RIGHT_EYE = 54
_NOSE_TIP = 55
_STRIDE = 4              # joints_flat: x, y, z, confidence per joint
HEAD_CONF_MIN = 0.1      # min face joint confidence for head heading


def _joint_xyz_conf(joints_flat, idx):
    """Extract (x, y, z), confidence for joint idx from joints_flat."""
    base = idx * _STRIDE
    x, y, z = joints_flat[base], joints_flat[base + 1], joints_flat[base + 2]
    conf = joints_flat[base + 3]
    return np.array([x, y, z]), conf


def _heading_from_lr(left_xyz, right_xyz, sign):
    """Forward heading of a left / right joint pair in the Y-up analysis world (conventions.heading_of):
    forward = sign * cross(left - right, Y_UP); heading = atan2(forward.x, forward.z): 0 = +Z, pi/2 = +X.
    None when degenerate."""
    return conventions.heading_of(left_xyz, right_xyz, conventions.UP_Y, sign)


def frames_sign(player_frames, default=conventions.REAL_ANALYSIS_SIGN, conf_min=HIP_CONF_MIN):
    """Handedness sign of one player's doc frames (SMPL layout joints_flat, Y-up), read from the feet
    (conventions.forward_sign_smpl; joints below conf_min ignored); `default` when no frame shows hips,
    shoulders and feet - pass conventions.analysis_sign(cams) when the cameras are known."""
    idx = [conventions.SMPL_L_HIP, conventions.SMPL_R_HIP, conventions.SMPL_L_ANKLE, conventions.SMPL_R_ANKLE,
           conventions.SMPL_L_FOOT, conventions.SMPL_R_FOOT, conventions.SMPL_L_SHOULDER,
           conventions.SMPL_R_SHOULDER]
    rows = []
    for fr in player_frames:
        jf = fr.get("joints_flat")
        if jf is None or len(jf) < 22 * _STRIDE:
            continue
        a = np.asarray(jf[:22 * _STRIDE], dtype=np.float64).reshape(22, _STRIDE)
        if (a[idx, 3] >= conf_min).all():
            rows.append(a[:, :3])
    if not rows:
        return conventions.forward_sign_smpl(np.zeros((0, 22, 3)), default)
    return conventions.forward_sign_smpl(np.stack(rows), default)


_FACE_58 = (_LEFT_EYE, _RIGHT_EYE, _NOSE_TIP)
_HIPS = (_L_HIP, _R_HIP)
_SHOULDERS = (_L_SHOULDER, _R_SHOULDER)


def doc_joint_indices(doc) -> dict:
    """fill_facing's face / hips / shoulders indices for a doc, by joint name from its skeleton (joint_names; else
    by the joint count of its first posed frame: 58 = SMPL-X 58, 28 / 24 = SMPL-28 / SMPL-24). face None when the
    skeleton has no eyes + nose tip (SMPL-24)."""
    from analysis import schema
    from analysis import skeleton24 as sk
    names = (doc.get("skeleton") or {}).get("joint_names")
    if not names:
        n = next((len(fr["joints_flat"]) // _STRIDE for p in doc.get("players", []) for fr in p.get("frames", [])
                  if fr.get("joints_flat")), None)
        names = {58: schema.JOINT_NAMES, 28: sk.SMPL28_NAMES, 24: sk.TYPES["smpl-24"][0]}.get(n)
        if names is None:
            raise ValueError(f"facing: no skeleton.joint_names and an unknown joint count {n}")
    ix = {n: i for i, n in enumerate(names)}
    face = tuple(ix[n] for n in ("left_eye", "right_eye", "nose_tip")) if all(
        n in ix for n in ("left_eye", "right_eye", "nose_tip")) else None
    return {"face": face, "hips": (ix["left_hip"], ix["right_hip"]),
            "shoulders": (ix["left_shoulder"], ix["right_shoulder"])}


def _head_heading_from_face(joints_flat, face=_FACE_58, conf_min=HEAD_CONF_MIN):
    """Compute head heading from face joints (eye midpoint → nose_tip).

    face = (left_eye, right_eye, nose_tip) joint indices (default SMPL-X 58; SMPL-28 passes 25, 26, 27).
    conf_min = minimum face joint confidence.
    Returns heading angle in radians or None if face joints are missing.
    """
    le_i, re_i, nose_i = face
    nose, nose_conf = _joint_xyz_conf(joints_flat, nose_i)
    if nose_conf < conf_min:
        return None

    le, le_conf = _joint_xyz_conf(joints_flat, le_i)
    re, re_conf = _joint_xyz_conf(joints_flat, re_i)

    if le_conf >= conf_min and re_conf >= conf_min:
        eye_mid = (le + re) * 0.5
    elif le_conf >= conf_min:
        eye_mid = le
    elif re_conf >= conf_min:
        eye_mid = re
    else:
        return None

    forward = nose - eye_mid
    xz_norm = math.sqrt(forward[0] ** 2 + forward[2] ** 2)
    if xz_norm < 1e-8:
        return None
    return math.atan2(forward[0], forward[2])


def compute_facing(joints_flat, *, sign, face=_FACE_58, conf_min=None, hips=_HIPS, shoulders=_SHOULDERS):
    """Compute facing direction for a single frame. `sign` = the handedness of the coordinates
    (conventions: +1 right-handed, -1 mirror-handed = real analysis docs); required, never assumed.

    Returns {"heading", "shoulder_heading", "confidence"} or None.
    Includes "head_heading" when face joints are available; face = (left_eye, right_eye, nose_tip)
    joint indices (default SMPL-X 58), None = no head heading (e.g. SMPL-24, which has no face joints).
    conf_min = minimum hip / shoulder / face joint confidence; None = HIP_CONF_MIN / HEAD_CONF_MIN
    (pose24's schema 4.1 exp(-sigma/40 mm) confidences pass 1e-3: a far player's hips sit near 0.02).
    hips / shoulders = (left, right) joint indices (default: the SMPL layout every skeleton shares); a doc's own
    face / hips / shoulders: doc_joint_indices(doc).
    """
    hip_min = HIP_CONF_MIN if conf_min is None else conf_min
    head_min = HEAD_CONF_MIN if conf_min is None else conf_min
    if joints_flat is None:
        return None

    l_hip, l_conf = _joint_xyz_conf(joints_flat, hips[0])
    r_hip, r_conf = _joint_xyz_conf(joints_flat, hips[1])

    min_conf = min(l_conf, r_conf)
    if min_conf < hip_min:
        return None

    heading = _heading_from_lr(l_hip, r_hip, sign)
    if heading is None:
        return None

    # Shoulder heading (may be None if shoulders are missing/degenerate)
    shoulder_heading = heading  # fallback
    l_sh, l_sh_conf = _joint_xyz_conf(joints_flat, shoulders[0])
    r_sh, r_sh_conf = _joint_xyz_conf(joints_flat, shoulders[1])
    if min(l_sh_conf, r_sh_conf) >= hip_min:
        sh = _heading_from_lr(l_sh, r_sh, sign)
        if sh is not None:
            shoulder_heading = sh

    result = {
        "heading": round(heading, 4),
        "shoulder_heading": round(shoulder_heading, 4),
        "confidence": round(min_conf, 3),
    }

    hh = _head_heading_from_face(joints_flat, face, head_min) if face else None
    if hh is not None:
        result["head_heading"] = round(hh, 4)

    return result


def fill_facing(player_frames, fps, *, sign, face=_FACE_58, conf_min=None, hips=_HIPS, shoulders=_SHOULDERS):
    """Compute facing direction for all frames of one player (sign / face / conf_min / hips / shoulders as in
    compute_facing).

    Modifies player_frames in place, adding "facing" key to each frame dict.
    Applies SG smoothing to heading angles with angle unwrapping.
    """
    # Pass 1: compute raw facing per frame
    for fr in player_frames:
        fr["facing"] = compute_facing(fr.get("joints_flat"), sign=sign, face=face, conf_min=conf_min, hips=hips,
                                      shoulders=shoulders)

    # Pass 2: SG smooth heading and shoulder_heading
    n = len(player_frames)
    if n < FACING_SG_WINDOW:
        return

    for key in ("heading", "shoulder_heading", "head_heading"):
        raw = np.full(n, np.nan)
        for i, fr in enumerate(player_frames):
            if fr["facing"] is not None and key in fr["facing"]:
                raw[i] = fr["facing"][key]

        valid = ~np.isnan(raw)
        if valid.sum() < FACING_SG_WINDOW:
            continue

        # Unwrap angles to avoid -pi/+pi discontinuity
        unwrapped = np.unwrap(raw[valid])
        smoothed = savgol_filter(unwrapped, FACING_SG_WINDOW, FACING_SG_ORDER)

        # Rewrap to [-pi, pi]
        smoothed = (smoothed + math.pi) % (2 * math.pi) - math.pi

        # Write back
        idx = 0
        for i in range(n):
            if valid[i]:
                player_frames[i]["facing"][key] = round(float(smoothed[idx]), 4)
                idx += 1
