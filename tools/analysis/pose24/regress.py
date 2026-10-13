"""Anatomical 35 fit keypoints -> SMPL-28 joints + AUX-14 (spec § 3; review H5).

Works in the analysis world (Y-up, metres). Frames (SMPL rest convention +x = subject's left, +y up, +z fwd):
  pelvis  x = hip line (L - R); y = world up orthogonalised to x; z = x × y. ASSUMPTION: pelvis tilt isn't
          observable from two hip keypoints, so the pelvis is taken upright and trunk lean/bend shows up in
          the spine curve (plan deviation 1).
  thorax  x = shoulder line; y = hip centre -> shoulder centre orthogonalised to x; z = x × y
  head    x = ear line; f = ear centre -> nose orthogonalised to x; y = f × x
The coordinate system's handedness is read from the feet (kpset.forward_sign); mirrored input is flipped
in x before the rules and back after. Every constant is an ASSUMPTION with its source and is written into
each doc's meta.pose24.regressor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from analysis import skeleton24 as sk
from analysis.pose24 import kpset

UP = np.array([0.0, 1.0, 0.0])

CONSTANTS = {
    # ASSUMPTION hip joint centre from the COCO hips (spec § 3 defaults, Harrington-style): medial fraction of
    # half the hip width, inferior / anterior fractions of the hip width. To be refit on 3DPW (§ 3.1, F0).
    "hjc_medial": 0.70, "hjc_inferior": 0.12, "hjc_anterior": 0.0,
    # ASSUMPTION SMPL neutral template (F0): pelvis 9.1 cm above, 0.56 cm in front of the HJC centre for a
    # 13.7 cm HJC width (x HJC width).
    "pelvis_up": 0.664, "pelvis_fwd": 0.041,
    # ASSUMPTION SMPL template: neck in the thorax frame (up, forward) x |shoulder centre - hip centre| (~C7).
    "neck_up": 0.114, "neck_fwd": 0.013,
    # ASSUMPTION SMPL template: spine1/2/3 at these fractions of a Hermite curve pelvis -> neck (tangents =
    # pelvis / thorax up x length), offset (up, forward) in the slerped pelvis->thorax frame x length.
    "spine_t": [0.216, 0.479, 0.577],
    "spine_off": [[-0.003, -0.035], [-0.001, -0.014], [0.002, 0.045]],
    # ASSUMPTION SMPL template: collar = shoulder centre + (lateral, up, forward) x shoulder width.
    "collar": [0.231, -0.092, 0.019],
    # ASSUMPTION SMPL template + Pose2Mesh vertex ears (F0): head = ear centre + (up, forward) x ear width.
    "head": [-0.464, 0.217],
    # ASSUMPTION without ears: head = nose + (up, forward) x ear width in the thorax frame, ear width =
    # 0.437 x shoulder width (SMPL template); the build halves its confidence.
    "head_from_nose": [-0.399, -0.707], "ear_per_shoulder": 0.437,
    # ASSUMPTION ball of the foot = ankle + 0.65 x (toe centre - ankle) (spec § 3).
    "foot_toe_frac": 0.65,
    # ASSUMPTION hand without knuckles = wrist + 8 cm along the forearm; the build sets conf 0.3 x wrist.
    "hand_fallback_m": 0.08,
}

DIRECT = {"left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
          "right_wrist": 10, "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
_TRUNK = [5, 6, 11, 12]
SOURCES_28 = dict(
    pelvis=[11, 12], left_hip=[11, 12], right_hip=[11, 12], spine1=_TRUNK, spine2=_TRUNK, spine3=_TRUNK,
    neck=_TRUNK, left_collar=_TRUNK, right_collar=_TRUNK, head=[0, 3, 4],
    left_foot=[15, 17, 18], right_foot=[16, 20, 21], left_hand=[23, 24, 25, 26], right_hand=[27, 28, 29, 30],
    jaw=[31], left_eye=[32], right_eye=[33], nose_tip=[34], **{k: [v] for k, v in DIRECT.items()})
SOURCE_LIST = [SOURCES_28[n] for n in sk.SMPL28_NAMES]
AUX_FIT = [19, 22, 17, 20, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30]


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.where(n > 1e-9, n, np.nan)


def _orth(v, axis):
    return _unit(v - np.sum(v * axis, -1, keepdims=True) * axis)


def _frame(x, y):
    return np.stack([x, y, np.cross(x, y)], axis=-1)          # (F,3,3), columns = axes


def _slerp(Ra, Rb, t):
    out = np.full(Ra.shape, np.nan)
    ok = np.isfinite(Ra).all((1, 2)) & np.isfinite(Rb).all((1, 2))
    if ok.any():
        ra, rb = Rotation.from_matrix(Ra[ok]), Rotation.from_matrix(Rb[ok])
        out[ok] = (ra * Rotation.from_rotvec(t * (ra.inv() * rb).as_rotvec())).as_matrix()
    return out


def _hermite(t, P0, P1, T0, T1):
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * P0 + (t ** 3 - 2 * t ** 2 + t) * T0
            + (-2 * t ** 3 + 3 * t ** 2) * P1 + (t ** 3 - t ** 2) * T1)


def regress(kp, observed=None, const=None, default_sign=1.0):
    """kp (F,35,3) Y-up fit keypoints -> J (F,28,3) SMPL-28, aux (F,14,3), info. `observed` (35,) marks
    keypoints some camera ever supported; the others are treated as missing (-> fallbacks / NaN).
    `default_sign` = the world's handedness in these coordinates (the build passes -kpset.world_sign(cams)
    for the Y-up analysis world), used when neither feet nor face were ever seen."""
    c = dict(CONSTANTS, **(const or {}))
    kp = np.array(kp, dtype=np.float64)
    if observed is not None:
        kp[:, ~np.asarray(observed, bool)] = np.nan
    mirror = kpset.forward_sign(kp, default=default_sign) < 0
    if mirror:
        kp[..., 0] *= -1.0
    F = len(kp)
    N = {n: i for i, n in enumerate(sk.SMPL28_NAMES)}
    J = np.full((F, 28, 3), np.nan)
    Lh, Rh, Ls, Rs = kp[:, 11], kp[:, 12], kp[:, 5], kp[:, 6]
    m, s = 0.5 * (Lh + Rh), 0.5 * (Ls + Rs)
    xp = _unit(Lh - Rh)
    yp = _orth(np.broadcast_to(UP, xp.shape), xp)
    zp = np.cross(xp, yp)
    wh = np.linalg.norm(Lh - Rh, axis=-1)[:, None]
    xt = _unit(Ls - Rs)
    yt = _orth(s - m, xt)
    zt = np.cross(xt, yt)
    ws = np.linalg.norm(Ls - Rs, axis=-1)[:, None]
    for name, sign in (("left_hip", 1.0), ("right_hip", -1.0)):
        J[:, N[name]] = (m + sign * xp * (c["hjc_medial"] * wh / 2) - yp * (c["hjc_inferior"] * wh)
                         + zp * (c["hjc_anterior"] * wh))
    hw = np.linalg.norm(J[:, 1] - J[:, 2], axis=-1)[:, None]
    J[:, 0] = 0.5 * (J[:, 1] + J[:, 2]) + yp * (c["pelvis_up"] * hw) + zp * (c["pelvis_fwd"] * hw)
    trunk = np.linalg.norm(s - m, axis=-1)[:, None]
    neck = s + yt * (c["neck_up"] * trunk) + zt * (c["neck_fwd"] * trunk)
    J[:, N["neck"]] = neck
    D = np.linalg.norm(neck - J[:, 0], axis=-1)[:, None]
    Rp, Rt = _frame(xp, yp), _frame(xt, yt)
    for name, t, (ou, of) in zip(("spine1", "spine2", "spine3"), c["spine_t"], c["spine_off"]):
        Rk = _slerp(Rp, Rt, t)
        J[:, N[name]] = (_hermite(t, J[:, 0], neck, yp * D, yt * D) + Rk[:, :, 1] * (ou * D)
                         + Rk[:, :, 2] * (of * D))
    lat, cu, cf = c["collar"]
    for name, sign in (("left_collar", 1.0), ("right_collar", -1.0)):
        J[:, N[name]] = s + sign * xt * (lat * ws) + yt * (cu * ws) + zt * (cf * ws)
    e = 0.5 * (kp[:, 3] + kp[:, 4])
    xh = _unit(kp[:, 3] - kp[:, 4])
    fh = _orth(kp[:, 0] - e, xh)
    yh = np.cross(fh, xh)
    we = np.linalg.norm(kp[:, 3] - kp[:, 4], axis=-1)[:, None]
    head = e + yh * (c["head"][0] * we) + fh * (c["head"][1] * we)
    we2 = c["ear_per_shoulder"] * ws
    alt = kp[:, 0] + yt * (c["head_from_nose"][0] * we2) + zt * (c["head_from_nose"][1] * we2)
    head_fb = ~np.isfinite(head).all(-1) & np.isfinite(alt).all(-1)
    J[:, N["head"]] = np.where(head_fb[:, None], alt, head)
    for name, k in DIRECT.items():
        J[:, N[name]] = kp[:, k]
    for name, a, bt, st in (("left_foot", 15, 17, 18), ("right_foot", 16, 20, 21)):
        J[:, N[name]] = kp[:, a] + c["foot_toe_frac"] * (0.5 * (kp[:, bt] + kp[:, st]) - kp[:, a])
    hand_fb = np.zeros((F, 2), bool)
    for i, (name, wr, el, mcps) in enumerate((("left_hand", 9, 7, [23, 24, 25, 26]),
                                              ("right_hand", 10, 8, [27, 28, 29, 30]))):
        cen = kp[:, mcps].mean(axis=1)
        alt = kp[:, wr] + c["hand_fallback_m"] * _unit(kp[:, wr] - kp[:, el])
        hand_fb[:, i] = ~np.isfinite(cen).all(-1) & np.isfinite(alt).all(-1)
        J[:, N[name]] = np.where(hand_fb[:, i, None], alt, cen)
    for name, k in (("jaw", 31), ("left_eye", 32), ("right_eye", 33), ("nose_tip", 34)):
        J[:, N[name]] = kp[:, k]
    aux = kp[:, AUX_FIT].copy()
    if mirror:
        J[..., 0] *= -1.0
        aux[..., 0] *= -1.0
    return J, aux, {"head_fallback": head_fb, "hand_fallback": hand_fb, "mirrored": bool(mirror)}
