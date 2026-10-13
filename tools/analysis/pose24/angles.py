"""Anatomical angles (degrees) on the 35 fit keypoints (kpset) and generic helpers for named joints.

`sgn` = kpset.forward_sign of the coordinates in use (+1 when cross(left, up) points forward); `up` is the
world up ((0,1,0) Y-up analysis world, (0,-1,0) Y-down court world). Used by the fit's joint-limit terms
(LIMITS) and by the evaluation."""
from __future__ import annotations

import numpy as np


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.where(n > 1e-9, n, np.nan)


def angle_deg(a, b):
    c = np.sum(unit(a) * unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _trunk_axis(X):
    return 0.5 * (X[:, 5] + X[:, 6]) - 0.5 * (X[:, 11] + X[:, 12])


def pelvis_forward(X, sgn):
    return sgn * unit(np.cross(X[:, 11] - X[:, 12], _trunk_axis(X)))


def knee_flexion(X, sgn, up, side):
    """Signed knee flexion: 0 = straight, + when the knee lies forward of the hip-ankle line, where
    "forward" is the leg's own foot direction (mid(big toe, small toe) - heel), both projected
    perpendicular to the hip-ankle line — so an abducted / externally rotated leg (side lunge) keeps
    its sign. NaN when the foot direction is unknown (the fit reads NaN as "no limit"). `sgn` and `up` are
    unused (kept so every LIMITS entry has the same signature)."""
    h, k, a, bt, st, heel = (11, 13, 15, 17, 18, 19) if side == "left" else (12, 14, 16, 20, 21, 22)
    theta = angle_deg(X[:, k] - X[:, h], X[:, a] - X[:, k])
    line = unit(X[:, a] - X[:, h])

    def perp(v):
        return v - np.sum(v * line, -1, keepdims=True) * line
    off = perp(X[:, k] - X[:, h])
    foot = perp(0.5 * (X[:, bt] + X[:, st]) - X[:, heel])
    d = np.sum(off * foot, -1)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(d), np.where(d >= 0, theta, -theta), np.nan)


def elbow_flexion(X, side):
    """Unsigned elbow flexion (0 = straight); the sign isn't observable from 3 keypoints."""
    s, e, w = (5, 7, 9) if side == "left" else (6, 8, 10)
    return angle_deg(X[:, e] - X[:, s], X[:, w] - X[:, e])


def ankle_dorsiflexion(X, side):
    """90 - angle(shank up, heel -> toe centre): + = toes up (dorsi), - = plantar."""
    k, a, heel, bt, st = (13, 15, 19, 17, 18) if side == "left" else (14, 16, 22, 20, 21)
    foot = 0.5 * (X[:, bt] + X[:, st]) - X[:, heel]
    return 90.0 - angle_deg(X[:, k] - X[:, a], foot)


def trunk_lean(axis, left, up, sgn):
    """(forward, lateral) lean of `axis` from vertical, in the horizontal frame of the `left` line
    (lateral + = towards the subject's left)."""
    up = unit(np.asarray(up, dtype=np.float64))
    xh = unit(left - np.sum(left * up, -1, keepdims=True) * up)
    fh = sgn * unit(np.cross(xh, np.broadcast_to(up, xh.shape)))
    vu = np.sum(axis * up, -1)
    return (np.degrees(np.arctan2(np.sum(axis * fh, -1), vu)),
            np.degrees(np.arctan2(np.sum(axis * xh, -1), vu)))


def trunk_flexion(X, sgn, up):
    """Forward lean of hip centre -> shoulder centre from vertical (the pelvis is taken upright)."""
    return trunk_lean(_trunk_axis(X), X[:, 11] - X[:, 12], up, sgn)[0]


def trunk_lateral_flexion(X):
    """Side bend relative to the hip line: 90 - angle(hip line, trunk axis); + = towards the left hip."""
    return 90.0 - angle_deg(X[:, 11] - X[:, 12], _trunk_axis(X))


def trunk_axial(X):
    """Unsigned twist between the hip line and the shoulder line about the trunk axis."""
    a = unit(_trunk_axis(X))

    def proj(v):
        return v - np.sum(v * a, -1, keepdims=True) * a
    return angle_deg(proj(X[:, 11] - X[:, 12]), proj(X[:, 5] - X[:, 6]))


# (name, fn(X, sgn, up) -> (T,) degrees, lower, upper, keypoints the angle depends on)
LIMITS = [
    ("left_knee_flexion", lambda X, s, u: knee_flexion(X, s, u, "left"), -5.0, 160.0, [11, 13, 15, 17, 18, 19]),
    ("right_knee_flexion", lambda X, s, u: knee_flexion(X, s, u, "right"), -5.0, 160.0, [12, 14, 16, 20, 21, 22]),
    ("left_elbow_flexion", lambda X, s, u: elbow_flexion(X, "left"), -np.inf, 150.0, [5, 7, 9]),
    ("right_elbow_flexion", lambda X, s, u: elbow_flexion(X, "right"), -np.inf, 150.0, [6, 8, 10]),
    ("left_ankle_dorsiflexion", lambda X, s, u: ankle_dorsiflexion(X, "left"), -50.0, 30.0, [13, 15, 17, 18, 19]),
    ("right_ankle_dorsiflexion", lambda X, s, u: ankle_dorsiflexion(X, "right"), -50.0, 30.0, [14, 16, 20, 21, 22]),
    ("trunk_flexion", lambda X, s, u: trunk_flexion(X, s, u), -45.0, 90.0, [5, 6, 11, 12]),
    ("trunk_lateral_flexion", lambda X, s, u: trunk_lateral_flexion(X), -40.0, 40.0, [5, 6, 11, 12]),
    ("trunk_axial", lambda X, s, u: trunk_axial(X), -np.inf, 50.0, [5, 6, 11, 12]),
]
