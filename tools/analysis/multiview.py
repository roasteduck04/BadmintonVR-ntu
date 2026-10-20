"""Multi-view camera model for court-calibrated triangulation.

Assumed-focal pinhole: intrinsics come from a field-of-view guess (no lens
metadata in the phone footage); extrinsics come from court-point PnP in the
Y-up court world frame (floor at Y=0). See
_archive/superpowers/specs/2026-08-06-real-footage-triangulation-pose-position-design.md.
"""
from __future__ import annotations

import numpy as np
import cv2


def intrinsics_from_fov(width, height, fov_deg, on="long"):
    """3x3 pinhole K with square pixels and centre principal point.

    fov_deg is the field of view across the chosen image axis:
    on='long' -> the larger of (width, height); on='short' -> the smaller.
    """
    axis = max(width, height) if on == "long" else min(width, height)
    f = (axis / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def projection_matrix(K, R, t):
    """3x4 projection P = K [R | t]. t shape (3,)."""
    Rt = np.hstack([np.asarray(R, dtype=np.float64),
                    np.asarray(t, dtype=np.float64).reshape(3, 1)])
    return np.asarray(K, dtype=np.float64) @ Rt


def _court_to_world(court_xz):
    court_xz = np.asarray(court_xz, dtype=np.float64)
    return np.hstack([court_xz[:, :1],
                      np.zeros((len(court_xz), 1)),
                      court_xz[:, 1:]])


def solve_extrinsics(court_xz, image_px, K):
    """PnP: floor court points (X,Z) + their pixels + K -> (R, t)."""
    world = _court_to_world(court_xz).astype(np.float64)
    image_px = np.asarray(image_px, dtype=np.float64)
    if len(world) < 4:
        raise ValueError("need >= 4 court points for PnP")
    ok, rvec, tvec = cv2.solvePnP(
        world.reshape(-1, 1, 3), image_px.reshape(-1, 1, 2),
        np.asarray(K, dtype=np.float64), None,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def find_best_fov(width, height, court_xz, image_px, on="long",
                   lo=40.0, hi=140.0, steps=200, fine_step=0.05):
    """Sweep FOV to find the value that minimises PnP reprojection RMS: a coarse sweep over [lo, hi], then a fine one
    (fine_step degrees) around the coarse best. fine_step=None skips the fine sweep."""
    def sweep(fovs):
        best = (None, float("inf"))
        for fov in fovs:
            K = intrinsics_from_fov(width, height, fov, on=on)
            try:
                R, t = solve_extrinsics(court_xz, image_px, K)
                rms = reprojection_rms(court_xz, image_px, K, R, t)
            except Exception:
                continue
            if rms < best[1]:
                best = (fov, rms)
        return best

    best_fov, best_rms = sweep([lo + (hi - lo) * i / steps for i in range(steps + 1)])
    if best_fov is not None and fine_step:
        step = (hi - lo) / steps
        a, b = max(lo, best_fov - step), min(hi, best_fov + step)
        n = max(1, int(round((b - a) / fine_step)))
        f_fov, f_rms = sweep([a + (b - a) * i / n for i in range(n + 1)])
        if f_fov is not None and f_rms <= best_rms:
            best_fov, best_rms = f_fov, f_rms
    return best_fov, best_rms


def solve_camera(width, height, court_xz, image_px, fov_deg=None, on="long", lo=40.0, hi=140.0):
    """Court-point calibration of one camera. fov_deg None = find the best field of view first (find_best_fov).

    Returns {fov_deg, K, R, t, rms_px, proj (N,2), err_px (N,), at_edge, notes}: proj / err_px = where each court point
    lands with the solved camera and how far from its click; at_edge = the best FOV is at the end of the search range
    (usually a wrong click); notes = plain-language warnings about the clicks."""
    court_xz = np.asarray(court_xz, dtype=np.float64)
    image_px = np.asarray(image_px, dtype=np.float64)
    at_edge = False
    if fov_deg is None:
        fov_deg, _ = find_best_fov(width, height, court_xz, image_px, on=on, lo=lo, hi=hi)
        if fov_deg is None:
            raise RuntimeError("no field of view fits these points")
        at_edge = fov_deg <= lo + 0.5 or fov_deg >= hi - 0.5
    K = intrinsics_from_fov(width, height, float(fov_deg), on=on)
    R, t = solve_extrinsics(court_xz, image_px, K)
    P = projection_matrix(K, R, t)
    world = _court_to_world(court_xz)
    hom = (P @ np.hstack([world, np.ones((len(world), 1))]).T).T
    proj = hom[:, :2] / hom[:, 2:3]
    err = np.linalg.norm(proj - image_px, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    notes = []
    if at_edge:
        notes.append(f"the best field of view is at the end of the search range ({fov_deg:.1f} deg): check the clicks")
    c = court_xz - court_xz.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False) if len(c) >= 2 else np.zeros(2)
    if sv[0] > 0 and sv[-1] / sv[0] < 0.15:
        notes.append("the points lie almost on one line: add points across the court for a reliable field of view")
    elif len(court_xz) < 6:
        notes.append(f"{len(court_xz)} points: 6 or more spread over the court give a more reliable field of view")
    return {"fov_deg": float(fov_deg), "K": K, "R": R, "t": t, "rms_px": rms, "proj": proj, "err_px": err,
            "at_edge": bool(at_edge), "notes": notes}


def reprojection_rms(court_xz, image_px, K, R, t):
    """RMS pixel error of the fitted camera over the court points."""
    P = projection_matrix(K, R, t)
    world = _court_to_world(court_xz)
    proj = (P @ np.hstack([world, np.ones((len(world), 1))]).T).T
    uv = proj[:, :2] / proj[:, 2:3]
    d = uv - np.asarray(image_px, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
