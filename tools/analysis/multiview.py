"""Multi-view camera model for court-calibrated triangulation.

Assumed-focal pinhole: intrinsics come from a field-of-view guess (no lens
metadata in the phone footage); extrinsics come from court-point PnP in the
Y-up court world frame (floor at Y=0). See
docs/superpowers/specs/2026-08-06-real-footage-triangulation-pose-position-design.md.
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
                   lo=40.0, hi=140.0, steps=200):
    """Sweep FOV to find the value that minimises PnP reprojection RMS."""
    best_fov, best_rms = None, float("inf")
    for i in range(steps + 1):
        fov = lo + (hi - lo) * i / steps
        K = intrinsics_from_fov(width, height, fov, on=on)
        try:
            R, t = solve_extrinsics(court_xz, image_px, K)
            rms = reprojection_rms(court_xz, image_px, K, R, t)
        except Exception:
            continue
        if rms < best_rms:
            best_rms, best_fov = rms, fov
    return best_fov, best_rms


def reprojection_rms(court_xz, image_px, K, R, t):
    """RMS pixel error of the fitted camera over the court points."""
    P = projection_matrix(K, R, t)
    world = _court_to_world(court_xz)
    proj = (P @ np.hstack([world, np.ones((len(world), 1))]).T).T
    uv = proj[:, :2] / proj[:, 2:3]
    d = uv - np.asarray(image_px, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
