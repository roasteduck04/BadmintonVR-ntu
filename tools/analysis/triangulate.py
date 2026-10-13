"""DLT triangulation of 2D keypoints across court-calibrated cameras.

Each joint is triangulated independently from the views that saw it with
sufficient score. Reprojection error is the metric-accuracy gate (the focal
length is an assumption). See the design spec.
"""
from __future__ import annotations

import numpy as np

ERR_SCALE = 50.0   # px, confidence falloff (tuned for assumed-focal PnP calibration)
MAX_REPROJ_PX = 50.0  # reject triangulations with mean reprojection above this


def triangulate_point(projs, uvs):
    """Linear DLT of one 3D point. projs: list of 3x4; uvs: list of (u, v)."""
    rows = []
    for P, (u, v) in zip(projs, uvs):
        P = np.asarray(P, dtype=np.float64)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    if len(A) < 4:
        raise ValueError("need >= 2 views to triangulate")
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def _per_view_errors(projs, uvs, X):
    """Per-view reprojection distance."""
    Xh = np.append(np.asarray(X, dtype=np.float64), 1.0)
    ds = []
    for P, (u, v) in zip(projs, uvs):
        p = np.asarray(P, dtype=np.float64) @ Xh
        uv = p[:2] / p[2]
        ds.append(float(np.hypot(uv[0] - u, uv[1] - v)))
    return ds


def reprojection_error(projs, uvs, X):
    """Mean pixel distance of X reprojected into the given views."""
    return float(np.mean(_per_view_errors(projs, uvs, X)))


INLIER_PX = 25.0  # a view must reproject within this to be added to the consensus


PRIOR_WEIGHT = 0.15  # blend weight toward prior 3D when available
PRIOR_MAX_DIST = 0.3  # meters — ignore prior if triangulation is further than this


def triangulate_joint(cams, kp2d_by_view, score_thr, prior_X=None):
    """Triangulate one joint from per-view (u, v, score).

    cams: {view: P(3x4)}; kp2d_by_view: {view: (u, v, score)}.
    prior_X: optional (3,) array — previous frame's 3D position for this joint.
        When provided, the result is pulled toward the prior to reduce jitter.
    Returns (X (3,) or None, confidence in [0,1], reproj err or None).

    Strategy: try every pair of views, pick the pair with the lowest
    reprojection error, then greedily add remaining views whose
    reprojection is within INLIER_PX. This avoids one bad view
    poisoning the triangulation.
    """
    projs, uvs, scores = [], [], []
    for view, P in cams.items():
        kp = kp2d_by_view.get(view)
        if kp is None:
            continue
        u, v, s = kp
        if s is None or s < score_thr or not np.isfinite([u, v]).all():
            continue
        projs.append(P)
        uvs.append((float(u), float(v)))
        scores.append(float(s))
    if len(projs) < 2:
        return None, 0.0, None

    n = len(projs)
    best_X, best_err, best_set = None, float("inf"), None

    for i in range(n):
        for j in range(i + 1, n):
            try:
                X = triangulate_point([projs[i], projs[j]],
                                      [uvs[i], uvs[j]])
            except Exception:
                continue
            if not np.isfinite(X).all():
                continue
            err = reprojection_error([projs[i], projs[j]],
                                     [uvs[i], uvs[j]], X)
            if err < best_err:
                best_err, best_X, best_set = err, X, {i, j}

    if best_X is None:
        return None, 0.0, None

    for k in range(n):
        if k in best_set:
            continue
        pv_err = _per_view_errors([projs[k]], [uvs[k]], best_X)
        if pv_err[0] <= INLIER_PX:
            inlier_projs = [projs[i] for i in sorted(best_set | {k})]
            inlier_uvs = [uvs[i] for i in sorted(best_set | {k})]
            X2 = triangulate_point(inlier_projs, inlier_uvs)
            if np.isfinite(X2).all():
                best_set.add(k)
                best_X = X2

    use_projs = [projs[i] for i in sorted(best_set)]
    use_uvs = [uvs[i] for i in sorted(best_set)]
    use_scores = [scores[i] for i in sorted(best_set)]

    err = reprojection_error(use_projs, use_uvs, best_X)
    if err > MAX_REPROJ_PX:
        return None, 0.0, None

    # temporal prior: blend toward previous frame to reduce jitter
    if prior_X is not None:
        prior_X = np.asarray(prior_X, dtype=np.float64)
        dist = np.linalg.norm(best_X - prior_X)
        if dist < PRIOR_MAX_DIST:
            best_X = best_X * (1.0 - PRIOR_WEIGHT) + prior_X * PRIOR_WEIGHT

    conf = (float(np.mean(use_scores))
            * min(1.0, len(use_projs) / 3.0)
            * float(np.exp(-err / ERR_SCALE)))
    return best_X, max(0.0, min(1.0, conf)), err
