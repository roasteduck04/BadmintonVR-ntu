"""pose24 per-joint 3D uncertainty -> confidence (spec § 5.2; review C2).

For a fitted keypoint: 2D sigma per view = SIGMA_PX / max(p, 0.05) px; information = sum over inlier
views of J^T J / sigma^2 with J the 2x3 projection Jacobian; sigma_mm = 1 / sqrt(smallest eigenvalue) =
the worst direction (captures the depth weakness of opposite-corner pairs), inflated by the fit's
residual (x max(1, e / SIGMA_PX)). Keypoints not measured (< 2 inlier views) decay from the nearest measured
frame (1 cm per frame). With a monocular depth prior (camera axis, sigma_d) every point's information gains
axis axis^T / sigma_d^2 and >= 1 inlier view counts as measured. Rule joints (spine, collars, pelvis, ...)
take the max sigma of their sources.
Doc confidence = exp(-sigma_mm / 40 mm) (0.78 at 10 mm, 0.37 at 40 mm)."""
from __future__ import annotations

import warnings

import numpy as np

SIGMA_PX = 8.0
W_FLOOR = 0.05
CAP_MM = 1000.0
FILL_MM_PER_FRAME = 10.0
CONF_SCALE_MM = 40.0


def settings() -> dict:
    return {"sigma_px": SIGMA_PX, "score_floor": W_FLOOR, "cap_mm": CAP_MM,
            "fill_mm_per_frame": FILL_MM_PER_FRAME, "conf": "exp(-sigma_mm / 40)", "conf_scale_mm": CONF_SCALE_MM}


def point_sigma_mm(P, X, w, inl, prior=None):
    """P (V,3,4), X (N,3) court world, w/inl (N,V) -> (N,) worst-direction 1-sigma error in mm.
    prior (axis (3,), sigma_m): depth prior information added to every point (then >= 1 inlier view suffices)."""
    Xh = np.c_[X, np.ones(len(X))]
    p = np.einsum("vij,nj->nvi", P, Xh)
    z = np.where(np.abs(p[..., 2]) < 1e-9, 1e-9, p[..., 2])
    u = p[..., :2] / z[..., None]
    Ju = (P[None, :, 0, :3] - u[..., 0:1] * P[None, :, 2, :3]) / z[..., None]
    Jv = (P[None, :, 1, :3] - u[..., 1:2] * P[None, :, 2, :3]) / z[..., None]
    s_v = SIGMA_PX / np.maximum(np.nan_to_num(w), W_FLOOR)
    wt = np.where(inl, 1.0 / s_v ** 2, 0.0)
    info = np.einsum("nv,nvi,nvj->nij", wt, Ju, Ju) + np.einsum("nv,nvi,nvj->nij", wt, Jv, Jv)
    info = np.nan_to_num(info)
    min_views = 2
    if prior is not None:
        axis = np.asarray(prior[0], float)
        info = info + np.outer(axis, axis) / float(prior[1]) ** 2
        min_views = 1
    lam = np.linalg.eigvalsh(info)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = 1000.0 / np.sqrt(lam)
    sig = np.where(np.isfinite(sig) & (lam > 0) & (inl.sum(axis=1) >= min_views), sig, CAP_MM)
    return np.minimum(sig, CAP_MM)


def fill_decay(sigma, measured, mm_per_frame=FILL_MM_PER_FRAME, cap=CAP_MM):
    """Frames not measured (< 2 inlier views; < 1 with a depth prior) get min(own sigma, the better of the
    previous / next measured frame's sigma + mm_per_frame x distance)."""
    out = np.array(sigma, dtype=np.float64)
    F = len(out)
    t = np.arange(F)
    for k in range(out.shape[1]):
        idx = np.nonzero(measured[:, k])[0]
        if not len(idx):                  # never measured in the clip
            out[:, k] = cap
            continue
        nxt = np.minimum(np.searchsorted(idx, t), len(idx) - 1)          # first measured frame >= t (or last)
        prv = np.maximum(np.searchsorted(idx, t, side="right") - 1, 0)   # last measured frame <= t (or first)
        decay = np.fmin(out[idx[prv], k] + mm_per_frame * np.abs(idx[prv] - t),
                        out[idx[nxt], k] + mm_per_frame * np.abs(idx[nxt] - t))
        um = ~measured[:, k]
        out[um, k] = np.minimum(np.nan_to_num(out[um, k], nan=cap), decay[um])
        out[:, k] = np.minimum(out[:, k], cap)
    return out


def keypoint_sigma(fitres, P, depth_axis=None, depth_sigma_m=None):
    """(sigma (F,K) mm, filled (F,K) = no inlier view, measured (F,K) = >= 2 inlier views without a depth
    prior, >= 1 with one). depth_axis (3,) camera optical axis in the court world + depth_sigma_m (m): the
    monocular depth prior; give both or neither."""
    if (depth_axis is None) != (depth_sigma_m is None):
        raise ValueError("keypoint_sigma: give both depth_axis and depth_sigma_m (monocular prior) or neither, got "
                         f"depth_axis={depth_axis!r}, depth_sigma_m={depth_sigma_m!r}")
    X, w, r, inl = fitres["X"], fitres["w"], fitres["resid"], fitres["inlier"]
    F, K, V = w.shape
    prior = None if depth_axis is None else (depth_axis, depth_sigma_m)
    sig = point_sigma_mm(P, np.nan_to_num(X.reshape(-1, 3)), w.reshape(-1, V), inl.reshape(-1, V),
                         prior=prior).reshape(F, K)
    wi = np.nan_to_num(w) * inl
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (np.nan_to_num(r) * wi).sum(-1) / wi.sum(-1)
    sig = sig * np.maximum(1.0, np.nan_to_num(e) / SIGMA_PX)
    n_inl = inl.sum(-1)
    measured = n_inl >= (2 if prior is None else 1)
    sig = fill_decay(sig, measured)
    sig[~np.isfinite(X).all(-1)] = np.nan
    return sig, n_inl == 0, measured


def joint_sigma(sig35, sources):
    """(F, len(sources)): each joint's sigma = the max over its source keypoints."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmax(sig35[:, src], axis=1) for src in sources], axis=1)


def to_conf(sigma_mm):
    return np.clip(np.nan_to_num(np.exp(-np.asarray(sigma_mm, float) / CONF_SCALE_MM), nan=0.0), 0.0, 1.0)
