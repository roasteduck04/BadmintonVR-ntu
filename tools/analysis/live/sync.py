"""Time-step schedule and per-view sync for the live pipeline (spec § 4, § 6.1).

Raw frame of view v at ref frame r = r + offset_v (60 fps). With sync "interp" a fractional position takes the two
source frames either side and pose3d interpolates the 2D (as offline track._resample / pose_v2.sample_at); "nearest"
takes floor(x + 0.5) (video_io.view_frame's rule)."""
from __future__ import annotations

import math

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import scores

EPS = 1e-6


def source_frames(ref_frame: int, offset: float, mode: str):
    x = ref_frame + float(offset)
    f0 = math.floor(x + EPS)
    a = x - f0
    if a < EPS:
        return [int(f0)], 0.0
    if mode == "nearest":
        return [int(math.floor(x + 0.5))], 0.0
    return [int(f0), int(f0) + 1], float(a)


def schedule(cfg) -> list:
    out = []
    for i, r in enumerate(cfg["ref_frames"]):
        views = {}
        for v in cfg["views"]:
            f, a = source_frames(r, cfg["offsets"][v], cfg["sync_mode"])
            views[v] = {"frames": f, "alpha": a}
        out.append({"step": i, "ref_frame": int(r), "views": views})
    return out


def nearest_index(n_frames: int, alpha: float) -> int:
    return 0 if n_frames < 2 or alpha < 0.5 else 1


def raw_det_array(det: dict) -> np.ndarray:
    k = np.zeros((133, 3), np.float32)
    k[:, :2] = np.asarray(det["kp"], np.float32).reshape(133, 2)
    k[:, 2] = np.asarray(det["score"], np.float32).reshape(133)
    return k


def calibrate_dets(raw, params):
    """As scores.load_kp2d24 per frame: per-group calibrated scores, detections with a mean calibrated body score
    < pose_v2.SCORE_GATE dropped. Returns (kept arrays, their indices in `raw`)."""
    out, idx = [], []
    for d, k in enumerate(raw):
        k = np.array(k, dtype=np.float32, copy=True)
        k[:, 2] = scores.calibrate(k[:, 2], params)
        if k[:17, 2].mean() >= pv.SCORE_GATE:
            out.append(k)
            idx.append(d)
    return out, idx


def _centre(k):
    m = k[:17, 2] >= pv.W_MIN
    return k[:17][m, :2].mean(axis=0) if m.any() else None


def partner(det, others, max_px: float = 150.0):
    """Index of the detection in `others` (the view's other source frame) nearest det's body centre, None when
    none is within max_px (then the single detection is used, never a mix of two people)."""
    c = _centre(det)
    if c is None:
        return None
    best, best_d = None, max_px
    for j, o in enumerate(others):
        co = _centre(o)
        if co is None:
            continue
        d = float(np.linalg.norm(co - c))
        if d <= best_d:
            best, best_d = j, d
    return best


def interp_rows(A, B, alpha):
    if A is None:
        return B
    if B is None:
        return A
    return pv.sample_at(np.stack([A, B]).astype(np.float32), np.array([float(alpha)]))[0]
