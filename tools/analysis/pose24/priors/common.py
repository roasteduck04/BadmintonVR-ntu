"""v0.7.5 monoprior: shared pieces of the learned 1-camera depth-prior producers (MotionBERT, WHAM converter).

  coco133_to_h36m   COCO-WholeBody-133 (x, y, score) -> H36M-17, following MotionBERT's halpe2h36m with the Halpe
                    extras derived from COCO: "Hip" = hip midpoint, "Neck" = shoulder midpoint, "Head" = ear midpoint
  H36M_TO_KP        (h36m index, kpset index) pairs of the 13 H36M joints that are fit keypoints
  write_prior       the mono prior .npz (contract in analysis.pose24.mono's docstring), atomic
  view_image_size   (w, h) of a view from the pose24 debug cache"""
from __future__ import annotations

import json
import os
import uuid

import numpy as np

# H36M-17 <- COCO-17 body indices (COCO-WholeBody 0-16 = kpset.NAMES[0:17]); 0, 7, 8, 10 are derived (below)
_H36M_DIRECT = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}
_H36M_MEAN = {0: (11, 12), 7: (5, 6, 11, 12), 8: (5, 6), 10: (3, 4)}   # 7 = mid(hip mid, shoulder mid)

H36M_TO_KP = [(1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),      # R hip/knee/ankle, L hip/knee/ankle
              (9, 0), (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10)]  # nose, L arm, R arm

META_KEYS = ("method", "clip", "view", "timeline_of", "fps", "frame_count", "build_id")


def coco133_to_h36m(x):
    """(F,133,3) or (F,>=17,3) pixels + scores -> (F,17,3) H36M-17. A derived point's score is the min of its
    sources' scores; a point whose (any) source is missing (score <= 0 or non-finite) has score 0 and NaN x, y."""
    x = np.asarray(x, float)
    xy, s = x[..., :17, :2], x[..., :17, 2]
    miss = ~(np.isfinite(xy).all(-1) & np.isfinite(s) & (s > 0))
    xy = np.where(miss[..., None], np.nan, xy)
    s = np.where(miss, 0.0, s)
    y = np.full(x.shape[:-2] + (17, 3), np.nan)
    for h, c in _H36M_DIRECT.items():
        y[..., h, :2], y[..., h, 2] = xy[..., c, :], s[..., c]
    for h, src in _H36M_MEAN.items():
        y[..., h, :2] = xy[..., list(src), :].mean(-2)        # 7: mean of the 4 = mid(hip mid, shoulder mid)
        y[..., h, 2] = s[..., list(src)].min(-1)
    bad = y[..., 2] <= 0
    y[..., :2][bad] = np.nan
    y[..., 2][bad] = 0.0
    return y


def view_image_size(cache, view):
    """(width, height) of `view` from the debug cache's image_size (track.py / build.py); KeyError when missing."""
    wh = (cache.get("image_size") or {}).get(view)
    if not wh or len(wh) != 2 or any(v is None for v in wh):
        raise KeyError(f"debug cache has no image_size for view {view!r} (rebuild the clip's pose24 cache)")
    return int(wh[0]), int(wh[1])


def write_prior(path, meta, players):
    """Write a mono prior .npz atomically (tmp file in the same dir + os.replace; parent dirs created).

    meta: dict with META_KEYS (stored as the JSON string `meta`, allow_nan=False). players: {pid: {"kp": (n,) int
    kpset indices, "rel_cam": (F,n,3), "conf": (F,), optional "contact": (F,6)}} with F = meta["frame_count"].
    ValueError on missing meta keys, non-JSON-safe meta or shape mismatches."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"prior meta lacks {missing}")
    try:
        meta_s = json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"prior meta is not JSON-safe: {e}") from e
    F = int(meta["frame_count"])
    arrays = {"meta": np.array(meta_s)}
    for pid, d in players.items():
        kp = np.asarray(d["kp"])
        if kp.ndim != 1 or not np.issubdtype(kp.dtype, np.integer):
            raise ValueError(f"{pid}: kp must be a 1-D int array, got {kp.dtype} {kp.shape}")
        n = len(kp)
        rel, conf = np.asarray(d["rel_cam"], float), np.asarray(d["conf"], float)
        if rel.shape != (F, n, 3):
            raise ValueError(f"{pid}: rel_cam shape {rel.shape} != {(F, n, 3)}")
        if conf.shape != (F,):
            raise ValueError(f"{pid}: conf shape {conf.shape} != {(F,)}")
        arrays.update({f"{pid}_kp": kp.astype(np.int64), f"{pid}_rel_cam": rel, f"{pid}_conf": conf})
        if d.get("contact") is not None:
            c = np.asarray(d["contact"], float)
            if c.shape != (F, 6):
                raise ValueError(f"{pid}: contact shape {c.shape} != {(F, 6)}")
            arrays[f"{pid}_contact"] = c
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
