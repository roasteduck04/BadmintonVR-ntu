"""Per-group RTMPose SimCC score -> probability calibration for pose24 (spec § 5.1; review C2).

Raw wholebody scores are SimCC values (median ~4.3, 0-10), not probabilities. Each keypoint group has
its own logistic p = 1 / (1 + exp(-(raw - mid) / width)), applied once when the 2D is loaded; every pose24
gate then uses calibrated values. Defaults for all groups = pose_v2's fit (mid 2.75, width 0.30);
`python -m analysis.pose24 calibrate-scores` refits them and writes score_calibration.json next to this file.
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis import pose_v2 as pv

GROUPS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91), "hands": (91, 133)}
DEFAULT_PARAMS = {g: {"mid": pv.SCORE_MID, "width": pv.SCORE_WIDTH} for g in GROUPS}
CALIB_PATH = os.path.join(os.path.dirname(__file__), "score_calibration.json")


def load_params(path: str | None = CALIB_PATH) -> dict:
    """{"groups": {group: {mid, width}}, "source": ...}; pose_v2 defaults for any group the file lacks."""
    groups = {g: dict(v) for g, v in DEFAULT_PARAMS.items()}
    source = f"default (pose_v2 fit, mid {pv.SCORE_MID} width {pv.SCORE_WIDTH})"
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        for g, v in doc.get("groups", {}).items():
            if g in groups:
                try:
                    mid, width = float(v["mid"]), float(v["width"])
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"{path}: group {g!r} needs numeric mid and width ({e})") from None
                if not (np.isfinite(mid) and np.isfinite(width) and width > 0):
                    raise ValueError(f"{path}: group {g!r} needs a finite mid and a width > 0, got {mid}, {width}")
                groups[g] = {"mid": mid, "width": width}
        source = os.path.basename(path)
    return {"groups": groups, "source": source}


def calibrate(raw, params: dict) -> np.ndarray:
    """(..., 133) raw SimCC scores -> calibrated probabilities (float64)."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    for g, (a, b) in GROUPS.items():
        p = params["groups"][g]
        out[..., a:b] = 1.0 / (1.0 + np.exp(-(raw[..., a:b] - p["mid"]) / p["width"]))
    return out


def to_raw(prob, params: dict) -> np.ndarray:
    """Inverse of `calibrate` (probabilities clipped to [1e-6, 1 - 1e-6])."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    out = np.empty_like(p)
    for g, (a, b) in GROUPS.items():
        q = params["groups"][g]
        out[..., a:b] = q["mid"] + q["width"] * np.log(p[..., a:b] / (1.0 - p[..., a:b]))
    return out


def load_kp2d24(path: str, params: dict):
    """kp2d.json -> (frames, fps or None, idx, header). As pose_v2.load_kp2d, with per-group calibration:
    frames[f] = list of (133,3) float32 arrays (calibrated scores); detections with a mean calibrated body
    score < pose_v2.SCORE_GATE dropped; idx[f] = the kept detections' original indices; fps is None when
    the header has none (the caller logs the fallback); header = every top-level key except frames, plus
    _track_ids[f] = the kept detections' rtmlib track ids (-1 = none)."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    frames, idx, tids = [], [], []
    for fr in doc["frames"]:
        dets, ids, tk = [], [], []
        for d, det in enumerate(fr):
            k = np.asarray(det["keypoints"], dtype=np.float32)
            k[:, 2] = calibrate(k[:, 2], params)
            if k[:17, 2].mean() >= pv.SCORE_GATE:
                dets.append(k)
                ids.append(d)
                tk.append(int(det.get("track_id", -1)) if det.get("track_id") is not None else -1)
        frames.append(dets)
        idx.append(ids)
        tids.append(tk)
    header = {k: v for k, v in doc.items() if k != "frames"}
    header["_track_ids"] = tids          # rtmlib track id per kept detection (-1 = none); tracker3d link hint
    fps = float(doc["fps"]) if doc.get("fps") else None
    return frames, fps, idx, header
