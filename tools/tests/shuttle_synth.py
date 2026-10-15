"""Synthetic 5-camera shuttle rally with known truth, for the shuttle variant/tracker tests.

Three drag-ballistic shots (two hits) in court coordinates (metres, Y-up), projected into
five cameras around the court, with the corruptions seen in real TrackNetV3 output:
per-view time offsets, a stuck run, a repeated default point, an isolated blip, and one
view stored at 1/1.5 pixel scale.
"""
import numpy as np

from analysis.shuttle_physics import integrate_shuttle
from tests.test_racket_rigidfit import _look_at_camera

FPS = 60.0
CAM_CENTERS = {
    "back": [0.0, 3.0, -11.0],
    "backleft": [-6.0, 3.0, -9.0],
    "backright": [6.0, 3.0, -9.0],
    "frontleft": [-6.0, 3.0, 9.0],
    "frontright": [6.0, 3.0, 9.0],
}
VIEWS = tuple(CAM_CENTERS)
START = [0.5, 2.3, -5.0]
SHOTS = [[-0.3, 5.0, 16.0], [0.4, 4.0, -15.0], [0.0, 6.0, 14.0]]
ARC_LEN, LEAD, TAIL = 50, 20, 20
DEFAULT_OFFSETS = {"back": 0, "backleft": -2, "backright": -1, "frontleft": 0, "frontright": -3}
IMG_W, IMG_H = 2160, 3840   # _look_at_camera default (portrait)


def make_cams():
    return {v: _look_at_camera(c, [0.0, 1.5, 0.0], f=1500.0) for v, c in CAM_CENTERS.items()}


def rally_truth():
    """(X (n,3) court Y-up, NaN outside flight; frames where shots 2 and 3 start = hits)."""
    n = LEAD + ARC_LEN * len(SHOTS) + TAIL
    X = np.full((n, 3), np.nan)
    p = np.array(START, dtype=float)
    t = np.arange(ARC_LEN) / FPS
    hits = []
    for i, v in enumerate(SHOTS):
        s = LEAD + i * ARC_LEN
        tr = integrate_shuttle(np.r_[p, v], (0.0, t[-1]), t)
        X[s:s + ARC_LEN] = tr
        if i > 0:
            hits.append(s)
        p = tr[-1]
    return X, hits


def project(cams, X, noise_px=2.0, dropout=0.3, rng=None):
    """Per-view analysis-frame-indexed detections [u, v] (source pixels) or None."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for v, cam in cams.items():
        fr = []
        for x in X:
            if not np.isfinite(x).all() or rng.random() < dropout:
                fr.append(None)
                continue
            uv, z = cam.project(x[None])
            u, w = uv[0] + rng.normal(0.0, noise_px, 2)
            inside = z[0] > 0.1 and 0 <= u < IMG_W and 0 <= w < IMG_H
            fr.append([float(u), float(w)] if inside else None)
        out[v] = fr
    return out


def unsync(frames, shift):
    """Raw-index copy (raw index = analysis frame + shift): raw[g] = frames[g - shift]."""
    n = len(frames)
    return [frames[g - shift] if 0 <= g - shift < n else None for g in range(n)]


def scene(seed=0, offsets=None):
    rng = np.random.default_rng(seed)
    offsets = dict(DEFAULT_OFFSETS if offsets is None else offsets)
    cams = make_cams()
    X, hits = rally_truth()
    clean = project(cams, X, rng=rng)
    raw = {v: unsync(fr, offsets[v]) for v, fr in clean.items()}
    raw["back"][0:8] = [[900.0, 700.0] for _ in range(8)]            # stuck run (lead-in)
    for g in (3, 45, 90, 130, 160, 185):
        raw["frontleft"][g] = [1027.0, 949.0]                          # TrackNet default point
    raw["backright"][10] = [100.0, 3000.0]                             # isolated blip (lead-in)
    raw["backleft"] = [None if p is None else [p[0] / 1.5, p[1] / 1.5]  # stored at 1/1.5 scale
                       for p in raw["backleft"]]
    return {"cams": cams, "truth": X, "hits": hits, "n": len(X), "fps": FPS,
            "clean": clean, "raw": raw, "offsets": offsets}
