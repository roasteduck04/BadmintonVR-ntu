"""Clothing-colour appearance features per 2D detection, for identity in the pose v2 tracker.

    PYTHONPATH=tools python -m analysis.appearance --clip doubles_2sides

For every `step`-th video frame and every RTMPose detection, a colour histogram of the torso
(shoulders -> hips) and of the shorts/thighs (hips -> knees): 12 hue bins for chromatic pixels
plus white / grey / black bins, L1-normalised per region (30 floats). Teammates in different
shirts or shorts, and bystanders, become separable where body proportions are not.
Cached to data/pose_cache/<clip>_appearance.npz (frame, det index, feature per view). Each view also stores
the sha1 of the kp2d file its det indices refer to ({view}_kp2d_sha1); a reader drops a view whose kp2d changed
since (review 1 L11). The OSNet cache (people.osnet) uses the same key.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

CACHE_DIR = os.path.join("data", "pose_cache")
VIDEO_DIRS = {
    "singles_1x": "singles/1", "singles_05x": "singles/0.5", "doubles_1side": "doubles/1 side",
    "doubles_2sides": "doubles/2 sides", "doubles_2sides_4angle": "doubles/2 sides (4 angle)",
}
N_HUE = 12
DIM = 2 * (N_HUE + 3)
RAW_SCORE_MIN = 2.5   # ~0.3 calibrated (see pose_v2.norm_score)


def region_hist(hsv: np.ndarray) -> np.ndarray:
    """(N,3) HSV pixels (OpenCV ranges) -> 15-bin histogram."""
    h, s, v = hsv[:, 0].astype(float), hsv[:, 1].astype(float), hsv[:, 2].astype(float)
    out = np.zeros(N_HUE + 3)
    chrom = (s > 50) & (v > 50)
    if chrom.any():
        out[:N_HUE] = np.bincount((h[chrom] * N_HUE / 180.0).astype(int).clip(0, N_HUE - 1), minlength=N_HUE)
    ach = ~chrom
    out[N_HUE] = np.sum(ach & (v > 170))
    out[N_HUE + 1] = np.sum(ach & (v > 70) & (v <= 170))
    out[N_HUE + 2] = np.sum(ach & (v <= 70))
    tot = out.sum()
    return out / tot if tot > 0 else out


def _box(kps, idx, shrink=0.15):
    pts = np.array([kps[j][:2] for j in idx if kps[j][2] >= RAW_SCORE_MIN])
    if len(pts) < 3:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    if size.min() < 6:
        c = (lo + hi) / 2
        size = np.maximum(size, 12)
        lo, hi = c - size / 2, c + size / 2
    return lo + size * shrink, hi - size * shrink


def det_feature(hsv_img, kps):
    """30-float feature for one detection or None when the torso is not visible."""
    H, W = hsv_img.shape[:2]
    feats = []
    for idx in ((5, 6, 11, 12), (11, 12, 13, 14)):
        b = _box(kps, idx)
        if b is None:
            return None
        (x0, y0), (x1, y1) = b
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(W, x1)), int(min(H, y1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        crop = hsv_img[y0:y1, x0:x1]
        stride = max(1, int(max(crop.shape[:2]) / 48))
        feats.append(region_hist(crop[::stride, ::stride].reshape(-1, 3)))
    return np.concatenate(feats)


def hellinger(a, b) -> float:
    """Mean Hellinger distance over the two regions, in [0, 1]."""
    d = []
    for s in (slice(0, N_HUE + 3), slice(N_HUE + 3, DIM)):
        bc = float(np.sum(np.sqrt(np.clip(a[s], 0, None) * np.clip(b[s], 0, None))))
        d.append(np.sqrt(max(0.0, 1.0 - bc)))
    return float(np.mean(d))


def extract_view(video_path, kp2d_path, step=2, log=print):
    import cv2
    with open(kp2d_path, "r", encoding="utf-8") as fh:
        frames = json.load(fh)["frames"]
    cap = cv2.VideoCapture(video_path)
    fi, rows = [], []
    f = 0
    while f < len(frames):
        if not cap.grab():
            break
        if f % step == 0 and frames[f]:
            ok, img = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                for d, det in enumerate(frames[f]):
                    ft = det_feature(hsv, det["keypoints"])
                    if ft is not None:
                        fi.append((f, d))
                        rows.append(ft)
        f += 1
        if f % 500 == 0:
            log(f"      {os.path.basename(video_path)}: {f}/{len(frames)}")
    cap.release()
    return np.array(fi, dtype=np.int32).reshape(-1, 2), np.array(rows, dtype=np.float16).reshape(-1, DIM)


def cache_path(clip):
    return os.path.join(CACHE_DIR, f"{clip}_appearance.npz")


def kp2d_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kp2d_key(v):
    return f"{v}_kp2d_sha1"


def read_npz(path) -> dict:
    """Every array of an .npz, file closed afterwards (np.load keeps the handle open on Windows otherwise)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def fresh_views(arrs, kp2d_paths, log=print, what="appearance"):
    """Views of a cache dict ({v}_idx / {v}_feat [/ {v}_kp2d_sha1]) whose det indices still match their kp2d file.
    A view without a key (cache older than v0.7.5) is kept with a note; one whose kp2d is missing can't be checked
    and is kept."""
    out, legacy = [], []
    for k in arrs:
        if not k.endswith("_idx"):
            continue
        v = k[:-4]
        key, p = kp2d_key(v), kp2d_paths.get(v)
        if key not in arrs:
            legacy.append(v)
        elif p and os.path.exists(p) and str(arrs[key]) != kp2d_sha1(p):
            log(f"    WARNING: {what} cache of {v} is stale (its kp2d changed since extraction): view ignored - "
                "re-extract it")
            continue
        out.append(v)
    if legacy:
        log(f"    note: {what} cache of {', '.join(sorted(legacy))} has no kp2d key (older than v0.7.5): can't tell "
            "if the 2D changed since; re-extract to add one")
    return out


def extract_clip(clip, views, step=2, log=print):
    """Features for every view of a base clip (views like back_31); returns the cache path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(clip)
    have = read_npz(path) if os.path.exists(path) else {}
    vdir = os.path.join("data", "15-08", VIDEO_DIRS[clip])
    for v in views:
        kp = os.path.join("data", "pose2d", f"{v}_kp2d.json")
        if f"{v}_idx" in have and v in fresh_views({k: a for k, a in have.items() if k.startswith(f"{v}_")},
                                                   {v: kp}, log=log):
            continue
        with open(kp, "r", encoding="utf-8") as fh:
            src = json.load(fh).get("source")
        vids = glob.glob(os.path.join(vdir, src)) if src else []
        if not vids:
            log(f"    appearance: no video for {v} in {vdir} — skipped")
            continue
        log(f"    appearance: {v} ({os.path.basename(vids[0])})")
        idx, feat = extract_view(vids[0], kp, step=step, log=log)
        have[f"{v}_idx"], have[f"{v}_feat"] = idx, feat
        have[kp2d_key(v)] = np.array(kp2d_sha1(kp))
        np.savez_compressed(path, **have)
    return path


def load_clip(clip, data_dir="data", kp2d_paths=None, log=print) -> dict:
    """{view: {(frame, det_index): feature}} or {} when not extracted. Views whose kp2d file changed since
    extraction are dropped (fresh_views); kp2d_paths {view: path} defaults to <data_dir>/pose2d/<view>_kp2d.json."""
    path = cache_path(clip)
    if not os.path.exists(path):
        return {}
    z = read_npz(path)
    paths = {k[:-4]: os.path.join(data_dir, "pose2d", f"{k[:-4]}_kp2d.json") for k in z if k.endswith("_idx")}
    paths.update(kp2d_paths or {})
    out = {}
    for v in fresh_views(z, paths, log=log):
        idx, feat = z[f"{v}_idx"], z[f"{v}_feat"].astype(np.float64)
        out[v] = {(int(a), int(b)): feat[i] for i, (a, b) in enumerate(idx)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args(argv)
    from analysis import pose_v2 as pv
    clips = list(VIDEO_DIRS) if args.all else (args.clip or [])
    for c in clips:
        cfg = pv.clip_config(c)
        print(f"[appearance] {c}")
        print("    ->", extract_clip(c, cfg["views"], step=args.step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
