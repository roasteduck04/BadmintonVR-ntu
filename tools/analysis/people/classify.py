"""Per-tracklet player / non-player / unknown score (SP3 spec § 2.2 + § 2.4).

Features in [0, 1]: in_lines (doubles lines + court_margin_m), one_half (same side of the net), support (views),
size (triangulated stature vs the clip's reference height - the seeded players' median, SP3 finding 1: the known
~15 % scale bias makes the spec's absolute 1.45-2.05 m reject real players; 1-view: pixel height vs a reference-
height person at that floor point), moving (90th-percentile speed), straight (net / path length for long paths:
walkers), hotspot (time near a bystander hotspot - a floor cell (+ its 3x3 neighbourhood) outside the lines where
candidates pile up >= 2 s;
SP3 finding 3: zones are learnt, never hard-coded), seed_app (appearance closeness to a seeded tracklet).
Score = logistic(bias + sum w x), the tracklet's own evidence shrunk towards 0.5 for tracklets shorter than
min_frames_full_evidence (logit scaled by (n/min)^2); the hotspot term is clip-level evidence and is not shrunk (T15:
bystander fragments inside a hotspot stayed "unknown").
Seed-pinned tracklets are players (score 1)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from analysis import pose_v2 as pv
from analysis.people.candidates import ANKLE_H

HALF_X, HALF_Z = 3.05, 6.70


def in_lines(fl, margin):
    fl = np.asarray(fl, float)
    return (np.abs(fl[:, 0]) <= HALF_X + margin) & (np.abs(fl[:, 1]) <= HALF_Z + margin)


def hotspots(cands_by_frame, fps, p) -> list:
    cell, cnt = p["hotspot_cell_m"], {}
    for C in cands_by_frame:
        for c in C:
            if in_lines(c.floor[None], p["court_margin_m"])[0]:
                continue
            cnt.setdefault(tuple(np.floor(c.floor / cell).astype(int)), []).append(c.floor)
    # a person standing on a cell boundary spreads over up to 4 cells: the densest free cell takes its free 3x3
    # neighbourhood (densest first, ties by cell), and that group is a hotspot when it holds >= hotspot_min_s
    need = p["hotspot_min_s"] * fps
    out, used = [], set()
    for key in sorted(cnt, key=lambda k: (-len(cnt[k]), k)):
        if key in used:
            continue
        group = [(key[0] + dx, key[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        group = [g for g in group if g in cnt and g not in used]
        pts = [q for g in group for q in cnt[g]]
        if len(pts) >= need:
            used.update(group)
            m = np.mean(pts, axis=0)
            out.append({"x": round(float(m[0]), 2), "z": round(float(m[1]), 2), "frames": len(pts)})
    return sorted(out, key=lambda h: (h["x"], h["z"]))


def band(r, lo, hi, soft=0.15):
    if lo <= r <= hi:
        return 1.0
    d = (lo - r) / lo if r < lo else (r - hi) / hi
    return float(max(0.0, 1.0 - d / soft))


def expected_pixel_height(cam, floor, ref_h, ankle_h=ANKLE_H, head_top_m=None):
    """Pixel span head keypoints -> ankles (ankle_h up) of a ref_h person standing at `floor` (court world, Y-down).
    ref_h is a candidate height (candidates._multi: ankle keypoint -> top head keypoint + head_top_m), so the head
    keypoints sit ref_h - head_top_m above the ankle keypoints (review 1 L12). head_top_m None: params.json's."""
    if head_top_m is None:
        from analysis import people
        head_top_m = float(people.load_params()["head_top_m"])
    x, z = float(floor[0]), float(floor[1])
    uv, depth = pv.project(pv.proj_mats({"v": cam}, ["v"]),
                           np.array([[x, -(ankle_h + ref_h - head_top_m), z], [x, -ankle_h, z]]))
    if not (depth > 0).all():
        return float("nan")
    return float(abs(uv[1, 0, 1] - uv[0, 0, 1]))


def size_score(c, ref_h, cams, p, ankle_h=ANKLE_H):
    if np.isfinite(c.height):
        lo = p["size_crouch_lo"] if (np.isfinite(c.knee_deg) and c.knee_deg > p["crouch_knee_deg"]) else p["size_lo"]
        return band(c.height / ref_h, lo, p["size_hi"])
    if c.support == 1 and np.isfinite(c.pix_h):
        e = expected_pixel_height(cams[next(iter(c.dets))], c.floor, ref_h, ankle_h, head_top_m=p["head_top_m"])
        if np.isfinite(e) and e > 0:
            return band(c.pix_h / e, p["pix_ratio_lo"], p["pix_ratio_hi"])
    return 0.5


def reference_height(tracklets, pins, p, log=print):
    hs = [c.height for t in tracklets if t.tid in pins for c in t.cands if np.isfinite(c.height)]
    if len(hs) >= p["ref_min_samples"]:
        ref = float(np.median(hs))
        log(f"    reference height {ref:.2f} m (median of {len(hs)} seeded-player candidates)")
        return ref
    log(f"    reference height: too few seeded samples ({len(hs)}) - using {p['ref_height_m']:.2f} m")
    return float(p["ref_height_m"])


def _smooth(fl, k=5):
    return fl if len(fl) < k else median_filter(fl, size=(k, 1), mode="nearest")


def features(t, ctx) -> dict:
    p, fps = ctx["p"], ctx["fps"]
    fl = _smooth(t.floor())
    band = p["net_band_m"]
    off = fl[np.abs(fl[:, 1]) > band, 1]
    side = np.sign(np.median(off if len(off) else fl[:, 1])) or 1.0
    if len(fl) > 1:
        seg = np.linalg.norm(np.diff(fl, axis=0), axis=1)
        speed = seg / np.maximum(np.diff(t.frames), 1) * fps
        path = float(seg.sum())
    else:
        speed, path = np.zeros(1), 0.0
    net = float(np.linalg.norm(fl[-1] - fl[0]))
    near = np.zeros(len(fl), bool)
    for h in ctx["hotspots"]:
        near |= np.hypot(fl[:, 0] - h["x"], fl[:, 1] - h["z"]) < p["hotspot_radius_m"]
    sd = [ctx["app"].dist(t, s) for s in ctx["seed_tracklets"] if s.tid != t.tid]
    sd = [d for d in sd if d is not None]
    return {"in_lines": float(in_lines(fl, p["court_margin_m"]).mean()),
            # the net line counts as the tracklet's own side; a tracklet never off the line (umpire at the post) is 0.5
            "one_half": float(((np.sign(fl[:, 1]) == side) | (np.abs(fl[:, 1]) <= band)).mean()) if len(off) else 0.5,
            "support": float(np.clip((np.mean([c.support for c in t.cands]) - 1) / 2, 0, 1)),
            "size": float(np.mean([size_score(c, ctx["ref_h"], ctx["cams"], p, ctx.get("ankle_h", ANKLE_H)) for c in t.cands])),
            "moving": float(np.clip(np.percentile(speed, 90) / p["moving_mps"], 0, 1)),
            "straight": net / path if path > p["straight_min_path_m"] else 0.0,
            "hotspot": float(near.mean()),
            "seed_app": 1.0 - float(np.clip(min(sd) / ctx["app"].gate, 0, 1)) if sd else 0.5}


CLIP_LEVEL = ("hotspot",)    # evidence learnt over the whole clip (>= hotspot_min_s at that spot): not shrunk (T15)


def score(feat, n, p):
    c = p["classify"]
    z = c["bias"] + sum(w * feat[k] for k, w in c["weights"].items() if k not in CLIP_LEVEL)
    z *= min(1.0, n / c["min_frames_full_evidence"]) ** 2
    z += sum(c["weights"][k] * feat[k] for k in CLIP_LEVEL if k in c["weights"])
    return float(1.0 / (1.0 + np.exp(-z)))


def label_of(s, p):
    c = p["classify"]
    return "player" if s > c["player"] else ("nonplayer" if s < c["nonplayer"] else "unknown")


def classify(tracklets, pins, ctx):
    labels, scores, feats = {}, {}, {}
    for t in tracklets:
        feats[t.tid] = features(t, ctx)
        if t.tid in pins:
            labels[t.tid], scores[t.tid] = "player", 1.0
            continue
        s = score(feats[t.tid], len(t), ctx["p"])
        labels[t.tid], scores[t.tid] = label_of(s, ctx["p"]), s
    return labels, scores, feats
