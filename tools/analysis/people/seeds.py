"""Seeds (<clip>_seeds.json, clicked in serve_calibrator / click_seeds) -> player sides and tracklet pins
(SP3 spec § 2.5). Every seed set is used: `frame_time` and each `additional_seeds` entry; frame_time is in seconds
of that view's video (ffmpeg -ss), so the raw frame is round(frame_time x fps) and the synced frame is raw - shift."""
from __future__ import annotations

import numpy as np

from analysis import people
from analysis import player_tracker as pt
from analysis import pose_v2 as pv


def player_ids(mode):
    return ["P1", "P2"] if mode == "singles" else ["P1", "P2", "P3", "P4"]


def player_sides(seeds_views, cams, mode, log=print) -> dict:
    zs = {}
    for v, e in (seeds_views or {}).items():
        if v not in cams:
            continue
        K, R, t = cams[v]
        for pid, px in (e.get("seeds") or {}).items():
            g = pt.pixel_to_ground(px[0], px[1], K, R, t)
            if g is not None:
                zs.setdefault(pid, []).append(g[1])
    ids, sides = player_ids(mode), {}
    for i, pid in enumerate(ids):
        if pid in zs:
            sides[pid] = "near" if np.median(zs[pid]) < 0 else "far"
        else:
            sides[pid] = "near" if i < len(ids) // 2 else "far"
            log(f"    WARNING: no seed for {pid}: assuming the {sides[pid]} half")
    return sides


def seed_events(seeds_views, views, fps) -> list:
    out = []
    for v in views:
        e = (seeds_views or {}).get(v) or {}
        sets = [(e.get("frame_time", 1.0), e.get("seeds") or {})]
        sets += [(a.get("frame_time"), a.get("seeds") or {}) for a in e.get("additional_seeds") or []]
        for ft, seeds in sets:
            if ft is None:
                continue
            r = int(round(float(ft) * fps))
            out += [(v, r, pid, (float(px[0]), float(px[1]))) for pid, px in seeds.items()]
    return out


def nearest_feet(dets, u, v, min_px, frac, s_min=pv.W_MIN):
    """(index, feet distance px, distance / keypoint-box height, accepted) of the detection whose feet (ankle mean;
    else the bottom centre of its keypoint box) are nearest the click relative to its limit max(min_px, frac x box
    height); (None, inf, inf, False) without usable detections."""
    best = (None, np.inf, np.inf, False)
    best_r = np.inf
    for i, k in enumerate(dets):
        ok = k[:23, 2] >= s_min
        if ok.sum() < 3:
            continue
        pts = k[:23][ok, :2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        feet = k[[15, 16]][k[[15, 16], 2] >= s_min, :2]
        anchor = feet.mean(axis=0) if len(feet) else np.array([(lo[0] + hi[0]) / 2, hi[1]])
        d = float(np.linalg.norm(anchor - [u, v]))
        h = max(float(hi[1] - lo[1]), 1e-6)
        lim = max(min_px, frac * h)
        if d / lim < best_r:
            best, best_r = (i, d, d / h, d <= lim), d / lim
    return best


def det_under_pixel(dets, u, v, min_px=None, frac=None, s_min=pv.W_MIN):
    """The detection whose feet are nearest the click, when they are within max(min_px, frac x its keypoint-box
    height) (params seed_min_px / seed_feet_frac by default); else None. Seeds are clicked at the feet: a click that is
    only inside someone's box is not evidence (T15: doubles_2sides' 1 s seeds, clicked on another moment, sit 0.16-0.38
    x the person's height from any feet and pinned the wrong people; good clicks are <= 0.11 x)."""
    if min_px is None or frac is None:
        p = people.load_params()
        min_px = p["seed_min_px"] if min_px is None else min_px
        frac = p["seed_feet_frac"] if frac is None else frac
    i, _, _, ok = nearest_feet(dets, u, v, min_px, frac, s_min)
    return i if ok else None


def pin_tracklets(events, tracklets, kp2d, shifts, F, log=print, p=None) -> dict:
    p = p or people.load_params()
    at = {}
    for t in tracklets:
        for f, c in zip(t.frames, t.cands):
            for v, i in c.dets.items():
                at[(v, f, i)] = t
    votes = {}
    for v, r, pid, (u, w) in events:
        if v not in shifts:
            continue
        f = r - shifts[v]
        if not 0 <= f < F:
            log(f"    seed {pid} {v} raw {r}: outside the synced timeline - skipped")
            continue
        i, d, fr, ok = nearest_feet(kp2d[v][f] if f < len(kp2d[v]) else [], u, w, p["seed_min_px"], p["seed_feet_frac"])
        if not ok:
            why = "no detection" if i is None else f"nearest feet {d:.0f} px = {fr:.2f} x its box height"
            log(f"    seed {pid} {v} raw {r}: {why} (limit max({p['seed_min_px']:.0f} px, {p['seed_feet_frac']:.2f} x)) "
                f"- skipped")
            continue
        if (v, f, i) not in at:
            log(f"    seed {pid} {v} raw {r}: feet {d:.0f} px = {fr:.2f} x its box height, but the detection is in no "
                f"tracklet - skipped")
            continue
        votes.setdefault(at[(v, f, i)].tid, []).append(pid)
    by_tid = {t.tid: t for t in tracklets}
    pins = {}
    for tid, pids in votes.items():
        top = max(sorted(set(pids)), key=pids.count)
        if 2 * pids.count(top) > len(pids):
            pins[tid] = top
        else:
            log(f"    tracklet {tid}: conflicting seeds {sorted(set(pids))} - not pinned")
    # a player can't be in two tracklets at once: keep the better-voted of overlapping pins
    for pid in sorted(set(pins.values())):
        mine = sorted((tid for tid, p in pins.items() if p == pid), key=lambda tid: -votes[tid].count(pid))
        kept = []
        for tid in mine:
            t = by_tid[tid]
            if any(not (t.last < by_tid[k].first or t.first > by_tid[k].last) for k in kept):
                log(f"    tracklet {tid}: {pid} is already pinned at the same time - pin dropped")
                del pins[tid]
            else:
                kept.append(tid)
    return pins
