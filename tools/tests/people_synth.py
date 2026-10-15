"""Synthetic multi-camera scenes for the SP3 people tests: players, bystanders, a walker, occlusion, seeds
(frame_time + additional_seeds), rtmlib-style track ids and an appearance cache, written as data/calib,
data/pose2d and data/pose_cache files under `root` in the pipeline's conventions (see pose24_synth: court
world Y-down, analysis world Y-up)."""
from __future__ import annotations

import json
import os

import numpy as np

import pose24_synth as syn
from analysis import appearance

FPS = 60.0
SEED_TIMES = (0.5, 1.5)
VIEWS = ("a", "b", "c", "d")
PIDS = {"p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}


def standing(F, x, z, arm=False):
    """A person standing still at (x, z) (Y-up world keypoints)."""
    return syn.motion(F, x0=x, z0=z, walk=False, arm=arm)


def along(F, pts, arm=False):
    """Walk through waypoints [(x, z), ...] at constant speed over the clip (piecewise linear)."""
    base = standing(F, 0.0, -1.0, arm=arm) - np.array([0.0, 0.0, -1.0])
    pts = np.asarray(pts, float)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0.0, cum[-1], F)
    x, z = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
    return base + np.stack([x, np.zeros(F), z], axis=1)[:, None, :]


def colour(hue_top, hue_bottom):
    """A 30-float appearance feature: one hue bin per region (appearance.region_hist layout)."""
    f = np.zeros(appearance.DIM)
    f[hue_top] = 1.0
    f[appearance.N_HUE + 3 + hue_bottom] = 1.0
    return f


def _hidden(span, f):
    return span is not None and span[0] <= f < span[1]


def write_scene(root, name, people, cams, views, mode="singles", seeds=None, hidden=None, shifts=None,
                noise_px=0.5, colours=None, seed_times=SEED_TIMES, seed=0):
    """people: {person: (F,133,3) Y-up world}; seeds: {pid: person} clicked at the feet at every seed time
    (first = frame_time, the rest = additional_seeds); hidden: {(person, view): (f0, f1)} synced frames [f0, f1)
    that camera doesn't detect the person; colours: {person: (30,)} -> data/pose_cache/<name>_appearance.npz
    (every 2nd raw frame, keyed (raw frame, det index) like analysis.appearance); shifts[v] = k: raw frame f + k
    of v shows synced frame f. Detections are shuffled per frame; track_id = 1 + person index.
    Returns (registry entry, truth {(view, raw frame, det index): person})."""
    rng = np.random.default_rng(seed)
    shifts, hidden, seeds, colours = shifts or {}, hidden or {}, seeds or {}, colours or {}
    views = list(views)
    calib, pose2d, cache = (os.path.join(root, "data", d) for d in ("calib", "pose2d", "pose_cache"))
    for d in (calib, pose2d, cache):
        os.makedirs(d, exist_ok=True)
    F = len(next(iter(people.values())))
    names = sorted(people)
    proj = {n: syn.project(people[n], cams, views, noise_px, rng) for n in names}
    truth, app = {}, {}
    for v in views:
        K, R, t = cams[v]
        with open(os.path.join(calib, f"{v}_cam.json"), "w", encoding="utf-8") as fh:
            json.dump({"K": K.tolist(), "R": R.tolist(), "t": t.tolist(), "width": syn.W, "height": syn.H}, fh)
        k = int(shifts.get(v, 0))
        frames, idx_rows, feat_rows = [], [], []
        for r in range(F):
            f, dets = r - k, []
            if 0 <= f < F:
                who = [n for n in names if not _hidden(hidden.get((n, v)), f) and (proj[n][v][f, :17, 2] > 0).all()]
                rng.shuffle(who)
                for d, n in enumerate(who):
                    dets.append({"track_id": 1 + names.index(n), "keypoints": proj[n][v][f].tolist()})
                    truth[(v, r, d)] = n
                    if n in colours and r % 2 == 0:
                        idx_rows.append((r, d))
                        feat_rows.append(colours[n])
            frames.append(dets)
        with open(os.path.join(pose2d, f"{v}_kp2d.json"), "w", encoding="utf-8") as fh:
            json.dump({"view": v, "clip": name, "fps": FPS, "width": syn.W, "height": syn.H, "frame_count": F,
                       "keypoint_format": "coco_wholebody_133", "multi_person": True, "frames": frames}, fh)
        if idx_rows:
            app[f"{v}_idx"] = np.array(idx_rows, np.int32)
            app[f"{v}_feat"] = np.array(feat_rows, np.float16)
    if app:
        np.savez_compressed(os.path.join(cache, f"{name}_appearance.npz"), **app)
    sv = {}
    for v in views:
        k, sets = int(shifts.get(v, 0)), []
        for ts in seed_times:
            f = int(round(ts * FPS)) - k
            if not 0 <= f < F:
                continue
            sets.append({"frame_time": ts,
                         "seeds": {pid: [float(x) for x in proj[n][v][f, [15, 16], :2].mean(axis=0)]
                                   for pid, n in seeds.items() if not _hidden(hidden.get((n, v)), f)}})
        if sets:
            sv[v] = {"frame_time": sets[0]["frame_time"], "seeds": sets[0]["seeds"], "additional_seeds": sets[1:]}
    with open(os.path.join(calib, f"{name}_seeds.json"), "w", encoding="utf-8") as fh:
        json.dump({"clip": name, "mode": mode, "views": sv}, fh)
    entry = {"mode": mode, "seeds": name, "time_ref": sorted(views)[0], "views": views,
             "videos": {v: None for v in views}}
    return entry, truth


def raise_ankles(X, dh):
    """Ankle keypoints dh m higher (real rigs triangulate them ~0.15 m up, the template has 0.08 m)."""
    X = np.array(X, float)
    X[:, [15, 16], 1] += dh
    return X


def singles_scene(root, F=180, shifts=None, ankle_dh=0.0, **kw):
    """P1 near (steps outside the doubles sideline: x up to 3.7 m), P2 far, a static bystander at (-3.5, 7.5)
    seen by b and d only, an umpire-like person beside the net post (4.2, 0.0), a walker crossing the near half.
    ankle_dh raises everyone's ankle keypoints."""
    cams = syn.rig(VIEWS)
    people = {"p1": syn.motion(F, x0=2.9, z0=-3.0), "p2": syn.motion(F, x0=0.0, z0=3.5),
              "by": standing(F, -3.5, 7.5), "ump": standing(F, 4.2, 0.0),
              "walk": along(F, [(-5.0, -1.0), (5.0, -1.0)])}
    people = {n: raise_ankles(X, ankle_dh) for n, X in people.items()}
    colours = {"p1": colour(0, 1), "p2": colour(4, 5), "by": colour(8, 9), "ump": colour(2, 2), "walk": colour(10, 11)}
    entry, truth = write_scene(root, "syns", people, cams, VIEWS, "singles", seeds={"P1": "p1", "P2": "p2"},
                               hidden={("by", "a"): (0, F), ("by", "c"): (0, F)},
                               shifts=shifts if shifts is not None else {"c": 2}, colours=colours, **kw)
    return {"syns": entry}, truth, people


def doubles_scene(root, F=180, **kw):
    """P1 / P2 (near) cross 0.35 m apart with different colours; P4 (far) is seen by camera d only for frames 60-120;
    P3 far; the static bystander."""
    cams = syn.rig(VIEWS)
    people = {"p1": along(F, [(-1.5, -3.0), (1.5, -3.0)], arm=True),
              "p2": along(F, [(1.5, -3.35), (-1.5, -3.35)], arm=True),
              "p3": syn.motion(F, x0=-1.2, z0=3.0), "p4": syn.motion(F, x0=1.2, z0=4.5),
              "by": standing(F, -3.5, 7.5)}
    colours = {"p1": colour(0, 0), "p2": colour(6, 6), "p3": colour(3, 3), "p4": colour(9, 9), "by": colour(1, 11)}
    hidden = {("p4", "a"): (60, 120), ("p4", "b"): (60, 120), ("p4", "c"): (60, 120),
              ("by", "a"): (0, F), ("by", "c"): (0, F)}
    entry, truth = write_scene(root, "synd", people, cams, VIEWS, "doubles",
                               seeds={"P1": "p1", "P2": "p2", "P3": "p3", "P4": "p4"}, hidden=hidden,
                               colours=colours, **kw)
    return {"synd": entry}, truth, people


def net_scene(root, F=180, **kw):
    """Singles: both players come to the net at the same x (0.4 m apart across it) with identical colours."""
    cams = syn.rig(VIEWS)
    people = {"p1": along(F, [(0.0, -3.0), (0.0, -0.2), (0.3, -3.0)]),
              "p2": along(F, [(0.0, 3.0), (0.0, 0.2), (-0.3, 3.0)])}
    colours = {"p1": colour(5, 5), "p2": colour(5, 5)}
    entry, truth = write_scene(root, "synn", people, cams, VIEWS, "singles", seeds={"P1": "p1", "P2": "p2"},
                               colours=colours, **kw)
    return {"synn": entry}, truth, people


def reseed_scene(root, F=240, **kw):
    """Doubles near-side teammates in identical colours cross at frame ~120; seeds at 0.5 s and 3.0 s: only the
    additional seed can say who is who after the crossing."""
    cams = syn.rig(VIEWS)
    people = {"p1": along(F, [(-1.5, -3.0), (1.5, -3.0)]), "p2": along(F, [(1.5, -3.35), (-1.5, -3.35)]),
              "p3": syn.motion(F, x0=-1.2, z0=3.0), "p4": syn.motion(F, x0=1.2, z0=4.5)}
    colours = {"p1": colour(7, 7), "p2": colour(7, 7), "p3": colour(3, 3), "p4": colour(9, 9)}
    entry, truth = write_scene(root, "synr", people, cams, VIEWS, "doubles",
                               seeds={"P1": "p1", "P2": "p2", "P3": "p3", "P4": "p4"}, colours=colours,
                               seed_times=(0.5, 3.0), **kw)
    return {"synr": entry}, truth, people
