# tools/tests/live_synth.py
"""Synthetic 2-player, 3-camera stream for the v0.8.0 live tests (on pose24_synth's rig + motion)."""
from __future__ import annotations

import numpy as np

import pose24_synth as syn
from analysis.live import sync

VIEWS = ("a", "b", "c")
BYSTANDER = dict(x0=5.5, z0=-1.0, walk=False, arm=False)


def scene(F=240, fps=60.0):
    cams = syn.rig(VIEWS)
    players = {"P1": syn.motion(F, fps, x0=0.0, z0=-3.0), "P2": syn.motion(F, fps, x0=0.5, z0=3.0, arm=False)}
    return cams, players


def seeds_views(cams, players):
    proj = {pid: syn.project(X[:1], cams, VIEWS) for pid, X in players.items()}
    return {v: {"seeds": {pid: [float(x) for x in proj[pid][v][0, [15, 16], :2].mean(axis=0)] for pid in players}}
            for v in VIEWS}


def _at(X, t):
    """Linear interpolation of (F,133,3) at fractional frame t (clamped)."""
    t = float(np.clip(t, 0, len(X) - 1))
    f0 = int(np.floor(t))
    f1 = min(f0 + 1, len(X) - 1)
    a = t - f0
    return (1 - a) * X[f0] + a * X[f1]


def raw_dets(cams, players, f, offset=0.0, noise_px=0.3, seed=0, bystander=True, drop=()):
    """{view: [raw (133,3)]} at that view's raw frame f; the scene time is f - offset (raw = ref + offset)."""
    rng = np.random.default_rng(seed + abs(int(f)))
    F = len(next(iter(players.values())))
    out = {}
    for v in VIEWS:
        dets = []
        if bystander:
            B = syn.motion(F, **BYSTANDER)
            dets.append(syn.project(_at(B, f - offset)[None], cams, [v], noise_px, rng)[v][0])
        for pid, X in players.items():
            if pid in drop:
                continue
            dets.append(syn.project(_at(X, f - offset)[None], cams, [v], noise_px, rng)[v][0])
        out[v] = dets
    return out


def frameset(step, ref_frame, cams, players, offsets=None, sync_mode="interp", drop=()):
    offsets = offsets or {v: 0.0 for v in VIEWS}
    views = {}
    for v in VIEWS:
        frames, a = sync.source_frames(ref_frame, offsets[v], sync_mode)
        views[v] = {"alpha": a, "frames": [
            {"src_frame": g, "dets": raw_dets(cams, players, g, offset=offsets[v], drop=drop)[v]} for g in frames]}
    return {"step": step, "ref_frame": ref_frame, "views": views}


def cfg(n_steps=120, stride=2, warmup=30, window=30, overlap=5, history=30):
    """A resolved live config for the synthetic scene: every nested section from configs/live/defaults.yaml plus
    the flat step-based keys the streaming core reads."""
    import os

    import yaml
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(root, "configs", "live", "defaults.yaml"), "r", encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    c.update({"views": list(VIEWS), "time_ref": VIEWS[0], "ref_frames": list(range(0, n_steps * stride, stride)),
              "stride": stride, "source_fps": 60.0, "fps": 60.0 / stride, "skeleton": "smpl-28",
              "warmup_steps": warmup, "window": window, "overlap": overlap, "history": history,
              "offsets": {v: 0.0 for v in VIEWS}, "offsets_source": "config", "sync_mode": "interp",
              "ankle_h_m": None, "mode": "singles", "clip": "synth", "fallback_max_hold": 5, "facing_window": 7,
              "workers": 0, "fit_budget_s": (window - overlap) / (60.0 / stride)})
    return c


def truth(players, pid, f):
    return players[pid][f]
