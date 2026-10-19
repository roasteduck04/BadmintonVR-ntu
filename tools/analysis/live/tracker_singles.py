# tools/analysis/live/tracker_singles.py
"""Online singles tracker: per time-step 3D candidates (people.candidates) -> one candidate per player by court half
+ distance to the player's last floor position (Hungarian). No seeds: P1 = near half (z < 0), P2 = far half, the
convention the offline seedless fallback (people.seeds.player_sides) uses. First sighting: the best-supported
candidate on the player's half. Every weight / gate comes from the run config's `tracker` section. Doubles is out
of scope.

Seedless start (config `warmup`): a player is only acquired from a plausible candidate (seen by >= min_support_views
cameras, standing height inside player_height_m when it can be measured). During the warm-up steps a tracked player
whose candidates keep being implausible (more than half of the last few judged steps) is dropped and re-acquired, so
a bystander or a ghost picked at the start is replaced. A step where fewer than min_support_views cameras see the player
is not judged (one camera cannot measure a height): v0.8.1 fix, at 60 fps a far player seen by 2 cameras was dropped
after a 0.1 s one-camera gap and only re-acquired 5 s later. Every acquire / drop is recorded in `events`."""
from __future__ import annotations

import numpy as np
from collections import deque

from scipy.optimize import linear_sum_assignment

from analysis import people
from analysis import pose_v2 as pv
from analysis.people import candidates

BIG = 1e6
SIDES = {"P1": "near", "P2": "far"}


class SinglesTracker:
    def __init__(self, cams, views, fps, tcfg, params=None, ankle_h=None, wcfg=None, warm_steps=0):
        self.p = params or people.load_params()
        self.t = dict(tcfg)
        self.w = dict(wcfg or {})
        self.warm = int(warm_steps)
        self.events = []
        self.views = sorted(views)
        self.cams = {v: cams[v] for v in self.views}
        self.fps = float(fps)
        self.ankle_auto = ankle_h is None
        self.ankle_h = float(ankle_h) if ankle_h is not None else float(self.p["ankle_h_m"])
        self.cand_hist = []
        self.sides = dict(SIDES)
        self.pids = sorted(self.sides)
        self.last = {pid: None for pid in self.pids}
        self.lost = {pid: 0 for pid in self.pids}
        self.gate_m, self.gate_growth = float(self.t["gate_m"]), float(self.t["gate_growth_mps"])
        self.check_n = max(3, min(10, self.warm // 3)) if self.warm else 0
        self.recent = {pid: deque(maxlen=self.check_n or 1) for pid in self.pids}

    def estimate_ankle(self) -> dict:
        """warmup.ankle_h = auto: the ankle-plane height from the warm-up candidates (people.candidates rule), used
        from the next step on. Returns what was decided (for the report)."""
        before = self.ankle_h
        est = candidates.estimate_ankle_height(self.cand_hist, self.p, n_views=len(self.views))
        n = sum(1 for C in self.cand_hist for c in C if c.tri and np.isfinite(c.ankle_h))
        self.ankle_h = float(est)
        self.cand_hist = []
        return {"ankle_h_m": self.ankle_h, "default_m": before, "samples": n,
                "used_default": n < int(self.p["ankle_min_samples"])}

    def plausible(self, c) -> bool:
        lo, hi = self.w.get("player_height_m") or (0.0, 99.0)
        if c.support < int(self.w.get("min_support_views", 1)):
            return False
        h = float(c.height)
        return (not np.isfinite(h)) or float(lo) <= h <= float(hi)

    def _on_side(self, pid, floor):
        z = float(floor[1])
        band = float(self.t["net_band_m"])
        return z < band if self.sides[pid] == "near" else z > -band

    def step(self, i, dets) -> dict:
        C = candidates.frame_candidates(i, {v: dets.get(v, []) for v in self.views}, self.cams, self.views, self.p,
                                        ankle_h=self.ankle_h)
        m = float(self.t["court_margin_m"])
        C = [c for c in C if abs(c.floor[0]) <= pv.COURT_X_MAX + m and abs(c.floor[1]) <= pv.COURT_Z_MAX + m]
        self.last_candidates = C
        if self.ankle_auto and i < self.warm:
            self.cand_hist.append(C)
        cost = np.full((len(self.pids), max(len(C), 1)), BIG)
        for a, pid in enumerate(self.pids):
            for b, c in enumerate(C):
                if not self._on_side(pid, c.floor):
                    continue
                if self.last[pid] is None:
                    if not self.plausible(c):
                        continue
                    cost[a, b] = (-float(self.t["first_support_weight"]) * c.support
                                  + float(self.t["first_depth_weight"]) * abs(float(c.floor[1])))
                else:
                    d = float(np.linalg.norm(np.asarray(c.floor, float) - self.last[pid]))
                    if d <= self.gate_m + self.gate_growth * self.lost[pid] / self.fps:
                        cost[a, b] = d - float(self.t["support_weight"]) * c.support
        chosen = {pid: None for pid in self.pids}
        if C:
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < BIG / 2:
                    chosen[self.pids[a]] = C[b]
        out = {}
        for pid in self.pids:
            c = chosen[pid]
            if c is None:
                self.lost[pid] += 1
                out[pid] = {}
                continue
            if self.last[pid] is None:
                self.events.append({"kind": "acquire", "step": int(i), "pid": pid, "floor": [float(x) for x in c.floor],
                                    "support": int(c.support), "height_m": float(c.height),
                                    "candidates": len(C)})
                self.recent[pid].clear()
            self.last[pid], self.lost[pid] = np.asarray(c.floor, float), 0
            out[pid] = {v: int(k) for v, k in c.dets.items()}
            if i < self.warm and self.check_n and c.support >= int(self.w.get("min_support_views", 1)):
                self.recent[pid].append(self.plausible(c))
                r = self.recent[pid]
                if len(r) == self.check_n and sum(r) < self.check_n / 2:
                    self.events.append({"kind": "drop", "step": int(i), "pid": pid, "reason": "implausible",
                                        "floor": [float(x) for x in c.floor], "height_m": float(c.height),
                                        "support": int(c.support)})
                    self.last[pid] = None
                    r.clear()
        return out
