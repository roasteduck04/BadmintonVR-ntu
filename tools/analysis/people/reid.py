"""Appearance per tracklet (SP3 spec § 2.6).

colour (default): the analysis.appearance cache (torso + shorts colour histograms, every 2nd raw frame, keyed
(raw frame, original det index)); a tracklet's appearance is the mean feature per view (lighting differs per camera),
distance = mean Hellinger over the views both tracklets have. osnet (--reid osnet): OSNet-x0.25 512-d embeddings from
data/pose_cache/<clip>_osnet.npz (same keying, see osnet.py); a tracklet's appearance is the mean L2-normalised
embedding over all views, distance = 1 - cosine. Used as a gate (> gate: never the same person) and a cost."""
from __future__ import annotations

import numpy as np

from analysis import appearance


class Appearance:
    def __init__(self, kind, feats, shifts, kp_idx, p):
        if kind not in ("colour", "osnet"):
            raise ValueError(f"unknown reid kind {kind!r} (colour | osnet)")
        self.kind, self.feats, self.shifts, self.kp_idx = kind, feats or {}, shifts, kp_idx
        a = p["appearance"]
        self.gate, self.weight = float(a[f"{kind}_gate"]), float(a[f"{kind}_weight_m"])
        self._cache = {}

    def of(self, t) -> dict:
        if t.tid not in self._cache:
            self._cache[t.tid] = self._compute(t)
        return self._cache[t.tid]

    def _compute(self, t):
        acc = {}
        for f, c in zip(t.frames, t.cands):
            for v, i in c.dets.items():
                ids = self.kp_idx.get(v)
                if ids is None or f >= len(ids) or i >= len(ids[f]):
                    continue
                ft = self.feats.get(v, {}).get((f + self.shifts.get(v, 0), int(ids[f][i])))
                if ft is not None:
                    acc.setdefault(v, []).append(np.asarray(ft, float))
        if self.kind == "colour":
            return {v: np.mean(a, axis=0) for v, a in acc.items()}
        allv = [x / (np.linalg.norm(x) + 1e-12) for a in acc.values() for x in a]
        if not allv:
            return {}
        m = np.mean(allv, axis=0)
        return {"all": m / (np.linalg.norm(m) + 1e-12)}

    def dist(self, a, b):
        A, B = self.of(a), self.of(b)
        shared = sorted(set(A) & set(B))
        if not shared:
            return None
        if self.kind == "colour":
            return float(np.mean([appearance.hellinger(A[v], B[v]) for v in shared]))
        return float(1.0 - np.dot(A["all"], B["all"]))
