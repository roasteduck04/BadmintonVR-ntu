"""Link per-frame candidates into short 3D tracklets (SP3 spec § 2.3).

Frame-to-frame Hungarian on floor distance to a velocity-predicted position (gate link_gate_m, link_gate_1view_m when
either end is a 1-view candidate, growing by gate_growth_m_per_frame per missed frame); a shared rtmlib track id in a
view lowers the cost by track_id_bonus_m. A tracklet ends on a gap > max_gap frames, when it enters or leaves an
ambiguity (another candidate < ambig_m; leaving needs > ambig_m + ambig_hyst_m), or when the next candidate is beyond
the net band (|z| > net_band_m) on the other side of the tracklet's side (the side of its last candidate outside the
band), inside the court width. The band keeps a player standing at the net line (floor z jitters around 0, T15) in
one tracklet. Identity decisions are never made inside a tracklet: identity.py joins
tracklets."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

COURT_HALF_X = 3.05        # doubles sideline (m); the net spans the court width
BIG = 1e6


@dataclass
class Tracklet:
    tid: int
    frames: list = field(default_factory=list)
    cands: list = field(default_factory=list)
    crowded: bool = False
    side: int = 0              # -1 near / +1 far: side of the last candidate outside the net band (0: none yet)

    def add(self, c):
        self.frames.append(int(c.frame))
        self.cands.append(c)

    @property
    def first(self):
        return self.frames[0]

    @property
    def last(self):
        return self.frames[-1]

    def __len__(self):
        return len(self.frames)

    def floor(self):
        return np.array([c.floor for c in self.cands], float)

    def predict(self, f):
        if len(self.cands) < 2:
            return self.cands[-1].floor
        k = min(5, len(self.cands) - 1)
        v = (self.cands[-1].floor - self.cands[-1 - k].floor) / max(1, self.frames[-1] - self.frames[-1 - k])
        return self.cands[-1].floor + v * (f - self.frames[-1])


def _shared_ids(a, b):
    return sum(1 for v, t in b.track_ids.items() if t >= 0 and a.track_ids.get(v) == t)


def _side(c, p):
    z = float(c.floor[1])
    return 0 if abs(z) <= p["net_band_m"] else (1 if z > 0 else -1)


def _crosses_net(t, b, p):
    a = t.cands[-1]
    inside = max(abs(a.floor[0]), abs(b.floor[0])) <= COURT_HALF_X + p["court_margin_m"]
    s = _side(b, p)
    return bool(inside and t.side and s and s != t.side)


def _add(t, c, p):
    t.add(c)
    t.side = _side(c, p) or t.side


def link(cands_by_frame, p) -> list:
    done, active, nid = [], [], [0]

    def new(c):
        t = Tracklet(tid=nid[0], crowded=bool(c.crowd_dist < p["ambig_m"]))
        nid[0] += 1
        _add(t, c, p)
        return t

    for f, C in enumerate(cands_by_frame):
        keep = []
        for t in active:
            (keep if f - t.last - 1 <= p["max_gap"] else done).append(t)      # f - t.last - 1 frames missing
        active, taken = keep, set()
        if active and C:
            cost = np.full((len(active), len(C)), BIG)
            for i, t in enumerate(active):
                q, prev, miss = t.predict(f), t.cands[-1], f - t.last - 1
                for j, c in enumerate(C):
                    if _crosses_net(t, c, p):
                        continue
                    gate = (p["link_gate_1view_m"] if min(c.support, prev.support) == 1 else p["link_gate_m"]) \
                        + p["gate_growth_m_per_frame"] * miss
                    d = float(np.linalg.norm(c.floor - q))
                    if d <= gate:
                        cost[i, j] = max(0.0, d - p["track_id_bonus_m"] * _shared_ids(prev, c))
            nxt = list(active)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= BIG:
                    continue
                taken.add(j)
                t, c = active[i], C[j]
                enter = (not t.crowded) and c.crowd_dist < p["ambig_m"]
                leave = t.crowded and c.crowd_dist > p["ambig_m"] + p["ambig_hyst_m"]
                if enter or leave:
                    done.append(t)
                    nxt[i] = new(c)
                else:
                    _add(t, c, p)
            active = nxt
        for j, c in enumerate(C):
            if j not in taken:
                active.append(new(c))
    done += active
    return sorted(done, key=lambda t: (t.first, t.tid))


def cut_handovers(T, p) -> list:
    """Hand-over overlaps: a tracklet that ends <= handover_max_frames after another starts, next to it (< handover_m at
    that frame), is usually the same person seen as two candidates for a few frames (T15: a 1-view + a 2-view candidate
    of a far player). The old tracklet's overlapping tail becomes its own tracklet, so identity (one tracklet per
    player at a time) and seed pins don't lose the new, long tracklet over a 3-frame overlap."""
    nid = max((t.tid for t in T), default=-1) + 1
    starts = sorted(T, key=lambda t: (t.first, t.tid))
    firsts = [t.first for t in starts]
    out = list(T)
    for a in list(starts):
        lo = bisect.bisect_left(firsts, a.last - p["handover_max_frames"] + 1)
        hi = bisect.bisect_right(firsts, a.last)
        cands = [b for b in starts[lo:hi] if b is not a and a.first < b.first <= a.last < b.last]
        for b in sorted(cands, key=lambda b: b.first):
            if b.first > a.last:
                continue                     # an earlier cut already ended a before b
            k = a.frames.index(b.first) if b.first in a.frames else next(i for i, f in enumerate(a.frames) if f > b.first)
            if np.linalg.norm(a.cands[k].floor - b.cands[0].floor) >= p["handover_m"]:
                continue
            tail = Tracklet(tid=nid, crowded=a.crowded, side=a.side)
            nid += 1
            for c in a.cands[k:]:
                tail.add(c)
            a.frames, a.cands = a.frames[:k], a.cands[:k]
            out.append(tail)
    return sorted((t for t in out if len(t)), key=lambda t: (t.first, t.tid))
