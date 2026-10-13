"""tracker3d (SP3): TrackerV2-compatible multi-view people tracker - 3D candidates -> tracklets -> seed pins ->
player score -> per-half identity. pose24.track.track_and_sync24 swaps it in with --tracker 3d (streams already
synced by integer shifts, as TrackerV2 with shifts=...). run() returns TrackerV2's output format and keeps the
loader's own arrays, so pose24.track.detection_indices works unchanged."""
from __future__ import annotations

import numpy as np

from analysis import people
from analysis import pose_v2 as pv
from analysis.people import candidates, classify, identity, seeds, tracklets
from analysis.people import reid as reidmod


def release_pins(tids, pins, labels, scores, feats, T, p) -> dict:
    """Pins identity did not use (overruled by an overlapping better-pinned piece of that player, other half) are not
    reported as pinned: the tracklet gets its classifier score and label back (in place); returns the used pins."""
    n = {t.tid: len(t) for t in T}
    for tid in tids:
        if tid in pins:
            scores[tid] = classify.score(feats[tid], n[tid], p)
            labels[tid] = classify.label_of(scores[tid], p)
    return {tid: pid for tid, pid in pins.items() if tid not in tids}


class Tracker3D:
    def __init__(self, cams, kp2d, seeds_views, mode, fps, shifts=None, kp_idx=None, appear=None, track_ids=None,
                 params=None, reid="colour", reid_feats=None, log=print):
        self.mode, self.fps, self.log = mode, float(fps), log
        self.p = params or people.load_params()
        self.players = seeds.player_ids(mode)
        self.views = sorted(kp2d)
        self.cams = cams
        shifts = shifts or {}
        self.shifts = {v: int(shifts.get(v, 0)) for v in self.views}
        self.kp2d = {v: pv._shift(kp2d[v], self.shifts[v]) for v in self.views}
        if kp_idx:
            self.kp_idx = {v: pv._shift(kp_idx[v], self.shifts[v]) for v in self.views}
        else:
            self.kp_idx = {v: [list(range(len(fr))) for fr in self.kp2d[v]] for v in self.views}
        self.track_ids = {v: pv._shift(track_ids[v], self.shifts[v]) for v in self.views
                          if track_ids and track_ids.get(v) is not None}
        self.frame_count = min(len(self.kp2d[v]) for v in self.views)
        self.seeds_views = seeds_views or {}
        self.sides = seeds.player_sides(self.seeds_views, cams, mode, log)
        feats = reid_feats if reid == "osnet" else appear
        self.app = reidmod.Appearance(reid, feats, self.shifts, self.kp_idx, self.p)
        self.reacquired, self.events, self.result, self.track = 0, [], None, {}

    def run(self) -> dict:
        F, p = self.frame_count, self.p

        def cands(frames, ankle_h):
            out = []
            for f in frames:
                dets = {v: self.kp2d[v][f] for v in self.views}
                tids = {v: self.track_ids[v][f] for v in self.track_ids if f < len(self.track_ids[v])}
                out.append(candidates.frame_candidates(f, dets, self.cams, self.views, p, tids, ankle_h=ankle_h))
            return out

        # the ankle plane of the floor points comes from the clip's own well-seen triangulated ankles (T15)
        ankle_h = candidates.estimate_ankle_height(cands(range(0, F, p["ankle_est_step"]), p["ankle_h_m"]), p,
                                                   n_views=len(self.views))
        self.log(f"    ankle keypoint height {ankle_h:.3f} m (floor points of single views)")
        C = cands(range(F), ankle_h)
        T = tracklets.cut_handovers(tracklets.link(C, p), p)
        pins = seeds.pin_tracklets(seeds.seed_events(self.seeds_views, self.views, self.fps), T, self.kp2d,
                                   self.shifts, F, log=self.log, p=p)
        hs = classify.hotspots(C, self.fps, p)
        ref_h = classify.reference_height(T, pins, p, log=self.log)
        ctx = {"p": p, "fps": self.fps, "cams": self.cams, "ref_h": ref_h, "ankle_h": ankle_h, "hotspots": hs, "app": self.app,
               "seed_tracklets": [t for t in T if t.tid in pins]}
        labels, scores, feats = classify.classify(T, pins, ctx)
        unpinned = set()
        player_of = identity.assign(T, labels, scores, pins, self.sides, self.fps, self.app, p, log=self.log,
                                    unpinned=unpinned)
        pins = release_pins(unpinned, pins, labels, scores, feats, T, p)
        final = {}
        for t in T:
            if t.tid in player_of:
                final[t.tid] = "player"
            elif labels[t.tid] == "player":
                final[t.tid] = "unknown"          # player-like, but no free identity (spec § 2.5)
            else:
                final[t.tid] = labels[t.tid]
        out = {pid: {v: [None] * F for v in self.views} for pid in self.players}
        self.track = {pid: np.full((F, 2), np.nan) for pid in self.players}
        for t in T:
            pid = player_of.get(t.tid)
            if pid is None:
                continue
            for f, c in zip(t.frames, t.cands):
                self.track[pid][f] = c.floor
                for v, i in c.dets.items():
                    out[pid][v][f] = self.kp2d[v][f][i]
        per = {}
        for tid, pid in player_of.items():
            per.setdefault(pid, []).append(tid)
        self.reacquired = sum(max(0, len(x) - 1) for x in per.values())
        self.result = {"tracklets": T, "labels": final, "scores": scores, "features": feats, "player_of": player_of,
                       "pins": pins, "hotspots": hs, "ref_height_m": ref_h, "ankle_h_m": ankle_h}
        n = {k: sum(1 for x in final.values() if x == k) for k in ("player", "nonplayer", "unknown")}
        self.log(f"    tracker3d: {len(T)} tracklets (player {n['player']}, nonplayer {n['nonplayer']}, "
                 f"unknown {n['unknown']}), {len(pins)} pinned, {len(hs)} bystander hotspots")
        return out
