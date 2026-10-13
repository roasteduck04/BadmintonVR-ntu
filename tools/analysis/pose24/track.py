"""pose24 tracking + sync (spec § 2 step 3). Uses pose_v2's TrackerV2 / estimate_offsets / view_health on
the per-group calibrated 2D, adding: a fixed sync reference, an optional shared timeline (--timeline-of:
a camera-subset build reuses the full build's reference and offsets, so frame f is the same instant in
both), per-frame detection indices, and a loud flag when < 3 views can't vote on view health.
pose_v2.track_and_sync itself is not changed (posev2 output stays identical)."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis import appearance
from analysis import pose_v2 as pv
from analysis import people
from analysis.people import persons as personsmod
from analysis.pose24 import scores

CACHE_SUFFIX = "_pose24"
SYNC_DISAGREE_M = 1.0      # SP3 finding 2: v2 (sync) vs tracker3d floor tracks; fallback when params.json lacks
                           # sync_disagree_m (the tracker3d run reads people.load_params()["sync_disagree_m"])


def _median_dist(a, b, min_frames):
    n = min(len(a), len(b))
    a, b = np.asarray(a)[:n], np.asarray(b)[:n]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))) if ok.sum() >= min_frames else None


def sync_matching(track_a, track_b, min_frames=30) -> dict:
    """One-to-one matching (Hungarian on the median floor distance) of track_b's players (tracker3d) to track_a's
    tracks (TrackerV2): {pid_b: (pid_a, median m)}, (None, None) when no track_a track overlaps it >= min_frames.
    One-to-one, not nearest: two tracker3d players must not both match one v2 track (the other v2 track may follow
    nobody); not same-id: the labels may be permuted when the two are seeded differently (T15)."""
    pb, pa = sorted(track_b or {}), sorted(track_a or {})
    out = {pid: (None, None) for pid in pb}
    if not pb or not pa:
        return out
    D = [[_median_dist(track_a[a], track_b[b], min_frames) for a in pa] for b in pb]
    big = 1e6
    cost = np.array([[big if d is None else d for d in row] for row in D])
    for i, j in zip(*linear_sum_assignment(cost)):
        if D[i][j] is not None:
            out[pb[i]] = (pa[j], D[i][j])
    return out


def sync_disagreement(track_a, track_b, min_frames=30) -> dict:
    """Per player of track_b: median floor distance to its one-to-one matched track_a track (sync_matching; None when
    unmatched)."""
    return {pid: d for pid, (_, d) in sync_matching(track_a, track_b, min_frames).items()}


def sync_matches(track_a, track_b, min_frames=30) -> dict:
    """Per player of track_b: the id of its one-to-one matched track_a track (None when unmatched)."""
    return {pid: q for pid, (q, _) in sync_matching(track_a, track_b, min_frames).items()}


def sync_check(track_v2, track_3d, threshold=SYNC_DISAGREE_M, min_frames=30, log=print):
    """Log the v2 (sync) vs tracker3d matching: a note per label permutation, a WARNING per tracker3d player whose
    matched v2 track is > threshold away or that has no v2 match, and per v2 track matched to no tracker3d player.
    Returns (disagreement {pid: m | None}, matches {pid: v2 id | None})."""
    m = sync_matching(track_v2, track_3d, min_frames)
    for pid, (q, d) in m.items():
        if q is None:
            log(f"    WARNING: {pid}: no TrackerV2 track follows tracker3d's {pid} - the sync offsets may come from a "
                f"wrong person (SP3 finding 2)")
        elif d > threshold:
            log(f"    WARNING: {pid}: its matched TrackerV2 track {q} is {d:.1f} m (median) from tracker3d's - the sync "
                f"offsets may come from a wrong person (SP3 finding 2)")
        elif q != pid:
            log(f"    note: tracker3d {pid} follows TrackerV2's {q} (their seeds disagree on the label; the sync only "
                f"needs one person per track)")
    used = {q for q, _ in m.values() if q is not None}
    for q in sorted(set(track_v2 or {}) - used):
        log(f"    WARNING: TrackerV2 track {q} matches no tracker3d player - it may follow a non-player (SP3 finding 2)")
    return {pid: d for pid, (_, d) in m.items()}, {pid: q for pid, (q, _) in m.items()}


def _make_tracker(kind, cams, kp2d, seeds_views, mode, fps, shifts, kp_idx, appear, track_ids, reid, reid_feats, log):
    if kind == "v2":
        return pv.TrackerV2(cams, kp2d, seeds_views, mode, fps, shifts=shifts, kp_idx=kp_idx, appear=appear)
    if kind == "3d":
        from analysis.people.tracker3d import Tracker3D
        return Tracker3D(cams, kp2d, seeds_views, mode, fps, shifts=shifts, kp_idx=kp_idx, appear=appear,
                         track_ids=track_ids, reid=reid, reid_feats=reid_feats, log=log)
    raise ValueError(f"unknown tracker {kind!r} (v2 | 3d)")


def cache_path(clip, cache_dir=pv.CACHE_DIR):
    return os.path.join(cache_dir, f"{clip}{CACHE_SUFFIX}.npz")


def load_timeline(clip, cache_dir=pv.CACHE_DIR) -> dict:
    """Sync reference + offsets + frame count of an earlier pose24 build of `clip`."""
    path = cache_path(clip, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--timeline-of {clip}: no pose24 cache at {path}; build {clip} first")
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    return {"clip": clip, "offsets": dict(meta.get("offsets_all") or meta["offsets"]),
            "time_ref_view": meta["time_ref_view"], "frame_count": int(meta["frame_count"])}


def detection_indices(tr, matched, n) -> dict:
    """{pid: (V, n) int32}: the original kp2d detection index each (view, tracker frame) used, -1 = none.
    Tracker frame f is raw kp2d frame f + tr.shifts[view]. Found by identity: the tracker stores the very
    array objects `load_kp2d24` returned."""
    out = {}
    for pid in matched:
        D = np.full((len(tr.views), n), -1, np.int32)
        for k, v in enumerate(tr.views):
            for f, kps in enumerate(matched[pid][v][:n]):
                if kps is None:
                    continue
                for i, cand in enumerate(tr.kp2d[v][f]):
                    if cand is kps:
                        D[k, f] = tr.kp_idx[v][f][i]
                        break
        out[pid] = D
    return out


def _resample(matched, views, offsets, shifts, n):
    obs = {}
    for pid in matched:
        obs[pid] = {}
        for v in views:
            D = pv.dense(matched[pid][v], n)
            r = offsets[v] - shifts[v]
            obs[pid][v] = pv.sample_at(D, np.arange(n) + r) if abs(r) > 1e-6 else D
    return obs


def track_and_sync24(cfg, params, timeline=None, log=print, tracker="v2", reid="colour", reid_feats=None) -> dict:
    """Load calibrated 2D, sync (or take the timeline's offsets), track, drop views whose calibration
    disagrees (>= 3 views only). Returns a pose_v2-style bundle plus pose24's extra keys. tracker: "v2"
    (TrackerV2) or "3d" (analysis.people.tracker3d) follows the synced stream on the kept views; the sync and
    the view-health vote always use TrackerV2, so both trackers get the same offsets and excluded views.
    Either way the bundle carries a persons/1 doc ("persons")."""
    if tracker not in ("v2", "3d"):
        raise ValueError(f"unknown tracker {tracker!r} (v2 | 3d)")
    views = list(cfg["views"])
    cams_all = pv.load_cams({v: cfg["cam_paths"][v] for v in views})
    kp2d, kp_idx, track_ids, sizes, fps = {}, {}, {}, {}, None
    for v in views:
        kp2d[v], fps_v, kp_idx[v], header = scores.load_kp2d24(cfg["kp2d_paths"][v], params)
        track_ids[v] = header.get("_track_ids")
        sizes[v] = [header.get("width"), header.get("height")]
        fps = fps or fps_v
    if fps is None:
        fps = 60.0
        log("    WARNING: no fps in any kp2d header - using 60")
    appear = appearance.load_clip(cfg["clip"], kp2d_paths=cfg["kp2d_paths"], log=log)
    log(f"    appearance features: {', '.join(v for v in views if v in appear) or 'none (run analysis.appearance)'}")
    with open(cfg["seeds_path"], "r", encoding="utf-8") as fh:
        seeds_views = json.load(fh).get("views", {})
    mode = cfg["mode"]
    if timeline:
        missing = [v for v in views if v not in timeline["offsets"]]
        if missing:
            raise ValueError(f"views {missing} have no offset in the {timeline['clip']} timeline")
        ref = timeline["time_ref_view"]
    else:
        ref = cfg["time_ref"]
    excluded, health, offsets_all, checked, sync_err = {}, {}, {}, len(views) >= 3, {}
    suspect, tr0 = False, None
    while True:
        cams = {v: cams_all[v] for v in views}
        if timeline:
            offsets = {v: float(timeline["offsets"][v]) for v in views}
        else:
            ref = ref if ref in views else views[0]
            tr0 = pv.TrackerV2(cams, {v: kp2d[v] for v in views}, seeds_views, mode, fps, kp_idx=kp_idx, appear=appear)
            raw = tr0.run()
            dense0 = {pid: {v: pv.dense(raw[pid][v], tr0.frame_count) for v in views} for pid in raw}
            offsets, sync_err = pv.estimate_offsets(cams, dense0, views, ref, log=log)
        if ref in offsets_all:
            # re-express the carried offsets (dropped views too) against `ref`: non-zero only when the old
            # reference was dropped, so every offsets_all entry stays on the time_ref_view timeline
            base = offsets_all[ref]
            offsets_all = {u: o - base for u, o in offsets_all.items()}
        offsets_all.update(offsets)
        shifts = {v: int(np.round(o)) for v, o in offsets.items()}
        # view health always votes on TrackerV2's synced tracks: tracker3d gates a badly calibrated view's
        # detections out of its 3D candidates, so that view would have no observations and could never be dropped
        tr = pv.TrackerV2(cams, {v: kp2d[v] for v in views}, seeds_views, mode, fps, shifts=shifts,
                          kp_idx=kp_idx, appear=appear)
        matched = tr.run()
        n = int(timeline["frame_count"]) if timeline else int(cfg.get("frame_count") or tr.frame_count)
        obs = _resample(matched, views, offsets, shifts, n)
        if len(views) < 3:
            if not checked:
                log(f"    WARNING: only {len(views)} views - view health can't vote, a bad calibration would go unnoticed")
            break
        health = pv.view_health(cams, views, obs)
        log("    view held-out residual: " + ", ".join(f"{v} {e:.1f}px" for v, e in health.items()))
        bad = [v for v, e in health.items() if e > pv.VIEW_BAD_PX]
        if bad and len(views) - len(bad) < 2:
            # nothing can be dropped; typical cause: the sync reference's calibration is bad, so every view's
            # offset (and every held-out residual) is off (ledger T5; a fallback reference is later work)
            suspect = True
            log(f"    WARNING: every view but {len(views) - len(bad)} fails view health (> {pv.VIEW_BAD_PX} px) - "
                f"check the calibration of the sync reference {ref}; offsets and poses are suspect")
        if not bad or len(views) - len(bad) < 2:
            break
        for v in bad:
            excluded[v] = round(float(health[v]), 1)
            log(f"    dropping {v}: calibration disagrees with the other views ({health[v]:.0f}px)")
        views = [v for v in views if v not in bad]
    tr_synced = tr                            # the last TrackerV2: synced (final shifts) on the kept views
    if tracker == "3d":                       # same sync, same views: tracker3d follows the synced stream
        cams = {v: cams_all[v] for v in views}
        tr = _make_tracker(tracker, cams, {v: kp2d[v] for v in views}, seeds_views, mode, fps, shifts,
                           kp_idx, appear, {v: track_ids[v] for v in views}, reid, reid_feats, log)
        matched = tr.run()
        obs = _resample(matched, views, offsets, shifts, n)
    log(f"    tracker re-acquisitions: {tr.reacquired}")
    det_index = detection_indices(tr, matched, n)
    if tracker == "3d":
        limit = float(people.load_params().get("sync_disagree_m", SYNC_DISAGREE_M))
        # vs the synced TrackerV2 run (same seeds as the sync run tr0, but views at the same instant; tr0's floor
        # track blends views up to +/-30 frames apart, a noisy reference for fast movers - review 1 L10)
        disagree = sync_check(getattr(tr_synced, "track", None), tr.track, limit, log=log)[0]
        pdoc = personsmod.from_tracker3d(tr, cfg["out_clip"], cfg["all_views"], offsets_all, n, time_ref=ref)
    else:
        disagree = {}
        pdoc = personsmod.from_v2(tr, det_index, cfg["out_clip"], cfg["all_views"], offsets_all, n, time_ref=ref)
    pdoc["stats"]["view_health_suspect"] = bool(suspect)
    pdoc["stats"]["sync_disagreement_m"] = [round(disagree[p], 3) if disagree.get(p) is not None else -1.0
                                            for p in pdoc["stats"]["player_ids"]]
    return {"views": views, "cams": {v: cams_all[v] for v in views}, "cams_all": cams_all, "obs": obs,
            "fps": float(fps), "frame_count": n, "offsets": offsets, "offsets_all": offsets_all,
            "sync_err": {v: (float(e) if np.isfinite(e) else None) for v, e in sync_err.items()},
            "excluded": excluded, "sides": dict(tr.sides), "mode": mode, "time_ref_view": ref,
            "timeline_of": timeline["clip"] if timeline else None, "view_health_checked": checked,
            "view_health": {v: (round(float(e), 2) if np.isfinite(e) else None) for v, e in health.items()},
            "view_health_suspect": suspect,
            "shifts": shifts, "det_index": det_index, "image_size": sizes,
            "reacquired": int(tr.reacquired), "tracker": tracker, "reid": reid, "persons": pdoc,
            "sync_disagreement": disagree}
