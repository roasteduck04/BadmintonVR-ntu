"""Alternative racket tracks written as sidecars next to an existing analysis.json.

The base analysis is never modified; it supplies player identity, cameras and court
position. Hands come from the raw pose2d detections (the analysis joints are causally
smoothed and lag ~3 frames in fast motion), matched to analysis players by reprojection.

    PYTHONPATH=tools python -m analysis.racket_variants build --clip singles_1x
    PYTHONPATH=tools python -m analysis.racket_variants build --all
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis import racket_rigidfit as rf

SIDECAR_FORMAT = "racket_variant/1"
DEFAULT_ANALYSIS_DIR = os.path.join("Assets", "StreamingAssets", "analysis")
DEFAULT_RACKET2D_DIR = os.path.join("data", "racket2d")
DEFAULT_POSE2D_DIR = os.path.join("data", "pose2d")
TRIANGULATED_CLIPS = ["singles_1x", "singles_05x", "doubles_1side", "doubles_2sides",
                      "doubles_2sides_4angle"]

HANDLE_GATE_M = 0.30    # handle keypoint within this of a hand (metric, at hand depth)
BOTTOM_GATE_M = 0.55    # or the throat within this
STICKY_M = 0.12         # same-player continuity radius
STICKY_BONUS_M = 0.05
PERSON_GATE_PX = 200.0  # analysis-to-raw person match (median body joint distance)
MAX_BODY_REPROJ_PX = 40.0  # a view whose best body reprojection is worse has an unusable calibration
MAX_SHIFT = 10
BODY_LAG_ALPHA = 0.25   # build_real._smooth_poses EMA

_COCO_BODY = {5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
              9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip", 13: "left_knee",
              14: "right_knee", 15: "left_ankle", 16: "right_ankle"}
_WRISTS = (9, 10)
_MID_MCP_WB = (100, 121)  # coco-wholebody left/right middle-finger MCP


def sidecar_path(analysis_path: str, variant: str) -> str:
    return analysis_path.replace(".analysis.json", f".racket_{variant}.json")


def cameras_from_doc(doc) -> dict:
    return {c["name"]: rf.Camera(c["K"], c["R"], c["t"]) for c in doc["meta"]["cameras"]}


def hand_joint_indices(joint_names) -> tuple:
    for left, right in (("left_hand", "right_hand"), ("left_middle1", "right_middle1"),
                        ("left_wrist", "right_wrist")):
        if left in joint_names and right in joint_names:
            return joint_names.index(left), joint_names.index(right)
    raise ValueError("analysis skeleton has no hand or wrist joints")


def hand_joint_candidates(joint_names) -> tuple:
    """Per side ([left], [right]) the joint indices to try in order: hand / middle MCP, then the wrist."""
    out = []
    for side in ("left", "right"):
        idx = [joint_names.index(f"{side}_{nm}") for nm in ("hand", "middle1", "wrist")
               if f"{side}_{nm}" in joint_names]
        if not idx:
            raise ValueError("analysis skeleton has no hand or wrist joints")
        out.append(idx)
    return tuple(out)


def player_world_hands(doc) -> dict:
    """{pid: (analysis hands (N,2,3) world [left,right] NaN when missing, positions (N,2))}. The hand is chosen per
    frame by confidence: the hand / middle-MCP joint when confident, else the wrist (review 2 M1: a 58-joint
    build_real doc names middle1 but never triangulates it)."""
    cands = hand_joint_candidates(doc["skeleton"]["joint_names"])
    out = {}
    for p in doc["players"]:
        n = len(p["frames"])
        hands = np.full((n, 2, 3), np.nan)
        pos = np.full((n, 2), np.nan)
        for f, pf in enumerate(p["frames"]):
            jf, ps = pf.get("joints_flat"), pf.get("position")
            if jf is None or ps is None:
                continue
            pos[f] = ps
            for k, idx in enumerate(cands):
                for j in idx:
                    x, y, z, c = jf[j * 4:j * 4 + 4]
                    if c is not None and c > 0.05:
                        hands[f, k] = [x + ps[0], y, z + ps[1]]
                        break
        out[p["id"]] = (hands, pos)
    return out


def world_body_joints(doc):
    """(coco indices, {pid: (N,J,3) world joints, NaN when missing})."""
    names = doc["skeleton"]["joint_names"]
    pairs = [(c, names.index(nm)) for c, nm in _COCO_BODY.items() if nm in names]
    out = {}
    for p in doc["players"]:
        arr = np.full((len(p["frames"]), len(pairs), 3), np.nan)
        for f, pf in enumerate(p["frames"]):
            jf, ps = pf.get("joints_flat"), pf.get("position")
            if jf is None or ps is None:
                continue
            for k, (_c, j) in enumerate(pairs):
                x, y, z, c = jf[j * 4:j * 4 + 4]
                if c > 0.1:
                    arr[f, k] = [x + ps[0], y, z + ps[1]]
        out[p["id"]] = arr
    return [c for c, _ in pairs], out


def load_racket2d(path) -> list:
    """Per-frame candidate arrays (k,5,3); empty array when no detection."""
    with open(path, "r", encoding="utf-8") as fh:
        frames = json.load(fh)["frames"]
    out = []
    for raw in frames:
        if not raw:
            out.append(np.zeros((0, 5, 3)))
            continue
        cands = raw if isinstance(raw[0][0], list) else [raw]
        out.append(np.asarray(cands, dtype=np.float64).reshape(-1, 5, 3))
    return out


def load_pose2d_compact(path, body_idx) -> tuple:
    """Per frame: (body (k,J,3), hands (k,2,3) [left,right]) keeping only what we use. Scores are the raw RTMPose
    SimCC values mapped to calibrated probabilities (`pose_v2.norm_score`), so every 0.3 gate downstream gates
    (review 2 L2)."""
    from analysis.pose_v2 import norm_score
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    wholebody = str(doc.get("keypoint_format", "")).startswith("coco_wholebody")
    body, hands = [], []
    for fr in doc["frames"]:
        if not fr:
            body.append(np.zeros((0, len(body_idx), 3)))
            hands.append(np.zeros((0, 2, 3)))
            continue
        kps = []
        for p in fr:
            k = np.asarray(p["keypoints"], dtype=np.float64)
            k[:, 2] = norm_score(k[:, 2])
            kps.append(k)
        body.append(np.array([[k[i] for i in body_idx] for k in kps], dtype=np.float64))
        hs = []
        for k in kps:
            pair = []
            for s in (0, 1):
                w = k[_WRISTS[s]]
                m = k[_MID_MCP_WB[s]] if wholebody and len(k) > _MID_MCP_WB[s] else None
                pair.append(m if m is not None and m[2] >= 0.3 else w)
            hs.append(pair)
        hands.append(np.array(hs, dtype=np.float64))
    return body, hands


def _project_joints(cam, arr):
    flat = arr.reshape(-1, 3)
    ok = np.isfinite(flat).all(axis=1)
    uv = np.full((len(flat), 2), np.nan)
    depth = np.full(len(flat), np.nan)
    if ok.any():
        u, z = cam.project(flat[ok])
        good = z > 0.1
        uv[ok] = np.where(good[:, None], u, np.nan)
        depth[ok] = np.where(good, z, np.nan)
    return uv.reshape(arr.shape[:-1] + (2,)), depth.reshape(arr.shape[:-1])


def _person_distance(persons, uv):
    """Median joint distance of each raw person to projected joints uv (J,2)."""
    ok = np.isfinite(uv).all(axis=1)
    if ok.sum() < 4 or len(persons) == 0:
        return np.full(len(persons), np.inf)
    d = np.linalg.norm(persons[:, ok, :2] - uv[ok][None], axis=2)
    d[persons[:, ok, 2] < 0.3] = np.nan
    enough = np.isfinite(d).sum(axis=1) >= 3
    out = np.full(len(persons), np.inf)
    if enough.any():
        out[enough] = np.nanmedian(d[enough], axis=1)
    return out


def estimate_offsets_pose2d(body_by_view: dict, body_proj: dict, n: int,
                            max_shift=MAX_SHIFT, step=2, log=print, health=None) -> dict:
    """Per view shift (raw index = analysis frame + shift) minimising body reprojection.

    `health`, if given, receives each view's best median body reprojection (px; inf when
    the analysis players never land on a detected person, i.e. a broken calibration).
    """
    offsets = {}
    for v, persons in body_by_view.items():
        curve = []
        for shift in range(-max_shift, max_shift + 1):
            ds = []
            for f in range(0, n, step):
                g = f + shift
                if g < 0 or g >= len(persons) or len(persons[g]) == 0:
                    continue
                for uv in body_proj[v].values():
                    d = _person_distance(persons[g], uv[f])
                    if len(d) and d.min() < 150:
                        ds.append(float(d.min()))
            curve.append(np.median(ds) if len(ds) > 30 else np.inf)
        k = int(np.argmin(curve))
        if health is not None:
            health[v] = float(curve[k])
        if not np.isfinite(curve[k]):
            log(f"    {v}: analysis players never match a detected person in this view")
            continue
        offsets[v] = k - max_shift
        edge = " (at search edge!)" if abs(offsets[v]) == max_shift else ""
        log(f"    {v}: shift {offsets[v]:+d} (body reproj {curve[k]:.1f}px, at 0: {curve[max_shift]:.1f}px){edge}")
    return offsets


def estimate_offsets_handles(r2d: dict, hand_px: dict, n: int, max_shift=MAX_SHIFT, step=2) -> dict:
    """Fallback when pose2d is unavailable: handle-to-hand distance (biased by body lag)."""
    offsets = {}
    for v, frames in r2d.items():
        best = (np.inf, 0)
        for shift in range(-max_shift, max_shift + 1):
            ds = []
            for f in range(0, n, step):
                g = f + shift
                if g < 0 or g >= len(frames) or len(frames[g]) == 0:
                    continue
                cands = frames[g][frames[g][:, rf.HANDLE, 2] >= 0.3]
                hands = np.concatenate([uv[f] for uv in hand_px[v].values()])
                hands = hands[np.isfinite(hands).all(axis=1)]
                if len(cands) == 0 or len(hands) == 0:
                    continue
                d = np.linalg.norm(cands[:, None, rf.HANDLE, :2] - hands[None], axis=2).min(axis=1)
                ds.extend(d[d < 400].tolist())
            if len(ds) > 30 and np.median(ds) < best[0]:
                best = (float(np.median(ds)), shift)
        offsets[v] = best[1]
    return offsets


def estimate_view_sync(doc, cams: dict, pose2d_dir: str | None, log=print):
    """Per-view pose2d sync for the given cameras (shared by racket and shuttle variants).

    Returns (offsets {view: shift}, health {view: best body reprojection px},
    body_proj {view: {pid: (N,J,2)}}, body_by_view, hands_by_view). Without pose2d files the
    offsets/health/raw dicts are empty.
    """
    n = int(doc["meta"]["frame_count"])
    pids = [p["id"] for p in doc["players"]]
    body_idx, body3d = world_body_joints(doc)
    body_proj = {v: {pid: _project_joints(cams[v], body3d[pid][:n])[0] for pid in pids} for v in cams}
    body_by_view, hands_by_view = {}, {}
    if pose2d_dir:
        for v in cams:
            p = os.path.join(pose2d_dir, f"{v}_kp2d.json")
            if os.path.exists(p):
                body_by_view[v], hands_by_view[v] = load_pose2d_compact(p, body_idx)
    health = {}
    offsets = estimate_offsets_pose2d(body_by_view, body_proj, n, log=log, health=health)
    return offsets, health, body_proj, body_by_view, hands_by_view


def _match_persons(persons_g, proj_v, pids, f):
    """[(pid, raw person index)]: Hungarian on the median body-joint distance (<= PERSON_GATE_PX) between the raw
    persons of one view frame and each player's projected analysis body at analysis frame f."""
    if len(persons_g) == 0:
        return []
    cost = np.stack([_person_distance(persons_g, proj_v[pid][f]) for pid in pids])
    ok = cost <= PERSON_GATE_PX
    if not ok.any():
        return []
    rows, cols = linear_sum_assignment(np.where(ok, cost, 1e6))
    return [(pids[r], c) for r, c in zip(rows, cols) if ok[r, c]]


def match_raw_hands(body_by_view, hands_by_view, body_proj, offsets, pids, n) -> dict:
    """{view: {pid: (N,2,3) raw 2D [left,right] hands}} via per-frame Hungarian person matching."""
    out = {}
    for v, persons in body_by_view.items():
        out[v] = {pid: np.full((n, 2, 3), np.nan) for pid in pids}
        for f in range(n):
            g = f + offsets.get(v, 0)
            if g < 0 or g >= len(persons):
                continue
            for pid, c in _match_persons(persons[g], body_proj[v], pids, f):
                out[v][pid][f] = hands_by_view[v][g][c]
    return out


def refine_offsets_to_ref(doc, cams, body_by_view, body_proj, coarse, ref, log=print) -> dict:
    """Fractional per-view sync {view: offset} (raw frame = reference frame + offset; ref 0.0) from the raw pose2d
    alone, with the estimator the pose docs use (`pose_v2.estimate_offsets`, cross-view two-view reprojection,
    referenced to the doc's sync view `ref`). The analysis body only gives each raw person its player identity
    (at the coarse, lag-biased shift), so the base body's ~1.7-frame causal lag no longer leaks into the sync
    (review 2 L14). {} when `ref` has no usable pose2d."""
    from analysis import pose_v2 as pv
    views = [v for v in cams if v in body_by_view and v in coarse]
    if ref not in views or len(views) < 2:
        log(f"    sync reference {ref} unusable (no pose2d / excluded): sidecar stays on the body-fit shifts")
        return {}
    n = int(doc["meta"]["frame_count"])
    pids = [p["id"] for p in doc["players"]]
    body_idx = world_body_joints(doc)[0]
    dense = {pid: {} for pid in pids}
    for v in views:
        persons = body_by_view[v]
        D = {pid: np.full((len(persons), 17, 3), np.nan, np.float32) for pid in pids}
        for f in range(n):
            g = f + coarse[v]
            if 0 <= g < len(persons):
                for pid, c in _match_persons(persons[g], body_proj[v], pids, f):
                    D[pid][g, body_idx] = persons[g][c]
        for pid in pids:
            dense[pid][v] = D[pid]
    camt = {v: (cams[v].K, cams[v].R, cams[v].t) for v in views}
    offs, errs = pv.estimate_offsets(camt, dense, views, ref, log=log)
    return {v: float(o) for v, o in offs.items() if np.isfinite(errs.get(v, np.inf))}


def reference_shifts(coarse: dict, frac: dict, ref) -> tuple:
    """Integer racket2d / shuttle2d shifts on the reference video timeline + a `sync` record for the sidecar.
    Views without a fractional offset take their body-fit shift plus the body lag measured on the others (the lag
    is common to every view); without any fractional offset the body-fit shifts stay, and the record says so."""
    if not frac:
        return dict(coarse), {"reference": None, "method": "body reprojection (lag-biased base body)"}
    common = [v for v in frac if v in coarse]
    lag = float(np.median([frac[v] - coarse[v] for v in common])) if common else 0.0
    shifts = {v: int(round(frac[v])) for v in frac}
    for v, s in coarse.items():
        shifts.setdefault(v, int(round(s + lag)))
    return shifts, {"reference": ref, "method": "pose_v2.estimate_offsets on raw pose2d (fractional)",
                    "fractional": {v: round(o, 2) for v, o in frac.items()}, "body_lag_frames": round(lag, 2)}


def triangulate_raw_hands(cams, raw_hands, pids, n) -> dict:
    """{pid: (N,2,3) world hands} from the raw 2D hands (NaN where < 2 views agree)."""
    from analysis.triangulate import triangulate_joint
    Ps = {v: cams[v].P() for v in raw_hands}
    out = {}
    for pid in pids:
        arr = np.full((n, 2, 3), np.nan)
        for f in range(n):
            for s in (0, 1):
                per_view = {v: raw_hands[v][pid][f, s] for v in raw_hands
                            if np.isfinite(raw_hands[v][pid][f, s]).all()}
                if len(per_view) < 2:
                    continue
                X, _c, _e = triangulate_joint(Ps, per_view, 0.3)
                if X is not None:
                    arr[f, s] = X * rf._Y_FLIP
        out[pid] = arr
    return out


def assign_candidates(r2d: dict, hand2d: dict, ppm: dict, offsets: dict, pids: list, n: int):
    """Per player, per frame: {view: [preferred obs, alternatives...]}, plus left/right votes."""
    frames = {pid: [dict() for _ in range(n)] for pid in pids}
    votes = {pid: [0, 0] for pid in pids}
    prev = {(pid, v): None for pid in pids for v in r2d}
    for f in range(n):
        for v, rk in r2d.items():
            g = f + offsets.get(v, 0)
            if g < 0 or g >= len(rk) or len(rk[g]) == 0:
                for pid in pids:
                    prev[(pid, v)] = None
                continue
            cands = rk[g]
            cost = np.full((len(cands), len(pids)), np.inf)
            which = np.zeros_like(cost, dtype=int)
            for pi, pid in enumerate(pids):
                scale = ppm[v][pid][f]
                if not np.isfinite(scale):
                    continue
                for s in (0, 1):
                    h = hand2d[v][pid][f, s]
                    if not np.isfinite(h[:2]).all():
                        continue
                    dh = np.linalg.norm(cands[:, rf.HANDLE, :2] - h[:2], axis=1) / scale
                    db = np.linalg.norm(cands[:, rf.BOTTOM, :2] - h[:2], axis=1) / scale
                    ok = (dh <= HANDLE_GATE_M) | (db <= BOTTOM_GATE_M)
                    c = np.where(ok, np.minimum(dh, db * HANDLE_GATE_M / BOTTOM_GATE_M), np.inf)
                    better = c < cost[:, pi]
                    cost[better, pi] = c[better]
                    which[better, pi] = s
                last = prev[(pid, v)]
                if last is not None:
                    near = np.linalg.norm(cands[:, rf.HANDLE, :2] - last, axis=1) / scale < STICKY_M
                    cost[near & np.isfinite(cost[:, pi]), pi] -= STICKY_BONUS_M
            finite = np.isfinite(cost)
            taken = {}
            if finite.any():
                rows, cols = linear_sum_assignment(np.where(finite, cost, 1e6))
                taken = {c: r for r, c in zip(rows, cols) if finite[r, c]}
            claimed = set(taken.values())
            for pi, pid in enumerate(pids):
                if pi not in taken:
                    prev[(pid, v)] = None
                    continue
                r = taken[pi]
                alts = [i for i in np.argsort(cost[:, pi]) if i != r and i not in claimed and finite[i, pi]]
                frames[pid][f][v] = [cands[r]] + [cands[i] for i in alts]
                votes[pid][which[r, pi]] += 1
                prev[(pid, v)] = cands[r, rf.HANDLE, :2].copy()
    return frames, votes


def _template_samples(frames_by_pid, limit=600):
    samples = [{v: cs[0] for v, cs in fr.items()}
               for frames in frames_by_pid.values() for fr in frames if len(fr) >= 3]
    if len(samples) > limit:
        samples = [samples[i] for i in np.linspace(0, len(samples) - 1, limit).astype(int)]
    return samples


def relag_track(track: dict, alpha=BODY_LAG_ALPHA) -> dict:
    """Apply the body's causal EMA to the racket so it stays on the (lagged) rendered hand."""
    from scipy.spatial.transform import Rotation
    g, R, valid = track["grip"].copy(), track["R"].copy(), track["valid"]
    q = Rotation.from_matrix(R).as_quat()
    prev_g = prev_q = None
    for i in range(len(g)):
        if not valid[i]:
            prev_g = prev_q = None
            continue
        if prev_g is not None:
            if q[i] @ prev_q < 0:
                q[i] = -q[i]
            g[i] = alpha * g[i] + (1 - alpha) * prev_g
            q[i] = alpha * q[i] + (1 - alpha) * prev_q
            q[i] /= np.linalg.norm(q[i])
        prev_g, prev_q = g[i].copy(), q[i].copy()
    out = dict(track)
    out["grip"], out["R"] = g, Rotation.from_quat(q).as_matrix()
    return out


def _r5(v):
    return [round(float(x), 5) for x in v]


def track_to_frames(track: dict, tpl: rf.RacketTemplate, positions: np.ndarray, hand: str) -> list:
    """Finalized world track -> per-frame body-local racket dicts (analysis racket schema)."""
    out = []
    hc = tpl.head_center()
    for f in range(len(track["grip"])):
        if not track["valid"][f] or not np.isfinite(positions[f]).all():
            out.append(None)
            continue
        R = track["R"][f]
        grip = track["grip"][f] - np.array([positions[f][0], 0.0, positions[f][1]])
        head = grip + R @ hc
        axis, across = R[:, 2], R[:, 0]
        normal = np.cross(axis, across)
        err = track["err"][f]
        out.append({
            "grip": _r5(grip), "head": _r5(head), "side": _r5(head + across * tpl.head_width / 2.0),
            "axis": _r5(axis), "face_normal": _r5(normal / np.linalg.norm(normal)),
            "which_hand": hand, "confidence": round(float(track["confidence"][f]), 4),
            "status": str(track["status"][f]), "views": int(track["views"][f]),
            "err": None if not np.isfinite(err) else round(float(err), 4),
        })
    return out


def prepare_inputs(doc, racket2d_dir: str, pose2d_dir: str | None, log=print) -> dict:
    """Everything the fitter needs, shared by `build_variant` and the evaluation."""
    n = int(doc["meta"]["frame_count"])
    cams = cameras_from_doc(doc)
    r2d = {v: load_racket2d(os.path.join(racket2d_dir, f"{v}_racket2d.json"))
           for v in cams if os.path.exists(os.path.join(racket2d_dir, f"{v}_racket2d.json"))}
    if not r2d:
        raise FileNotFoundError(f"no racket2d files for views {list(cams)} in {racket2d_dir}")
    cams = {v: cams[v] for v in r2d}
    pids = [p["id"] for p in doc["players"]]
    hands3d, positions = {}, {}
    for pid, (h, pos) in player_world_hands(doc).items():
        hands3d[pid], positions[pid] = h, pos

    ppm = {}
    for v in cams:
        ppm[v] = {}
        for pid in pids:
            _uv, depth = _project_joints(cams[v], hands3d[pid][:n])
            cnt = np.isfinite(depth).sum(axis=1)
            mean_depth = np.where(cnt > 0, np.nansum(depth, axis=1) / np.maximum(cnt, 1), np.nan)
            ppm[v][pid] = cams[v].K[0, 0] / mean_depth

    log("  sync offsets (raw index = analysis frame + shift):")
    offsets, health, body_proj, body_by_view, hands_by_view = estimate_view_sync(doc, cams, pose2d_dir, log)
    bad = [v for v, h in health.items() if not h <= MAX_BODY_REPROJ_PX]
    if bad:
        log(f"    excluding {bad}: body reprojection > {MAX_BODY_REPROJ_PX:.0f}px (calibration unusable)")
        for v in bad:
            cams.pop(v)
            r2d.pop(v)
            offsets.pop(v, None)
            body_by_view.pop(v, None)
            hands_by_view.pop(v, None)
    from analysis.pose_v2 import doc_time_ref
    ref = doc_time_ref(doc)
    frac = refine_offsets_to_ref(doc, cams, body_by_view, body_proj, offsets, ref, log) if offsets else {}
    offsets, sync = reference_shifts(offsets, frac, ref)
    if frac:
        log(f"    shifts on the {ref} video timeline: {offsets} (body lag absorbed by the body fit: "
            f"{sync['body_lag_frames']:+.2f} frames)")
    hand_px = {v: {pid: _project_joints(cams[v], hands3d[pid][:n])[0] for pid in pids} for v in cams}
    missing = [v for v in cams if v not in offsets]
    if missing:
        fb = estimate_offsets_handles({v: r2d[v] for v in missing}, hand_px, n)
        lag = sync.get("body_lag_frames", 0.0)        # the handle fit is biased by the same body lag
        fb = {v: int(round(s + lag)) for v, s in fb.items()}
        log(f"    handle-based fallback for {missing}: {fb}")
        offsets.update(fb)

    raw = match_raw_hands(body_by_view, hands_by_view, body_proj, offsets, pids, n)
    hand2d = {}
    for v in cams:
        hand2d[v] = {}
        for pid in pids:
            proj = np.concatenate([hand_px[v][pid], np.full((n, 2, 1), 0.5)], axis=2)
            r = raw.get(v, {}).get(pid)
            hand2d[v][pid] = proj if r is None else np.where(np.isfinite(r).all(axis=2, keepdims=True), r, proj)
    raw3d = triangulate_raw_hands(cams, raw, pids, n) if raw else {}
    world_hands = {}
    for pid in pids:
        r = raw3d.get(pid)
        world_hands[pid] = hands3d[pid][:n] if r is None else \
            np.where(np.isfinite(r).all(axis=2, keepdims=True), r, hands3d[pid][:n])
        if r is not None:
            n_raw = int(np.isfinite(r).all(axis=2).any(axis=1).sum())
            log(f"    {pid}: raw triangulated hands on {n_raw}/{n} frames")

    frames_by_pid, votes = assign_candidates(r2d, hand2d, ppm, offsets, pids, n)
    return {"n": n, "fps": float(doc["meta"]["fps"]), "cams": cams, "pids": pids,
            "offsets": offsets, "sync": sync, "view_health_px": health, "frames": frames_by_pid, "votes": votes,
            "world_hands": world_hands, "positions": positions}


def build_variant(analysis_path: str, racket2d_dir: str, variant: str = "rigid",
                  cfg: rf.FitConfig | None = None, pose2d_dir: str | None = DEFAULT_POSE2D_DIR,
                  display_lag: str = "video", log=print) -> dict:
    cfg = cfg or rf.FitConfig()
    with open(analysis_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    inp = prepare_inputs(doc, racket2d_dir, pose2d_dir, log)
    n, fps = inp["n"], inp["fps"]
    tpl, n_tpl = rf.estimate_template(inp["cams"], _template_samples(inp["frames"]))
    log(f"  template from {n_tpl} clean frames: {tpl.as_dict()}")

    fitter = rf.RacketFitter(inp["cams"], tpl, cfg)
    players_out, stats = [], {}
    for pid in inp["pids"]:
        vl, vr = inp["votes"][pid]
        hand_name = "left" if vl > vr else "right"
        hi = 0 if hand_name == "left" else 1
        wh = inp["world_hands"][pid]
        hand_seq = [wh[f, hi] if np.isfinite(wh[f, hi]).all() else None for f in range(n)]
        t0 = time.time()
        track = rf.finalize_track(rf.fit_track(fitter, inp["frames"][pid], hand_seq, cfg), fps, cfg)
        if display_lag == "body":
            track = relag_track(track)
        pframes = track_to_frames(track, tpl, inp["positions"][pid], hand_name)
        st = {s: int((track["status"] == s).sum()) for s in ("detected", "partial", "interpolated")}
        st["coverage"] = round(sum(1 for x in pframes if x) / max(n, 1), 4)
        used = track["views"][track["status"] == "detected"]
        st["mean_views"] = round(float(used.mean()), 2) if len(used) else 0.0
        errs = track["err"][np.isfinite(track["err"])]
        st["median_err"] = round(float(np.median(errs)), 4) if len(errs) else None
        st["hand"] = hand_name
        stats[pid] = st
        log(f"  {pid}: {st}  ({time.time() - t0:.0f}s)")
        players_out.append({"id": pid, "which_hand": hand_name, "frames": pframes})

    return {
        "format": SIDECAR_FORMAT, "variant": variant, "clip_id": doc.get("clip_id"),
        "source_analysis": os.path.basename(analysis_path),
        "racket2d_dir": racket2d_dir.replace("\\", "/"), "producer": "racket_rigidfit",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "time_base": display_lag, "fps": fps, "frame_count": n,
        "template": tpl.as_dict(), "sync_offsets": inp["offsets"], "sync": inp["sync"],
        "view_health_px": {v: (round(h, 1) if np.isfinite(h) else None) for v, h in inp["view_health_px"].items()},
        "stats": stats,
        "players": players_out,
    }


def validate_sidecar(sc: dict, analysis_doc: dict | None = None) -> list:
    errs = []
    if sc.get("format") != SIDECAR_FORMAT:
        errs.append(f"format must be {SIDECAR_FORMAT}")
    n = sc.get("frame_count")
    for p in sc.get("players", []):
        fr = p.get("frames", [])
        if len(fr) != n:
            errs.append(f"{p.get('id')}: {len(fr)} frames != frame_count {n}")
        for i, r in enumerate(fr):
            if r is None:
                continue
            for k in ("grip", "head", "side", "axis", "face_normal"):
                if not (isinstance(r.get(k), list) and len(r[k]) == 3):
                    errs.append(f"{p.get('id')} frame {i}: {k} must be [x,y,z]")
                    break
            if r.get("which_hand") not in ("left", "right"):
                errs.append(f"{p.get('id')} frame {i}: bad which_hand")
    if analysis_doc is not None:
        if [p["id"] for p in sc.get("players", [])] != [p["id"] for p in analysis_doc["players"]]:
            errs.append("player ids differ from analysis")
        if analysis_doc["meta"]["frame_count"] != n:
            errs.append("frame_count differs from analysis")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build racket variant sidecars.")
    sub = ap.add_subparsers(dest="command")
    b = sub.add_parser("build", help="fit a rigid racket track and write a sidecar")
    b.add_argument("--clip", action="append", default=None)
    b.add_argument("--all", action="store_true", help=f"all of {TRIANGULATED_CLIPS}")
    b.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    b.add_argument("--racket2d-dir", default=DEFAULT_RACKET2D_DIR)
    b.add_argument("--pose2d-dir", default=DEFAULT_POSE2D_DIR,
                   help="raw pose2d files for sync + unlagged hands ('' to skip)")
    b.add_argument("--variant", default="rigid")
    b.add_argument("--display-lag", choices=["video", "body"], default="video",
                   help="video = racket in true video time; body = re-lagged to match the smoothed body")
    args = ap.parse_args(argv)
    if args.command != "build":
        ap.print_help()
        return 1
    clips = TRIANGULATED_CLIPS if args.all else (args.clip or [])
    if not clips:
        ap.error("give --clip or --all")
    for clip in clips:
        src = os.path.join(args.analysis_dir, f"{clip}.analysis.json")
        print(f"[{args.variant}] {clip}")
        sc = build_variant(src, args.racket2d_dir, args.variant,
                           pose2d_dir=args.pose2d_dir or None, display_lag=args.display_lag)
        with open(src, "r", encoding="utf-8") as fh:
            errs = validate_sidecar(sc, json.load(fh))
        if errs:
            raise SystemExit("sidecar invalid:\n  " + "\n  ".join(errs[:20]))
        out = sidecar_path(src, args.variant)
        text = json.dumps(sc, separators=(",", ":"), allow_nan=False)   # Unity can't parse NaN
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
