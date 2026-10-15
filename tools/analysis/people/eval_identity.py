"""Identity evaluation (SP3 spec § 4.2).

Labelled (data/gt/labels/<clip>_ids.json, idlabels/1 from the labelling page): per labelled detection the truth
("P1".."P4", "N", "S" = skipped) vs a prediction map {(view, raw frame, det): "P1".. | "N" | "U"} (missing = "N").
IDF1 with the best one-to-one GT -> predicted id mapping (Hungarian), HOTA-lite = sqrt(DetA x AssA) (DetA =
player-vs-not detection accuracy, AssA = correctly identified share of detected players), non-player FP rate, player
FN rate (a player predicted N or U), identity switches along each GT player's labelled sequence per view.
Label-free (any persons doc + calibrations + raw kp2d): coverage, net crossings (0.5 m hysteresis), per (frame, view)
reprojection of each player's triangulated body into its assigned detections (inconsistent = > err_px: a wrong-person
assignment), the same leave-one-view-out (held-out-view agreement), detections shared by two players, appearance jumps (1 s window means, Hellinger > 0.35), hotspots."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis import appearance
from analysis import pose_v2 as pv
from analysis.people import persons as pm

FORMAT = "idlabels/1"
BODY = list(range(5, 17))
JUMP_HELLINGER = 0.35


def load_labels(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: label file format {doc.get('format')!r}, expected {FORMAT!r}")
    return doc.get("labels", [])


IGNORED = ("S", "U")        # ground-truth rows that are no decision: skipped, or an unknown proposal accepted as is


def _pooled(labels, pred) -> dict:
    rows = [(l, pred.get((l["view"], int(l["raw_frame"]), int(l["det"])), "N")) for l in labels
            if l["label"] not in IGNORED]
    gp = [(l, p) for l, p in rows if l["label"] != "N"]
    gn = [(l, p) for l, p in rows if l["label"] == "N"]
    is_p = lambda x: x not in ("N", "U")
    tp = [(l, p) for l, p in gp if is_p(p)]
    fn, fp = len(gp) - len(tp), sum(is_p(p) for _, p in gn)
    gids = sorted({l["label"] for l, _ in gp})
    pids = sorted({p for _, p in rows if is_p(p)})
    idtp = 0
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        for l, p in tp:
            M[gids.index(l["label"]), pids.index(p)] += 1
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    n_pred = sum(is_p(p) for _, p in rows)
    idfn, idfp = len(gp) - idtp, n_pred - idtp
    deta = len(tp) / (len(tp) + fn + fp) if (tp or fn or fp) else 0.0
    assa = idtp / len(tp) if tp else 0.0
    switches = 0
    for key in sorted({(l["view"], l["label"]) for l, _ in gp}):
        seq = [p for l, p in sorted(gp, key=lambda x: x[0]["raw_frame"]) if (l["view"], l["label"]) == key and is_p(p)]
        switches += sum(a != b for a, b in zip(seq, seq[1:]))
    return {"n_player": len(gp), "n_nonplayer": len(gn),
            "idf1": 2 * idtp / (2 * idtp + idfp + idfn) if (idtp or idfp or idfn) else 0.0,
            "deta": deta, "assa": assa, "hota_lite": float(np.sqrt(deta * assa)),
            "nonplayer_fp_rate": fp / len(gn) if gn else 0.0, "player_fn_rate": fn / len(gp) if gp else 0.0,
            "switches": switches}


def labelled_metrics(labels, pred) -> dict:
    """Pooled metrics at the top level + the same metric dict per view under "per_view"."""
    labels = [l for l in labels if l["label"] not in IGNORED]
    out = _pooled(labels, pred)
    out["per_view"] = {v: _pooled([l for l in labels if l["view"] == v], pred)
                       for v in sorted({l["view"] for l in labels})}
    return out


def net_crossings(z_seq, margin=0.5) -> int:
    side, n = 0, 0
    for z in z_seq:
        s = 1 if z > margin else (-1 if z < -margin else 0)
        if s == 0:
            continue
        if side and s != side:
            n += 1
        side = s
    return n


def _player_frames(doc):
    V, names = len(doc["views"]), doc["views"]
    by, floors, seen, shared = {}, {}, {}, 0
    for t in doc["tracks"]:
        if t["label"] != "player" or not t["player_id"]:
            continue
        pid = t["player_id"]
        L = t["last_frame"] - t["first_frame"] + 1
        D = np.asarray(t["det_index"]).reshape(V, L)
        fl = np.asarray(t["floor_xz"], float).reshape(L, 2)
        for k in range(L):
            f = t["first_frame"] + k
            if fl[k, 0] != pm.NONE_XZ:
                floors.setdefault(pid, {})[f] = fl[k]
            for ci in range(V):
                if D[ci, k] < 0:
                    continue
                key = (names[ci], f + doc["det_shift"][ci], int(D[ci, k]))
                if key in seen and seen[key] != pid:
                    shared += 1
                seen[key] = pid
                by.setdefault(pid, {}).setdefault(f, {})[names[ci]] = key[1:]
    return by, floors, shared


def _reproj(frames, cams, raw):
    errs = []
    for f, vd in frames.items():
        vs = sorted(v for v, (r, d) in vd.items() if v in cams and 0 <= r < len(raw[v]) and d < len(raw[v][r]))
        if len(vs) < 2:
            continue
        K = np.stack([raw[v][vd[v][0]][vd[v][1]][BODY] for v in vs], axis=1).astype(float)   # (12, Vp, 3)
        X, _, _, _ = pv.triangulate_batch({v: cams[v] for v in vs}, vs, K[..., :2], K[..., 2])
        ok = np.isfinite(X).all(axis=1)
        if ok.sum() < 4:
            continue
        uv, _ = pv.project(pv.proj_mats(cams, vs), X[ok])
        for j, v in enumerate(vs):
            vis = K[ok, j, 2] >= pv.W_MIN
            if vis.sum() >= 3:
                errs.append(float(np.median(np.linalg.norm(uv[vis, j] - K[ok, j, :2][vis], axis=1))))
    return np.array(errs)


def _heldout(frames, cams, raw):
    """Leave-one-view-out: triangulate the body from the other assigned views, score in the left-out one."""
    errs = []
    for f, vd in frames.items():
        vs = sorted(v for v, (r, d) in vd.items() if v in cams and 0 <= r < len(raw[v]) and d < len(raw[v][r]))
        if len(vs) < 3:
            continue
        K = {v: raw[v][vd[v][0]][vd[v][1]][BODY].astype(float) for v in vs}
        for v in vs:
            others = [o for o in vs if o != v]
            U = np.stack([K[o] for o in others], axis=1)
            X, _, _, _ = pv.triangulate_batch({o: cams[o] for o in others}, others, U[..., :2], U[..., 2])
            ok = np.isfinite(X).all(axis=1)
            if ok.sum() < 4:
                continue
            uv, _ = pv.project(pv.proj_mats(cams, [v]), X[ok])
            vis = K[v][ok, 2] >= pv.W_MIN
            if vis.sum() >= 3:
                errs.append(float(np.median(np.linalg.norm(uv[vis, 0] - K[v][ok, :2][vis], axis=1))))
    return np.array(errs)


def _jumps(frames, appear, fps_window=60):
    if appear is None:
        return -1
    n = 0
    for v in sorted({v for vd in frames.values() for v in vd}):
        seq = sorted((f, appear.get(v, {}).get(vd[v])) for f, vd in frames.items() if v in vd)
        seq = [(f, x) for f, x in seq if x is not None]
        wins = {}
        for f, x in seq:
            wins.setdefault(f // fps_window, []).append(x)
        means = [np.mean(wins[k], axis=0) for k in sorted(wins) if len(wins[k]) >= 5]
        n += sum(appearance.hellinger(a, b) > JUMP_HELLINGER for a, b in zip(means, means[1:]))
    return int(n)


def label_free(doc, cams, raw, appear=None, err_px=40.0) -> dict:
    by, floors, shared = _player_frames(doc)
    out = {"players": {}, "shared_detections": int(shared), "hotspots": doc.get("stats", {}).get("hotspots", [])}
    for pid in sorted(by):
        e = _reproj(by[pid], cams, raw)
        h = _heldout(by[pid], cams, raw)
        fl = floors.get(pid, {})
        out["players"][pid] = {
            "coverage": len(by[pid]) / max(1, doc["frame_count"]),
            "net_crossings": net_crossings([fl[f][1] for f in sorted(fl)]) if fl else -1,
            "reproj_px_median": float(np.median(e)) if len(e) else -1.0,
            "reproj_px_p90": float(np.percentile(e, 90)) if len(e) else -1.0,
            "inconsistent_frac": float((e > err_px).mean()) if len(e) else -1.0,
            "heldout_px_median": float(np.median(h)) if len(h) else -1.0,
            "heldout_px_p90": float(np.percentile(h, 90)) if len(h) else -1.0,
            "heldout_inconsistent_frac": float((h > err_px).mean()) if len(h) else -1.0,
            "appearance_jumps": _jumps(by[pid], appear)}
    return out
