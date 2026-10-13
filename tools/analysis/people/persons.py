"""persons/1 sidecar: every tracked person of a clip with its label (SP3 spec § 2.7; SP2 findings F1/F2).

<out_clip>.persons.json next to the analysis docs (pose24 build --tracker 3d) or data/people_eval/<clip>.<tracker>
.persons.json (python -m analysis.people track). JsonUtility-safe: objects only as the fixed-class fields
tracks[] / stats / stats.hotspots[]; per track flat arrays: floor_xz[2*L] (court m, -9999 = none), support[L],
det_index[V*L] (original kp2d detection index in raw frame f + det_shift[v], -1 = none), L = last - first + 1.
views[V] = every view of the base clip, sorted (= cvdebug cameras[])."""
from __future__ import annotations

import json
import os

import numpy as np

from analysis import cvdebug, video_io

FORMAT = "persons/1"
NONE_XZ = -9999.0
TRACK_FIELDS = ("track_id", "label", "player_id", "score", "first_frame", "last_frame", "ambiguous", "pinned",
                "floor_xz", "support", "det_index")
_OBJECTS = {"tracks", "stats", "hotspots"}


def path(analysis_dir, out_clip):
    return os.path.join(analysis_dir, f"{out_clip}.persons.json")


def _header(clip_id, tracker, fps, n, views_all, build_views, shifts, offsets, time_ref):
    return {"format": FORMAT, "clip_id": clip_id, "tracker": tracker, "source_build_id": "", "fps": float(fps),
            "frame_count": int(n), "views": list(views_all), "views_used": sorted(build_views),
            "det_shift": [int(shifts[v]) if v in shifts else int(video_io.view_frame(0, offsets.get(v, 0.0))[0])
                          for v in views_all],
            "offsets": [round(float(offsets.get(v, 0.0)), 3) for v in views_all], "time_ref_view": time_ref or ""}


def _stats(tracks, hotspots, players):
    cov = []
    for pid in players:
        fr = set()
        for t in tracks:
            if t["player_id"] == pid:
                V = len(t["det_index"]) // max(1, t["last_frame"] - t["first_frame"] + 1)
                D = np.asarray(t["det_index"]).reshape(V, -1)
                fr |= {t["first_frame"] + int(k) for k in np.nonzero((D >= 0).any(axis=0))[0]}
        cov.append(len(fr))
    return {"n_tracks": len(tracks), "n_player": sum(t["label"] == "player" for t in tracks),
            "n_nonplayer": sum(t["label"] == "nonplayer" for t in tracks),
            "n_unknown": sum(t["label"] == "unknown" for t in tracks),
            "player_ids": list(players), "player_frames": cov, "hotspots": list(hotspots),
            "view_health_suspect": False, "sync_disagreement_m": [-1.0] * len(players), "ref_height_m": -1.0}


def from_tracker3d(tr, clip_id, views_all, offsets, n, time_ref=""):
    res, V = tr.result, len(views_all)
    col = {v: views_all.index(v) for v in tr.views}
    tracks = []
    for t in res["tracklets"]:
        frames = [(f, c) for f, c in zip(t.frames, t.cands) if f < n]
        if not frames:
            continue
        f0, f1 = frames[0][0], frames[-1][0]
        L = f1 - f0 + 1
        fl, sup, det = np.full((L, 2), NONE_XZ), np.zeros(L, np.int64), np.full((V, L), -1, np.int64)
        for f, c in frames:
            k = f - f0
            fl[k], sup[k] = np.round(c.floor, 3), c.support
            for v, i in c.dets.items():
                det[col[v], k] = tr.kp_idx[v][f][i]
        lab = res["labels"][t.tid]
        tracks.append({"track_id": int(t.tid), "label": lab,
                       "player_id": res["player_of"].get(t.tid, "") if lab == "player" else "",
                       "score": round(float(res["scores"][t.tid]), 3), "first_frame": int(f0), "last_frame": int(f1),
                       "ambiguous": int(t.crowded), "pinned": int(t.tid in res["pins"]),
                       "floor_xz": fl.ravel().tolist(), "support": sup.tolist(), "det_index": det.ravel().tolist()})
    doc = _header(clip_id, "3d", tr.fps, n, views_all, tr.views, tr.shifts, offsets, time_ref)
    doc["tracks"] = tracks
    doc["stats"] = _stats(tracks, res["hotspots"], tr.players)
    doc["stats"]["ref_height_m"] = round(float(res["ref_height_m"]), 3)
    if res.get("ankle_h_m") is not None and np.isfinite(res["ankle_h_m"]):   # pose24 caches it for mono (L2)
        doc["stats"]["ankle_h_m"] = round(float(res["ankle_h_m"]), 3)
    return doc


def from_v2(tr, det_index, clip_id, views_all, offsets, n, time_ref=""):
    """TrackerV2 players as persons tracks (one per player, first..last detected frame; no non-players)."""
    V, col = len(views_all), [views_all.index(v) for v in tr.views]
    tracks = []
    for k, pid in enumerate(sorted(det_index)):
        D = np.asarray(det_index[pid])[:, :n]
        seen = np.nonzero((D >= 0).any(axis=0))[0]
        if not len(seen):
            continue
        f0, f1 = int(seen[0]), int(seen[-1])
        L = f1 - f0 + 1
        det = np.full((V, L), -1, np.int64)
        det[col] = D[:, f0:f1 + 1]
        fl = np.full((L, 2), NONE_XZ)
        tk = (getattr(tr, "track", None) or {}).get(pid)
        if tk is not None:
            seg = np.asarray(tk)[f0:f1 + 1]
            m = np.isfinite(seg).all(axis=1) & (det[:, :len(seg)] >= 0).any(axis=0)
            fl[:len(seg)][m] = np.round(seg[m], 3)
        tracks.append({"track_id": k, "label": "player", "player_id": pid, "score": 1.0, "first_frame": f0,
                       "last_frame": f1, "ambiguous": 0, "pinned": 0, "floor_xz": fl.ravel().tolist(),
                       "support": (det >= 0).sum(axis=0).astype(int).tolist(), "det_index": det.ravel().tolist()})
    doc = _header(clip_id, "v2", tr.fps, n, views_all, tr.views, tr.shifts, offsets, time_ref)
    doc["tracks"] = tracks
    doc["stats"] = _stats(tracks, [], sorted(det_index))
    return doc


def from_sidecar(sc):
    """A cvdebug sidecar's players (det_index V x F) - e.g. the base doc's PlayerTracker - as a persons doc."""
    names, F = [c["name"] for c in sc["cameras"]], int(sc["frame_count"])
    V = len(names)
    tracks = []
    for k, p in enumerate(sc.get("players", [])):
        D = np.asarray(p["det_index"]).reshape(V, F)
        seen = np.nonzero((D >= 0).any(axis=0))[0]
        if not len(seen):
            continue
        f0, f1 = int(seen[0]), int(seen[-1])
        L = f1 - f0 + 1
        tracks.append({"track_id": k, "label": "player", "player_id": p["id"], "score": 1.0, "first_frame": f0,
                       "last_frame": f1, "ambiguous": 0, "pinned": 0, "floor_xz": [NONE_XZ] * (2 * L),
                       "support": (D[:, f0:f1 + 1] >= 0).sum(axis=0).astype(int).tolist(),
                       "det_index": D[:, f0:f1 + 1].ravel().astype(int).tolist()})
    doc = {"format": FORMAT, "clip_id": sc.get("clip_id", ""), "tracker": "sidecar:" + str(sc.get("variant", "")),
           "source_build_id": sc.get("source_build_id", ""), "fps": float(sc.get("fps", 60.0)), "frame_count": F,
           "views": names, "views_used": names, "det_shift": [int(c["det_shift"]) for c in sc["cameras"]],
           "offsets": [float(c.get("offset", 0.0)) for c in sc["cameras"]], "time_ref_view": sc.get("time_ref_view", ""),
           "tracks": tracks}
    doc["stats"] = _stats(tracks, [], [p["id"] for p in sc.get("players", [])])
    return doc


def validate(doc):
    errs = cvdebug.validate_sidecar(doc, object_fields=_OBJECTS)
    V = len(doc.get("views", []))
    for t in doc.get("tracks", []):
        L = t["last_frame"] - t["first_frame"] + 1
        if len(t["floor_xz"]) != 2 * L or len(t["support"]) != L or len(t["det_index"]) != V * L:
            errs.append(f"track {t['track_id']}: array lengths don't match L={L}, V={V}")
        if t["label"] not in ("player", "nonplayer", "unknown"):
            errs.append(f"track {t['track_id']}: bad label {t['label']!r}")
        if t["label"] != "player" and t.get("player_id"):
            errs.append(f"track {t['track_id']}: player_id {t['player_id']!r} on a {t['label']!r} track")
    if len(doc.get("det_shift", [])) != V:
        errs.append("det_shift length != len(views)")
    return errs


def write(p, doc):
    errs = validate(doc)
    if errs:
        raise ValueError("persons doc invalid:\n  " + "\n  ".join(errs[:10]))
    cvdebug.write_sidecar(p, doc)


def load(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def predictions(doc) -> dict:
    """{(view, raw frame, det index): "P1".. | "N" | "U"} for every detection some track holds."""
    V, names, out = len(doc["views"]), doc["views"], {}
    for t in doc["tracks"]:
        L = t["last_frame"] - t["first_frame"] + 1
        D = np.asarray(t["det_index"]).reshape(V, L)
        lab = t["player_id"] if t["label"] == "player" and t["player_id"] else ("U" if t["label"] == "unknown" else "N")
        for ci in range(V):
            for k in np.nonzero(D[ci] >= 0)[0]:
                out[(names[ci], t["first_frame"] + int(k) + doc["det_shift"][ci], int(D[ci, k]))] = lab
    return out
