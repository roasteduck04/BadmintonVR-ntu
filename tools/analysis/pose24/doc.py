"""pose24 outputs (spec § 6, § 8): the schema 4.1 analysis doc, racket / shuttle / rallies carried from a
base doc with players matched by pelvis position (never by id, review L3), and the debug cache that keeps
the per-joint CV metadata for the cvdebug sidecar (next release)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis import pose_v2 as pv
from analysis import schema
from analysis import skeleton24 as sk
from analysis.schema24 import SCHEMA_VERSION

PRODUCER = "pose24_multiview_fit"
CACHE_FORMAT = "pose24_cache/1"
MATCH_MAX_M = 0.5            # players further apart (median court distance) stay unmatched


def json_safe(obj):
    """Recursively: non-finite floats -> None, numpy scalars/arrays -> Python types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def write_json(path, doc):
    """Atomic write (temp file + os.replace): a reader never sees a half-written doc."""
    text = json.dumps(doc, allow_nan=False)      # Unity's JsonUtility can't parse NaN
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


DOC_DECIMALS = 5        # joints_flat / aux_flat rounding (conf too: a conf < 5e-6 is written 0 = "missing")


def _flat(loc, conf):
    flat = np.empty(len(loc) * 4)
    flat[0::4], flat[1::4], flat[2::4], flat[3::4] = loc[:, 0], loc[:, 1], loc[:, 2], conf
    return [round(float(x), DOC_DECIMALS) for x in flat]


def frames_from_world24(J, C, aux, auxc, interp, fps):
    """World (Y-up) joints (F,NJ,3) + confidences -> doc frames: body-local joints_flat / aux_flat (pelvis XZ
    removed, heights kept), court position, interpolated flag. A missing joint is [0, pelvis y, 0] with
    confidence 0 (as pose_v2); a missing aux keypoint is [0, 0, 0] with confidence 0."""
    frames = []
    for f in range(len(J)):
        base = {"frame_id": f, "time": round(f / fps, 4)}
        if not np.isfinite(J[f, 0]).all():
            frames.append(dict(base, joints_flat=None, position=None, interpolated=False, aux_flat=None,
                               racket=None, grip=None, stroke=None))
            continue
        pel = J[f, 0]
        off = np.array([pel[0], 0.0, pel[2]])
        loc = J[f] - off
        miss = ~np.isfinite(loc).all(axis=1)
        loc[miss] = [0.0, pel[1], 0.0]
        a = aux[f] - off
        amiss = ~np.isfinite(a).all(axis=1)
        aux_flat = None
        if not amiss.all():
            a[amiss] = 0.0
            aux_flat = _flat(a, np.where(amiss, 0.0, np.nan_to_num(auxc[f])))
        frames.append(dict(base, joints_flat=_flat(loc, np.where(miss, 0.0, np.nan_to_num(C[f]))),
                           position=[round(float(pel[0]), 4), round(float(pel[2]), 4)],
                           interpolated=bool(interp[f]), aux_flat=aux_flat, racket=None, grip=None, stroke=None))
    return frames


def camera_meta(bundle):
    """Every configured camera, incl. ones view health dropped (excluded: true)."""
    out = []
    for v in sorted(bundle["cams_all"]):
        K, R, t = bundle["cams_all"][v]
        w, h = (bundle.get("image_size") or {}).get(v, [None, None])
        out.append({"name": v, "K": np.asarray(K).tolist(), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist(),
                    "width": w, "height": h, "excluded": v in bundle["excluded"]})
    return out


def doc_label(kind, views, excluded, method=None):
    """1 view: the monocular label, `method` = the learned prior's name (None: no prior)."""
    sk_name = 'SMPL-28' if kind == 'smpl-28' else 'SMPL-24'
    if len(views) == 1:
        return f"{sk_name} · monocular ({method or 'no'} prior) · 1 cam {views[0]}"
    s = f"{sk_name} · triangulated · {len(views)} cams ({', '.join(views)})"
    return s + (f" · dropped {', '.join(sorted(excluded))}" if excluded else "")


def make_doc24(out_clip, fps, n, players, bundle, kind, pose_variant, pose24_meta):
    """players = [(pid, frames)] -> schema 4.1 doc (validate with schema24 after carry-over + facing)."""
    meta = {"fps": round(float(fps), 3), "frame_count": int(n), "source_clip": out_clip, "camera": None,
            "mode": bundle["mode"], "label": doc_label(kind, bundle["views"], bundle["excluded"],
                                                     method=(pose24_meta.get("mono") or {}).get("prior")),
            "producers": {"pose": PRODUCER, "position": PRODUCER, "racket": None, "grip": None, "stroke": None,
                          "shuttle": None},
            "cameras": camera_meta(bundle), "pose_variant": pose_variant, "pose24": pose24_meta}
    return {"schema_version": SCHEMA_VERSION, "clip_id": out_clip, "meta": json_safe(meta),
            "court": schema.court_block(), "skeleton": sk.skeleton_block(kind),
            "players": [{"id": pid, "side": bundle["sides"].get(pid, "near"), "frames": fr} for pid, fr in players],
            "shuttle": {"frames": []}, "rallies": []}


def match_players(base, doc, max_m=MATCH_MAX_M):
    """{doc player id: base player id}: Hungarian on the median court distance of their positions over
    frames where both have one (>= 10); pairs further apart than max_m stay unmatched."""
    bp, dp = base["players"], doc["players"]
    cost = np.full((len(dp), len(bp)), 1e6)
    for i, p in enumerate(dp):
        for j, q in enumerate(bp):
            d = [math.hypot(a["position"][0] - b["position"][0], a["position"][1] - b["position"][1])
                 for a, b in zip(p["frames"], q["frames"]) if a.get("position") and b.get("position")]
            if len(d) >= 10:
                cost[i, j] = float(np.median(d))
    out = {}
    if cost.size:
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= max_m:
                out[dp[i]["id"]] = bp[j]["id"]
    return out


def base_to_doc(doc_to_base):
    """Invert match_players' {doc id: base id} into write_racket_sidecars' {base id: doc id}."""
    out = {b: d for d, b in doc_to_base.items()}
    if len(out) != len(doc_to_base):
        raise ValueError(f"player match is not one-to-one: {doc_to_base}")
    return out


def carry_over24(base, doc, log=print, gate_m=pv.CARRY_GATE_M):
    """Shuttle / rallies from the base doc; per-frame racket (re-expressed for the new pelvis), grip and
    stroke from the base player at the same place in that frame (`pose_v2.match_frames`: identity-safe when the
    base tracker swaps players mid-clip). Everything carried is sanitised (old base docs may hold NaN).
    Returns the per-frame match (pass it to `pose_v2.write_racket_sidecars`)."""
    for k in ("shuttle", "shuttle_legacy", "rallies"):
        if k in base:
            doc[k] = json_safe(base[k])
    for key in ("racket", "grip", "stroke", "shuttle"):
        doc["meta"]["producers"][key] = (base["meta"].get("producers") or {}).get(key)
    return pv.carry_rackets(base, doc, clean=json_safe, log=log, gate_m=gate_m)


def save_debug_cache(path, bundle, per_player, extra_meta):
    """The pose_v2 cache format (per-player synced 133-kpt uv + calibrated w per view, cams, offsets, excluded)
    plus per-player arrays `<pid>_<key>` and extra meta (format pose24_cache/1). Player ids must not contain
    "_" (a pid "P1_X" would collide with player P1's key "X")."""
    bad = [pid for pid in bundle["obs"] if "_" in str(pid)]
    if bad:
        raise ValueError(f"debug cache: player ids must not contain '_': {bad}")
    arrs = {}
    for pid in bundle["obs"]:
        uv, w = pv.obs_arrays(bundle, pid, idx=list(range(133)))
        arrs[f"{pid}_uv"] = uv.astype(np.float32)
        arrs[f"{pid}_w"] = w.astype(np.float32)
        for key, arr in (per_player.get(pid) or {}).items():
            arrs[f"{pid}_{key}"] = np.asarray(arr)
    meta = {k: bundle[k] for k in ("views", "fps", "frame_count", "offsets", "excluded", "sides", "mode")}
    meta["cams"] = {v: [np.asarray(x).tolist() for x in bundle["cams"][v]] for v in bundle["views"]}
    meta["pids"] = list(bundle["obs"])
    meta.update(extra_meta)
    meta["format"] = CACHE_FORMAT
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(json_safe(meta)), **arrs)


def load_debug_cache(path):
    """The cache as a pose_v2-style bundle (views, cams, obs, fps, ...) + "extra" {pid: {key: array}}."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        cams = {v: tuple(np.array(x) for x in meta["cams"][v]) for v in meta["views"]}
        obs, extra = {}, {}
        for pid in meta["pids"]:
            uv, w = z[f"{pid}_uv"], z[f"{pid}_w"]
            obs[pid] = {v: np.concatenate([uv[:, :, k], w[:, :, k, None]], axis=-1) for k, v in enumerate(meta["views"])}
            pre = f"{pid}_"
            extra[pid] = {k[len(pre):]: z[k] for k in z.files
                          if k.startswith(pre) and k not in (f"{pid}_uv", f"{pid}_w")}
    return dict(meta, cams=cams, obs=obs, extra=extra)
