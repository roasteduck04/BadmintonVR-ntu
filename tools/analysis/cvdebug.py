"""cvdebug/1 sidecar (SP2 spec § 3): per camera / player / frame / joint facts of what the CV pipeline saw and
concluded, for the video overlay (analysis.overlay_video) and Unity (SP4). Written next to the analysis doc as
<clip>[_variant].cvdebug.json, JsonUtility-safe: flat int arrays, -1 = unknown, bitmasks over cameras[] (bit v =
camera v), no NaN / null-in-array / nested arrays / free-key dicts.

Unity camera pose (Python does the Y flip, never Unity): the analysis world is the court world with Y negated, and
it is mirror-handed (SP1 F1), like Unity's left-handed world. With F = diag(1,-1,1), camera centre c = F(-R^T t) and
rotation M = F R^T F (columns = Unity camera right, up, forward; det +1): a world point p has Unity camera-local
l = M^T (p - c) and pixel u = cx + fx lx/lz, v = cy - fy ly/lz, the same as racket_rigidfit.Camera.project(p)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from analysis import pose_v2 as pv
from analysis import racket_rigidfit as rf
from analysis import racket_variants as rv
from analysis import skeleton24 as sk
from analysis import video_io
from analysis.pose24 import config, kpset, regress, track
from analysis.pose24 import doc as docmod
from analysis.pose24 import scores as scoremod

FORMAT = "cvdebug/1"
MAX_VIEWS = 8
STATE_MISSING, STATE_MEASURED, STATE_FILLED, STATE_PRIOR = 0, 1, 2, 3
INLIER_PX = pv.INLIER_PX              # variant path: a seen view counts as inlier below this
GATE_PX = 150.0                       # variant path: raw detection -> doc player (median body px), racket_variants'
_BODY_KP = np.arange(5, 17)           # COCO body shoulders .. ankles: cache 2D -> raw detection matching
SYNTH_JOINTS = {"spine1", "spine2", "spine3", "neck", "left_collar", "right_collar"}
BODY_ONLY_J = 22                      # SMPL body without hands/face: pelvis .. right_wrist (same order in 24/28/58)
MAX_FULL_J = 28                       # variant docs with more joints (SMPL-X 58) default to body-only (size target)
_F = np.array([1.0, -1.0, 1.0])
_VARIANT_KP = dict(regress.DIRECT, left_big_toe=17, right_big_toe=20)


def unity_camera(K, R, t):
    K, R, t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float).reshape(3)
    pos = _F * (-R.T @ t)
    M = np.diag(_F) @ R.T @ np.diag(_F)
    quat = Rotation.from_matrix(M).as_quat()          # x, y, z, w
    vfov = float(np.degrees(2.0 * np.arctan(K[1, 2] / K[1, 1])))   # principal point at the centre -> h / 2 = cy
    return [round(float(x), 6) for x in pos], [round(float(x), 10) for x in quat], round(vfov, 4)


def unity_project(pos, quat, K, X):
    M = Rotation.from_quat(quat).as_matrix()
    loc = (np.asarray(X, float) - np.asarray(pos, float)) @ M          # rows: M^T (p - c)
    K = np.asarray(K, float)
    return np.stack([K[0, 2] + K[0, 0] * loc[:, 0] / loc[:, 2], K[1, 2] - K[1, 1] * loc[:, 1] / loc[:, 2]], axis=1)


def project(cam: rf.Camera, X):
    X = np.asarray(X, float)
    flat = X.reshape(-1, 3)
    uv = np.full((len(flat), 2), np.nan)
    ok = np.isfinite(flat).all(axis=1)
    if ok.any():
        u, z = cam.project(flat[ok])
        uv[ok] = np.where((z > 0.1)[:, None], u, np.nan)
    return uv.reshape(X.shape[:-1] + (2,))


def pack_bits(mask):
    mask = np.asarray(mask, bool)
    if mask.shape[-1] > MAX_VIEWS:
        raise ValueError(f"cvdebug: at most {MAX_VIEWS} cameras per clip, got {mask.shape[-1]}")
    return (mask.astype(np.int64) << np.arange(mask.shape[-1])).sum(axis=-1)


def unpack_bits(bits, V):
    return ((np.asarray(bits, np.int64)[..., None] >> np.arange(V)) & 1).astype(bool)


def quantise(a, scale):
    a = np.asarray(a, float)
    out = np.full(a.shape, -1, np.int64)
    ok = np.isfinite(a) & (a >= 0)
    out[ok] = np.round(a[ok] * scale).astype(np.int64)
    return out


def doc_world_joints(doc):
    """{pid: (J (F,Jn,3) Y-up world, NaN where missing or confidence 0; C (F,Jn))}."""
    nj = len(doc["skeleton"]["joint_names"])
    out = {}
    for p in doc["players"]:
        F = len(p["frames"])
        J, C = np.full((F, nj, 3), np.nan), np.zeros((F, nj))
        for f, fr in enumerate(p["frames"]):
            jf, ps = fr.get("joints_flat"), fr.get("position")
            if jf is None or ps is None:
                continue
            a = np.asarray(jf, float).reshape(nj, 4)
            ok = a[:, 3] > 0
            J[f, ok] = a[ok, :3] + np.array([ps[0], 0.0, ps[1]])
            C[f] = a[:, 3]
        out[p["id"]] = (J, C)
    return out


def variant_joint_keypoints(joint_names):
    return [[_VARIANT_KP[n]] if n in _VARIANT_KP else [] for n in joint_names]


def pose24_joint_sources(joint_names):
    src = dict(zip(sk.SMPL28_NAMES, regress.SOURCE_LIST))
    mask_src = [list(src.get(n, [])) for n in joint_names]
    err_src = [[] if n in SYNTH_JOINTS else list(s) for n, s in zip(joint_names, mask_src)]
    return mask_src, err_src


# --- pose24 path: sidecar from the analysis doc + its debug cache (paired by meta.pose24.build_id) ---------------

class PairingError(ValueError):
    pass


# fields allowed to hold objects (fixed classes on the Unity side); everything else is a scalar or a flat array
_OBJECT_FIELDS = {"cameras", "players", "persons", "stats", "views", "hotspots"}


def doc_path(analysis_dir, clip, variant):
    return os.path.join(analysis_dir, f"{clip}.analysis.json" if variant in (None, "", "base")
                        else f"{clip}_{variant}.analysis.json")


def sidecar_path(analysis_dir, clip, variant):
    return os.path.join(analysis_dir, f"{clip}.cvdebug.json" if variant in (None, "", "base")
                        else f"{clip}_{variant}.cvdebug.json")


def camera_views(clip, registry):
    return sorted(registry[registry[clip].get("base", clip)]["views"])


def _calib(calib_dir, v):
    with open(os.path.join(calib_dir, f"{v}_cam.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cameras(views_all, calib_dir, videos, offsets, offset_source, excluded, held_out, health, sizes,
             det_shifts=None):
    """det_shift: raw kp2d frame of det_index = f + det_shift (the tracker's integer shift where it has one, else
    floor(offset + 0.5)); the video frame is always floor(f + offset + 0.5)."""
    out = []
    for v in views_all:
        c = _calib(calib_dir, v)
        K, R, t = np.array(c["K"], float), np.array(c["R"], float), np.array(c["t"], float).reshape(3)
        pos, quat, vfov = unity_camera(K, R, t)
        w, h = sizes.get(v) or (None, None)
        if w is None or h is None:
            w, h = c.get("width"), c.get("height")
        if v in excluded:
            health_px = excluded[v]
        elif health.get(v) is not None:
            health_px = health[v]
        else:
            health_px = -1.0
        out.append({"name": v, "video": videos.get(v) or "", "width": int(w or 0), "height": int(h or 0),
                    "K": K.ravel().tolist(), "R": R.ravel().tolist(), "t": t.tolist(),
                    "offset": round(float(offsets.get(v, 0.0)), 3), "offset_source": offset_source.get(v, "none"),
                    "det_shift": int((det_shifts or {}).get(v, video_io.view_frame(0, offsets.get(v, 0.0))[0])),
                    "excluded": v in excluded, "held_out": v in held_out, "health_px": round(float(health_px), 2),
                    "unity_pos": pos, "unity_rot": quat, "vfov_deg": vfov})
    return out


def _expand(arr_vc, cols, V):
    """(..., Vc) over the build's views -> (..., V) over cameras[] (missing views False / NaN)."""
    shape = arr_vc.shape[:-1] + (V,)
    out = np.zeros(shape, bool) if arr_vc.dtype == bool else np.full(shape, np.nan)
    out[..., cols] = arr_vc
    return out


def _nanmean(a, axis):
    with warnings.catch_warnings():           # all-NaN slice -> NaN, silently
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _per_joint(kp_fkv, sources, reduce):
    """(F,K,V) over keypoints -> (F,J,V) over joints via `sources` (keypoint index lists; [] -> False / NaN)."""
    F, _, V = kp_fkv.shape
    out = np.zeros((F, len(sources), V), bool) if reduce == "any" else np.full((F, len(sources), V), np.nan)
    for j, s in enumerate(sources):
        if s:
            out[:, j] = kp_fkv[:, s].any(axis=1) if reduce == "any" else _nanmean(kp_fkv[:, s], axis=1)
    return out


def _stats(views_all, players, F, J):
    V = len(views_all)
    used, denom, cov = np.zeros(V), 0, []
    for p in players:
        st = np.array(p["frame_state"])
        vm = unpack_bits(np.array(p["view_mask"]).reshape(F, J), V)
        meas = st == STATE_MEASURED
        denom += int(meas.sum())
        used += vm[meas].any(axis=1).sum(axis=0)
        cov.append(round(float((st != STATE_MISSING).mean()) if F else 0.0, 4))
    views = [{"name": v, "used_frac": round(float(used[k] / denom) if denom else 0.0, 4)}
             for k, v in enumerate(views_all)]
    return {"views": views, "coverage_per_player": cov}


def _frame_state(frames, n_views_hips):
    st = np.full(len(frames), STATE_MISSING, np.int64)
    for f, fr in enumerate(frames):
        if fr.get("joints_flat") is None:
            continue
        st[f] = STATE_FILLED if fr.get("interpolated") else (STATE_MEASURED if n_views_hips[f] >= 2 else STATE_PRIOR)
    return st


def _header(doc, variant, build_id, joint_names, banner):
    return {"format": FORMAT, "clip_id": doc["clip_id"], "source_analysis": "", "source_build_id": build_id,
            "variant": variant or "base", "skeleton_type": (doc.get("skeleton") or {}).get("type", ""),
            "fps": float(doc["meta"]["fps"]), "frame_count": int(doc["meta"]["frame_count"]),
            "time_ref_view": "", "joint_names": list(joint_names), "aux_names": list(sk.AUX_NAMES),
            "banner": banner}


def _contact(extra, F):
    """(F*6,) 0/1 planted-foot flags in kpset.CONTACT order (toes + heels); 0 where the cache has none."""
    c = np.zeros((F, len(kpset.CONTACT)), np.int64)
    if extra.get("contact") is not None:
        a = np.asarray(extra["contact"])[:F]
        c[:len(a)] = a.astype(bool)
    return c.ravel().tolist()


def build_pose24(doc, cache, views_all, calib_dir, videos, timeline=None, body_only=False):
    names = list(doc["skeleton"]["joint_names"])
    J = BODY_ONLY_J if body_only else len(names)
    names = names[:J]
    F, V = int(doc["meta"]["frame_count"]), len(views_all)
    build_views = list(cache["views"])
    cols = [views_all.index(v) for v in build_views]
    excluded = {v: float(h) for v, h in (cache.get("excluded") or {}).items()}
    held_out = [v for v in views_all if v not in build_views and v not in excluded]
    offsets, source = {}, {}
    for v, o in (cache.get("offsets_all") or cache["offsets"]).items():
        offsets[v], source[v] = float(o), "stored"
    for v, o in ((timeline or {}).get("offsets") or {}).items():
        if v not in offsets:
            offsets[v], source[v] = float(o), "timeline"
    mask_src, err_src = pose24_joint_sources(names)
    players = []
    for pid in cache["pids"]:
        ex = cache["extra"][pid]
        w133 = np.stack([np.nan_to_num(cache["obs"][pid][v][..., 2]) for v in build_views], axis=-1)   # (F,133,Vc)
        w35 = kpset.scores35(w133)                     # (F,35,Vc): ring centres = ring minimum, as the fit (L6)
        seen35 = _expand(w35 >= pv.W_MIN, cols, V)
        inl35 = _expand(np.asarray(ex["inlier_mask"], bool), cols, V) & seen35
        res35 = np.where(inl35, _expand(np.asarray(ex["resid_px"], float), cols, V), np.nan)
        seen = _per_joint(seen35, mask_src, "any")
        view = _per_joint(inl35, mask_src, "any")
        err = _nanmean(np.where(view, _per_joint(res35, err_src, "mean"), np.nan), axis=-1)
        ferr = _nanmean(res35.reshape(F, -1), axis=1)
        hips_views = inl35[:, [11, 12]].any(axis=1).sum(axis=1)
        det = np.full((V, F), -1, np.int64)
        det[cols] = np.asarray(ex["det_index"])[:, :F]
        lr = np.zeros((V, F), np.int64)
        lr[cols] = np.asarray(ex["lr_swapped"])[:, :F].astype(np.int64)
        frames = next(p for p in doc["players"] if p["id"] == pid)["frames"]
        players.append({"id": pid, "label": "player",
                        "frame_state": _frame_state(frames, hips_views).tolist(),
                        "seen_mask": pack_bits(seen).ravel().tolist(), "view_mask": pack_bits(view).ravel().tolist(),
                        "err_dpx": quantise(err, 10).ravel().tolist(),
                        "sigma_mm": quantise(np.asarray(ex["sigma_mm"], float)[:, :J], 1).ravel().tolist(),
                        "filled": np.asarray(ex["filled"])[:, :J].astype(np.int64).ravel().tolist(),
                        "det_index": det.ravel().tolist(), "lr_swapped": lr.ravel().tolist(),
                        "frame_err_px": quantise(ferr, 10).tolist(),
                        "contact": _contact(ex, F)})
    health = {v: float(h) for v, h in (cache.get("view_health") or {}).items() if h is not None}
    sizes = {c["name"]: (c.get("width"), c.get("height")) for c in doc["meta"].get("cameras") or []}
    suspect = bool((doc["meta"].get("pose_variant") or {}).get("view_health_suspect"))
    banner = ("VIEW HEALTH SUSPECT: every view failed the health check - calibration or sync reference bad"
              if suspect else "")
    sc = _header(doc, "pose24", doc["meta"]["pose24"]["build_id"], names, banner)
    sc["time_ref_view"] = cache.get("time_ref_view") or ""
    det_shifts = {v: int(s) for v, s in (cache.get("shifts") or {}).items() if v in build_views}
    sc["cameras"] = _cameras(views_all, calib_dir, videos, offsets, source, excluded, held_out, health, sizes,
                             det_shifts)
    sc["players"], sc["persons"] = players, []
    sc["stats"] = _stats(views_all, players, F, J)
    return sc


def write_sidecar(path, sc):
    """Compact (no ", " / ": " padding: ~15 % smaller) atomic write (temp file + os.replace), no NaN."""
    text = json.dumps(sc, allow_nan=False, separators=(",", ":"))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def validate_sidecar(sc, object_fields=None):
    """Problems Unity's JsonUtility would trip on (empty list = safe): objects only as the fixed-class fields
    (object_fields, default _OBJECT_FIELDS, or the top level), keys that are field names, flat arrays without null,
    finite numbers."""
    allowed = _OBJECT_FIELDS if object_fields is None else set(object_fields)
    errs = []

    def walk(x, path, field):
        if isinstance(x, dict):
            if path and field not in allowed:
                errs.append(f"{path}: object not allowed here (JsonUtility has no dictionaries)")
            for k, v in x.items():
                if not (isinstance(k, str) and k.isidentifier()):
                    errs.append(f"{path}.{k}: key is not a field name")
                walk(v, f"{path}.{k}" if path else str(k), k)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                if isinstance(v, (list, tuple)):
                    errs.append(f"{path}[{i}]: nested array")
                elif v is None:
                    errs.append(f"{path}[{i}]: null in array")
                else:
                    walk(v, f"{path}[{i}]", field)        # array elements are instances of the field's class
        elif isinstance(x, float) and not np.isfinite(x):
            errs.append(f"{path}: non-finite number")

    walk(sc, "", "")
    return errs


def _doc_build_id(doc, raw_bytes):
    b = ((doc.get("meta") or {}).get("pose24") or {}).get("build_id")
    return b or "sha1:" + hashlib.sha1(raw_bytes).hexdigest()[:16]


def attach_persons(sc, pdoc, build_id, min_frames=0) -> str:
    """Copy a persons/1 doc's tracks into sc["persons"] when it pairs with this sidecar: same camera order and frame
    count, same build (when the persons doc names one) and the same det_shift for every view it used. Returns ""
    when attached, else why not (a mismatched file would colour the wrong people). min_frames > 0 drops nonplayer /
    unknown tracks shorter than that (players are always kept; SP3 F19: ~2000 short tracks made singles sidecars
    2.5-3.3 MB); the clip's bystander hotspots are copied so the zones survive the pruning."""
    from analysis.people import persons as pm
    names = [c["name"] for c in sc["cameras"]]
    if list(pdoc.get("views", [])) != names:
        return "camera order differs"
    if int(pdoc.get("frame_count", -1)) != int(sc["frame_count"]):
        return "frame count differs"
    if len(pdoc.get("det_shift", [])) != len(names):
        return "det_shift length differs from the camera count"
    bid = pdoc.get("source_build_id") or ""
    if bid and build_id and bid != build_id:
        return f"persons from another build ({bid} != {build_id})"
    for k, v in enumerate(names):
        if v in pdoc.get("views_used", []) and int(pdoc["det_shift"][k]) != int(sc["cameras"][k]["det_shift"]):
            return f"{v} det_shift differs ({pdoc['det_shift'][k]} vs {sc['cameras'][k]['det_shift']})"
    keep, dropped = [], 0
    for t in pdoc.get("tracks", []):
        n = int(t["last_frame"]) - int(t["first_frame"]) + 1
        if t["label"] != "player" and n < int(min_frames):
            dropped += 1
            continue
        keep.append({k: t[k] for k in pm.TRACK_FIELDS})
    sc["persons"] = keep
    sc["persons_dropped"] = dropped
    sc["hotspots"] = [{"x": float(h["x"]), "z": float(h["z"]), "frames": int(h["frames"])}
                      for h in (pdoc.get("stats") or {}).get("hotspots", [])]
    return ""


def build(clip, variant="pose24", analysis_dir=config.DEFAULT_ANALYSIS_DIR, cache_dir=pv.CACHE_DIR,
          data_dir="data", registry=None, body_only=False, all_joints=False, write=True, log=print,
          persons_path=None, persons_min_frames=30):
    """persons_min_frames: drop nonplayer / unknown tracks shorter than this when attaching persons (0 = keep all).
    all_joints: keep every joint of a > MAX_FULL_J-joint variant doc (default: body-only, BODY_ONLY_J)."""
    t0 = time.perf_counter()
    reg = registry if registry is not None else config.load_registry()
    src = doc_path(analysis_dir, clip, variant)
    if not os.path.exists(src):
        raise FileNotFoundError(f"no analysis doc {src}")
    with open(src, "rb") as fh:
        raw = fh.read()
    doc = json.loads(raw)
    views_all = camera_views(clip, reg)
    # clips.json paths are repo-relative (run from the repo root); absolute registry paths (tests) pass through
    videos = {v: (p or "") for v, p in video_io.resolve_videos(clip, registry=reg, root="").items()}
    calib_dir = os.path.join(data_dir, "calib")
    if variant == "pose24":
        cpath = track.cache_path(clip, cache_dir)
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"no pose24 debug cache {cpath}; run pose24 build --clip {clip}")
        cache = docmod.load_debug_cache(cpath)
        want = ((doc.get("meta") or {}).get("pose24") or {}).get("build_id")
        if not want:
            raise PairingError(f"{os.path.basename(src)} has no meta.pose24.build_id: cannot pair it with "
                               f"{os.path.basename(cpath)} - rerun pose24 build --clip {clip}")
        if cache.get("build_id") != want:
            raise PairingError(f"{os.path.basename(src)} build_id {want} != cache {os.path.basename(cpath)} build_id "
                               f"{cache.get('build_id')}: they come from different builds - rerun pose24 build "
                               f"--clip {clip}")
        timeline = track.load_timeline(cache["timeline_of"], cache_dir) if cache.get("timeline_of") else None
        sc = build_pose24(doc, cache, views_all, calib_dir, videos, timeline=timeline, body_only=body_only)
    else:
        sc = build_variant(doc, variant, views_all, calib_dir, videos, cache_dir=cache_dir,
                           pose2d_dir=os.path.join(data_dir, "pose2d"), body_only=body_only, all_joints=all_joints,
                           log=log)
    sc["source_analysis"] = os.path.basename(src)
    sc["source_build_id"] = _doc_build_id(doc, raw)
    sc["proxy_clip"] = reg[clip].get("base", clip)         # data/video_proxy/<proxy_clip>/<view>.mp4 (SP2 F5)
    sc.setdefault("hotspots", [])
    sc.setdefault("persons_dropped", 0)
    from analysis.people import persons as pm
    ppath = persons_path or pm.path(analysis_dir, clip)
    if persons_path and not os.path.exists(persons_path):
        raise FileNotFoundError(f"persons file {persons_path} not found")
    if os.path.exists(ppath):
        why = attach_persons(sc, pm.load(ppath), sc["source_build_id"], min_frames=persons_min_frames)
        log(f"    persons: {len(sc['persons'])} tracks from {os.path.basename(ppath)}, "
            f"{sc['persons_dropped']} short dropped" if not why
            else f"    persons: {os.path.basename(ppath)} not attached - {why}")
    errs = validate_sidecar(sc)
    if errs:
        raise ValueError("cvdebug sidecar not JsonUtility-safe:\n  " + "\n  ".join(errs[:10]))
    out = sidecar_path(analysis_dir, clip, variant)
    if write:
        write_sidecar(out, sc)
        log(f"[cvdebug] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)")
    return out, sc


# --- variant path (base / posev2 / posefit): the doc's 3D reprojected into every camera ----------------------------
# 2D observations per view: the pose_v2 cache (<clip>.npz) when it is on the doc's timeline (det_index by matching
# the cache's own 2D to the raw detections); else the raw kp2d file at the doc's stored offset (floor(o + 0.5)), or
# at a shift estimated by racket_variants.estimate_view_sync. Excluded views (bad calibration) stay unmeasured.

def measure_reprojection(Jw, uv, w, cams, joint_kp, w_min=pv.W_MIN):
    """Jw (F,Jn,3), uv (F,133,V,2), w (F,133,V) -> seen (F,Jn,V), view (F,Jn,V) = seen and < INLIER_PX,
    err (F,Jn) mean inlier px, ferr (F,) mean over every inlier joint-view. Joints without a keypoint: unseen, NaN."""
    F, Jn, _ = Jw.shape
    V = len(cams)
    seen = np.zeros((F, Jn, V), bool)
    errv = np.full((F, Jn, V), np.nan)
    for j, ks in enumerate(joint_kp):
        if not ks:
            continue
        k = ks[0]
        for v, cam in enumerate(cams):
            p = project(cam, Jw[:, j])                                     # (F,2)
            ok = (w[:, k, v] >= w_min) & np.isfinite(uv[:, k, v]).all(axis=1)
            seen[:, j, v] = ok
            e = np.linalg.norm(uv[:, k, v] - p, axis=1)
            errv[:, j, v] = np.where(ok & np.isfinite(e), e, np.nan)
    view = seen & (np.nan_to_num(errv, nan=np.inf) < INLIER_PX)
    inl = np.where(view, errv, np.nan)
    return seen, view, _nanmean(inl, axis=-1), _nanmean(inl.reshape(F, -1), axis=1)


def load_raw_kp2d(path, params=None):
    """kp2d.json -> (frames[f] (D,133,3) float32 with calibrated scores, every detection kept; header)."""
    params = params or scoremod.load_params()
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    frames = []
    for fr in d["frames"]:
        if not fr:
            frames.append(np.zeros((0, 133, 3), np.float32))
            continue
        k = np.asarray([det["keypoints"] for det in fr], np.float32)
        k[..., 2] = scoremod.calibrate(k[..., 2], params)
        frames.append(k)
    return frames, {k: v for k, v in d.items() if k != "frames"}


def assign_raw(frames_raw, uv_proj, coco_idx, shift):
    """Nearest raw detection (median body-joint px distance < GATE_PX) per analysis frame, at raw frame f + shift."""
    F = len(uv_proj)
    uv, w, det = np.full((F, 133, 2), np.nan), np.zeros((F, 133)), np.full(F, -1, np.int64)
    for f in range(F):
        g = f + shift
        if g < 0 or g >= len(frames_raw) or len(frames_raw[g]) == 0:
            continue
        d = rv._person_distance(frames_raw[g][:, coco_idx].astype(np.float64), uv_proj[f])
        if len(d) and np.isfinite(d.min()) and d.min() < GATE_PX:
            i = int(np.argmin(d))
            uv[f], w[f], det[f] = frames_raw[g][i, :, :2], frames_raw[g][i, :, 2], i
    return uv, w, det


def _timeline_offsets(clip, views_all, offsets, source, cache_dir, log=print):
    """Views still without an offset (excluded ones: the doc stores none) take it from the same clip's pose24 cache
    (offsets_all), moved onto the doc's timeline by the median difference over the views both have."""
    todo = [v for v in views_all if v not in offsets]
    if not todo or not os.path.exists(track.cache_path(clip, cache_dir)):
        return
    tl = track.load_timeline(clip, cache_dir)["offsets"]
    both = [v for v in offsets if v in tl]
    if not both:
        return
    delta = float(np.median([offsets[v] - float(tl[v]) for v in both]))
    for v in todo:
        if v in tl:
            offsets[v], source[v] = float(tl[v]) + delta, "timeline"
            log(f"    {v}: offset {offsets[v]:+.2f} from {os.path.basename(track.cache_path(clip, cache_dir))}")


def build_variant(doc, variant, views_all, calib_dir, videos, cache_dir=pv.CACHE_DIR, pose2d_dir="data/pose2d",
                  body_only=False, all_joints=False, log=print):
    names = list(doc["skeleton"]["joint_names"])
    if len(names) > MAX_FULL_J and not all_joints:
        body_only = True
    J = BODY_ONLY_J if body_only else len(names)
    names = names[:J]
    F, V = int(doc["meta"]["frame_count"]), len(views_all)
    cams = [rf.Camera(*(np.array(_calib(calib_dir, v)[k], float) for k in ("K", "R", "t"))) for v in views_all]
    world = {pid: Jw[:, :J] for pid, (Jw, _C) in doc_world_joints(doc).items()}
    pvm = doc["meta"].get("pose_variant") or {}
    excluded = {v: float(h) for v, h in (pvm.get("excluded_views") or {}).items()}
    offsets, source, health = {}, {}, {}
    for v, o in (pvm.get("offsets") or {}).items():
        offsets[v], source[v] = float(o), "stored"
    cache = None
    cpath = os.path.join(cache_dir, f"{doc['clip_id']}.npz")
    if offsets and os.path.exists(cpath):
        c = docmod.load_debug_cache(cpath)
        same = int(c["frame_count"]) == F and all(abs(float(c["offsets"].get(v, 1e9)) - o) < 0.01
                                                  for v, o in offsets.items() if v in c["views"])
        if same and set(c["pids"]) >= set(world):
            cache = c
        else:
            log(f"    {os.path.basename(cpath)}: offsets differ from the doc's (or other players) - using raw kp2d")
    cache_views = set(cache["views"]) if cache is not None else set()
    # every calibrated, non-excluded view is measured: views outside the cache via raw kp2d, estimating the
    # offsets the doc does not store (once, over just those views); cache views load raw kp2d for det_index only
    raw_frames = {}
    params = scoremod.load_params()
    for v in views_all:
        p = os.path.join(pose2d_dir, f"{v}_kp2d.json")
        if v not in excluded and os.path.exists(p):
            raw_frames[v] = load_raw_kp2d(p, params)[0]
    missing = [v for v in views_all if v not in cache_views and v not in excluded and v not in offsets]
    if missing:
        est, hl, *_ = rv.estimate_view_sync(doc, {v: cams[views_all.index(v)] for v in missing}, pose2d_dir, log=log)
        for v, o in est.items():
            offsets[v], source[v] = float(o), "estimated"
        health.update({v: float(h) for v, h in hl.items() if np.isfinite(h)})
    _timeline_offsets(doc["clip_id"], views_all, offsets, source, cache_dir, log)
    joint_kp = variant_joint_keypoints(names)
    body_coco, body_j = (list(x) for x in zip(*[(ks[0], j) for j, ks in enumerate(joint_kp) if ks and ks[0] < 17]))
    players = []
    for p in doc["players"]:
        pid = p["id"]
        Jw = world[pid]
        uv = np.full((F, 133, V, 2), np.nan)
        w = np.zeros((F, 133, V))
        det = np.full((V, F), -1, np.int64)
        for k, v in enumerate(views_all):
            if v in excluded:
                continue
            shift = video_io.view_frame(0, offsets[v])[0] if v in offsets else None   # floor(o + 0.5), as the overlay
            if v in cache_views:
                o = cache["obs"][pid][v]
                uv[:, :, k], w[:, :, k] = o[:F, :, :2], np.nan_to_num(o[:F, :, 2])
                if v in raw_frames and shift is not None:      # the detection the tracker used: match its own 2D
                    own = np.where((w[:, _BODY_KP, k] >= pv.W_MIN)[..., None], uv[:, _BODY_KP, k], np.nan)
                    det[k] = assign_raw(raw_frames[v], own, list(_BODY_KP), shift)[2]
            elif v in raw_frames and shift is not None:
                proj = project(cams[k], Jw[:, body_j])
                uv[:, :, k], w[:, :, k], det[k] = assign_raw(raw_frames[v], proj, body_coco, shift)
        seen, view, err, ferr = measure_reprojection(Jw, uv, w, cams, joint_kp)
        n_views = seen[:, body_j].any(axis=1).sum(axis=1)
        interp = np.array([bool(fr.get("interpolated")) for fr in p["frames"]], bool)
        players.append({"id": pid, "label": "player", "frame_state": _frame_state(p["frames"], n_views).tolist(),
                        "seen_mask": pack_bits(seen).ravel().tolist(), "view_mask": pack_bits(view).ravel().tolist(),
                        "err_dpx": quantise(err, 10).ravel().tolist(), "sigma_mm": [-1] * (F * J),
                        "filled": np.repeat(interp.astype(np.int64), J).tolist(),
                        "det_index": det.ravel().tolist(), "lr_swapped": [0] * (V * F),
                        "frame_err_px": quantise(ferr, 10).tolist(), "contact": [0] * (F * len(kpset.CONTACT))})
    held_out = [v for v in views_all if pvm.get("views") and v not in pvm["views"] and v not in excluded]
    sizes = {c["name"]: (c.get("width"), c.get("height")) for c in doc["meta"].get("cameras") or []}
    stored_health = {v: float(h) for v, h in (pvm.get("view_health") or {}).items() if h is not None}
    sc = _header(doc, variant, "", names, "")
    sc["time_ref_view"] = pvm.get("time_ref_view") or next(
        (v for v in views_all if abs(float((pvm.get("offsets") or {}).get(v, 1.0))) < 1e-6), "")
    sc["cameras"] = _cameras(views_all, calib_dir, videos, offsets, source, excluded, held_out,
                             {**stored_health, **health}, sizes)
    sc["players"], sc["persons"] = players, []
    sc["stats"] = _stats(views_all, players, F, J)
    return sc


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="analysis.cvdebug", description="Build cvdebug/1 sidecars")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--clip")
    g.add_argument("--all", action="store_true")
    b.add_argument("--variant", default="pose24", help="comma list: pose24, posefit, posev2, base")
    b.add_argument("--body-only", action="store_true")
    b.add_argument("--all-joints", action="store_true",
                   help=f"keep every joint of a > {MAX_FULL_J}-joint variant doc (default: body-only)")
    b.add_argument("--persons", default=None,
                   help="persons.json to attach (default <clip>.persons.json); one --clip only")
    b.add_argument("--persons-min-frames", type=int, default=30,
                   help="drop nonplayer / unknown tracks shorter than N frames (0 = keep all)")
    b.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    b.add_argument("--cache-dir", default=pv.CACHE_DIR)
    b.add_argument("--data-dir", default="data")
    a = ap.parse_args(argv)
    if a.persons and a.all:
        ap.error("--persons is only valid with one --clip")
    reg = config.load_registry()
    clips = config.all_clips(reg) if a.all else [a.clip]
    failed = 0
    for clip in clips:
        for var in [s.strip() for s in a.variant.split(",") if s.strip()]:
            try:
                build(clip, var, analysis_dir=a.analysis_dir, cache_dir=a.cache_dir, data_dir=a.data_dir,
                      registry=reg, body_only=a.body_only, all_joints=a.all_joints, persons_path=a.persons,
                      persons_min_frames=a.persons_min_frames)
            except FileNotFoundError as e:
                if not a.all:
                    raise
                print(f"[cvdebug] skip {clip} {var}: {e}")
            except Exception as e:                    # isolate one clip's failure in --all
                if not a.all:
                    raise
                failed += 1
                print(f"[cvdebug] FAILED {clip} {var}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
