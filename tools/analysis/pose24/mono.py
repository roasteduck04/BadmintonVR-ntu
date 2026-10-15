"""pose24 monocular (1-camera) fit, Stage B (spec § 4; SP5 D2 / D4): the SP1 window problem with one view.

  root on the court   planted heels / toes -> pixel ray meets the plane at the keypoint's floor height (FOOT_H),
                      used as the fit's anchor (sigma 0.3 m, as in fit); no anchor off contact (jumps: the
                      acceleration + limb terms carry the body)
  foot contact        from 2D (slow + the lower foot in the image), a prior's contact probabilities, and in
                      pass 2 fit.detect_contact on the pass-1 3D (OR-ed)
  limb lengths        fixed per subject: "oracle" (the full multi-camera build's lengths, flagged oracle) or
                      "prior" (rest template scaled to a 1.70 m stature; honest, carries the H6 scale mismatch)
  depth prior         optional learned prior (WHAM / GVHMR, Phase C): root-relative camera depths -> fit depth
                      rows (sigma DEPTH_FIT_SIGMA); the confidence sigma along the axis is depth_sigma_m

Handedness: kpset.world_sign works with one camera (the sign of its det R), so the single view's own sign is used.

Prior file contract (Phase C producers): data/smpl_mono/<clip>/<view>_<method>.npz with `meta` (JSON: method,
clip, view, timeline_of, fps, frame_count) and per player <pid>_kp (n,) int fit-keypoint indices, <pid>_rel_cam
(F,n,3) root-relative (hip-midpoint) camera-frame metres (OpenCV axes), <pid>_conf (F,) in [0,1], optional
<pid>_contact (F,6) probabilities (kpset.CONTACT order). meta build_id (v0.7.5) = the debug cache's build_id the
prior was made from: build_mono / camsubsets refuse a mismatch (check_prior_meta); a file without one is accepted
with a warning.

build_mono (`pose24 build --mono <view>`): a real 1-camera doc + debug cache from the clip's own full pose24 cache
(oracle identity + 2D, D2); no persons file, no racket sidecars (racket / shuttle 3D need >= 2 views)."""
from __future__ import annotations

import json
import os
import time
import uuid
import warnings

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import fit, kpset

DEFAULT_ANKLE_H = 0.13          # ankle keypoint height (m) without a cache ankle_h_m: SP3 F12 measured 0.123-0.151 m
FOOT_H = {15: DEFAULT_ANKLE_H, 16: DEFAULT_ANKLE_H, 17: 0.02, 18: 0.02, 19: 0.03, 20: 0.02, 21: 0.02, 22: 0.03}
PRIOR_DEPTH_SIGMA = 0.08        # confidence only: depth sigma (m) along the camera axis with a learned prior
NO_PRIOR_DEPTH_SIGMA = 0.25     # ... without one
DEPTH_FIT_SIGMA = 0.04          # the fit's depth-row sigma (m): "a tight prior, a few cm" (spec § 4)
PRIOR_CONF_MIN = 0.3            # prior frames below this confidence give no depth rows
CONTACT_V_FRAC = 0.015          # 2D contact: speed < 1.5 % of the median body bbox height per (60 fps) frame
LOWER_FOOT_PX = 15.0
LENGTH_SOURCES = ("oracle", "prior")
IDENTITY = "oracle (full-rig tracking)"


def settings() -> dict:
    """The mono fit constants (camsubsets freshness key of mono results; fit.settings() holds the shared ones)."""
    return {"depth_fit_sigma_m": DEPTH_FIT_SIGMA, "default_ankle_h_m": DEFAULT_ANKLE_H,
            "foot_h_m": {kpset.NAMES[k]: h for k, h in FOOT_H.items()}, "prior_depth_sigma_m": PRIOR_DEPTH_SIGMA,
            "no_prior_depth_sigma_m": NO_PRIOR_DEPTH_SIGMA, "prior_conf_min": PRIOR_CONF_MIN,
            "contact": {"v_frac": CONTACT_V_FRAC, "lower_foot_px": LOWER_FOOT_PX}}


def _foot_h(k, ankle_h):
    return ankle_h if k in (15, 16) else FOOT_H[k]


def ray_plane(uv, cam, height):
    """(N,2) pixels -> (N,3) Y-down court-world points where the pixel rays meet the plane Y = -height
    (height scalar or (N,)); NaN for NaN pixels, rays parallel to the plane or meeting it behind the camera."""
    K, R, t = (np.asarray(x, float) for x in cam)
    uv = np.asarray(uv, float).reshape(-1, 2)
    ray = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T @ R          # rows: R.T @ K^-1 @ [u, v, 1]
    c = -R.T @ t
    h = np.broadcast_to(np.asarray(height, float), (len(uv),))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (-h - c[1]) / ray[:, 1]
    ok = np.isfinite(s) & (np.abs(ray[:, 1]) > 1e-10) & (s > 0)
    return np.where(ok[:, None], c + np.where(ok, s, 0.0)[:, None] * ray, np.nan)


def _seen(uv, w):
    return (np.nan_to_num(w) >= pv.W_MIN) & np.isfinite(uv).all(-1)


def contact_from_2d(uv, w, fps, v_max_px=None, min_run=4):
    """(F,35,2) pixels + (F,35) scores of one view -> (F,6) bool (kpset.CONTACT): a heel / toe is planted when its 2D
    speed (px per frame at 60 fps; the larger of both neighbouring frames) is below v_max_px for >= min_run frames
    and its foot is the lower one in the image (larger v) or within 15 px of it. Default v_max_px = 1.5 % of the
    median body bbox height, per clip and view (the player's own image scale: a far camera sees smaller strides)."""
    seen = _seen(uv, w)
    Pc = np.where(seen[:, kpset.CONTACT, None], uv[:, kpset.CONTACT], np.nan)
    if v_max_px is None:
        body = np.where(seen[:, :17], uv[:, :17, 1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            h = np.nanmax(body, axis=1) - np.nanmin(body, axis=1)
            v_max_px = CONTACT_V_FRAC * float(np.nanmedian(h)) if np.isfinite(h).any() else 0.0
    d = np.linalg.norm(np.diff(Pc, axis=0), axis=-1) * (60.0 / fps)
    pad = np.full((1, Pc.shape[1]), np.nan)
    v = np.fmax(np.vstack([pad, d]), np.vstack([d, pad]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.stack([np.nanmean(Pc[:, :3, 1], axis=1), np.nanmean(Pc[:, 3:, 1], axis=1)], axis=1)   # left, right
    with np.errstate(invalid="ignore"):
        lower = np.stack([~(low[:, 0] < low[:, 1] - LOWER_FOOT_PX), ~(low[:, 1] < low[:, 0] - LOWER_FOOT_PX)], 1)
        c = (v < v_max_px) & np.repeat(lower & np.isfinite(low), 3, axis=1)
    return np.stack([fit._clean_runs(c[:, j], min_run) for j in range(c.shape[1])], axis=1)


def _stature(edges, prior=fit.STATURE_PRIOR, measured=None):
    implied = {k: edges[k] / r for k, r in fit.SEGMENT_RATIO.items() if edges.get(k)}
    h = measured if measured is not None else (float(np.median(list(implied.values()))) if implied else None)
    return {"prior_m": list(prior), "measured_m": h, "posterior_m": h if h is not None else prior[0],
            "implied_m": implied}


def lengths_from_points(X35_down) -> dict:
    """Per-subject lengths (fit.estimate_lengths format) of (F,35,3) Y-down keypoints: per-edge medians over the
    frames where both ends are finite (with hundreds of frames the stature prior barely moves them)."""
    X = np.asarray(X35_down, float)
    return fit.estimate_lengths(X, np.isfinite(X).all(-1))


def lengths_for(source, cache, pid, stature_m=fit.STATURE_PRIOR[0]) -> dict:
    """{"edges", "soft", "stature"} for the mono fit (SP5 D4).
    "oracle": the full build's per-player lengths (cache meta `players_lengths[pid]` = {"edges", "soft", "stature"},
    fit.estimate_lengths' format, stored by build.py from v0.7.4 on), else (older caches) lengths_from_points on the
    cache's fitted X (Y-up -> Y-down). Flagged oracle: they come from the multi-camera build of the same player.
    "prior": the rest template kpset.REST35 scaled so its segment-implied stature is `stature_m` (1.70 m). Honest,
    but it carries the known H6 scale mismatch (the court world's bones imply ~1.45-1.5 m)."""
    if source == "oracle":
        stored = ((cache or {}).get("players_lengths") or {}).get(pid)
        if stored:
            return {"edges": dict(stored["edges"]), "soft": dict(stored["soft"]), "stature": dict(stored["stature"])}
        X = (cache or {}).get("extra", {}).get(pid, {}).get("X")
        if X is None:
            raise ValueError(f"lengths 'oracle': the cache has neither players_lengths nor a fitted X for {pid}")
        return lengths_from_points(np.asarray(X, float) * np.array([1.0, -1.0, 1.0]))
    if source == "prior":
        L = lengths_from_points(np.repeat(kpset.REST35[None] * np.array([1.0, -1.0, 1.0]), fit.CONTACT_MIN_RUN + 10, 0))
        s = stature_m / L["stature"]["measured_m"]
        edges = {k: v * s for k, v in L["edges"].items()}
        return {"edges": edges, "soft": {k: v * s for k, v in L["soft"].items()},
                "stature": dict(_stature(edges, measured=float(stature_m)), source="prior template")}
    raise ValueError(f"unknown lengths source {source!r} (expected one of {LENGTH_SOURCES} or a lengths dict)")


def load_prior(path, pid):
    """One player's learned mono prior from a prior .npz (module docstring), or None when the file lacks the player."""
    with np.load(path) as z:
        if f"{pid}_kp" not in z.files:
            return None
        c = z[f"{pid}_contact"] if f"{pid}_contact" in z.files else None
        return {"kp": z[f"{pid}_kp"].astype(int), "rel_cam": z[f"{pid}_rel_cam"].astype(float),
                "conf": z[f"{pid}_conf"].astype(float), "contact": c, "meta": json.loads(str(z["meta"]))}


def read_prior_meta(path) -> dict:
    """The prior file's `meta` (JSON) without loading its arrays."""
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def check_prior_meta(meta, build_id, view, path):
    """Refuse a prior made for another view or from another build of the clip (meta build_id != the debug cache's
    build_id: its 2D / timing may differ); a prior without build_id (older file) is accepted with a UserWarning."""
    if meta.get("view") not in (None, view):
        raise ValueError(f"mono prior {path} is for view {meta.get('view')!r}, not {view!r}")
    pb = meta.get("build_id")
    if pb is None:
        warnings.warn(f"mono prior {path} has no build_id in its meta (older file): not checked against the "
                      f"cache's build {build_id}", UserWarning, stacklevel=2)
    elif pb != build_id:
        raise ValueError(f"mono prior {path} was made from build_id {pb}, the pose24 cache is build_id {build_id}: "
                         "re-run the prior producer on the current cache")


def _interp(v, t):
    ok = np.isfinite(v).all(-1)
    if not ok.any():
        return None
    return np.stack([np.interp(t, t[ok], v[ok, c]) for c in range(v.shape[1])], axis=1)


def _points(uv, seen, cam, k, h, mask=None):
    m = seen[:, k] if mask is None else seen[:, k] & mask
    return np.where(m[:, None], ray_plane(np.nan_to_num(uv[:, k]), cam, h), np.nan)


def init_mono(uv, w, cam, lengths, sgn, contact, ankle_h=DEFAULT_ANKLE_H):
    """(F,35,2) + (F,35) one view -> (F,35,3) Y-down initial keypoints: the rest template (rest_offsets_down(sgn),
    scaled to the leg length) turned to the hip line's yaw and standing on the root floor point. Root = median of
    the planted heels' / toes' ray_plane points, else the ankles on the ankle plane (else the hips); yaw = right-hip
    -> left-hip point on the plane Y = -(thigh + shank + ankle_h). Gaps are interpolated."""
    F = len(w)
    t = np.arange(F)
    seen = _seen(uv, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feet = np.stack([_points(uv, seen, cam, k, FOOT_H[k], contact[:, j]) for j, k in enumerate(kpset.CONTACT)], 1)
        root = np.nanmedian(feet, axis=1)
        ank = np.nanmean(np.stack([_points(uv, seen, cam, k, ankle_h) for k in (15, 16)], 1), axis=1)
        root = np.where(np.isfinite(root), root, ank)
        E = lengths["edges"]
        hip_h = E.get("thigh", 0.42) + E.get("shank", 0.42) + ankle_h
        hipL, hipR = _points(uv, seen, cam, 11, hip_h), _points(uv, seen, cam, 12, hip_h)
        root = np.where(np.isfinite(root), root, 0.5 * (hipL + hipR))
    root = _interp(root[:, [0, 2]], t)
    root = np.zeros((F, 2)) if root is None else root
    D = kpset.rest_offsets_down(sgn)
    leg = np.linalg.norm(D[11] - D[13]) + np.linalg.norm(D[13] - D[15])
    D = D * ((E["thigh"] + E["shank"]) / leg if E.get("thigh") and E.get("shank") else 1.0)
    a = (hipL - hipR)[:, [0, 2]]
    a[~(np.linalg.norm(a, axis=-1) > 0.5 * np.linalg.norm(D[11] - D[12]))] = np.nan
    a = _interp(a / np.linalg.norm(a, axis=-1, keepdims=True), t)
    a0 = (D[11] - D[12])[[0, 2]]
    phi = np.zeros(F) if a is None else np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(a0[1], a0[0])
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    X = np.empty((F, kpset.K, 3))
    X[..., 0] = c * D[None, :, 0] - s * D[None, :, 2]            # proper rotation about the vertical axis
    X[..., 2] = s * D[None, :, 0] + c * D[None, :, 2]
    X[..., 1] = D[None, :, 1] - 0.02 - D[kpset.CONTACT, 1].mean()
    foot = X[:, kpset.CONTACT][..., [0, 2]].mean(1)
    X[..., [0, 2]] += (root - foot)[:, None]
    return X


def _ankle_h(cache):
    h = (cache or {}).get("ankle_h_m")
    return float(h) if h is not None and np.isfinite(h) else DEFAULT_ANKLE_H


def _foot_anchor(uv, w, cam, contact, ankle_h):
    """(F,35,3): planted heels / toes on their floor-height planes; NaN elsewhere."""
    A = np.full(uv.shape[:2] + (3,), np.nan)
    seen = _seen(uv, w)
    for j, k in enumerate(kpset.CONTACT):
        A[:, k] = _points(uv, seen, cam, k, _foot_h(k, ankle_h), contact[:, j])
    return A


def _depth_rows(prior, lengths, axis, F, log=None):
    """Prior -> depth rows over clip frames; rel_cam depths scaled by s = median(our thigh+shank / prior's)
    (1.0, logged, when the lengths or the prior's legs give no ratio)."""
    kp, rel, conf = np.asarray(prior["kp"], int), np.asarray(prior["rel_cam"], float), np.asarray(prior["conf"], float)
    if rel.shape[0] != F or len(conf) != F:
        raise ValueError(f"mono prior has {rel.shape[0]} frames, the clip {F}")
    col = {k: i for i, k in enumerate(kp)}
    s = None
    E = lengths["edges"]
    if E.get("thigh") and E.get("shank") and all(k in col for k in (11, 12, 13, 14, 15, 16)):
        leg = [np.linalg.norm(rel[:, col[h]] - rel[:, col[kn]], axis=-1) + np.linalg.norm(rel[:, col[kn]] - rel[:, col[a]], axis=-1)
               for h, kn, a in ((11, 13, 15), (12, 14, 16))]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (E["thigh"] + E["shank"]) / np.concatenate(leg)
        r = r[np.isfinite(r) & (r > 0)]
        s = float(np.median(r)) if len(r) else None
    if s is None:
        s = 1.0
        if log is not None:
            log("    mono prior: no leg-length ratio (lengths or prior legs missing): depth scale 1.0")
    tt, jj = np.nonzero((conf[:, None] >= PRIOR_CONF_MIN) & np.isfinite(rel[..., 2]))
    return {"axis": axis, "t": tt, "k": kp[jj], "d": s * rel[tt, jj, 2], "sigma": DEPTH_FIT_SIGMA, "scale": s}


def _slice_depth(depth, s, e):
    if depth is None:
        return None
    m = (depth["t"] >= s) & (depth["t"] < e)
    return dict(depth, t=depth["t"][m] - s, k=depth["k"][m], d=depth["d"][m])


def fit_mono(bundle, pid, view, cache=None, lengths="oracle", prior=None, passes=2, depth_sigma_m=None,
             log=print) -> dict:
    """One player's 1-view fit -> fit.fit_player's result keys (view axis of length 1) + "depth_axis" (unit camera
    optical axis R[2] in the court world) + "depth_sigma_m" (confidence only: PRIOR_DEPTH_SIGMA with a prior,
    NO_PRIOR_DEPTH_SIGMA without). lengths: "oracle" | "prior" | a lengths dict (passed through)."""
    if depth_sigma_m is not None and not depth_sigma_m > 0:
        raise ValueError(f"depth_sigma_m must be > 0, got {depth_sigma_m}")
    cams = bundle["cams"]
    cam = cams[view]
    P = pv.proj_mats(cams, [view])
    uv, w = kpset.player_obs(dict(bundle, views=[view], cams={view: cam}), pid)
    F = len(w)
    fps = float(bundle["fps"])
    sgn = kpset.world_sign({view: cam})
    L = lengths if isinstance(lengths, dict) else lengths_for(lengths, cache, pid)
    if not (L.get("stature") or {}).get("measured_m"):
        L = dict(L, stature=_stature(L["edges"]))
    ankle_h = _ankle_h(cache)
    c2d = contact_from_2d(uv[:, :, 0], w[:, :, 0], fps)
    if prior is not None and prior.get("contact") is not None:
        c2d = c2d | (np.nan_to_num(np.asarray(prior["contact"], float)) > 0.5)
    X0 = init_mono(uv[:, :, 0], w[:, :, 0], cam, L, sgn, c2d, ankle_h=ankle_h)
    axis = np.asarray(cam[1], float)[2]
    axis = axis / np.linalg.norm(axis)
    depth = _depth_rows(prior, L, axis, F, log=log) if prior is not None else None
    contact = c2d
    segments = fit.support_segments(w)
    edges = fit.edge_arrays(L)
    X = np.full_like(X0, np.nan)
    for p in range(passes):
        anchor = _foot_anchor(uv[:, :, 0], w[:, :, 0], cam, contact, ankle_h)
        for s, e in segments:
            init = X0[s:e] if p == 0 else X[s:e].copy()
            X[s:e] = fit._fit_segment(P, uv[s:e], w[s:e], init, edges, anchor[s:e], contact[s:e], sgn, fps, True,
                                      depth=_slice_depth(depth, s, e))
        if p < passes - 1:
            contact = c2d | fit.detect_contact(X, fps)
    r, inlier = fit._residuals(P, X, uv, w)
    ds = depth_sigma_m if depth_sigma_m is not None else (PRIOR_DEPTH_SIGMA if prior is not None
                                                          else NO_PRIOR_DEPTH_SIGMA)
    peak = {"raw": None, "fit": fit._p99_speed(X[:, [9, 10]], fps)}
    log(f"    {pid}: mono fit ({view}) {len(segments)} segment(s), p99 wrist speed {peak['fit']} m/s, "
        f"contact frames {int(contact.any(axis=1).sum())}, depth prior "
        + ("none" if depth is None else f"{len(depth['t'])} rows (scale {depth['scale']:.3f})"))
    return {"X": X, "uv": uv, "w": w, "resid": r, "inlier": inlier, "raw": np.full_like(X, np.nan),
            "contact": contact, "lengths": L, "sgn": sgn, "segments": segments, "peak_wrist_mps": peak,
            "depth_axis": axis, "depth_sigma_m": float(ds)}


def _view_rows(arr, col, F, fill):
    """A cache (V, F) per-view array -> the one view's (1, F) rows (fill when an older cache lacks it)."""
    return np.full((1, F), fill) if arr is None else np.asarray(arr)[[col]]


def _image_size(cache, view, full_doc_path):
    """[width, height] of `view` from the full cache's image_size, else the full doc's meta.cameras; None if neither."""
    wh = (cache.get("image_size") or {}).get(view)
    if wh and any(x is None for x in wh):                  # e.g. [None, None] from a cache without sizes: missing
        wh = None
    if not wh and os.path.exists(full_doc_path):
        with open(full_doc_path, "r", encoding="utf-8") as fh:
            cams = (json.load(fh).get("meta") or {}).get("cameras") or []
        wh = next(([c.get("width"), c.get("height")] for c in cams if c.get("name") == view), None)
    return [int(x) for x in wh] if wh and all(x is not None for x in wh) else None


def build_mono(clip, view, out_clip, lengths="oracle", prior_path=None, analysis_dir=None, cache_dir=None,
               skeleton="smpl-28", passes=2, log=print) -> str:
    """`pose24 build --mono <view>` -> the doc path. Reads `clip`'s own pose24 debug cache (its identity, sync and
    2D; D2), fits each player with fit_mono on the one view and writes, like build.build: the debug cache
    (pose24_cache/1, new build_id, timeline_of = clip), then removes stale racket sidecars / persons file of
    out_clip, then the doc last. lengths: "oracle" | "prior"; prior_path: a learned prior .npz (module docstring)."""
    from analysis.pose24 import build as pb
    from analysis.pose24 import config, track
    from analysis.pose24 import doc as docmod
    from analysis.pose24.camsubsets import oracle
    analysis_dir = analysis_dir or config.DEFAULT_ANALYSIS_DIR
    cache_dir = cache_dir or pv.CACHE_DIR
    if lengths not in LENGTH_SOURCES:
        raise ValueError(f"build --mono: lengths {lengths!r} not one of {LENGTH_SOURCES}")
    if not out_clip or out_clip == clip:
        raise ValueError(f"build --mono {view}: out_clip must name a new clip, not {clip!r}")
    t_all = time.perf_counter()
    times = {}
    src = track.cache_path(clip, cache_dir)
    if not os.path.exists(src):
        raise FileNotFoundError(f"build --mono {view}: no pose24 cache of {clip} at {src}; build {clip} first "
                                f"(`python -m analysis.pose24 build --clip {clip}`): the 1-camera build takes the "
                                "full build's identity and 2D (SP5 D2)")
    cache = docmod.load_debug_cache(src)
    b = oracle.slice_bundle(cache, [view])
    col, F, fps = cache["views"].index(view), int(cache["frame_count"]), float(cache["fps"])
    timeline_of = cache.get("timeline_of") or clip
    ex = cache["extra"]
    size = _image_size(cache, view, pb.output_path(analysis_dir, clip))
    b.update(offsets={view: cache["offsets"][view]}, cams_all={view: cache["cams"][view]},
             image_size={view: size} if size else {},
             det_index={p: _view_rows(ex.get(p, {}).get("det_index"), col, F, -1) for p in cache["obs"]},
             lr_swapped={p: _view_rows(ex.get(p, {}).get("lr_swapped"), col, F, False) for p in cache["obs"]})
    if prior_path:
        check_prior_meta(read_prior_meta(prior_path), cache.get("build_id"), view, prior_path)
    log(f"[pose24] {out_clip}: mono {view} of {clip} (lengths {lengths}"
        + (f", prior {os.path.basename(prior_path)}" if prior_path else ", no prior") + ")")
    method = None
    players, cache_arrays, pmeta, plengths = [], {}, {}, {}
    for pid in cache["obs"]:
        prior = load_prior(prior_path, pid) if prior_path else None
        if prior_path and prior is None:
            log(f"    {pid}: not in {prior_path}: fitted without a prior")
        if prior is not None:
            method = method or prior["meta"].get("method")
        t0 = time.perf_counter()
        fr = fit_mono(b, pid, view, cache=cache, lengths=lengths, prior=prior, passes=passes, log=log)
        times["fit"] = round(times.get("fit", 0.0) + time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        frames, arrays, meta = pb.player_outputs(b, pid, fr, skeleton, depth_axis=fr["depth_axis"],
                                                 depth_sigma_m=fr["depth_sigma_m"])
        times["regress_confidence"] = round(times.get("regress_confidence", 0.0) + time.perf_counter() - t0, 2)
        meta["depth_sigma_m"] = fr["depth_sigma_m"]
        pb.log_player(log, pid, meta)
        players.append((pid, frames))
        cache_arrays[pid] = arrays
        pmeta[pid] = meta
        plengths[pid] = pb.lengths_meta(fr)
    off = float(cache["offsets"][view])
    params = cache.get("score_params") or {}
    pose_variant = {"name": pb.VARIANT, "views": [view], "excluded_views": dict(cache.get("excluded") or {}),
                    "view_health_checked": cache.get("view_health_checked"),
                    "view_health": cache.get("view_health") or {}, "view_health_suspect": None,
                    "time_ref_view": cache["time_ref_view"], "timeline_of": timeline_of,
                    "offsets": {view: round(off, 2)}, "offsets_list": [{"view": view, "offset": round(off, 2)}],
                    "lr_swaps_per_view": {view: int(sum(b["lr_swapped"][p].sum() for p in cache["obs"]))},
                    "score_calibration": "meta.pose24.score_calibration", "base": None, "tracker": "oracle",
                    "reid": None}
    mono_meta = {"view": view, "lengths": lengths, "prior": method, "identity": IDENTITY}
    build_id = uuid.uuid4().hex
    p24 = pb.pose24_meta(build_id, skeleton, params, pmeta, mono=mono_meta, source_build_id=cache.get("build_id"))
    d = docmod.make_doc24(out_clip, fps, F, players, b, skeleton, pose_variant, p24)
    pb.finish_doc(d, fps, skeleton, times, t_all)
    cmeta = {"build_id": build_id, "out_clip": out_clip, "time_ref_view": cache["time_ref_view"],
             "offsets_all": cache.get("offsets_all") or cache["offsets"], "timeline_of": timeline_of,
             "view_health_checked": cache.get("view_health_checked"), "view_health": cache.get("view_health") or {},
             "shifts": cache.get("shifts") or {}, "score_params": params, "kpset_names": kpset.NAMES,
             "joint_names": list(d["skeleton"]["joint_names"]),
             "players_lengths": plengths, "mono": mono_meta, "source_build_id": cache.get("build_id"),
             "image_size": b["image_size"]}
    if cache.get("ankle_h_m") is not None:
        cmeta["ankle_h_m"] = cache["ankle_h_m"]
    from analysis.people import persons as personsmod
    out = pb.write_outputs(d, b, cache_arrays, cmeta, out_clip, analysis_dir, cache_dir,
                           lambda lg: pb.remove_persons(personsmod.path(analysis_dir, out_clip), lg), log)
    log(f"    wrote {out}; runtime " + ", ".join(f"{k} {v:.0f}s" for k, v in times.items()))
    return out
