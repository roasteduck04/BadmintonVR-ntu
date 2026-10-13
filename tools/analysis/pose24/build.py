"""One-command pose24 build (spec § 2): config -> calibrated 2D -> track + sync -> L/R swap fix -> multi-view
fit -> anatomical regression -> confidence -> schema 4.1 doc + racket sidecars + debug cache.
`mono=<view>` (or a registry entry with "mono": true): the 1-camera build of mono.build_mono (SP5 D2)."""
from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np

from analysis import conventions
from analysis import pose_v2 as pv
from analysis import schema24
from analysis import skeleton24 as sk
from analysis.facing import fill_facing
from analysis.people import persons as personsmod
from analysis.pose24 import confidence, config, fit, kpset, lrswap, regress, scores, track
from analysis.pose24 import doc as docmod

VARIANT = "pose24"
FACE_IDX_28 = tuple(sk.SMPL28_NAMES.index(n) for n in ("left_eye", "right_eye", "nose_tip"))
FACING_CONF_MIN = 1e-3          # exp(-sigma/40 mm): far / 2-camera hips sit near 0.02, below facing's 0.1
COURT_MARK = (5.0, 8.0)          # |x|, |z| (m) beyond which a pelvis frame is flagged out of court (pose_v2's cut)
_N = {n: i for i, n in enumerate(sk.SMPL28_NAMES)}


def output_path(analysis_dir, out_clip):
    return os.path.join(analysis_dir, f"{out_clip}_{VARIANT}.analysis.json")


def _nanmedian(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


HEAD_FALLBACK_SOURCES = [0] + regress.SOURCES_28["neck"]     # nose + trunk: what regress's head-from-nose uses


def joint_sigma_conf(sig35, info):
    """(F,35) keypoint sigma + regress info -> SMPL-28 joint sigma (F,28) and confidence (F,28).
    Head-from-nose fallback frames take their sigma from the sources that rule uses (nose + trunk; the ears were
    never measured, so theirs sits at CAP) and halve the confidence (review 1 L1). Hands without knuckles get
    0.3 x the wrist's confidence."""
    sigJ = confidence.joint_sigma(sig35, regress.SOURCE_LIST)
    hfb = np.asarray(info["head_fallback"], bool)
    if hfb.any():
        sigJ[hfb, _N["head"]] = confidence.joint_sigma(sig35[hfb], [HEAD_FALLBACK_SOURCES])[:, 0]
    C = confidence.to_conf(sigJ)
    C[hfb, _N["head"]] *= 0.5
    for i, (hand, wrist) in enumerate((("left_hand", "left_wrist"), ("right_hand", "right_wrist"))):
        fb = info["hand_fallback"][:, i]
        C[fb, _N[hand]] = 0.3 * C[fb, _N[wrist]]
    return sigJ, C


def regress_player(bundle, fitres, kind, depth_axis=None, depth_sigma_m=None):
    """Fit result -> world joints + confidence (pure: no doc frames, no cache arrays)."""
    P = pv.proj_mats(bundle["cams"], bundle["views"])
    sig35, filled35, measured35 = confidence.keypoint_sigma(fitres, P, depth_axis=depth_axis,
                                                            depth_sigma_m=depth_sigma_m)
    Xup = fitres["X"] * conventions.Y_FLIP
    J, aux, info = regress.regress(Xup, observed=measured35.any(axis=0),
                                   default_sign=conventions.analysis_sign(bundle["cams"]))   # Y flip mirrors
    sigJ, C = joint_sigma_conf(sig35, info)
    filledJ = np.stack([filled35[:, s].all(axis=1) for s in regress.SOURCE_LIST], axis=1)
    aux_sig = sig35[:, regress.AUX_FIT]
    nj = len(sk.TYPES[kind][0])
    return {"J": J[:, :nj], "C": C[:, :nj], "aux": aux, "aux_sigma": aux_sig, "sigma": sigJ[:, :nj],
            "filled": filledJ[:, :nj], "measured35": measured35, "sig35": sig35, "info": info, "Xup": Xup,
            "J_all": J}


def player_outputs(bundle, pid, fitres, kind, depth_axis=None, depth_sigma_m=None):
    """Fit result -> (doc frames, debug-cache arrays, per-player meta)."""
    r = regress_player(bundle, fitres, kind, depth_axis=depth_axis, depth_sigma_m=depth_sigma_m)
    J, Xup, info, aux_sig = r["J_all"], r["Xup"], r["info"], r["aux_sigma"]
    frames = docmod.frames_from_world24(r["J"], r["C"], r["aux"], confidence.to_conf(aux_sig), r["filled"][:, 0],
                                        bundle["fps"])
    resid = np.where(fitres["w"] >= pv.W_MIN, fitres["resid"], np.nan).astype(np.float16)
    with np.errstate(invalid="ignore"):       # marked, never blanked (spec § 4)
        out_of_court = (np.abs(J[:, 0, 0]) > COURT_MARK[0]) | (np.abs(J[:, 0, 2]) > COURT_MARK[1])
    arrays = {"X": Xup.astype(np.float32), "inlier_mask": fitres["inlier"], "resid_px": resid,
              "out_of_court": out_of_court,
              "sigma_mm": r["sigma"].astype(np.float32), "filled": r["filled"],
              "aux_sigma_mm": aux_sig.astype(np.float32), "contact": fitres["contact"],
              "det_index": bundle["det_index"][pid], "lr_swapped": bundle["lr_swapped"][pid]}
    n = len(frames)
    n_ok = sum(fr["joints_flat"] is not None for fr in frames)
    n_meas = sum(fr["joints_flat"] is not None and not fr["interpolated"] for fr in frames)
    meta = {"coverage": n_ok / n, "measured": n_meas / n, "stature": fitres["lengths"]["stature"],
            "limb_lengths_m": fitres["lengths"]["edges"], "trunk_lengths_m": fitres["lengths"]["soft"],
            "peak_wrist_mps": fitres["peak_wrist_mps"], "contact_frames": int(fitres["contact"].any(axis=1).sum()),
            "head_fallback_frames": int(info["head_fallback"].sum()),
            "forward_sign": -1.0 if info["mirrored"] else 1.0,       # handedness the regressor read (facing uses it)
            "out_of_court_frames": int(out_of_court.sum()),
            "lr_swapped_frames": int(bundle["lr_swapped"][pid].sum()),
            "median_sigma_mm": _nanmedian(r["sigma"])}
    return frames, arrays, meta


def log_player(log, pid, meta):
    h, sg = meta["stature"]["measured_m"], meta["median_sigma_mm"]
    log(f"    {pid}: pose {100 * meta['coverage']:.1f}% (measured {100 * meta['measured']:.1f}%), "
        f"stature ~{'-' if h is None else f'{h:.2f}'} m, median sigma {'-' if sg is None else f'{sg:.0f}'} mm")


def lengths_meta(fitres) -> dict:
    """The debug cache's players_lengths[pid] (fit.estimate_lengths format): mono.lengths_for("oracle") reads it."""
    return {k: fitres["lengths"][k] for k in ("edges", "soft", "stature")}


def pose24_meta(build_id, skeleton, params, pmeta, **extra) -> dict:
    return dict({"build_id": build_id, "skeleton": skeleton, "regressor": regress.CONSTANTS, "fit": fit.settings(),
                 "confidence": confidence.settings(), "score_calibration": params,
                 "stature_assumption": {"prior_m": list(fit.STATURE_PRIOR), "segment_ratios": fit.SEGMENT_RATIO,
                                        "source": "Winter 2009 / de Leva 1996",
                                        "note": "scale bias H6 not corrected; estimated stature reported per player"},
                 "players": pmeta, "runtime_s": {}}, **extra)


def finish_doc(d, fps, skeleton, times, t_all):
    """Facing (with each player's handedness from meta.pose24.players[pid].forward_sign), runtime, schema check
    (raises on an invalid doc: nothing written yet)."""
    face = FACE_IDX_28 if skeleton == "smpl-28" else None
    pmeta = d["meta"]["pose24"]["players"]
    for p in d["players"]:
        fill_facing(p["frames"], fps, sign=pmeta[p["id"]]["forward_sign"], face=face, conf_min=FACING_CONF_MIN)
    times["total"] = round(time.perf_counter() - t_all, 2)
    d["meta"]["pose24"]["runtime_s"] = dict(times)
    errs = schema24.validate_analysis24(d)
    if errs:
        raise ValueError("pose24 doc invalid:\n  " + "\n  ".join(errs[:10]))


def write_outputs(d, bundle, cache_arrays, cache_meta, out_clip, analysis_dir, cache_dir, sidecars, log):
    """Write order: debug cache, then racket sidecars / persons file (`sidecars(log)`; stale racket sidecars are
    removed first), then the doc last (atomically): a failed build never leaves a new doc without its cache, and no
    sidecar from an earlier build sits next to it. A failure after the cache write can leave the old doc next to the
    new cache - readers compare meta.pose24.build_id."""
    cpath = track.cache_path(out_clip, cache_dir)
    docmod.save_debug_cache(cpath, bundle, cache_arrays, cache_meta)
    log(f"    cached {cpath}")
    for var in pv.RACKET_SIDECARS:
        stale = os.path.join(analysis_dir, f"{out_clip}_{VARIANT}.racket_{var}.json")
        if os.path.exists(stale):
            os.remove(stale)
            log(f"    removed old {stale}")
    sidecars(log)
    out = output_path(analysis_dir, out_clip)
    docmod.write_json(out, d)
    return out


def remove_persons(ppath, log):
    if os.path.exists(ppath):
        os.remove(ppath)                     # from an earlier --tracker 3d build: no longer pairs with this doc
        log(f"    removed old {ppath}")


def _registry_mono(clip, registry, out_clip, mono_opts):
    """A registry entry with "mono": true -> (base clip, view, out_clip, mono_opts), else None."""
    e = (registry or {}).get(clip) or {}
    if not e.get("mono"):
        return None
    if len(e.get("views") or ()) != 1 or "base" not in e:
        raise ValueError(f"registry mono entry {clip!r} needs a base and exactly 1 view: {e}")
    opts = {"lengths": e.get("mono_lengths", "oracle")}
    if e.get("mono_prior"):
        opts["prior_path"] = e["mono_prior"]
    return e["base"], e["views"][0], out_clip or clip, dict(opts, **(mono_opts or {}))


def build(clip, views=None, out_clip=None, timeline_of=None, skeleton="smpl-28",
          analysis_dir=config.DEFAULT_ANALYSIS_DIR, cache_dir=pv.CACHE_DIR, data_dir="data", registry=None,
          fit_opts=None, log=print, tracker="3d", reid="colour", mono=None, mono_opts=None) -> str:
    """mono_opts (with mono): mono.build_mono's lengths / prior_path / passes."""
    if mono is None:
        registry = registry if registry is not None else config.load_registry()
        rm = _registry_mono(clip, registry, out_clip, mono_opts)
        if rm is not None:
            clip, mono, out_clip, mono_opts = rm
    if mono is not None:
        if views is not None or timeline_of is not None:
            raise ValueError("build --mono takes the full build's views and timeline: no views / timeline_of")
        if not out_clip:
            raise ValueError(f"build --mono {mono} needs an out_clip (e.g. {clip}_1cam_<view>)")
        from analysis.pose24 import mono as monomod
        return monomod.build_mono(clip, mono, out_clip, analysis_dir=analysis_dir, cache_dir=cache_dir,
                                  skeleton=skeleton, log=log, **(mono_opts or {}))
    times = {}

    def lap(name, t0):
        times[name] = round(times.get(name, 0.0) + time.perf_counter() - t0, 2)
    t_all = time.perf_counter()
    cfg = config.resolve(clip, views=views, out_clip=out_clip, timeline_of=timeline_of, registry=registry,
                         data_dir=data_dir, analysis_dir=analysis_dir)
    out_clip = cfg["out_clip"]
    params = scores.load_params()
    timeline = track.load_timeline(cfg["timeline_of"], cache_dir) if cfg["timeline_of"] else None
    base = None
    if cfg["base_doc_path"] and timeline is None:
        with open(cfg["base_doc_path"], "r", encoding="utf-8") as fh:
            base = json.load(fh)
        cfg["frame_count"] = int(base["meta"]["frame_count"])
    log(f"[pose24] {out_clip}: {cfg['mode']}, views {', '.join(cfg['views'])}"
        + (f", timeline of {cfg['timeline_of']}" if timeline else ""))
    if base is None and cfg["base_doc_path"]:
        # e.g. doubles_2sides_4angle_2cam: its base / posev2 / posefit docs are on the old 2-view sync timeline
        log(f"    {os.path.basename(cfg['base_doc_path'])} exists but is on its own sync timeline, not the "
            f"--timeline-of {cfg['timeline_of']} one: racket / shuttle / rallies not carried")
    elif base is None:
        log("    no base doc on this timeline: racket / shuttle / rallies not carried")
    t0 = time.perf_counter()
    reid_feats = None
    if tracker == "3d" and reid == "osnet":        # only tracker3d uses re-ID features; v2 never needs the cache
        from analysis.people import osnet          # Task 13 (SP3): cached OSNet embeddings of the clip
        reid_feats = osnet.load_cache(cfg["clip"], data_dir=data_dir, log=log)
    bundle = track.track_and_sync24(cfg, params, timeline, log=log, tracker=tracker, reid=reid, reid_feats=reid_feats)
    lap("track", t0)
    if base is not None and not pv.same_timeline(base, bundle["time_ref_view"], log=log):
        base = None                          # review 1 L13: copied by frame index, so only on the same timeline
    t0 = time.perf_counter()
    swaps = {pid: lrswap.detect_swaps(bundle, pid) for pid in bundle["obs"]}
    counts = lrswap.apply_swaps(bundle, swaps)
    bundle["lr_swapped"] = swaps
    lap("lrswap", t0)
    log("    L/R swaps fixed (frames): " + ", ".join(f"{v} {c}" for v, c in counts.items()))
    fps, n = bundle["fps"], bundle["frame_count"]
    players, cache_arrays, pmeta, plengths = [], {}, {}, {}
    for pid in bundle["obs"]:
        t0 = time.perf_counter()
        fr = fit.fit_player(bundle, pid, log=log, **(fit_opts or {}))
        lap("fit", t0)
        t0 = time.perf_counter()
        frames, arrays, meta = player_outputs(bundle, pid, fr, skeleton)
        lap("regress_confidence", t0)
        log_player(log, pid, meta)
        players.append((pid, frames))
        cache_arrays[pid] = arrays
        pmeta[pid] = meta
        plengths[pid] = lengths_meta(fr)
    pose_variant = {"name": VARIANT, "views": bundle["views"], "excluded_views": bundle["excluded"],
                    "view_health_checked": bundle["view_health_checked"], "view_health": bundle["view_health"],
                    "view_health_suspect": bundle["view_health_suspect"], "time_ref_view": bundle["time_ref_view"], "timeline_of": bundle["timeline_of"],
                    "offsets": {v: round(float(o), 2) for v, o in bundle["offsets"].items()},
                    "offsets_list": [{"view": v, "offset": round(float(bundle["offsets"][v]), 2)} for v in bundle["views"]],
                    "lr_swaps_per_view": counts, "score_calibration": "meta.pose24.score_calibration",
                    "base": os.path.basename(cfg["base_doc_path"]) if base is not None else None,
                    "tracker": tracker, "reid": reid if tracker == "3d" else None}
    build_id = uuid.uuid4().hex         # also in the debug cache: a reader checks the pair belongs together
    p24 = pose24_meta(build_id, skeleton, params, pmeta)
    d = docmod.make_doc24(out_clip, fps, n, players, bundle, skeleton, pose_variant, p24)
    carry_match = None
    if base is not None:
        carry_match = docmod.carry_over24(base, d, log=log)      # per-frame, identity-safe (review 2 H1)
    finish_doc(d, fps, skeleton, times, t_all)
    cmeta = {"build_id": build_id, "out_clip": out_clip, "time_ref_view": bundle["time_ref_view"],
             "offsets_all": bundle["offsets_all"], "timeline_of": bundle["timeline_of"],
             "view_health_checked": bundle["view_health_checked"], "view_health": bundle["view_health"],
             "shifts": bundle["shifts"], "score_params": params, "kpset_names": kpset.NAMES,
             "joint_names": list(sk.TYPES[skeleton][0]), "players_lengths": plengths,
             "image_size": bundle.get("image_size") or {}}
    ankle_h = ((bundle.get("persons") or {}).get("stats") or {}).get("ankle_h_m")
    if ankle_h is not None:                  # tracker3d's estimate (persons stats); v2 builds have none (mono default)
        cmeta["ankle_h_m"] = ankle_h

    def sidecars(log):
        if base is not None:
            pv.write_racket_sidecars(analysis_dir, out_clip, f"{out_clip}_{VARIANT}", base, d, log=log,
                                     match=carry_match)
        ppath = personsmod.path(analysis_dir, out_clip)
        if tracker == "3d":
            bundle["persons"]["source_build_id"] = build_id
            personsmod.write(ppath, bundle["persons"])
            st = bundle["persons"]["stats"]
            log(f"    wrote {ppath} ({st['n_tracks']} tracks: player {st['n_player']}, nonplayer {st['n_nonplayer']}, "
                f"unknown {st['n_unknown']})")
        else:
            remove_persons(ppath, log)
    out = write_outputs(d, bundle, cache_arrays, cmeta, out_clip, analysis_dir, cache_dir, sidecars, log)
    log(f"    wrote {out}; runtime " + ", ".join(f"{k} {v:.0f}s" for k, v in times.items()))
    return out
