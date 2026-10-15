"""End-to-end camera subsets (spec § 8): a real `pose24 build` on the subset views (own sync + own tracking, no
--timeline-of) into scratch dirs, re-timed onto the full build's timeline, players matched to the reference by
pelvis position; stored like the oracle results (mode "e2e", method "fit"). The scratch builds have no base doc:
the rallies (swing windows) come from the full doc at score time. Freshness covers the full build's build_id +
fit settings only: a change to the e2e pipeline code (build, sync, tracking, this module) needs a manual re-run."""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis.pose24 import build as pb
from analysis.pose24 import doc as docmod
from analysis.pose24 import track
from analysis.people import persons as personsmod
from analysis.pose24.camsubsets import metrics, oracle, retime
from analysis.pose24.camsubsets import subsets as cs

HIPS = (11, 12)                                            # kpset left_hip / right_hip: pelvis = their midpoint
_FLOAT = ("J", "C", "X", "sigma")
_BOOL = ("inlier", "contact", "lr_swapped")


def scratch_dirs(out_dir, clip):
    root = os.path.join(out_dir, clip, "e2e", "_builds")
    return os.path.join(root, "analysis"), os.path.join(root, "cache")


def _abbr(v):
    try:
        return cs.abbr(v)
    except ValueError:                                     # e.g. synthetic views "a", "b": the name itself
        return v


def out_clip(clip, subset):
    return f"{clip}_e2e_{'_'.join(_abbr(v) for v in sorted(subset))}"


def build_one(clip, subset, out_dir, registry=None, data_dir="data", log=print) -> str:
    adir, cdir = scratch_dirs(out_dir, clip)
    os.makedirs(adir, exist_ok=True)
    return pb.build(clip, views=sorted(subset), out_clip=out_clip(clip, subset), timeline_of=None, analysis_dir=adir,
                    cache_dir=cdir, data_dir=data_dir, registry=registry, log=log)


def _cleanup(clip, subset, out_dir):
    """This subset's scratch files only (doc, cache, persons, racket sidecars, partial temp files)."""
    adir, cdir = scratch_dirs(out_dir, clip)
    oc = out_clip(clip, subset)
    files = [pb.output_path(adir, oc), track.cache_path(oc, cdir), personsmod.path(adir, oc)]
    files += glob.glob(os.path.join(adir, f"{oc}_{pb.VARIANT}.*")) + glob.glob(os.path.join(cdir, f"{oc}{track.CACHE_SUFFIX}.*"))
    for f in set(files):
        if os.path.exists(f):
            os.remove(f)


def _remove_empty_scratch(clip, out_dir):
    adir, cdir = scratch_dirs(out_dir, clip)
    for d in (adir, cdir, os.path.dirname(adir)):          # only when empty (keep_builds / a stray file stays)
        try:
            os.rmdir(d)
        except OSError:
            pass


def match_by_pelvis(pel_sub_by_pid, pel_ref_by_pid, max_m=1.0, min_frames=30):
    subs, refs = list(pel_sub_by_pid), list(pel_ref_by_pid)
    cost = np.full((len(subs), len(refs)), 1e6)
    for i, s in enumerate(subs):
        for j, r in enumerate(refs):
            a, b = pel_sub_by_pid[s], pel_ref_by_pid[r]
            n = min(len(a), len(b))
            d = np.linalg.norm((a[:n] - b[:n])[:, [0, 2]], axis=-1)
            d = d[np.isfinite(d)]
            if len(d) >= min_frames:
                cost[i, j] = float(np.median(d))
    out = {}
    if cost.size:
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= max_m:
                out[subs[i]] = refs[j]
    return out


def _pelvis(X):
    return np.asarray(X, float)[:, list(HIPS)].mean(axis=1)


def _shift(full_cache, e2e_cache):
    shift, spread = retime.frame_shift(full_cache["offsets"], e2e_cache["offsets"])
    if shift is None:
        raise ValueError(f"e2e build has no shared view offsets with the full build "
                         f"(full {sorted(full_cache['offsets'])}, e2e {sorted(e2e_cache['offsets'])}): cannot re-time")
    return shift, spread


def doc_joints(doc, pid, names):
    """(J (F, n, 3) Y-up world, C (F, n)) of named joints from the doc; NaN where the frame or the joint is absent
    and J NaN where C == 0 exactly (the doc's never-seen placeholders and confs that round to 0; the oracle drops
    the same joints, oracle.doc_parity_joints, review 1 L3;
    joints with C > 0 are kept without a confidence threshold, R11)."""
    jn = doc["skeleton"]["joint_names"]
    p = next(p for p in doc["players"] if p["id"] == pid)
    idx = [(i, jn.index(n)) for i, n in enumerate(names) if n in jn]
    J = np.full((len(p["frames"]), len(names), 3), np.nan)
    C = np.full((len(p["frames"]), len(names)), np.nan)
    for f, fr in enumerate(p["frames"]):
        pos = fr.get("position")
        if pos is None or not fr.get("joints_flat"):
            continue
        a = np.asarray(fr["joints_flat"], float).reshape(-1, 4)
        off = np.array([pos[0], 0.0, pos[1]])
        for i, k in idx:
            C[f, i] = a[k, 3]
            if a[k, 3] != 0:
                J[f, i] = a[k, :3] + off
    return J, C


def to_result(doc_path, cache_path, full_cache, skeleton_names=None):
    """(result like oracle.fit_subset's, e2e_meta {"shift", "spread", "match", "identity", "build_id",
    "runtime_s" (the e2e doc's per-step seconds)}); pids = e2e ids."""
    with open(doc_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    sub = docmod.load_debug_cache(cache_path)
    shift, spread = _shift(full_cache, sub)
    n = int(full_cache["frame_count"])
    names = list(skeleton_names or doc["skeleton"]["joint_names"])
    views = list(sub["views"])
    out = {}
    for pid in sub["pids"]:
        ex = sub["extra"][pid]
        J, C = doc_joints(doc, pid, names)
        lr = ex.get("lr_swapped")
        src = {"J": J, "C": C, "X": ex["X"], "sigma": ex["sigma_mm"],
               "inlier": ex["inlier_mask"], "contact": ex["contact"],
               "lr_swapped": (np.asarray(lr).T if lr is not None                          # cache: (V, F) (R6)
                              else np.zeros((len(C), len(views))))}
        p = {k: retime.resample(src[k], shift, n).astype(np.float32) for k in _FLOAT}
        with np.errstate(invalid="ignore"):
            for k in _BOOL:
                p[k] = retime.resample(np.asarray(src[k], float), shift, n) > 0.5
        cov, meas, full = metrics.coverage_stats(p["J"], p["inlier"])             # the oracle definitions (R16)
        st = (doc["meta"]["pose24"]["players"].get(pid) or {}).get("stature") or {}
        p.update(coverage=cov, measured=meas, full_body=full, stature_m=st.get("measured_m"),
                 limb_lengths_m=doc["meta"]["pose24"]["players"].get(pid, {}).get("limb_lengths_m"))
        out[pid] = p
    pel_sub = {pid: _pelvis(p["X"]) for pid, p in out.items()}
    pel_ref = {pid: _pelvis(full_cache["extra"][pid]["X"]) for pid in full_cache["pids"]}
    match = match_by_pelvis(pel_sub, pel_ref)
    ident = {r: metrics.identity_proxy(pel_sub[s], pel_ref, r) for s, r in match.items()}
    res = {"players": out, "views": views, "method": "fit",
           "runtime_s": float((doc["meta"]["pose24"].get("runtime_s") or {}).get("total", 0.0))}
    meta = {"shift": shift, "spread": spread, "match": match, "identity": ident,
            "build_id": doc["meta"]["pose24"]["build_id"],
            "runtime_s": dict(doc["meta"]["pose24"].get("runtime_s") or {})}
    return res, meta


def expected_meta(full_cache):
    return oracle.expected_meta(full_cache, "fit", None, None)


def _build_and_convert(clip, subset, out_dir, full_cache_path, registry, data_dir, keep_builds=False, log=None,
                       full_cache=None, workers=1):
    """Build + convert + save one subset; the stored meta carries cpu_s (process time: the build's CPU seconds
    under `workers` competing single-BLAS-thread workers, R18) and workers."""
    t0, c0 = time.perf_counter(), time.process_time()
    full = full_cache if full_cache is not None else docmod.load_debug_cache(full_cache_path)
    subset = tuple(sorted(subset))
    try:
        doc_path = build_one(clip, subset, out_dir, registry=registry, data_dir=data_dir,
                             log=log or (lambda *a: None))
        res, em = to_result(doc_path, track.cache_path(out_clip(clip, subset), scratch_dirs(out_dir, clip)[1]), full)
        res["runtime_s"] = round(time.perf_counter() - t0, 2)
        e2e_id, e2e_rt = em.pop("build_id"), em.pop("runtime_s")
        res["cpu_s"] = round(time.process_time() - c0, 2)
        meta = dict(expected_meta(full), e2e=em, e2e_build_id=e2e_id, ref_build_id=full.get("build_id"),
                    e2e_runtime_s=e2e_rt,                  # the e2e doc's track / fit / ... seconds (cost table)
                    cpu_s=res["cpu_s"], workers=workers)
        oracle.save_result(oracle.result_path(out_dir, clip, "e2e", "fit", subset), res, meta)
    finally:                                               # also a failed build's partial files
        if not keep_builds:
            try:
                _cleanup(clip, subset, out_dir)
            except Exception as e:                         # never turns a saved result into FAILED
                (log or print)(f"    e2e {cs.key(subset)}: scratch cleanup failed ({e}); files left in "
                               f"{os.path.dirname(scratch_dirs(out_dir, clip)[0])}")
    return res["runtime_s"]


def run_e2e(clip, subsets, full_cache_path, out_dir=oracle.DEFAULT_OUT, workers=3, registry=None, data_dir="data",
            log=print, keep_builds=False) -> dict:
    """Build + convert every subset not already stored fresh; returns {subset key: result path}. A failed build is
    logged and skipped."""
    full = docmod.load_debug_cache(full_cache_path)
    want = expected_meta(full)
    done, todo = {}, []
    for s in subsets:
        s = tuple(sorted(s))
        path = oracle.result_path(out_dir, clip, "e2e", "fit", s)
        if oracle.load_result(path, want) is not None:
            done[cs.key(s)] = path
        else:
            todo.append((s, path))
    log(f"[camsubsets] {clip} e2e: {len(done)} fresh, {len(todo)} to build ({workers} workers)")
    if not todo:
        return done
    if workers <= 1:
        for s, path in todo:
            try:
                sec = _build_and_convert(clip, s, out_dir, full_cache_path, registry, data_dir, keep_builds,
                                         full_cache=full, workers=1, log=None)
                done[cs.key(s)] = path
                log(f"    e2e {cs.key(s)}: {sec:.0f}s")
            except Exception:
                log(f"    e2e {cs.key(s)}: FAILED\n{traceback.format_exc()}")
        _remove_empty_scratch(clip, out_dir)
        return done
    old = {k: os.environ.get(k) for k in oracle._THREAD_ENV}           # as oracle.run_fits: 1 BLAS thread per worker
    os.environ.update({k: "1" for k in oracle._THREAD_ENV})
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(_build_and_convert, clip, s, out_dir, full_cache_path, registry, data_dir,
                              keep_builds, workers=workers): (s, p) for s, p in todo}
            for f in as_completed(futs):
                s, p = futs[f]
                try:
                    log(f"    e2e {cs.key(s)}: {f.result():.0f}s")
                    done[cs.key(s)] = p
                except Exception:
                    log(f"    e2e {cs.key(s)}: FAILED\n{traceback.format_exc()}")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _remove_empty_scratch(clip, out_dir)
    return done


_LIMITS = None


def _init_worker():
    global _LIMITS
    if oracle.threadpool_limits is not None:
        _LIMITS = oracle.threadpool_limits(1)
