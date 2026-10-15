"""Oracle camera subsets (spec § 3): slice views out of a full build's pose24 debug cache (identity, sync and
view-health verdicts from the full build) and refit with the unchanged SP1 fit. Results -> compact .npz.
A job's method is a label: "fit" (multi-view SP1 fit, >= 2 views) or "mono-*" (1-view mono fit).
Per player: coverage / measured / full_body as metrics.coverage_stats (pelvis-based, R16; score.py recomputes them
from J / inlier); stature_m None for mono (its lengths are an input). Per result: runtime_s (wall) and cpu_s
(time.process_time of the job, R18); the stored meta also carries the pool's `workers`.
Learned mono prior (v0.7.5): a "mono-len-<src>+<prior>" job's mono_opts carry "prior_path" (one file) or, for a
run_fits batch, "prior_paths" {view: file} (job_mono_opts picks the job's); fit_subset loads it per player in the
worker (mono.load_prior; a player missing from the file is fitted without one) after mono.check_prior_meta. The
freshness key of a prior job adds the file's sha1 (not its path); no-prior keys are unchanged."""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
try:                                                       # optional (not a declared dependency)
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from analysis.pose24 import build as pb
from analysis.pose24 import confidence, fit, regress
from analysis.pose24 import doc as docmod
from analysis.pose24.camsubsets import metrics
from analysis.pose24.camsubsets import subsets as cs

DEFAULT_OUT = os.path.join("data", "pose_eval", "subsets")
# Bump when the code that turns a fit into stored J / C / sigma changes without a constant changing (it is part of
# the freshness key). 1 = v0.7.5: head-fallback sigma (review 1 L1), J NaN where the doc's conf rounds to 0 (L3).
RESULT_REV = 1
PER_PLAYER = ("J", "C", "X", "sigma", "inlier", "contact", "lr_swapped")
_CACHE = None
_LIMITS = None                                             # keeps a worker's BLAS limit alive
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def slice_bundle(cache, views):
    views = sorted(views)
    bad = [v for v in views if v not in cache["views"]]
    if bad:
        raise ValueError(f"views {bad} not in the full build ({', '.join(cache['views'])})")
    return dict(cache, views=views, cams={v: cache["cams"][v] for v in views}, excluded=dict(cache.get("excluded") or {}))


def _vcols(cache, views):
    return [cache["views"].index(v) for v in views]


def _is_mono(method):
    if method == "fit":
        return False
    if isinstance(method, str) and method.startswith("mono"):
        return True
    raise ValueError(f"unknown method {method!r} (expected 'fit' or 'mono-*')")


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_mono_opts(mono_opts, views):
    """A job's mono_opts: "prior_paths" {view: file} -> "prior_path" of the job's (1-view) subset (none when the
    view has no entry; run_fits then skips the job)."""
    if not mono_opts or "prior_paths" not in mono_opts:
        return mono_opts
    o = {k: v for k, v in mono_opts.items() if k != "prior_paths"}
    p = mono_opts["prior_paths"].get(tuple(views)[0]) if len(tuple(views)) == 1 else None
    if p:
        o["prior_path"] = p
    return o


def doc_parity_joints(J, C):
    """J with NaN where the doc would write confidence 0 (it rounds conf to DOC_DECIMALS; e2e.doc_joints then reads
    J as NaN): oracle and e2e rows of the same fit see the same joints (review 1 L3)."""
    J = np.array(J, dtype=float, copy=True)
    J[np.round(np.nan_to_num(np.asarray(C, float)), docmod.DOC_DECIMALS) == 0] = np.nan
    return J


def fit_subset(cache, views, method="fit", fit_opts=None, mono_opts=None, skeleton="smpl-28"):
    t0, c0 = time.perf_counter(), time.process_time()
    views = sorted(views)
    mono = _is_mono(method)
    if not mono and len(views) < 2:
        raise ValueError("method 'fit' needs >= 2 views (1 view: a 'mono-*' method)")
    if mono and len(views) != 1:
        raise ValueError(f"method {method!r} takes exactly 1 view")
    b = slice_bundle(cache, views)
    cols = _vcols(cache, views)
    mono_opts = dict(mono_opts or {})
    prior_path = mono_opts.pop("prior_path", None)
    if prior_path:
        from analysis.pose24 import mono as monomod
        monomod.check_prior_meta(monomod.read_prior_meta(prior_path), cache.get("build_id"), views[0], prior_path)
    out = {}
    for pid in cache["obs"]:
        if not mono:
            fr = fit.fit_player(b, pid, log=lambda *a: None, **(fit_opts or {}))
            r = pb.regress_player(b, fr, skeleton)
        else:
            from analysis.pose24 import mono as monomod
            prior = monomod.load_prior(prior_path, pid) if prior_path else None
            fr = monomod.fit_mono(b, pid, views[0], cache=cache, prior=prior, log=lambda *a: None, **mono_opts)
            r = pb.regress_player(b, fr, skeleton, depth_axis=fr["depth_axis"], depth_sigma_m=fr["depth_sigma_m"])
        r = dict(r, J=doc_parity_joints(r["J"], r["C"]))
        cov, meas, full = metrics.coverage_stats(r["J"], fr["inlier"], 1 if mono else 2)
        lr = cache["extra"].get(pid, {}).get("lr_swapped")
        out[pid] = {"J": r["J"].astype(np.float32), "C": r["C"].astype(np.float32),
                    "X": r["Xup"].astype(np.float32), "sigma": r["sigma"].astype(np.float32),
                    "inlier": np.asarray(fr["inlier"], bool), "contact": np.asarray(fr["contact"], bool),
                    "lr_swapped": (np.asarray(lr, bool)[cols].T if lr is not None     # cache: (V, F)
                                   else np.zeros((len(r["J"]), len(views)), bool)),
                    "coverage": cov, "measured": meas, "full_body": full,
                    "stature_m": None if mono else fr["lengths"]["stature"].get("measured_m"),
                    "limb_lengths_m": fr["lengths"]["edges"]}
    return {"players": out, "views": views, "method": method, "runtime_s": round(time.perf_counter() - t0, 2),
            "cpu_s": round(time.process_time() - c0, 2)}


def result_path(out_dir, clip, mode, method, subset):
    return os.path.join(out_dir, clip, mode, method, f"{cs.key(subset)}.npz")


def expected_meta(cache, method, fit_opts, mono_opts, skeleton="smpl-28"):
    """The freshness key: everything a stored J / C / sigma depends on - the fit settings and options, the
    regressor constants, the confidence model, the skeleton and RESULT_REV (review 1 M3). Mono methods also carry
    mono.settings() (M4) and, with a learned prior, the prior file's sha1."""
    mo = {k: v for k, v in (mono_opts or {}).items() if k not in ("prior_path", "prior_paths")}
    st = {"fit": fit.settings(), "fit_opts": fit_opts or {}, "mono_opts": mo, "regressor": regress.CONSTANTS,
          "confidence": confidence.settings(), "skeleton": skeleton, "rev": RESULT_REV}
    if _is_mono(method):
        from analysis.pose24 import mono as monomod
        st["mono"] = monomod.settings()
        p = (mono_opts or {}).get("prior_path")
        if p:                                              # prior jobs only: no-prior keys stay byte-identical
            st["prior"] = {"sha1": file_sha1(p) if os.path.exists(p) else None}
    return {"build_id": cache.get("build_id"), "method": method, "fit_settings": docmod.json_safe(st)}


def save_result(path, res, meta):
    """Atomic: a reader never sees a half-written result (temp file + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrs, scal = {}, {}
    for pid, p in res["players"].items():
        for k in PER_PLAYER:
            arrs[f"{pid}__{k}"] = np.asarray(p[k])
        scal[pid] = {k: p.get(k) for k in ("coverage", "measured", "full_body", "stature_m", "limb_lengths_m")}
    m = dict(meta, method=res["method"], views=res["views"], runtime_s=res["runtime_s"], players=scal)
    if res.get("cpu_s") is not None:
        m["cpu_s"] = res["cpu_s"]
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, meta=json.dumps(docmod.json_safe(m), allow_nan=False), **arrs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_result(path, expect_meta):
    """The stored result, or None when missing, corrupt, or stale vs `expect_meta` (None: no freshness check)."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            if expect_meta is not None and any(meta.get(k) != expect_meta.get(k)
                                               for k in ("build_id", "method", "fit_settings")):
                return None
            players = {pid: dict(sc, **{k: z[f"{pid}__{k}"] for k in PER_PLAYER})
                       for pid, sc in meta["players"].items()}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    return {"players": players, "views": meta["views"], "method": meta["method"], "runtime_s": meta["runtime_s"],
            "meta": meta}


def _init_worker(cache_path):
    """Pool worker: 1 BLAS thread (numpy may already be imported, so env vars alone can be too late)."""
    global _CACHE, _LIMITS
    if threadpool_limits is not None:                      # else the env vars run_fits sets suffice
        _LIMITS = threadpool_limits(1)
    _CACHE = docmod.load_debug_cache(cache_path)


def _job(method, subset, path, meta, fit_opts, mono_opts, cache=None):
    res = fit_subset(_CACHE if cache is None else cache, subset, method=method, fit_opts=fit_opts,
                     mono_opts=mono_opts)
    save_result(path, res, meta)
    return res["runtime_s"]


def run_fits(clip, jobs, cache_path, out_dir=DEFAULT_OUT, mode="oracle", workers=6, fit_opts=None, mono_opts=None,
             log=print):
    """Fit every (method, subset) job not already stored fresh; returns {subset key: result path}."""
    cache = docmod.load_debug_cache(cache_path)
    done, todo = {}, []
    skipped = 0
    for method, subset in jobs:
        path = result_path(out_dir, clip, mode, method, subset)
        mo = job_mono_opts(mono_opts, subset)
        if mono_opts and "prior_paths" in mono_opts and not (mo.get("prior_path")
                                                              and os.path.exists(mo["prior_path"])):
            log(f"    {method} {cs.key(subset)}: skipped: no prior ({mo.get('prior_path') or 'no path'})")
            skipped += 1
            continue
        meta = expected_meta(cache, method, fit_opts, mo)
        if load_result(path, meta) is not None:
            done[cs.key(subset)] = path
        else:                                              # stored meta: + the competing worker count (R18)
            todo.append((method, tuple(sorted(subset)), path, dict(meta, workers=max(1, int(workers))), mo))
    log(f"[camsubsets] {clip} {mode}: {len(done)} fresh, {len(todo)} to fit ({workers} workers)"
        + (f", {skipped} skipped (no prior file)" if skipped else ""))
    if not todo:
        return done
    if workers <= 1:
        for method, subset, path, meta, mo in todo:
            try:
                s = _job(method, subset, path, meta, fit_opts, mo, cache=cache)
                done[cs.key(subset)] = path
                log(f"    {method} {cs.key(subset)}: {s:.0f}s")
            except Exception:
                log(f"    {method} {cs.key(subset)}: FAILED\n{traceback.format_exc()}")
        return done
    # spawned children (Windows) get the parent's sys.path from multiprocessing, so `analysis` imports there too;
    # the thread env vars are set before they start (they read them when importing numpy), then restored
    old = {k: os.environ.get(k) for k in _THREAD_ENV}
    os.environ.update({k: "1" for k in _THREAD_ENV})
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cache_path,)) as ex:
            futs = {ex.submit(_job, m, s, p, meta, fit_opts, mo): (m, s, p) for m, s, p, meta, mo in todo}
            for f in as_completed(futs):
                m, s, p = futs[f]
                try:
                    log(f"    {m} {cs.key(s)}: {f.result():.0f}s")
                    done[cs.key(s)] = p
                except Exception:
                    log(f"    {m} {cs.key(s)}: FAILED\n{traceback.format_exc()}")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return done
