"""Offline post-processing time on a live run's section: the reference build's debug cache (already tracked + synced)
sliced to the section, then lrswap -> fit_player (2 passes) -> regress_player timed, at stride 1 (the source fps, as
built) and stride 2. Tracking + sync time comes from the doc's own meta.pose24.runtime_s (whole clip)."""
from __future__ import annotations

import argparse
import json
import os
import time

from analysis import pose_v2 as pv
from analysis.pose24 import build, fit, lrswap, track
from analysis.pose24 import doc as docmod


def slice_bundle(bundle, a, b, stride):
    out = dict(bundle)
    out["obs"] = {pid: {v: arr[a:b:stride].copy() for v, arr in views.items()} for pid, views in bundle["obs"].items()}
    out["frame_count"] = len(range(a, b, stride))
    out["fps"] = float(bundle["fps"]) / stride
    return out


def run(clip, ref_range, strides=(1, 2), cache_dir=pv.CACHE_DIR, analysis_dir=pv.DEFAULT_ANALYSIS_DIR,
        kind="smpl-28", log=print) -> dict:
    bundle = docmod.load_debug_cache(track.cache_path(clip, cache_dir))
    doc_path = build.output_path(analysis_dir, clip)
    rt = None
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as fh:
            rt = json.load(fh)["meta"]["pose24"].get("runtime_s")
    a, b = ref_range
    runs = {}
    for s in strides:
        sub = slice_bundle(bundle, a, b, s)
        t0 = time.perf_counter()
        swaps = {pid: lrswap.detect_swaps(sub, pid) for pid in sub["obs"]}
        lrswap.apply_swaps(sub, swaps)
        t1 = time.perf_counter()
        fits = {pid: fit.fit_player(sub, pid, log=lambda *x: None) for pid in sub["obs"]}
        t2 = time.perf_counter()
        for pid, fr in fits.items():
            build.regress_player(sub, fr, kind)
        t3 = time.perf_counter()
        n = sub["frame_count"]
        runs[str(s)] = {"frames": n, "fps": sub["fps"], "lrswap_s": t1 - t0, "fit_s": t2 - t1, "regress_s": t3 - t2,
                        "total_s": t3 - t0, "per_frame_ms": 1000 * (t3 - t0) / n}
        log(f"    stride {s}: {n} frames, lrswap {t1 - t0:.1f}s fit {t2 - t1:.1f}s regress {t3 - t2:.1f}s")
    return {"clip": clip, "section": [a, b], "doc_runtime_s": rt, "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.live.offline_timing")
    ap.add_argument("--clip", required=True, help="a pose24 build with a debug cache (data/pose_cache)")
    ap.add_argument("--section", required=True, help="first:stop source frames, e.g. 390:990")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.section.split(":"))
    res = run(a.clip, (lo, hi))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
