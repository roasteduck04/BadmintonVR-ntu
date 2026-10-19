"""v0.8.1: fit cost per run folder from timing.jsonl window events (solve time, solver evaluations, ms per new frame)
next to the report's quality numbers. Run from the repository root:

    python tools/live_bench/fit_stats.py runs/v081_fit/* [--out runs/v081_fit/fit_stats.md]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def window_stats(run):
    rows = []
    with open(os.path.join(run, "timing.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            if '"window"' not in line:
                continue
            e = json.loads(line)
            if e.get("ev") != "window":
                continue
            new = max(1, int(e["commit_stop"]) - int(e["start"]))
            for pid, ns in (e.get("solve_ns") or {}).items():
                rows.append({"ms": ns / 1e6, "per_frame_ms": ns / 1e6 / new, "nfev": (e.get("nfev") or {}).get(pid),
                             "status": (e.get("status") or {}).get(pid)})
    if not rows:
        return None
    ms = np.array([r["ms"] for r in rows])
    pf = np.array([r["per_frame_ms"] for r in rows])
    nf = np.array([r["nfev"] for r in rows if r["nfev"] is not None], float)
    return {"solves": len(rows), "solve_ms_p50": float(np.median(ms)), "solve_ms_mean": float(ms.mean()),
            "ms_per_new_frame_p50": float(np.median(pf)), "nfev_mean": float(nf.mean()) if len(nf) else None,
            "nfev_at_cap": float((nf >= nf.max()).mean()) if len(nf) else None}


def quality(run):
    p = os.path.join(run, "report.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        r = json.load(fh)
    return r


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    L = ["| run | solves | solve ms p50 | ms per new frame p50 | evaluations mean | at the cap | wall s |", "|---|---|---|---|---|---|---|"]
    for run in a.runs:
        if not os.path.isdir(run):
            continue
        s = window_stats(run)
        summ = {}
        sp = os.path.join(run, "run_summary.json")
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as fh:
                summ = json.load(fh)
        if s is None:
            continue
        wall = summ.get("loop_s") or summ.get("wall_s")
        L.append(f"| {os.path.basename(os.path.normpath(run))} | {s['solves']} | {s['solve_ms_p50']:.0f} | "
                 f"{s['ms_per_new_frame_p50']:.1f} | {s['nfev_mean'] if s['nfev_mean'] is None else round(s['nfev_mean'], 1)} | "
                 f"{'–' if s['nfev_at_cap'] is None else f'{100 * s['nfev_at_cap']:.0f} %'} | "
                 f"{'–' if wall is None else f'{wall:.1f}'} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
