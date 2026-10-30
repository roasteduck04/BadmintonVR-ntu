"""Round-5 experiment (test branch only): fit the raw-score map of wholebody at a smaller input (rtmpose.mode=balanced)
onto the wholebody-performance scale every live score gate uses, from two bench2d runs of the same frames.

    python tools/experiments/score_map_fit.py runs/campaign_in/lab_5090_r5/A01_perf/dets.npz \
        runs/campaign_in/lab_5090_r5/A02_balanced/dets.npz [--mode balanced]

Round 5 found the balanced streams invalid: pose2d_fast.score_to_wb passes every wholebody score through unchanged,
but RTMW-x at 256x192 scores ~5.5x lower (median 0.78 vs 4.30), so no keypoint passed a gate and no player was acquired.
Persons are matched per camera and frame by body centre (< 40 px); for each keypoint group (body 0-16, feet 17-22,
face 23-90, hands 91-132 - their score scales differ) 21 quantiles of the smaller model's scores are paired with
wholebody-performance's (the same quantile maps as pose2d_fast.SCORE_TO_WB, which only covers body / body_feet).
Written to tools/experiments/score_map_wb.json, where hooks.py reads it."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "score_map_wb.json")
GROUPS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91), "hands": (91, 133)}


def fit(ref_npz, small_npz, match_px=40.0):
    a, b = np.load(ref_npz), np.load(small_npz)
    xs = {g: [] for g in GROUPS}
    ys = {g: [] for g in GROUPS}
    n = 0
    for k in sorted(set(a.files) & set(b.files)):
        A, B = a[k].astype(np.float32), b[k].astype(np.float32)   # (persons, 133, 3) int16, score x 1000
        if A.ndim != 3 or not len(A) or not len(B):
            continue
        ca, cb = A[:, :17, :2].mean(1), B[:, :17, :2].mean(1)
        for i in range(len(B)):
            d = np.linalg.norm(ca - cb[i], axis=1)
            j = int(d.argmin())
            if d[j] < match_px:
                n += 1
                for g, (lo, hi) in GROUPS.items():
                    xs[g] += (B[i, lo:hi, 2] / 1000.0).tolist()
                    ys[g] += (A[j, lo:hi, 2] / 1000.0).tolist()
    q = np.linspace(0, 100, 21)
    out = {"persons": n, "groups": {}}
    for g, (lo, hi) in GROUPS.items():
        x, y = np.asarray(xs[g]), np.asarray(ys[g])
        out["groups"][g] = {"range": [lo, hi], "n": int(len(x)), "r": round(float(np.corrcoef(x, y)[0, 1]), 3),
                            "x": np.round(np.maximum.accumulate(np.percentile(x, q)), 4).tolist(),
                            "y": np.round(np.maximum.accumulate(np.percentile(y, q)), 4).tolist()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", help="dets.npz of the wholebody-performance bench2d run")
    ap.add_argument("small", help="dets.npz of the smaller-input bench2d run (same frames)")
    ap.add_argument("--mode", default="balanced", help="rtmpose.mode of the smaller run")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    m = fit(a.ref, a.small)
    doc = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    doc[f"wholebody/{a.mode}"] = {"ref": os.path.relpath(a.ref).replace("\\", "/"),
                                  "small": os.path.relpath(a.small).replace("\\", "/"), **m}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{m['persons']} matched persons -> {a.out}")
    for g, v in m["groups"].items():
        print(f"  {g:5s} n {v['n']:6d}  r {v['r']:.3f}  median {v['x'][10]:.3f} -> {v['y'][10]:.3f}")


if __name__ == "__main__":
    main()
