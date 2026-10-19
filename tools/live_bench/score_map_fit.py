#!/usr/bin/env python
"""Fit the raw-score maps of the smaller rtmlib models onto the wholebody-performance scale (pose2d_fast.SCORE_TO_WB).
Runs wholebody-performance and each smaller model on the same frames, matches persons by body centre, and prints
21 quantiles of each model's body-17 raw scores against wholebody's (JSON). GPU PC, from the repository root:
    python tools/live_bench/score_map_fit.py "data/15-08/singles/0.5/*back 0.5-1.mov" ...
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(patterns, frames=range(420, 1000, 40), match_px=40.0):
    from analysis.live.pose2d_fast import check_cuda
    check_cuda("cuda")                   # preloads the pip CUDA / cuDNN libraries
    from rtmlib import Body, BodyWithFeet, Wholebody
    imgs = []
    for pat in patterns:
        cap = cv2.VideoCapture(sorted(glob.glob(pat))[0])
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                imgs.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    ref = [Wholebody(mode="performance", backend="onnxruntime", device="cuda")(im) for im in imgs]
    out = {}
    for name, cls in (("body_feet", BodyWithFeet), ("body", Body)):
        for mode in ("balanced", "performance"):
            sol = cls(mode=mode, backend="onnxruntime", device="cuda")
            xs, ys = [], []
            for (rk, rs), im in zip(ref, imgs):
                gk, gs = sol(im)
                rk, rs, gk, gs = map(np.asarray, (rk, rs, gk, gs))
                if len(rk) == 0 or len(gk) == 0:
                    continue
                rc, gc = rk[:, :17].mean(axis=1), gk[:, :17].mean(axis=1)
                for i in range(len(gk)):
                    d = np.linalg.norm(rc - gc[i], axis=1)
                    j = int(d.argmin())
                    if d[j] < match_px:
                        xs += gs[i, :17].tolist()
                        ys += rs[j, :17].tolist()
            q = np.linspace(0, 100, 21)
            out[f"{name}/{mode}"] = {"n": len(xs), "x": np.round(np.percentile(xs, q), 4).tolist(),
                                     "y": np.round(np.percentile(ys, q), 4).tolist(),
                                     "corr": round(float(np.corrcoef(xs, ys)[0, 1]), 3)}
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:] or [f"data/15-08/singles/0.5/*{v} 0.5-1.mov" for v in ("back", "back right", "front left")])
