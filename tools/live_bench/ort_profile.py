"""v0.8.1: time the rtmlib detector and pose model alone with onnxruntime, and count which execution provider runs
each node (ORT profiling), for several session options.

    python tools/live_bench/ort_profile.py [--mode performance] [--batches 1,8,32] [--out runs/bench2d/ort_profile.md]

Variants: cudnn_conv_algo_search EXHAUSTIVE (ORT default) / HEURISTIC / DEFAULT; one fixed batch size vs a new size on
every call. Each number is the median of --reps runs after --warmup runs."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def model_paths(mode):
    from rtmlib import Wholebody
    sol = Wholebody(mode=mode, backend="onnxruntime", device="cpu")
    return sol.det_model.onnx_model, sol.pose_model.onnx_model


def make_session(path, search, profile_dir=None):
    import onnxruntime as ort
    from analysis.live.pose2d_fast import check_cuda
    check_cuda("cuda")                   # preloads the pip CUDA / cuDNN libraries (else the CUDA provider cannot load)
    so = ort.SessionOptions()
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(profile_dir, os.path.basename(path) + "_" + search)
    prov = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=prov)


def timed(sess, feed, warmup, reps):
    names = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(names, feed)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(names, feed)
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def provider_split(profile_json):
    """Sum of node time (ms) per execution provider from an ORT profile file."""
    with open(profile_json, "r", encoding="utf-8") as fh:
        ev = json.load(fh)
    out = {}
    for e in ev:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        p = (e.get("args") or {}).get("provider", "?")
        out[p] = out.get(p, 0.0) + e.get("dur", 0) / 1000.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="performance")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="runs/bench2d/ort_profile.md")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    prof_dir = os.path.join(os.path.dirname(a.out) or ".", "ort_profiles")
    os.makedirs(prof_dir, exist_ok=True)
    det_path, pose_path = model_paths(a.mode)
    batches = [int(b) for b in a.batches.split(",")]
    rng = np.random.default_rng(0)
    L = [f"# ORT profile — Wholebody {a.mode}", "", f"detector `{os.path.basename(det_path)}`, pose `{os.path.basename(pose_path)}`", ""]
    L += ["| model | input | cudnn search | ms (median) | ms per crop |", "|---|---|---|---|---|"]
    def row(r):
        L.append(r)
        print(r, flush=True)
    for search in ("HEURISTIC", "DEFAULT", "EXHAUSTIVE"):
        det = make_session(det_path, search)
        x = rng.integers(0, 255, (1, 3, 640, 640)).astype(np.float32)
        ms = timed(det, {det.get_inputs()[0].name: x}, a.warmup, a.reps)
        row(f"| detector | 1x3x640x640 | {search} | {ms:.1f} | – |")
        pose = make_session(pose_path, search)
        h, w = pose.get_inputs()[0].shape[2], pose.get_inputs()[0].shape[3]
        for b in batches:
            x = rng.standard_normal((b, 3, h, w)).astype(np.float32)
            ms = timed(pose, {pose.get_inputs()[0].name: x}, a.warmup, a.reps)
            row(f"| pose | {b}x3x{h}x{w} | {search} | {ms:.1f} | {ms / b:.1f} |")
        # a new batch size on every call (what live crops look like)
        names = [o.name for o in pose.get_outputs()]
        t = []
        sizes = [3, 5, 7, 9, 11, 13]
        for b in sizes:
            x = rng.standard_normal((b, 3, h, w)).astype(np.float32)
            t0 = time.perf_counter()
            pose.run(names, {pose.get_inputs()[0].name: x})
            t.append((time.perf_counter() - t0) * 1000 / b)
        row(f"| pose | new batch size each call {sizes} | {search} | first call per size, median {statistics.median(t):.1f} ms/crop | – |")
    # which provider runs the nodes (one profiled pass each)
    L += ["", "| model | provider split (ms of node time, one run) |", "|---|---|"]
    for name, path, shape in (("detector", det_path, (1, 3, 640, 640)), ("pose", pose_path, None)):
        s = make_session(path, "HEURISTIC", prof_dir)
        shp = shape or (8, 3, s.get_inputs()[0].shape[2], s.get_inputs()[0].shape[3])
        x = rng.standard_normal(shp).astype(np.float32)
        names = [o.name for o in s.get_outputs()]
        for _ in range(3):
            s.run(names, {s.get_inputs()[0].name: x})
        prof = s.end_profiling()
        row(f"| {name} {shp} | {json.dumps({k: round(v, 1) for k, v in provider_split(prof).items()})} |")
    md = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
