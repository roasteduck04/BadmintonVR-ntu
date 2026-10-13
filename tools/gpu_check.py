"""Check that the 2D pose model (RTMPose Wholebody, rtmlib + onnxruntime) really runs on the NVIDIA GPU.

Builds the model exactly like tools/cost/pose2d_gpu_timed.py (and _archive/phase4_gpu/pose2d_gpu.py), prints which
execution providers each sub-model got, and with --video times a few frames. Exit 0 = both sub-models on CUDA,
exit 1 = not (onnxruntime falls back to the CPU silently, so the 2D step would take hours).

    python tools/gpu_check.py
    python tools/gpu_check.py --video "data/15-08/singles/0.5/15.8 back 0.5-1.mov"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _timed_module():
    spec = importlib.util.spec_from_file_location("pose2d_gpu_timed", os.path.join(HERE, "cost", "pose2d_gpu_timed.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", help="time a few frames of this video (optional)")
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args(argv)
    print(f"python {sys.version.split()[0]} on {platform.system()} ({sys.executable})")
    try:
        import onnxruntime as ort
    except ImportError:
        print("FAIL: onnxruntime is not installed in this Python (pip install -r tools/requirements-pc.txt)")
        return 1
    print(f"onnxruntime {ort.__version__}, package providers: {ort.get_available_providers()}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("FAIL: this onnxruntime has no CUDA provider (the CPU-only 'onnxruntime' package is installed, or it shadows "
              "'onnxruntime-gpu'): pip uninstall -y onnxruntime onnxruntime-gpu, then pip install -r tools/requirements-pc.txt")
        return 1
    _timed_module().preload_cuda_dlls()
    from rtmlib import PoseTracker, Wholebody
    pose = PoseTracker(Wholebody, det_frequency=1, tracking=False, to_openpose=False, mode="performance",
                       backend="onnxruntime", device="cuda")      # the same arguments as the 2D script
    ok = True
    for name in ("det_model", "pose_model"):
        prov = getattr(pose, name).session.get_providers()
        print(f"{name}: {prov}")
        ok = ok and "CUDAExecutionProvider" in prov
    if a.video:
        import cv2
        cap = cv2.VideoCapture(a.video)
        okf, frame = cap.read()
        if not okf:
            print(f"FAIL: could not read a frame from {a.video}")
            return 1
        t = time.perf_counter()
        kps, _ = pose(frame)
        print(f"first frame (cold): {1000 * (time.perf_counter() - t):.0f} ms, {len(kps)} people")
        times = []
        for _ in range(a.frames):
            okf, frame = cap.read()
            if not okf:
                break
            t = time.perf_counter()
            pose(frame)
            times.append(1000 * (time.perf_counter() - t))
        if times:
            times.sort()
            print(f"next {len(times)} frames: median {times[len(times) // 2]:.0f} ms per frame (model only)")
    print("gpu-check:", "OK (CUDA)" if ok else "FAIL (a sub-model runs without CUDAExecutionProvider)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
