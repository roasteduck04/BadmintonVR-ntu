"""Check that the 2D pose model (RTMPose Wholebody, rtmlib + onnxruntime) really runs on the NVIDIA GPU.

Builds the model exactly like tools/cost/pose2d_gpu_timed.py (and _archive/phase4_gpu/pose2d_gpu.py), prints which
execution providers each sub-model got, and with --video times a few frames. Exit 0 = both sub-models on CUDA,
exit 1 = not (onnxruntime falls back to the CPU silently, so the 2D step would take hours).

    python tools/gpu_check.py
    python tools/gpu_check.py --video "data/15-08/singles/0.5/15.8 back 0.5-1.mov"
    python tools/gpu_check.py --video "..." --trt [--fp32]     # v0.8.3: both models on TensorRT (exit 0 = TensorRT
                                                               # active; the first run builds the engines, minutes)
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


def trt_diagnostics(pf):
    """What onnxruntime's TensorRT provider needs vs what the TensorRT wheel installed (for the FAIL report)."""
    import glob
    import subprocess
    import onnxruntime as ort
    dirs, ver = pf.tensorrt_lib_dirs()
    print(f"--- diagnostics: onnxruntime {ort.__version__}, TensorRT wheel {ver}")
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
    for d in dirs:
        print(f"TensorRT libs in {d}: {sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, 'lib*.so*')))}")
    prov = glob.glob(os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime_providers_tensorrt*"))
    for pth in prov:
        try:
            out = subprocess.run(["ldd", pth], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError) as e:
            print(f"ldd failed: {e}")
            continue
        need = [ln.strip() for ln in out.splitlines() if any(k in ln for k in ("nvinfer", "nvonnx", "not found"))]
        print(f"{os.path.basename(pth)} needs:")
        for ln in need:
            print(f"  {ln}")
    if not prov:
        print("this onnxruntime build has no TensorRT provider library")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", help="time a few frames of this video (optional)")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--trt", action="store_true", help="put both models on onnxruntime's TensorRT provider")
    ap.add_argument("--fp32", action="store_true", help="with --trt: TensorRT in FP32 (default FP16)")
    a = ap.parse_args(argv)
    if a.trt:                                   # TensorRT's folders on LD_LIBRARY_PATH (restarts this command once)
        sys.path.insert(0, HERE)
        from analysis.live import pose2d_fast as pf
        pf.ensure_trt_library_path()
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
    want = "CUDAExecutionProvider"
    if a.trt:
        sys.path.insert(0, HERE)
        from analysis.live import pose2d_fast as pf
        try:
            ver = pf.preload_tensorrt()
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return 1
        print(f"TensorRT {ver or '(version unknown)'} libraries loaded; building / loading the engines "
              f"({'FP32' if a.fp32 else 'FP16'}, cache {os.path.expanduser(pf.TRT_CACHE)})...")
        rc = {"trt_fp16": not a.fp32}
        t = time.perf_counter()
        notes = {}
        try:                                      # the pose model must build; the detector may stay on CUDA
            pm = pose.pose_model
            pw, ph = (int(x) for x in pm.model_input_size)
            notes = pf.trt_models(pose, rc, batch=1, in_hw=(ph, pw))
        except pf.TensorRTUnavailable as e:
            print(f"FAIL: {e}")
            trt_diagnostics(pf)
            print("gpu-check: FAIL (TensorRT not usable for the pose model; paste everything above)")
            return 1
        for name, note in notes.items():
            print(f"{name}: stays on {note}")
        print(f"sessions ready in {time.perf_counter() - t:.1f} s")
        want = "TensorrtExecutionProvider"
    ok = True
    for name in ("det_model", "pose_model"):
        prov = getattr(pose, name).session.get_providers()
        print(f"{name}: {prov}")
        if a.trt:
            ok = ok and (prov[:1] == [want] if name == "pose_model" else "CUDAExecutionProvider" in prov)
        else:
            ok = ok and want in prov
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
    if a.trt:
        det_trt = pose.det_model.session.get_providers()[:1] == ["TensorrtExecutionProvider"]
        print("gpu-check:", ("OK (TensorRT" + (")" if det_trt else ": pose model; detector on CUDA)")) if ok
              else "FAIL (the pose model is not on TensorRT)")
        return 0 if ok else 1
    print("gpu-check:", "OK (CUDA)" if ok else "FAIL (a sub-model runs without CUDAExecutionProvider)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
