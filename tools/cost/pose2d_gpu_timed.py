"""Phase 4 multi-person RTMPose 2D pose extraction (GPU), timed for the SP5 cost table (v0.7.4 / camsubsets).

Timed copy of _archive/phase4_gpu/pose2d_gpu.py with per-view timing (the archive stays untouched): each
run_view_multi call is wrapped in time.perf_counter() and <out>/<clip>_gpu_timing.json =
{"clip", "device", "views": {view: {"seconds", "frames", "fps_processed"[, "warmup_included": true on the first
view]}}} is written after the kp2d files
(read by `python -m analysis.pose24.camsubsets cost --gpu-timing`). The kp2d JSON output is unchanged.

Standalone script — runs on any machine with a GPU. No Colab dependencies.

Usage:
    python tools/cost/pose2d_gpu_timed.py --src /path/to/videos --clip singles_1x --out data/pose2d

Processes all camera views of the clip (or --views ...), outputs multi-person kp2d JSON files. With --views the
timing file is <out>/<clip>_<views>_gpu_timing.json, so parallel runs (tools/pose2d_parallel.py) never overwrite
each other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Run as `python tools/cost/pose2d_gpu_timed.py`, sys.path[0] is tools/cost/: put tools/ on the path for byte_tracker.
_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

CLIP_CAMERAS = {
    "singles_1x": {
        "back_11":       "15.8 back 1-1.mov",
        "backleft_11":   "15.8 back left 1-1.mov",
        "backright_11":  "15.8 back right 1-1.mov",
        "frontleft_11":  "15.8 front left 1-1.mov",
        "frontright_11": "15.8 front right 1-1.mov",
    },
    "singles_05x": {
        "back_051":       "15.8 back 0.5-1.mov",
        "backleft_051":   "15.8 back left 0.5-1.mov",
        "backright_051":  "15.8 back right 0.5-1.mov",
        "frontleft_051":  "15.8 front left 0.5-1.mov",
        "frontright_051": "15.8 front right 0.5-1.mov",
    },
    "doubles_1side": {
        "back_21":       "15.8 back 2-1.mov",
        "backleft_21":   "15.8 back left 2-1.mov",
        "backright_21":  "15.8 back right 2-1.mov",
        "frontleft_21":  "15.8 front left 2-1.mov",
        "frontright_21": "15.8 front right 2-1.mov",
    },
    "doubles_2sides": {
        "back_31":       "15.8 back 3-1.mov",
        "backleft_31":   "15.8 back left 3-1.mov",
        "backright_31":  "15.8 back right 3-1.mov",
        "frontleft_31":  "15.8 front left 3-1.mov",
        "frontright_31": "15.8 front right 3-1.mov",
    },
    "doubles_2sides_4angle": {
        "back_41":       "15.8 back 4-1.mov",
        "backright_41":  "15.8 back right 4-1.mov",
        "frontleft_41":  "15.8 front left 4-1.mov",
        "frontright_41": "15.8 front right 4-1.mov",
    },
}


def run_view_multi(pose, path, tracker):
    import cv2

    from byte_tracker import bboxes_from_keypoints

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    i = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        keypoints, scores = pose(frame)

        if keypoints is None or len(keypoints) == 0:
            frames.append([])
            i += 1
            if i % 100 == 0 or i == frame_count:
                elapsed = time.time() - t0
                fps_actual = i / elapsed if elapsed > 0 else 0
                print(f"  {i}/{frame_count}  ({fps_actual:.1f} fps)")
            continue

        bboxes = bboxes_from_keypoints(keypoints, scores)
        pairs = tracker.update(bboxes)

        frame_dets = []
        matched_indices = {idx for _, idx in pairs}
        for tid, idx in pairs:
            kps = [
                [float(keypoints[idx][j][0]),
                 float(keypoints[idx][j][1]),
                 float(scores[idx][j])]
                for j in range(keypoints.shape[1])
            ]
            frame_dets.append({"track_id": tid, "keypoints": kps})
        # Include unmatched detections with unique temporary IDs
        for idx in range(len(keypoints)):
            if idx not in matched_indices:
                kps = [
                    [float(keypoints[idx][j][0]),
                     float(keypoints[idx][j][1]),
                     float(scores[idx][j])]
                    for j in range(keypoints.shape[1])
                ]
                frame_dets.append({"track_id": -(idx + 1), "keypoints": kps})
        frames.append(frame_dets)

        i += 1
        if i % 100 == 0 or i == frame_count:
            elapsed = time.time() - t0
            fps_actual = i / elapsed if elapsed > 0 else 0
            print(f"  {i}/{frame_count} ({len(frame_dets)} persons, "
                  f"{fps_actual:.1f} fps)")

    cap.release()
    return frames, width, height, fps, frame_count


def run_view_range(pose, path, tracker, start_frame=0, max_frames=None):
    """run_view_multi on frames [start_frame, start_frame + max_frames) only, for short timing tests. Frame-accurate:
    the frames before start_frame are decoded with grab() (never a seek, imprecise on HEVC), so frame k is the k-th
    decoded frame as in a full run. Returns run_view_multi's tuple + the per-frame model + tracking ms."""
    import cv2

    from byte_tracker import bboxes_from_keypoints

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_skip = time.perf_counter()
    for _ in range(start_frame):
        if not cap.grab():
            break
    if start_frame:
        print(f"  skipped to frame {start_frame} in {time.perf_counter() - t_skip:.1f} s (decoding, not timed below)")
    frames, frame_ms = [], []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        keypoints, scores = pose(frame)
        frame_dets = []
        if keypoints is not None and len(keypoints) > 0:
            bboxes = bboxes_from_keypoints(keypoints, scores)
            pairs = tracker.update(bboxes)
            matched = {idx for _, idx in pairs}
            for tid, idx in pairs:
                frame_dets.append({"track_id": tid, "keypoints": [
                    [float(keypoints[idx][j][0]), float(keypoints[idx][j][1]), float(scores[idx][j])]
                    for j in range(keypoints.shape[1])]})
            for idx in range(len(keypoints)):
                if idx not in matched:
                    frame_dets.append({"track_id": -(idx + 1), "keypoints": [
                        [float(keypoints[idx][j][0]), float(keypoints[idx][j][1]), float(scores[idx][j])]
                        for j in range(keypoints.shape[1])]})
        frames.append(frame_dets)
        frame_ms.append(round(1000 * (time.perf_counter() - t0), 1))
        note = " (first frame: includes GPU warm-up)" if len(frames) == 1 else ""
        print(f"  frame {start_frame + len(frames) - 1}: {len(frame_dets)} persons, {frame_ms[-1]:.0f} ms{note}",
              flush=True)
    cap.release()
    return frames, width, height, fps, frame_count, frame_ms


def preload_cuda_dlls():
    """onnxruntime-gpu >= 1.21 installed with the [cuda,cudnn] pip extras keeps CUDA / cuDNN inside site-packages; rtmlib
    never loads them, so without this onnxruntime silently runs on the CPU (Windows above all). No-op on older builds."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:          # libraries from the system CUDA install are fine too
            print(f"onnxruntime.preload_dlls skipped: {e}")


def device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def write_timing(out_dir, clip, device, timings, name=None, extra=None):
    """timings {view: (seconds, frames)} in processing order -> <out_dir>/<name or clip>_gpu_timing.json; returns its path.
    The first view's seconds include the CUDA / model warm-up: it carries "warmup_included": true."""
    views = {v: {"seconds": round(s, 3), "frames": n, "fps_processed": round(n / s, 3) if s > 0 else None}
             for v, (s, n) in timings.items()}
    if views:
        views[next(iter(views))]["warmup_included"] = True
    for v, fields in (extra or {}).items():          # e.g. model_load_s, start_frame, frame_ms
        if v in views:
            views[v].update(fields)
    path = os.path.join(out_dir, f"{name or clip}_gpu_timing.json")
    with open(path, "w") as fh:
        json.dump({"clip": clip, "device": device, "views": views}, fh, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="Multi-person RTMPose 2D pose (GPU)")
    ap.add_argument("--src", required=True,
                    help="Directory containing the .mov video files")
    ap.add_argument("--clip", required=True, choices=list(CLIP_CAMERAS.keys()),
                    help="Clip batch to process")
    ap.add_argument("--out", default="data/pose2d",
                    help="Output directory for kp2d JSON files")
    ap.add_argument("--views", nargs="*", default=None,
                    help="View IDs to process (default: all views of the clip)")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="first frame to process (short timing tests; default 0)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only this many frames per view (short timing tests; default: the whole clip)")
    args = ap.parse_args()
    limited = args.start_frame > 0 or args.max_frames is not None

    cameras = CLIP_CAMERAS[args.clip]
    timing_name = None
    if args.views:
        cameras = {k: v for k, v in cameras.items() if k in args.views}
        if not cameras:
            print(f"ERROR: none of {args.views} in clip {args.clip}")
            sys.exit(1)
        if len(cameras) < len(CLIP_CAMERAS[args.clip]):
            timing_name = "_".join([args.clip] + list(cameras))
    os.makedirs(args.out, exist_ok=True)

    # Check all videos exist before starting
    for view_id, filename in cameras.items():
        path = os.path.join(args.src, filename)
        if not os.path.isfile(path):
            print(f"ERROR: video not found: {path}")
            sys.exit(1)
    print(f"All {len(cameras)} videos found in {args.src}")

    t_load = time.perf_counter()
    preload_cuda_dlls()
    from rtmlib import PoseTracker, Wholebody

    from byte_tracker import ByteTracker

    pose = PoseTracker(
        Wholebody,
        det_frequency=1,
        tracking=False,      # detection only — ByteTrack handles tracking
        to_openpose=False,
        mode="performance",
        backend="onnxruntime",
        device="cuda",
    )
    model_load_s = time.perf_counter() - t_load
    print(f"PoseTracker ready in {model_load_s:.1f} s (Wholebody/performance, ByteTrack tracking), "
          f"processing {args.clip}...\n")

    timings, extra = {}, {}
    for view_id, filename in cameras.items():
        video_path = os.path.join(args.src, filename)
        print(f"=== {view_id} ({filename}) ===")
        tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        t_view = time.perf_counter()
        if limited:
            frames, width, height, fps, frame_count, frame_ms = run_view_range(
                pose, video_path, tracker, args.start_frame, args.max_frames)
            timings[view_id] = (sum(frame_ms) / 1000, len(frames))
            extra[view_id] = {"model_load_s": round(model_load_s, 3), "start_frame": args.start_frame,
                              "frame_ms": frame_ms}
        else:
            frames, width, height, fps, frame_count = run_view_multi(
                pose, video_path, tracker)
            timings[view_id] = (time.perf_counter() - t_view, len(frames))
            extra[view_id] = {"model_load_s": round(model_load_s, 3)}

        n_with_det = sum(1 for f in frames if len(f) > 0)
        max_persons = max((len(f) for f in frames), default=0)
        n_kpts = len(frames[0][0]["keypoints"]) if frames and frames[0] else 0
        print(f"  {len(frames)} frames, {n_with_det} with detections, "
              f"max {max_persons} persons/frame, {n_kpts} keypoints/person")

        out = {
            "view": view_id,
            "source": filename,
            "clip": args.clip,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "keypoint_format": "coco_wholebody_133",
            "multi_person": True,
            "frames": frames,
        }
        if limited:      # a short test: frames[0] is video frame start_frame
            out.update(start_frame=args.start_frame, frame_count=len(frames), video_frame_count=frame_count)
        out_path = os.path.join(args.out, f"{view_id}_kp2d.json")
        with open(out_path, "w") as fh:
            json.dump(out, fh)
        print(f"  wrote {out_path}\n")

    print(f"  wrote {write_timing(args.out, args.clip, device_name(), timings, timing_name, extra)}")
    print("Done! Copy the kp2d files back to your local data/pose2d/")


if __name__ == "__main__":
    main()
