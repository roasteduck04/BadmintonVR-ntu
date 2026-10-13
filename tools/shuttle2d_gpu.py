"""TrackNetV3 shuttlecock detection (GPU).

Standalone script — runs on any machine with a GPU.
Uses the pretrained TrackNetV3 model (trained on ShuttleSet22).

First-time setup (run once):
    git clone https://github.com/qaz812345/TrackNetV3.git ~/TrackNetV3
    pip install opencv-python pandas tqdm parse gdown

    # Download checkpoints
    python -c "
    import gdown, zipfile
    gdown.download('https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA', 'ckpts.zip')
    zipfile.ZipFile('ckpts.zip').extractall('$HOME/TrackNetV3')
    "

Usage:
    python tools/shuttle2d_gpu.py --src videos --clip singles_1x --out data/shuttle
    python tools/shuttle2d_gpu.py --src "data/3cams + 2 cctv" --clip 3-1 --out data/shuttle

By default processes the "back" camera (widest court view). Use --views
to select specific cameras. Output is per-view JSON with [x, y] pixel
positions per frame (or null).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

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
    "3-1": {
        "cctv31": "cctv 3-1.mov",
    },
}

DEFAULT_VIEWS = {
    "singles_1x":     ["back_11"],
    "singles_05x":    ["back_051"],
    "doubles_1side":  ["back_21"],
    "doubles_2sides": ["back_31"],
    "doubles_2sides_4angle": ["back_41"],
    "3-1":            ["cctv31"],
}

CHUNK_FRAMES = 400
DEFAULT_RESOLUTION = 1080
BATCH_SIZE = 2


def find_tracknet(tn_dir=None):
    candidates = [
        tn_dir,
        os.path.expanduser("~/TrackNetV3"),
        os.path.join(os.path.dirname(__file__), "..", "TrackNetV3"),
        "/content/TrackNetV3",
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            predict = os.path.join(d, "predict.py")
            tracknet_ckpt = os.path.join(d, "ckpts", "TrackNet_best.pt")
            inpaint_ckpt = os.path.join(d, "ckpts", "InpaintNet_best.pt")
            if os.path.isfile(predict) and os.path.isfile(tracknet_ckpt):
                return d, predict, tracknet_ckpt, inpaint_ckpt
    print("ERROR: TrackNetV3 not found. Run setup first:")
    print("  git clone https://github.com/qaz812345/TrackNetV3.git ~/TrackNetV3")
    print("  # Then download checkpoints (see script docstring)")
    sys.exit(1)


def downscale_video(src_path, dst_path, target_h=DEFAULT_RESOLUTION):
    """Downscale to target height, preserving aspect ratio."""
    if os.path.exists(dst_path):
        return
    print(f"  downscaling to {target_h}p ...")
    r = subprocess.run([
        "ffmpeg", "-i", src_path,
        "-vf", f"scale=-2:{target_h}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an", "-y", dst_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed: {r.stderr[-500:]}")
        sys.exit(1)


def run_tracknet_segmented(predict_py, tracknet_ckpt, inpaint_ckpt,
                           video_path, out_dir, total_frames, fps):
    """Run TrackNetV3 in segments to avoid OOM on long videos."""
    seg_dir = os.path.join(out_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    chunk_sec = CHUNK_FRAMES / fps
    total_sec = total_frames / fps
    segments = []
    start_sec = 0.0
    while start_sec < total_sec:
        end_sec = min(start_sec + chunk_sec, total_sec)
        seg_out = os.path.join(seg_dir, f"seg_{int(start_sec)}")
        os.makedirs(seg_out, exist_ok=True)
        segments.append((start_sec, end_sec, seg_out))
        start_sec = end_sec

    print(f"  {total_frames} frames ({total_sec:.1f}s), "
          f"{len(segments)} segments of ~{chunk_sec:.1f}s")

    for i, (start_sec, end_sec, seg_out) in enumerate(segments):
        existing = [f for f in os.listdir(seg_out) if f.endswith("_ball.csv")]
        if existing:
            print(f"  seg {i + 1}/{len(segments)} "
                  f"[{start_sec:.0f}-{end_sec:.0f}s) — cached")
            continue

        cmd = [
            sys.executable, predict_py,
            "--video_file", video_path,
            "--tracknet_file", tracknet_ckpt,
            "--inpaintnet_file", inpaint_ckpt,
            "--save_dir", seg_out,
            "--batch_size", str(BATCH_SIZE),
            "--large_video",
            "--video_range", f"{int(start_sec)},{int(end_sec)}",
        ]

        print(f"  seg {i + 1}/{len(segments)} "
              f"[{start_sec:.0f}-{end_sec:.0f}s) ...", end=" ", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"FAILED ({elapsed:.0f}s)")
            print(f"  stderr: {result.stderr[-500:]}")
            return None
        print(f"OK ({elapsed:.0f}s)")

    merged_path = os.path.join(out_dir, "merged_ball.csv")
    with open(merged_path, "w", newline="") as out_f:
        writer = None
        for start, end, seg_out in segments:
            csv_files = [f for f in os.listdir(seg_out) if f.endswith("_ball.csv")]
            if not csv_files:
                print(f"  WARNING: no CSV for segment {start}-{end}")
                continue
            with open(os.path.join(seg_out, csv_files[0])) as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    return merged_path


def run_shuttle_view(predict_py, tracknet_ckpt, inpaint_ckpt,
                     video_path, view_id, work_dir, resolution=DEFAULT_RESOLUTION):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  source: {src_w}x{src_h} @ {fps} fps, {src_frames} frames")

    view_dir = os.path.join(work_dir, view_id)
    os.makedirs(view_dir, exist_ok=True)

    ds_path = os.path.join(view_dir, f"downscaled_{resolution}p.mp4")
    downscale_video(video_path, ds_path, resolution)

    cap2 = cv2.VideoCapture(ds_path)
    ds_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    ds_h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ds_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.release()

    scale_x = src_w / ds_w
    scale_y = src_h / ds_h
    print(f"  downscaled: {ds_w}x{ds_h}, {ds_frames} frames")

    ds_fps = cv2.VideoCapture(ds_path).get(cv2.CAP_PROP_FPS) or fps

    merged_csv = run_tracknet_segmented(
        predict_py, tracknet_ckpt, inpaint_ckpt,
        ds_path, view_dir, ds_frames, ds_fps)

    if merged_csv is None:
        return None

    detections = {}
    with open(merged_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["Frame"])
            vis = int(row["Visibility"])
            x = int(row["X"])
            y = int(row["Y"])
            if vis == 1 and (x != 0 or y != 0):
                detections[frame] = [round(x * scale_x), round(y * scale_y)]

    frames = [detections.get(i) for i in range(src_frames)]
    detected = sum(1 for f in frames if f is not None)
    print(f"  shuttle detected in {detected}/{src_frames} frames "
          f"({100 * detected / max(src_frames, 1):.1f}%)")

    return {
        "view": view_id,
        "model": "TrackNetV3",
        "weights": "pretrained_shuttleset22",
        "width": src_w,
        "height": src_h,
        "fps": fps,
        "frame_count": src_frames,
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser(description="TrackNetV3 shuttle detection (GPU)")
    ap.add_argument("--src", required=True,
                    help="Directory containing the .mov video files")
    ap.add_argument("--clip", required=True, choices=list(CLIP_CAMERAS.keys()),
                    help="Clip batch to process")
    ap.add_argument("--out", default="data/shuttle",
                    help="Output directory for shuttle JSON files")
    ap.add_argument("--tn-dir", default=None,
                    help="Path to TrackNetV3 repo (default: ~/TrackNetV3)")
    ap.add_argument("--views", nargs="*", default=None,
                    help="View IDs to process (default: back camera only)")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help=f"Downscale height in pixels (default: {DEFAULT_RESOLUTION})")
    args = ap.parse_args()

    all_cameras = CLIP_CAMERAS[args.clip]

    if args.views:
        cameras = {k: v for k, v in all_cameras.items() if k in args.views}
        if not cameras:
            print(f"ERROR: none of {args.views} in clip {args.clip}")
            print(f"  available: {list(all_cameras.keys())}")
            sys.exit(1)
    else:
        default = DEFAULT_VIEWS.get(args.clip, list(all_cameras.keys())[:1])
        cameras = {k: all_cameras[k] for k in default if k in all_cameras}
        print(f"Default views: {list(cameras.keys())} "
              f"(use --views to change)")

    os.makedirs(args.out, exist_ok=True)

    for view_id, filename in cameras.items():
        path = os.path.join(args.src, filename)
        if not os.path.isfile(path):
            print(f"ERROR: video not found: {path}")
            sys.exit(1)
    print(f"All {len(cameras)} videos found in {args.src}\n")

    tn_dir, predict_py, tracknet_ckpt, inpaint_ckpt = find_tracknet(args.tn_dir)
    print(f"TrackNetV3: {tn_dir}")

    work_dir = os.path.join(args.out, "_work")
    os.makedirs(work_dir, exist_ok=True)

    for view_id, filename in cameras.items():
        video_path = os.path.join(args.src, filename)
        print(f"\n=== {view_id} ({filename}) ===")
        doc = run_shuttle_view(
            predict_py, tracknet_ckpt, inpaint_ckpt,
            video_path, view_id, work_dir, resolution=args.resolution)

        if doc is None:
            print(f"  FAILED — skipping {view_id}")
            continue

        out_path = os.path.join(args.out, f"{view_id}_shuttle.json")
        with open(out_path, "w") as fh:
            json.dump(doc, fh)
        print(f"  wrote {out_path}")

    print("\nDone! Copy the shuttle files back to your local data/shuttle/")


if __name__ == "__main__":
    main()
