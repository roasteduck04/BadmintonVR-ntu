"""Phase 4 multi-person RTMPose 2D pose extraction (GPU).

Standalone script — runs on any machine with a GPU. No Colab dependencies.

Usage:
    python tools/pose2d_gpu.py --src /path/to/videos --clip singles_1x --out data/pose2d

Processes all 5 camera views, outputs multi-person kp2d JSON files.
"""
from __future__ import annotations

import argparse
import json
import os
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
}


def run_view_multi(pose, path, tracker):
    import cv2

    from byte_tracker import bboxes_from_keypoints
    # Image transport ROS image message to a cv2 matrix
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


def main():
    ap = argparse.ArgumentParser(description="Multi-person RTMPose 2D pose (GPU)")
    ap.add_argument("--src", required=True,
                    help="Directory containing the .mov video files")
    ap.add_argument("--clip", required=True, choices=list(CLIP_CAMERAS.keys()),
                    help="Clip batch to process")
    ap.add_argument("--out", default="data/pose2d",
                    help="Output directory for kp2d JSON files")
    args = ap.parse_args()

    cameras = CLIP_CAMERAS[args.clip]
    os.makedirs(args.out, exist_ok=True)

    # Check all videos exist before starting
    # Here you would check if each stream is working
    for view_id, filename in cameras.items():
        path = os.path.join(args.src, filename)
        if not os.path.isfile(path):
            print(f"ERROR: video not found: {path}")
            sys.exit(1)
    print(f"All {len(cameras)} videos found in {args.src}")

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
    print(f"PoseTracker ready (Wholebody/performance, ByteTrack tracking), "
          f"processing {args.clip}...\n")

    for view_id, filename in cameras.items(): # From processing video by video --> you do frame by frame
        video_path = os.path.join(args.src, filename)
        print(f"=== {view_id} ({filename}) ===")
        tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8) 

        # You need a consistent tracker, out of the loop 
        frames, width, height, fps, frame_count = run_view_multi(
            pose, video_path, tracker) #

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

         # TF2 ROS, from json... you would publish human poses
        out_path = os.path.join(args.out, f"{view_id}_kp2d.json")


        with open(out_path, "w") as fh:
            json.dump(out, fh)
        print(f"  wrote {out_path}\n")

    print("Done! Copy the kp2d files back to your local data/pose2d/")


if __name__ == "__main__":
    main()
