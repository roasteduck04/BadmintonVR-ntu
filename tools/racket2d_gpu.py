"""RacketVision 5-keypoint racket detection (GPU).

Standalone script — runs on any machine with a GPU + OpenMMLab stack.
Uses the pretrained RacketVision model (RTMDet-M detector + RTMPose-M,
5 keypoints: top, bottom, handle, left, right).

First-time setup (run once):
    # Clone and download checkpoints
    git clone --depth 1 https://github.com/OrcustD/RacketVision.git ~/RacketVision
    cd ~/RacketVision/source && python download_checkpoints.py --module RacketPose

    # Install OpenMMLab (in your conda env)
    pip install openmim
    mim install mmengine "mmcv>=2.0.0rc4,<2.2.0"
    pip install "mmdet>=3.0.0,<3.3.0" --no-deps
    pip install "mmpose>=1.1.0,<1.4.0" --no-deps
    pip install pycocotools munkres json_tricks

Usage:
    python tools/racket2d_gpu.py --src videos --clip singles_1x --out data/racket2d
    python tools/racket2d_gpu.py --src videos --clip 3-1 --out data/racket2d

Outputs per-view JSON with 5 keypoints per frame (or null).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import numpy as np

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
        "frontleft31":  "front left 3-1.mov",
        "backleft31":   "back left 3-1.mov",
        "backright31":  "back right 3-1.mov",
    },
}

CAT_ID = 0
DET_THR = 0.03
MAX_RATIO = 0.5
TOPK = 5
TARGET_H = 1080


def _apply_shims():
    """Stub out xtcocotools (mmpose imports it but inference never uses it)."""
    import importlib.util
    import types
    if importlib.util.find_spec("xtcocotools") is None:
        import sys
        for n in ("xtcocotools", "xtcocotools.coco",
                  "xtcocotools.cocoeval", "xtcocotools.mask"):
            sys.modules[n] = types.ModuleType(n)
        sys.modules["xtcocotools.coco"].COCO = type(
            "COCO", (), {"__init__": lambda s, *a, **k: None})
        sys.modules["xtcocotools.cocoeval"].COCOeval = type(
            "COCOeval", (), {"__init__": lambda s, *a, **k: None})


def find_racketvision(rv_dir=None):
    candidates = [
        rv_dir,
        os.path.expanduser("~/RacketVision/source"),
        os.path.join(os.path.dirname(__file__), "..", "RacketVision", "source"),
        "/content/RacketVision/source",
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            det_cfg = os.path.join(d, "RacketPose/configs/detection/rtmdet_m_racket_infer.py")
            pose_cfg = os.path.join(d, "RacketPose/configs/pose/rtmpose_m_racket_infer.py")
            det_ckpt = os.path.join(d, "RacketPose/checkpoints/epoch_300.pth")
            pose_ckpt = os.path.join(d, "RacketPose/checkpoints/best_PCK_epoch_90.pth")
            if all(os.path.isfile(p) for p in (det_cfg, pose_cfg, det_ckpt, pose_ckpt)):
                return det_cfg, pose_cfg, det_ckpt, pose_ckpt
    print("ERROR: RacketVision not found. Run setup first:")
    print("  git clone --depth 1 https://github.com/OrcustD/RacketVision.git ~/RacketVision")
    print("  cd ~/RacketVision/source && python download_checkpoints.py --module RacketPose")
    sys.exit(1)


def _load_wrist_positions(pose2d_path):
    """Load per-frame wrist (x, y) positions from a pose2d JSON.

    Returns list of lists of (x, y) tuples — one list of wrist positions per
    frame (one per detected person). Uses right_wrist (index 10 in COCO-17,
    index 10 in wholebody133) as the crop center.
    """
    with open(pose2d_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    fmt = doc.get("keypoint_format", "coco17")
    if "wholebody" in fmt:
        wrist_idx = 10  # right_wrist in wholebody133
    else:
        wrist_idx = 10  # right_wrist in COCO-17

    wrists = []
    for frame_data in doc["frames"]:
        frame_wrists = []
        if frame_data is None:
            wrists.append(frame_wrists)
            continue
        persons = frame_data.items() if isinstance(frame_data, dict) else frame_data
        for person in persons:
            if isinstance(person, dict):
                kps = person.get("keypoints", [])
            else:
                _, kps = person
            if len(kps) > wrist_idx:
                x, y, score = kps[wrist_idx]
                if score > 0.3:
                    frame_wrists.append((x, y))
        wrists.append(frame_wrists)
    return wrists


def _run_crop_inference(detector, poser, frame_full, wrist_xy, crop_size, crop_target, src_w, src_h):
    """Run detection + pose on a crop centered on wrist_xy.

    Returns list of candidate dicts (same format as full-frame), with
    keypoints mapped back to full-frame coordinates. Returns [] if no
    detection in crop.
    """
    import cv2
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.structures import merge_data_samples

    wx, wy = wrist_xy
    half = crop_size // 2

    # Crop boundaries (clamp to frame)
    x1 = max(0, int(wx - half))
    y1 = max(0, int(wy - half))
    x2 = min(src_w, x1 + crop_size)
    y2 = min(src_h, y1 + crop_size)
    if x2 - x1 < 50 or y2 - y1 < 50:
        return []

    crop = frame_full[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    # Resize to inference target
    scale_x = crop_target / crop_w
    scale_y = crop_target / crop_h
    img = cv2.resize(crop, (crop_target, crop_target), interpolation=cv2.INTER_LINEAR)

    det = inference_detector(detector, img)
    inst = det.pred_instances.cpu().numpy()
    keep = (inst.labels == CAT_ID) & (inst.scores >= DET_THR)
    bb, sc = inst.bboxes[keep], inst.scores[keep]

    if len(bb) == 0:
        return []

    # Size filter
    w = (bb[:, 2] - bb[:, 0]) / crop_target
    h = (bb[:, 3] - bb[:, 1]) / crop_target
    ok_size = (w < MAX_RATIO) & (h < MAX_RATIO)
    bb, sc = bb[ok_size], sc[ok_size]

    if len(bb) == 0:
        return []

    # Take top-1 from crop (one crop = one player region)
    best = int(np.argmax(sc))
    bbox = bb[best:best + 1]
    pr = inference_topdown(poser, img, bbox, bbox_format="xyxy")
    ds = merge_data_samples(pr)
    kpts = ds.pred_instances.keypoints[0].tolist()
    kscores = ds.pred_instances.keypoint_scores[0].tolist()

    # Map back to full-frame coordinates
    kpts_full = []
    for kp in kpts:
        fx = kp[0] / scale_x + x1
        fy = kp[1] / scale_y + y1
        kpts_full.append([round(fx, 2), round(fy, 2)])

    return [{
        "bbox": [round(bb[best][0] / scale_x + x1, 1),
                 round(bb[best][1] / scale_y + y1, 1),
                 round(bb[best][2] / scale_x + x1, 1),
                 round(bb[best][3] / scale_y + y1, 1)],
        "det_score": round(float(sc[best]), 4),
        "keypoints": kpts_full,
        "keypoint_scores": [round(float(s), 4) for s in kscores],
        "source": "crop",
    }]


def _merge_candidates(full_cands, crop_cands, iou_thr=0.3):
    """Merge full-frame and crop candidates, preferring crop when they overlap.

    Returns merged list of candidate dicts.
    """
    if not crop_cands:
        for c in full_cands:
            c["source"] = c.get("source", "full")
        return full_cands
    if not full_cands:
        return crop_cands

    merged = list(crop_cands)
    for fc in full_cands:
        fb = fc["bbox"]
        overlaps = False
        for cc in crop_cands:
            cb = cc["bbox"]
            # IoU check
            ix1 = max(fb[0], cb[0])
            iy1 = max(fb[1], cb[1])
            ix2 = min(fb[2], cb[2])
            iy2 = min(fb[3], cb[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_f = (fb[2] - fb[0]) * (fb[3] - fb[1])
                area_c = (cb[2] - cb[0]) * (cb[3] - cb[1])
                union = area_f + area_c - inter
                if union > 0 and inter / union > iou_thr:
                    overlaps = True
                    break
        if not overlaps:
            fc["source"] = "full"
            merged.append(fc)
    return merged


def run_racket_view(detector, poser, video_path, view_id, kp_names,
                    wrist_positions=None, crop_size=400, crop_target=800):
    import cv2
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.structures import merge_data_samples

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = TARGET_H / src_h
    inf_w = int(round(src_w * scale))
    inf_h = TARGET_H

    print(f"  source: {src_w}x{src_h} @ {fps} fps, {frame_count} frames")
    print(f"  inference at: {inf_w}x{inf_h}")

    results = []
    i = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)

        det = inference_detector(detector, img)
        inst = det.pred_instances.cpu().numpy()
        keep = (inst.labels == CAT_ID) & (inst.scores >= DET_THR)
        bb, sc = inst.bboxes[keep], inst.scores[keep]
        if len(bb):
            w = (bb[:, 2] - bb[:, 0]) / inf_w
            h = (bb[:, 3] - bb[:, 1]) / inf_h
            ok_size = (w < MAX_RATIO) & (h < MAX_RATIO)
            bb, sc = bb[ok_size], sc[ok_size]

        order = list(np.argsort(-sc)[:TOPK]) if len(sc) else []
        cands = []
        for b in order:
            bbox = bb[b:b + 1]
            pr = inference_topdown(poser, img, bbox, bbox_format="xyxy")
            ds = merge_data_samples(pr)
            kpts = ds.pred_instances.keypoints[0].tolist()
            kscores = ds.pred_instances.keypoint_scores[0].tolist()
            kpts_src = [[kp[0] / scale, kp[1] / scale] for kp in kpts]
            cands.append({
                "bbox": [round(v / scale, 1) for v in [bb[b][0], bb[b][1], bb[b][2], bb[b][3]]],
                "det_score": round(float(sc[b]), 4),
                "keypoints": [[round(kp[0], 2), round(kp[1], 2)] for kp in kpts_src],
                "keypoint_scores": [round(float(s), 4) for s in kscores],
            })

        # Pass 2: crop-based detection around wrist positions (new)
        if wrist_positions is not None and i < len(wrist_positions):
            frame_wrists = wrist_positions[i]
            crop_cands_all = []
            for wx, wy in frame_wrists:
                cc = _run_crop_inference(
                    detector, poser, frame, (wx, wy),
                    crop_size, crop_target, src_w, src_h)
                crop_cands_all.extend(cc)
            if crop_cands_all or cands:
                merged_cands = _merge_candidates(cands, crop_cands_all)
                merged_cands.sort(key=lambda c: c["det_score"], reverse=True)
                cands = merged_cands[:TOPK]

        if cands:
            merged = []
            for c in cands:
                merged.append([[c["keypoints"][j][0], c["keypoints"][j][1],
                                c["keypoint_scores"][j]] for j in range(5)])
            results.append(merged)
        else:
            results.append(None)

        i += 1
        if i % 50 == 0 or i == frame_count:
            elapsed = time.time() - t0
            fps_actual = i / elapsed if elapsed > 0 else 0
            hit = sum(1 for r in results if r is not None)
            print(f"  {i}/{frame_count} ({fps_actual:.1f} fps, {hit} hits)",
                  flush=True)

    cap.release()

    hit = sum(1 for r in results if r is not None)
    print(f"  racket detected in {hit}/{len(results)} frames "
          f"({100 * hit / max(len(results), 1):.1f}%)")

    return {
        "view": view_id,
        "width": src_w,
        "height": src_h,
        "fps": fps,
        "frame_count": len(results),
        "keypoint_format": "racketvision5",
        "keypoint_names": kp_names,
        "det": {"cat_id": CAT_ID, "score_thr": DET_THR,
                "max_box_ratio": MAX_RATIO, "topk": TOPK},
        "frames": results,
    }


def main():
    ap = argparse.ArgumentParser(description="RacketVision 5-keypoint racket detection (GPU)")
    ap.add_argument("--src", required=True,
                    help="Directory containing the .mov video files")
    ap.add_argument("--clip", required=True, choices=list(CLIP_CAMERAS.keys()),
                    help="Clip batch to process")
    ap.add_argument("--out", default="data/racket2d",
                    help="Output directory for racket2d JSON files")
    ap.add_argument("--rv-dir", default=None,
                    help="Path to RacketVision/source (default: ~/RacketVision/source)")
    ap.add_argument("--views", nargs="*", default=None,
                    help="Specific view IDs to process (default: all)")
    ap.add_argument("--pose2d-dir", default=None,
                    help="Path to pose2d directory for crop-based detection. "
                         "When provided, enables two-pass mode: full-frame + "
                         "wrist-centered crops at original resolution.")
    ap.add_argument("--crop-size", type=int, default=400,
                    help="Crop region size in original-resolution pixels (default: 400)")
    ap.add_argument("--crop-target", type=int, default=800,
                    help="Resize target for crops before inference (default: 800)")
    args = ap.parse_args()

    cameras = CLIP_CAMERAS[args.clip]
    if args.views:
        cameras = {k: v for k, v in cameras.items() if k in args.views}
        if not cameras:
            print(f"ERROR: none of {args.views} found in clip {args.clip}")
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    for view_id, filename in cameras.items():
        path = os.path.join(args.src, filename)
        if not os.path.isfile(path):
            print(f"ERROR: video not found: {path}")
            sys.exit(1)
    print(f"All {len(cameras)} videos found in {args.src}")

    det_cfg, pose_cfg, det_ckpt, pose_ckpt = find_racketvision(args.rv_dir)
    print(f"RacketVision: {os.path.dirname(det_cfg)}")

    _apply_shims()

    import torch
    torch.load = functools.partial(torch.load, weights_only=False)

    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose
    from mmpose.utils import adapt_mmdet_pipeline

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"torch: {torch.__version__}, device: {dev}")

    detector = init_detector(det_cfg, det_ckpt, device=dev)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    poser = init_pose(pose_cfg, pose_ckpt, device=dev)

    meta = poser.dataset_meta
    n = meta.get("num_keypoints", 0)
    kp_names = [meta["keypoint_id2name"][i] for i in range(n)]
    print(f"Keypoints ({n}): {kp_names}\n")

    # Load wrist positions for crop-based detection
    wrist_data = {}
    if args.pose2d_dir:
        for view_id in cameras:
            pose2d_path = os.path.join(args.pose2d_dir, f"{view_id}_kp2d.json")
            if os.path.isfile(pose2d_path):
                wrist_data[view_id] = _load_wrist_positions(pose2d_path)
                print(f"  loaded {len(wrist_data[view_id])} frames of wrist data for {view_id}")
            else:
                print(f"  WARNING: no pose2d for {view_id}, crop detection disabled for this view")

    for view_id, filename in cameras.items():
        video_path = os.path.join(args.src, filename)
        print(f"=== {view_id} ({filename}) ===")
        doc = run_racket_view(detector, poser, video_path, view_id, kp_names,
                              wrist_positions=wrist_data.get(view_id),
                              crop_size=args.crop_size,
                              crop_target=args.crop_target)

        out_path = os.path.join(args.out, f"{view_id}_racket2d.json")
        with open(out_path, "w") as fh:
            json.dump(doc, fh)
        print(f"  wrote {out_path}\n")

    print("Done! Copy the racket2d files back to your local data/racket2d/")


if __name__ == "__main__":
    main()
