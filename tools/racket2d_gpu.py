"""RacketVision 5-keypoint racket detection (GPU).

Standalone script — runs on any machine with a GPU + OpenMMLab stack.
Uses the pretrained RacketVision model (RTMDet-M detector + RTMPose-M,
5 keypoints: top, bottom, handle, left, right).

First-time setup (run once):
    # Clone and download checkpoints
    git clone --depth 1 https://github.com/OrcustD/RacketVision.git ~/RacketVision
    cd ~/RacketVision/source && python download_checkpoints.py --module RacketPose
    # (record `git -C ~/RacketVision rev-parse HEAD`; tools/racket_finetune/run_runpod.sh pins it via
    #  RACKETVISION_REF. The checkpoints' SHA-256 are printed at start; RACKET_DET_SHA256 /
    #  RACKET_POSE_SHA256 make a run refuse a different file.)

    # Install OpenMMLab (in your conda env)
    pip install openmim
    mim install mmengine "mmcv>=2.0.0rc4,<2.2.0"
    pip install "mmdet>=3.0.0,<3.3.0" --no-deps
    pip install "mmpose>=1.1.0,<1.4.0" --no-deps
    pip install pycocotools munkres json_tricks

Usage:
    python tools/racket2d_gpu.py --src videos --clip singles_1x --out data/racket2d
    python tools/racket2d_gpu.py --src videos --clip 3-1 --out data/racket2d

    # bootstrap variant: fine-tuned pose model + boxes guided by the rigid multi-view fit
    python tools/racket2d_gpu.py --src videos --clip singles_1x --out data/racket2d_bootstrap \
        --pose-ckpt work_dirs/rtmpose_m_racket_ft/best.pth --guides-dir data/racket_guides --pose-fullres

Outputs per-view JSON with 5 keypoints per frame (or null).
"""
from __future__ import annotations

import argparse
import contextlib
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


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _pose_on_boxes(poser, img, boxes, source):
    """Run the pose model on xyxy boxes (in img pixels); returns candidate dicts."""
    from mmpose.apis import inference_topdown
    if len(boxes) == 0:
        return []
    out = []
    results = inference_topdown(poser, img, np.asarray(boxes, dtype=np.float32), bbox_format="xyxy")
    for ds, bb in zip(results, boxes):
        kp = ds.pred_instances.keypoints[0].tolist()
        ks = ds.pred_instances.keypoint_scores[0].tolist()
        out.append({"bbox": [round(float(v), 1) for v in bb], "det_score": 0.0,
                    "keypoints": [[round(p[0], 2), round(p[1], 2)] for p in kp],
                    "keypoint_scores": [round(float(s), 4) for s in ks], "source": source})
    return out


def run_racket_view(detector, poser, video_path, view_id, kp_names,
                    wrist_positions=None, crop_size=400, crop_target=800,
                    guides=None, pose_fullres=False, max_frames=None):
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
        if max_frames is not None and i >= max_frames:
            break
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
        if pose_fullres:
            boxes = [[float(v) / scale for v in bb[b]] for b in order]
            cands = _pose_on_boxes(poser, frame, boxes, "full")
            for c, b in zip(cands, order):
                c["det_score"] = round(float(sc[b]), 4)
        else:
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

        # Guided pass: pose at full resolution where the multi-view rigid fit puts the racket
        if guides is not None:
            gboxes = [g["bbox"] for g in guides.get(str(i), [])]
            gboxes = [[max(0.0, b[0]), max(0.0, b[1]), min(src_w - 1.0, b[2]), min(src_h - 1.0, b[3])]
                      for b in gboxes if b[2] > 0 and b[3] > 0 and b[0] < src_w and b[1] < src_h]
            guided = _pose_on_boxes(poser, frame, gboxes, "guide")
            if guided:
                cands = guided + [c for c in cands if all(_iou(c["bbox"], g["bbox"]) < 0.5 for g in guided)]
                cands = cands[:TOPK]

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


@contextlib.contextmanager
def full_pickle_torch_load(torch):
    """torch >= 2.6 defaults to weights_only=True, which the OpenMMLab checkpoints (pickled configs) fail.

    Allow full unpickling only while the two models are built, instead of replacing torch.load for the
    whole process (the checkpoints are hash-reported / pinnable, see main)."""
    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        yield
    finally:
        torch.load = original


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
    ap.add_argument("--pose-ckpt", default=None,
                    help="Fine-tuned RTMPose checkpoint (default: RacketVision best_PCK_epoch_90.pth)")
    ap.add_argument("--guides-dir", default=None,
                    help="Per-view <view>_guides.json boxes from tools/racket_pseudolabel.py guides")
    ap.add_argument("--pose-fullres", action="store_true",
                    help="Detect at 1080p but run the pose model on the full-resolution frame")
    ap.add_argument("--max-frames", type=int, default=None, help="stop early (smoke tests)")
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
    if args.pose_ckpt:
        pose_ckpt = args.pose_ckpt
        print(f"pose checkpoint override: {pose_ckpt}")

    _apply_shims()

    import cv2
    import torch
    # Containers often expose every host core but enforce a small CPU quota.
    n_threads = int(os.environ.get("RACKET_THREADS", "0"))
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        cv2.setNumThreads(n_threads)

    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose
    from mmpose.utils import adapt_mmdet_pipeline

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"torch: {torch.__version__}, device: {dev}")

    from integrity import verify_sha256
    verify_sha256(det_ckpt, os.environ.get("RACKET_DET_SHA256"), "RacketVision detector checkpoint")
    verify_sha256(pose_ckpt, os.environ.get("RACKET_POSE_SHA256"), "racket pose checkpoint")
    with full_pickle_torch_load(torch):
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
        guides = None
        if args.guides_dir:
            gpath = os.path.join(args.guides_dir, f"{view_id}_guides.json")
            if os.path.isfile(gpath):
                with open(gpath, "r", encoding="utf-8") as fh:
                    guides = json.load(fh)["frames"]
                print(f"  guides: {len(guides)} frames")
            else:
                print(f"  WARNING: no guides for {view_id}")
        doc = run_racket_view(detector, poser, video_path, view_id, kp_names,
                              wrist_positions=wrist_data.get(view_id),
                              crop_size=args.crop_size,
                              crop_target=args.crop_target,
                              guides=guides, pose_fullres=args.pose_fullres,
                              max_frames=args.max_frames)
        doc["pose_checkpoint"] = os.path.basename(pose_ckpt)
        doc["guided"] = guides is not None
        doc["pose_fullres"] = bool(args.pose_fullres)

        out_path = os.path.join(args.out, f"{view_id}_racket2d.json")
        with open(out_path, "w") as fh:
            json.dump(doc, fh)
        print(f"  wrote {out_path}\n")

    print("Done! Copy the racket2d files back to your local data/racket2d/")


if __name__ == "__main__":
    main()
