"""Round-6 experiment (test branch only): the racket / shuttle models alone - speed on CUDA vs TensorRT FP16 for the
batch sizes a live run uses, and how far TensorRT FP16's answers move from CUDA FP32's on the same real frames.

    python tools/experiments/rs_bench.py --clip singles_05x --preset rtx5090 --out <folder> [--views 3|5] [--frames 64]

Frames: `--frames` consecutive frames of each camera from the clip's live section, decoded and shrunk to the live
1080 short side (as the live 2D does). Per execution provider (cuda, trt):
- racket detector: one camera per call and every camera in one call (ms per call, per camera frame)
- racket keypoints: 4 / 8 / 16 / 32 crops (the detector's boxes of these frames, repeated), flip test on and off
- TrackNet: 1 window and one window per camera in one call; per camera frame = call / stride for stride 8 / 4 / 1
Agreement TensorRT vs CUDA: detector boxes (IoU of the best box), keypoints (px at 1080p), TrackNet positions (px).
Writes rs_bench.json + rs_bench.md into --out."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import rs_live as L  # noqa: E402


def _frames(path, first, n, short=1080):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = []
    while len(out) < n:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        s = short / min(h, w)
        out.append(cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def _time(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        a = time.perf_counter()
        fn()
        t.append((time.perf_counter() - a) * 1000)
    return {"median": round(float(np.median(t)), 3), "p90": round(float(np.percentile(t, 90)), 3)}


def run(cfg, out, n_frames=64, reps=30, eps=("cuda", "trt"), log=print):
    views = list(cfg["views"])
    first = int(round(cfg["section_s"][0] * cfg["source_fps"]))
    t0 = time.perf_counter()
    imgs = {v: _frames(cfg["videos"][v], first, n_frames) for v in views}
    log(f"decoded {n_frames} frames x {len(views)} cameras in {time.perf_counter() - t0:.1f} s "
        f"({imgs[views[0]][0].shape[1]}x{imgs[views[0]][0].shape[0]})")
    rcfg = dict(cfg["rtmpose"])
    nv = len(views)
    res = {"views": views, "frames": n_frames, "decode_hw": list(imgs[views[0]][0].shape[:2]), "ep": {}}
    keep = {}
    for ep in eps:
        R = res["ep"][ep] = {}
        try:
            det_run, pd = L.session("racket_det", ep, rcfg,
                                    {"input": ((1, 3, 640, 640), (3, 3, 640, 640), (5, 3, 640, 640))})
            pose_run, pp = L.session("racket_pose", ep, rcfg,
                                     {"input": ((1, 3, 256, 256), (16, 3, 256, 256), (64, 3, 256, 256))})
            tn_run, pt = L.session("tracknet", ep, rcfg,
                                   {"input": ((1, 27, 288, 512), (3, 27, 288, 512), (5, 27, 288, 512))})
        except Exception as e:  # noqa: BLE001
            R["error"] = f"{type(e).__name__}: {e}"
            log(f"{ep}: {R['error']}")
            continue
        R["providers"] = {"racket_det": pd, "racket_pose": pp, "tracknet": pt}
        det, pose = L.RacketDet(det_run), L.RacketPose(pose_run, flip=True)
        pose_nf = L.RacketPose(pose_run, flip=False)
        one = [imgs[v][0] for v in views]
        R["det_1"] = _time(lambda: det.batch(one[:1]), reps)
        R["det_all"] = _time(lambda: det.batch(one), reps)
        R["det_all"]["per_camera"] = round(R["det_all"]["median"] / nv, 3)
        # every frame's detector boxes (for crops and for the TensorRT vs CUDA check)
        boxes = {v: [det(im)[0] for im in imgs[v]] for v in views}
        crops = []
        for v in views:
            for im, b in zip(imgs[v], boxes[v]):
                if len(b):
                    crops.append(pose.warp(im, b[:2])[0])
        crops = np.concatenate(crops) if crops else np.zeros((1, 256, 256, 3), np.uint8)
        for n in (4, 8, 16, 32):
            c = np.resize(crops, (n, 256, 256, 3))
            R[f"pose_{n}_flip"] = _time(lambda: pose.infer(c), reps)
            R[f"pose_{n}"] = _time(lambda: pose_nf.infer(c), reps)
        # TrackNet windows: background = median of the frames, 8 consecutive frames per camera
        win = {}
        for v in views:
            r = L.TrackNetRunner(tn_run, stride=8)
            r.set_background(L.median_background(imgs[v][::2]))
            for k, im in enumerate(imgs[v][:8]):
                r.add(k, im)
            win[v] = r.window()
        R["tn_1"] = _time(lambda: tn_run(win[views[0]][None]), reps)
        xb = np.stack([win[v] for v in views])
        R["tn_all"] = _time(lambda: tn_run(xb), reps)
        R["tn_per_camera_frame"] = {f"stride_{s}": round(R["tn_all"]["median"] / nv / s, 3) for s in (8, 4, 1)}
        # outputs for the agreement check
        kp = {v: [pose(im, b[:2])[0] if len(b) else np.zeros((0, 5, 2)) for im, b in zip(imgs[v], boxes[v])]
              for v in views}
        sh = {}
        for v in views:
            r = L.TrackNetRunner(tn_run, stride=8)
            r.set_background(L.median_background(imgs[v][::2]))
            sh[v] = r.offline(imgs[v])
        keep[ep] = {"boxes": boxes, "kp": kp, "sh": sh}
        log(f"{ep}: det 1 cam {R['det_1']['median']:.2f} ms, {nv} cams {R['det_all']['median']:.2f} ms; "
            f"racket 16 crops {R['pose_16_flip']['median']:.2f} ms (flip) / {R['pose_16']['median']:.2f} ms; "
            f"TrackNet 1 window {R['tn_1']['median']:.2f} ms, {nv} windows {R['tn_all']['median']:.2f} ms")
    if "cuda" in keep and "trt" in keep:
        a, b = keep["cuda"], keep["trt"]
        ious, kd, sd, sh_both, sh_one = [], [], [], 0, 0
        for v in views:
            for ba, bb, ka, kb in zip(a["boxes"][v], b["boxes"][v], a["kp"][v], b["kp"][v]):
                if len(ba) and len(bb):
                    ious.append(_iou(ba[0], bb[0]))
                    if len(ka) and len(kb):
                        kd.extend(np.linalg.norm(ka[0] - kb[0], axis=1).tolist())
            for pa, pb in zip(a["sh"][v], b["sh"][v]):
                if pa is not None and pb is not None:
                    sh_both += 1
                    sd.append(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
                elif (pa is None) != (pb is None):
                    sh_one += 1
        res["trt_vs_cuda"] = {"det_best_box_iou_median": _med(ious), "racket_kp_px_median": _med(kd),
                              "racket_kp_px_p95": _pct(kd, 95), "shuttle_px_median": _med(sd),
                              "shuttle_px_p95": _pct(sd, 95), "shuttle_found_both": sh_both,
                              "shuttle_found_one_only": sh_one}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rs_bench.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    md = markdown(res)
    with open(os.path.join(out, "rs_bench.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    log(md)
    return res


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / u) if u > 0 else 0.0


def _med(x):
    return round(float(np.median(x)), 3) if len(x) else None


def _pct(x, q):
    return round(float(np.percentile(x, q)), 3) if len(x) else None


def markdown(res):
    nv = len(res["views"])
    L_ = [f"# Racket + shuttle models alone ({nv} cameras, {res['frames']} frames, decoded "
          f"{res['decode_hw'][1]}x{res['decode_hw'][0]})", "",
          "| model | " + " | ".join(res["ep"]) + " |", "|---|" + "---|" * len(res["ep"])]

    def row(name, key, sub="median"):
        vals = []
        for ep, R in res["ep"].items():
            v = R.get(key)
            vals.append("–" if not isinstance(v, dict) or v.get(sub) is None else f"{v[sub]:.2f}")
        L_.append(f"| {name} | " + " | ".join(vals) + " |")
    row("racket detector, 1 camera (ms/call)", "det_1")
    row(f"racket detector, {nv} cameras in one call (ms/call)", "det_all")
    row(f"racket detector per camera frame ({nv} in one call)", "det_all", "per_camera")
    for n in (4, 8, 16, 32):
        row(f"racket keypoints, {n} crops, flip test (ms/call)", f"pose_{n}_flip")
        row(f"racket keypoints, {n} crops, no flip (ms/call)", f"pose_{n}")
    row("TrackNet, 1 window (ms/call)", "tn_1")
    row(f"TrackNet, {nv} windows in one call (ms/call)", "tn_all")
    for s in (8, 4, 1):
        vals = []
        for ep, R in res["ep"].items():
            v = (R.get("tn_per_camera_frame") or {}).get(f"stride_{s}")
            vals.append("–" if v is None else f"{v:.2f}")
        L_.append(f"| TrackNet per camera frame, a call every {s} frame(s) | " + " | ".join(vals) + " |")
    if res.get("trt_vs_cuda"):
        a = res["trt_vs_cuda"]
        L_ += ["", f"TensorRT FP16 vs CUDA FP32 on the same frames: best racket box IoU {a['det_best_box_iou_median']}, "
                   f"racket keypoints {a['racket_kp_px_median']} px median ({a['racket_kp_px_p95']} p95), shuttle "
                   f"{a['shuttle_px_median']} px median ({a['shuttle_px_p95']} p95; found by both "
                   f"{a['shuttle_found_both']}, by one only {a['shuttle_found_one_only']})."]
    for ep, R in res["ep"].items():
        if R.get("error"):
            L_.append(f"\n{ep}: {R['error']}")
        elif R.get("providers"):
            L_.append(f"\n{ep} providers: {R['providers']}")
    return "\n".join(L_) + "\n"


def main(argv=None):
    from analysis.live import config as live_config
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", default="singles_05x")
    ap.add_argument("--preset", default="rtx5090")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--views", type=int, default=None, help="5 = every camera of the clip (default: the live views)")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--ep", default="cuda,trt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if "trt" in a.ep:
        from analysis.live import pose2d_fast as pf
        pf.ensure_trt_library_path()          # Linux: restarts this command once with TensorRT on LD_LIBRARY_PATH
    sets = list(a.set)
    if a.views == 5:
        sets.append("views=[back_051, backleft_051, backright_051, frontleft_051, frontright_051]")
    cfg = live_config.load(a.clip, preset=a.preset, overrides=sets)
    run(cfg, a.out, a.frames, a.reps, tuple(a.ep.split(",")))


if __name__ == "__main__":
    main()
