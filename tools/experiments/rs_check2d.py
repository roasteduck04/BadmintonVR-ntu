"""Round-6 experiment (test branch only, laptop / CPU): what the live racket boxes cost in 2D, before any 3D.

    python tools/experiments/rs_check2d.py --view back_051 [--frames 30] [--max-frame 900]

Reference: the offline bootstrap racket of the same frames (data/racket2d_bootstrap: fine-tuned model on the guide
boxes of the multi-view rigid fit, 4K crops, flip test) - the best 2D racket we have. Each live way of finding the
racket runs on the frame shrunk to the live 1080 short side:
    guide4k    the offline guide boxes on the 4K frame (should reproduce the reference exactly)
    guide1080  the same boxes on the 1080 frame (what the live resolution alone costs)
    noflip     guide1080 without the flip test (half the racket-model work)
    det        RacketVision detector + keypoints on the 1080 frame
    wrist k/p  boxes around both wrists of every offline-2D person (k racket lengths wide, centred p racket lengths
               beyond the wrist), no detector
For every reference racket: found (some candidate within 40 px median keypoint distance at 4K, ~10 px at 1080) and
the error of the nearest candidate (4K px). Writes tools/experiments/models/check2d_<view>.json."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import rs_live as L  # noqa: E402

VIDEOS = {"back_051": "15.8 back 0.5-1.mov", "backleft_051": "15.8 back left 0.5-1.mov",
          "backright_051": "15.8 back right 0.5-1.mov", "frontleft_051": "15.8 front left 0.5-1.mov",
          "frontright_051": "15.8 front right 0.5-1.mov"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", default="back_051")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--max-frame", type=int, default=900)
    ap.add_argument("--found-px", type=float, default=40.0)
    a = ap.parse_args(argv)
    v = a.view
    boot = json.load(open(os.path.join(REPO, "data", "racket2d_bootstrap", f"{v}_racket2d.json")))["frames"]
    guides = json.load(open(os.path.join(REPO, "data", "racket_guides", f"{v}_guides.json")))["frames"]
    kp2d = json.load(open(os.path.join(REPO, "data", "pose2d", f"{v}_kp2d.json")))["frames"]
    cand = [int(k) for k, g in guides.items() if g and boot[int(k)] and int(k) < a.max_frame]
    idx = sorted(np.random.default_rng(1).choice(cand, size=min(a.frames, len(cand)), replace=False).tolist())
    want, imgs = set(idx), {}
    cap = cv2.VideoCapture(os.path.join(REPO, "data", "15-08", "singles", "0.5", VIDEOS[v]))
    i = 0
    while want - imgs.keys():
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            imgs[i] = img
        i += 1
    cap.release()
    import onnxruntime as ort
    mk = lambda n: (lambda s: (lambda x: s.run(None, {"input": x})))(  # noqa: E731
        ort.InferenceSession(L.model_path(n), providers=["CPUExecutionProvider"]))
    pose, pose_nf, det = L.RacketPose(mk("racket_pose")), L.RacketPose(mk("racket_pose"), flip=False), \
        L.RacketDet(mk("racket_det"))
    cfgs = (["guide4k", "guide1080", "noflip", "det", "det+refine"]
            + [f"wrist {k}/{p}" for k in (1.2, 1.6, 2.0) for p in (0.0, 0.3)]
            + [f"wrist {k}/0.3+refine" for k in (1.2, 1.6, 2.0)])
    acc = {c: {"found": 0, "refs": 0, "err": [], "crops": 0} for c in cfgs}
    for f in idx:
        img = imgs[f]
        h, w = img.shape[:2]
        s = 1080.0 / min(h, w)
        small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        gb = [g["bbox"] for g in guides[str(f)]]
        gb = np.asarray([[max(0.0, b[0]), max(0.0, b[1]), min(w - 1.0, b[2]), min(h - 1.0, b[3])]
                         for b in gb if b[2] > 0 and b[3] > 0 and b[0] < w and b[1] < h], np.float32).reshape(-1, 4)
        refs = [np.asarray(c, np.float32)[:, :2] for c in boot[f][:len(gb)]]
        if not refs:
            continue
        persons = [np.asarray(p["keypoints"], np.float32) for p in (kp2d[f] or [])]
        out = {"guide4k": pose(img, gb)[0], "guide1080": pose(small, gb * s)[0] / s,
               "noflip": pose_nf(small, gb * s)[0] / s}
        db, _ = det(small)
        out["det"] = pose(small, db)[0] / s if len(db) else np.zeros((0, 5, 2))
        out["det+refine"] = pose(small, db, refine=True)[0] / s if len(db) else np.zeros((0, 5, 2))
        for k in (1.2, 1.6, 2.0):
            for p in (0.0, 0.3):
                bx = [L.wrist_boxes(q[:17, :2] * s, q[:17, 2], k, p)[0] for q in persons]
                bx = np.concatenate(bx) if bx else np.zeros((0, 4), np.float32)
                out[f"wrist {k}/{p}"] = pose(small, bx)[0] / s if len(bx) else np.zeros((0, 5, 2))
                if p == 0.3:
                    out[f"wrist {k}/{p}+refine"] = (pose(small, bx, refine=True)[0] / s if len(bx)
                                                     else np.zeros((0, 5, 2)))
        for c in cfgs:
            got = out[c]
            acc[c]["crops"] += len(got)
            for r in refs:
                acc[c]["refs"] += 1
                if not len(got):
                    continue
                d = [float(np.median(np.linalg.norm(g - r, axis=1))) for g in got]
                j = int(np.argmin(d))
                if d[j] < a.found_px:
                    acc[c]["found"] += 1
                    acc[c]["err"].append(d[j])
    rep = {"view": v, "frames": len(idx), "found_px_4k": a.found_px, "configs": {}}
    print(f"{v}: {len(idx)} frames; reference = offline bootstrap racket (4K px; /{1 / s:.1f} for 1080 px)")
    for c in cfgs:
        e = np.asarray(acc[c]["err"])
        rep["configs"][c] = {"found": acc[c]["found"], "refs": acc[c]["refs"],
                             "found_frac": round(acc[c]["found"] / max(acc[c]["refs"], 1), 3),
                             "err_median_px": round(float(np.median(e)), 2) if len(e) else None,
                             "err_p90_px": round(float(np.percentile(e, 90)), 2) if len(e) else None,
                             "crops_per_frame": round(acc[c]["crops"] / len(idx), 2)}
        r = rep["configs"][c]
        print(f"  {c:18s} found {r['found']}/{r['refs']} ({100 * r['found_frac']:.0f}%), error median "
              f"{r['err_median_px']} p90 {r['err_p90_px']} px, {r['crops_per_frame']} crops/frame")
    out_p = os.path.join(HERE, "models", f"check2d_{v}.json")
    with open(out_p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("->", out_p)


if __name__ == "__main__":
    main()
