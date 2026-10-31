"""Round-6 experiment (test branch only): export the racket and shuttle networks to ONNX (rs_nets.py) and check them.

    python tools/experiments/rs_export.py export        # -> tools/experiments/models/*.onnx (+ models.json)
    python tools/experiments/rs_export.py check --video <back_051 .mov> --view back_051 [--frames 40]

Inputs (needs torch, CPU is enough; nothing here runs in the live processes):
    models/racketvision/epoch_300.pth            RacketVision RTMDet-M racket detector (Hugging Face
                                                 linfeng302/RacketVision-Models, MIT)
    data/racket_finetune_run/best_final_epoch_60.pth   our fine-tuned RTMPose-M racket model ("bootstrap")
    models/tracknetv3/ckpts/TrackNet_best.pt     TrackNetV3 (github.com/qaz812345/TrackNetV3, MIT)

The normalisation is inside each graph, so every model takes float32 0-255 BGR pixels (TrackNet: 0-1 RGB, its own
training input) and the live code only resizes / warps:
    racket_det.onnx   (N,3,640,640) BGR 0-255 -> boxes (N,8400,4) xyxy input px, scores (N,8400,3) sigmoid
    racket_pose.onnx  (N,3,256,256) BGR 0-255 -> simcc_x (N,5,512), simcc_y (N,5,512)
    tracknet.onnx     (N,27,288,512) RGB 0-1 (background, then 8 frames) -> heatmaps (N,8,288,512)

check: the exported models against the stored offline results of the same frames - racket keypoints on the guide
boxes (data/racket_guides) vs data/racket2d_bootstrap (same model, 4K crops, flip test), the detector + keypoints vs
the bootstrap file's detector candidates, TrackNet vs data/shuttle (TrackNetV3 + InpaintNet, mixed pixel scale)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "models")
SRC = {
    "racket_det": os.path.join(REPO, "models", "racketvision", "epoch_300.pth"),
    "racket_pose": os.path.join(REPO, "data", "racket_finetune_run", "best_final_epoch_60.pth"),
    "tracknet": os.path.join(REPO, "models", "tracknetv3", "ckpts", "TrackNet_best.pt"),
}


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(name):
    import torch
    sys.path.insert(0, HERE)
    import rs_nets as N

    class Wrap(torch.nn.Module):
        def __init__(self, net, mean, std, rgb):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
            self.rgb = rgb

        def forward(self, x):
            if self.rgb:
                x = x.flip(1)                  # BGR -> RGB
            return self.net((x - self.mean) / self.std)

    if name == "racket_pose":
        sd = N.load_mm_state(SRC[name])
        net = N.RacketPose(sd)
        net.load_state_dict({k: v for k, v in sd.items() if k.split(".")[0] in ("backbone", "head")}, strict=True)
        return Wrap(net.eval(), N.RacketPose.MEAN, N.RacketPose.STD, rgb=True).eval()
    if name == "racket_det":
        sd = N.load_mm_state(SRC[name])
        net = N.RacketDet(sd).load(sd)
        return Wrap(net.eval(), N.RacketDet.MEAN, N.RacketDet.STD, rgb=False).eval()
    if name == "tracknet":
        net, _ = N.load_tracknet(SRC[name])
        return net
    raise ValueError(name)


SHAPES = {"racket_det": (1, 3, 640, 640), "racket_pose": (1, 3, 256, 256), "tracknet": (1, 27, 288, 512)}
OUTPUTS = {"racket_det": ["boxes", "scores"], "racket_pose": ["simcc_x", "simcc_y"], "tracknet": ["heatmaps"]}


def export(names, out_dir=OUT, opset=17):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "models.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    for name in names:
        t0 = time.time()
        m = build(name)
        x = torch.rand(*SHAPES[name]) * (255.0 if name != "tracknet" else 1.0)
        path = os.path.join(out_dir, f"{name}.onnx")
        dyn = {"input": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS[name]}}
        with torch.no_grad():
            torch.onnx.export(m, (x,), path, input_names=["input"], output_names=OUTPUTS[name], dynamic_axes=dyn,
                              opset_version=opset, do_constant_folding=True, dynamo=False)
            ref = [t.numpy() for t in (m(x) if isinstance(m(x), tuple) else (m(x),))]
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": x.numpy()})
        diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
        meta[name] = {"file": f"{name}.onnx", "input": list(SHAPES[name]), "outputs": OUTPUTS[name],
                      "source": os.path.relpath(SRC[name], REPO).replace("\\", "/"), "source_sha256": _sha(SRC[name]),
                      "onnx_sha256": _sha(path), "mb": round(os.path.getsize(path) / 2 ** 20, 1),
                      "torch_vs_onnx_max_abs": diff}
        print(f"{name}: {path} {meta[name]['mb']} MB, torch vs onnxruntime max |diff| {diff:.2e} "
              f"({time.time() - t0:.0f} s)", flush=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    for name in names:
        split(os.path.join(out_dir, f"{name}.onnx"))


def split(path, max_mb=95):
    """A model over GitHub's 100 MB file limit -> <file>.part0, .part1, ... (committed; the whole file is gitignored
    and rs_live.model_path joins the parts again, checking models.json's sha256)."""
    size = os.path.getsize(path)
    if size <= max_mb * 2 ** 20:
        return []
    n = -(-size // (max_mb * 2 ** 20))
    step = -(-size // n)
    parts = []
    with open(path, "rb") as fh:
        for i in range(n):
            p = f"{path}.part{i}"
            with open(p, "wb") as out:
                out.write(fh.read(step))
            parts.append(p)
    print(f"{os.path.basename(path)}: {size / 2 ** 20:.1f} MB -> {n} parts of <= {step / 2 ** 20:.1f} MB")
    return parts


# ------------------------------------------------------------------------------------------------ check
def _frames(video, idx):
    """{frame index: BGR image} read sequentially (cv2, as the offline scripts read)."""
    import cv2
    want, out = set(idx), {}
    cap = cv2.VideoCapture(video)
    i = 0
    while want - out.keys():
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = img
        i += 1
    cap.release()
    return out


def check(args):
    import onnxruntime as ort
    sys.path.insert(0, HERE)
    import rs_live as L
    sess = {n: ort.InferenceSession(os.path.join(OUT, f"{n}.onnx"), providers=["CPUExecutionProvider"])
            for n in ("racket_det", "racket_pose", "tracknet")}
    run = {n: (lambda s: (lambda x: s.run(None, {"input": x})))(s) for n, s in sess.items()}
    rep = {}
    view = args.view
    boot = json.load(open(os.path.join(REPO, "data", "racket2d_bootstrap", f"{view}_racket2d.json")))["frames"]
    guides = json.load(open(os.path.join(REPO, "data", "racket_guides", f"{view}_guides.json")))["frames"]
    cand = [int(k) for k, v in guides.items() if v and boot[int(k)]]
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice([c for c in cand if c < args.max_frame], size=min(args.frames, len(cand)), replace=False))
    imgs = _frames(args.video, idx)
    print(f"{view}: {len(imgs)} frames {idx[:6]}...", flush=True)

    # 1) racket keypoints on the guide boxes (4K) vs the bootstrap file's guided candidates (they come first)
    pose = L.RacketPose(run["racket_pose"], flip=True)
    errs, n_c = [], 0
    for f in idx:
        img = imgs[f]
        h, w = img.shape[:2]
        gb = [g["bbox"] for g in guides[str(f)]]
        gb = [[max(0.0, b[0]), max(0.0, b[1]), min(w - 1.0, b[2]), min(h - 1.0, b[3])]
              for b in gb if b[2] > 0 and b[3] > 0 and b[0] < w and b[1] < h]
        if not gb:
            continue
        kp, sc = pose(img, np.asarray(gb, np.float32))
        for j in range(len(gb)):
            ref = np.asarray(boot[f][j], np.float32)
            errs.append(np.linalg.norm(kp[j] - ref[:, :2], axis=1))
            n_c += 1
    e = np.concatenate(errs) if errs else np.zeros(0)
    rep["racket_pose_vs_bootstrap_px"] = {"candidates": n_c, "median": float(np.median(e)),
                                          "p95": float(np.percentile(e, 95)), "max": float(e.max())}
    print("racket keypoints on guide boxes vs bootstrap file:", rep["racket_pose_vs_bootstrap_px"], flush=True)

    # 2) detector (1080-tall image as offline) + keypoints at 4K vs the bootstrap file's detector candidates
    det = L.RacketDet(run["racket_det"])
    errs, matched, total = [], 0, 0
    for f in idx:
        img = imgs[f]
        h, w = img.shape[:2]
        s = 1080.0 / h
        import cv2
        small = cv2.resize(img, (int(round(w * s)), 1080), interpolation=cv2.INTER_AREA)
        boxes, scores = det(small, score_min=0.03, max_ratio=0.5, topk=5)
        n_g = len(guides[str(f)])
        ref_c = [np.asarray(c, np.float32) for c in boot[f][n_g:]]
        if not len(boxes) or not ref_c:
            total += len(ref_c)
            continue
        kp, _ = pose(img, boxes / s)
        for rc in ref_c:
            total += 1
            d = [float(np.median(np.linalg.norm(k - rc[:, :2], axis=1))) for k in kp]
            j = int(np.argmin(d))
            if d[j] < 20:
                matched += 1
                errs.append(np.linalg.norm(kp[j] - rc[:, :2], axis=1))
    e = np.concatenate(errs) if errs else np.zeros(1)
    rep["racket_det_vs_bootstrap"] = {"detector_candidates": total, "matched": matched,
                                      "kp_median_px": float(np.median(e)), "kp_p95_px": float(np.percentile(e, 95))}
    print("detector + keypoints vs the bootstrap file's detector candidates:", rep["racket_det_vs_bootstrap"],
          flush=True)

    # 3) TrackNet: one non-overlapping 8-frame window at a time from frame 0, background = median of the frames
    if args.shuttle_frames:
        import cv2
        shut = json.load(open(os.path.join(REPO, "data", "shuttle", f"{view}_shuttle.json")))
        cap = cv2.VideoCapture(args.video)
        seq = []
        for _ in range(args.shuttle_frames):
            ok, img = cap.read()
            if not ok:
                break
            seq.append(img)
        cap.release()
        tn = L.TrackNetRunner(run["tracknet"], stride=8)
        tn.set_background(L.median_background(seq[::max(1, len(seq) // 60)]))
        pos = tn.offline(seq)
        rows = []
        for sc_ in (1.0, 1 / 1.5):
            d = [np.linalg.norm(np.asarray(p, float) - np.asarray(shut["frames"][i], float) / sc_)
                 for i, p in enumerate(pos) if p is not None and shut["frames"][i] is not None]
            rows.append((sc_, len(d), float(np.median(d)) if d else None,
                         float(np.mean(np.asarray(d) < 10)) if d else None))
        both = sum(p is not None and shut["frames"][i] is not None for i, p in enumerate(pos))
        rep["tracknet_vs_stored"] = {"frames": len(pos), "live_found": sum(p is not None for p in pos),
                                     "stored_found": sum(shut["frames"][i] is not None for i in range(len(pos))),
                                     "both": both,
                                     "by_scale": [{"stored_scale": round(1 / s_, 3), "n": n, "median_px": m,
                                                   "within_10px": w_} for s_, n, m, w_ in rows]}
        print("TrackNet vs stored shuttle 2D:", rep["tracknet_vs_stored"], flush=True)
    out = os.path.join(OUT, f"check_{view}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("->", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--only", nargs="*", default=["racket_det", "racket_pose", "tracknet"])
    c = sub.add_parser("check")
    c.add_argument("--video", required=True)
    c.add_argument("--view", default="back_051")
    c.add_argument("--frames", type=int, default=30)
    c.add_argument("--max-frame", type=int, default=600)
    c.add_argument("--shuttle-frames", type=int, default=240)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export(a.only)
    else:
        check(a)


if __name__ == "__main__":
    main()
