"""Round-6 experiment (test branch only): run the exported racket / shuttle models (rs_export.py) on frames - the
pre- and post-processing of the offline scripts (racket2d_gpu.py: mmdet / mmpose; shuttle2d_gpu.py: TrackNetV3 predict.py)
in NumPy + OpenCV, so a live 2D process needs only onnxruntime (CUDA or TensorRT).

Each model wrapper takes `run(x) -> list of outputs` (an onnxruntime session's run, see `session()`), so the same code
runs on the CPU (laptop check), CUDA or TensorRT.

- RacketDet: RTMDet-M 640x640 (keep-ratio resize, pad 114), class 0 (badminton racket), score >= 0.05 then NMS 0.6
  (mmdet test_cfg), offline filters on top (score >= 0.03, box under half the image, best 5).
- RacketPose: RTMPose-M 256x256 on boxes (centre, x1.25 padding, square), optional flip test (offline: on), SimCC
  argmax / 2, score = min(max_x, max_y) (raw SimCC, not a probability).
- wrist_boxes: racket boxes from the live body keypoints (no detector): a square around each wrist, pushed along the
  forearm, side = k x forearm length (guide boxes of the offline bootstrap racket are ~1.2-1.6 racket lengths).
- TrackNetRunner: TrackNetV3, 8 frames + background at 288x512 (frames squashed to it, as offline), a call every
  `stride` frames per camera; overlapping calls (stride < 8) are averaged (offline: stride 1, positional weights);
  heatmap > 0.5 -> largest blob's box centre."""
from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
FLIP_IDX = [0, 1, 2, 4, 3]          # top, bottom, handle, left <-> right


def _nms(boxes, scores, iou_thr):
    """Greedy NMS (same as analysis.live.det_split._nms) -> kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(area[i] + area[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, np.int64)


def model_path(name, model_dir=MODELS):
    """models/<name>.onnx; a model committed in parts (<name>.onnx.part0, .part1, ... - GitHub refuses files over
    100 MB) is joined once into the same folder (the joined file is gitignored) and checked against models.json."""
    path = os.path.join(model_dir, f"{name}.onnx")
    if os.path.exists(path):
        return path
    parts = sorted((p for p in os.listdir(model_dir) if p.startswith(f"{name}.onnx.part")),
                   key=lambda p: int(p.rsplit("part", 1)[1]))
    if not parts:
        raise FileNotFoundError(f"{path} (run tools/experiments/rs_export.py export)")
    import hashlib
    import json
    h = hashlib.sha256()
    tmp = path + ".tmp"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(os.path.join(model_dir, p), "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
                    out.write(b)
    want = (json.load(open(os.path.join(model_dir, "models.json"), encoding="utf-8")).get(name) or {}).get("onnx_sha256")
    if want and h.hexdigest() != want:
        os.remove(tmp)
        raise RuntimeError(f"{name}: joined parts do not match models.json (sha256 {h.hexdigest()[:12]} != {want[:12]})")
    os.replace(tmp, path)
    return path


def session(name, ep="cpu", rcfg=None, shapes=None, model_dir=MODELS):
    """onnxruntime session for models/<name>.onnx: ep cpu | cuda | trt (TensorRT FP16 through pose2d_fast's helpers,
    engines cached like the pose model's) | trt32 (TensorRT FP32, its own engine cache). Returns (run, providers)."""
    import onnxruntime as ort
    if ep == "trt32":
        ep, rcfg = "trt", dict(rcfg or {}, trt_fp16=False)
    path = model_path(name, model_dir)
    if ep in ("cuda", "trt"):
        from analysis.live import pose2d_fast as pf
        pf.check_cuda("cuda")
    if ep == "trt":               # (a command of its own calls pf.ensure_trt_library_path() first: it restarts)
        pf.preload_tensorrt()
        sess = pf.trt_session(path, dict(rcfg or {}), shapes)
    else:
        # cuDNN: onnxruntime's default EXHAUSTIVE search runs for every new batch size (minutes over a stream's
        # warm-up sizes); HEURISTIC picks at once (rtmpose.exp_rs_cudnn_search exhaustive | heuristic | default)
        search = str((rcfg or {}).get("exp_rs_cudnn_search") or "heuristic").upper()
        prov = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}), "CPUExecutionProvider"]
                if ep == "cuda" else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        if ep == "cpu" and rcfg and rcfg.get("_threads"):
            so.intra_op_num_threads = int(rcfg["_threads"])
        sess = ort.InferenceSession(path, sess_options=so, providers=prov)
        if ep == "cuda" and sess.get_providers()[:1] != ["CUDAExecutionProvider"]:
            raise RuntimeError(f"{name}: onnxruntime could not use CUDA (got {sess.get_providers()})")
    inp = sess.get_inputs()[0].name
    return (lambda x: sess.run(None, {inp: np.ascontiguousarray(x, np.float32)})), sess.get_providers()


# ------------------------------------------------------------------------------------------------ racket
class RacketDet:
    def __init__(self, run, size=640, cls=0, score_thr=0.05, iou=0.6):
        self.run, self.size, self.cls, self.thr, self.iou = run, size, cls, score_thr, iou

    def prep(self, img):
        h, w = img.shape[:2]
        sf = min(self.size / max(h, w), self.size / min(h, w))
        nw, nh = int(w * sf + 0.5), int(h * sf + 0.5)
        x = np.full((self.size, self.size, 3), 114, np.uint8)
        x[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return x.transpose(2, 0, 1)[None].astype(np.float32), (nw / w, nh / h), (nw, nh)

    def post(self, boxes, scores, fac, shape, img_hw, score_min=0.03, max_ratio=0.5, topk=5):
        s = scores[:, self.cls]
        keep = s > max(self.thr, score_min)
        b, s = boxes[keep].copy(), s[keep]
        if not len(s):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, shape[0])
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, shape[1])
        ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        b, s = b[ok], s[ok]
        k = _nms(b, s, self.iou)[:100]
        b, s = b[k] / np.array([fac[0], fac[1], fac[0], fac[1]], np.float32), s[k]
        h, w = img_hw
        if max_ratio:
            ok = ((b[:, 2] - b[:, 0]) / w < max_ratio) & ((b[:, 3] - b[:, 1]) / h < max_ratio)
            b, s = b[ok], s[ok]
        return b[:topk].astype(np.float32), s[:topk].astype(np.float32)

    def __call__(self, img, score_min=0.03, max_ratio=0.5, topk=5):
        x, fac, shape = self.prep(img)
        boxes, scores = self.run(x)
        return self.post(boxes[0], scores[0], fac, shape, img.shape[:2], score_min, max_ratio, topk)

    def batch(self, imgs, **kw):
        """Several images in one call (one per camera)."""
        if not imgs:
            return []
        preps = [self.prep(im) for im in imgs]
        boxes, scores = self.run(np.concatenate([p[0] for p in preps]))
        return [self.post(boxes[i], scores[i], p[1], p[2], im.shape[:2], **kw)
                for i, (p, im) in enumerate(zip(preps, imgs))]


class RacketPose:
    def __init__(self, run, flip=True, size=256, padding=1.25, split=2.0):
        self.run, self.flip, self.size, self.pad, self.split = run, flip, size, padding, split

    def warp(self, img, boxes):
        """Crops (n, 3, S, S) + (centre, side) per box, mmpose GetBBoxCenterScale + TopdownAffine (aspect 1)."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * self.pad
        side = np.maximum(side, 1.0)
        S = self.size
        crops = np.empty((len(boxes), S, S, 3), np.uint8)
        for i in range(len(boxes)):
            s = S / side[i]
            M = np.array([[s, 0, S / 2 - s * c[i, 0]], [0, s, S / 2 - s * c[i, 1]]], np.float32)
            crops[i] = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR)
        return crops, c, side

    def infer(self, crops):
        """crops (n, S, S, 3) uint8 BGR -> locations in crop pixels (n, 5, 2), scores (n, 5)."""
        n = len(crops)
        x = crops.transpose(0, 3, 1, 2).astype(np.float32)
        if self.flip:
            x = np.concatenate([x, x[..., ::-1]])
        sx, sy = self.run(x)
        if self.flip:
            fx, fy = sx[n:][:, FLIP_IDX, ::-1], sy[n:][:, FLIP_IDX]
            sx, sy = (sx[:n] + fx) * 0.5, (sy[:n] + fy) * 0.5
        loc = np.stack([sx.argmax(-1), sy.argmax(-1)], -1).astype(np.float32) / self.split
        vals = np.minimum(sx.max(-1), sy.max(-1))
        loc[vals <= 0] = -1
        return loc, vals

    def to_image(self, loc, c, side):
        return loc / self.size * side[:, None, None] + c[:, None, :] - 0.5 * side[:, None, None]

    def __call__(self, img, boxes, refine=False):
        if not len(boxes):
            return np.zeros((0, 5, 2), np.float32), np.zeros((0, 5), np.float32)
        crops, c, side = self.warp(img, boxes)
        loc, vals = self.infer(crops)
        kp = self.to_image(loc, c, side)
        return self.refine(img, kp, vals) if refine else (kp, vals)

    def refine(self, img, kp, vals, scale=1.15, min_side=24.0):
        """Second pass on a tight box around each first-pass racket (the model was fine-tuned on tight guide boxes;
        a loose wrist box leaves the racket small in the crop). Kept where the mean score rises."""
        if not len(kp):
            return kp, vals
        boxes = np.stack([keypoint_box(k, scale, min_side) for k in kp])
        crops, c, side = self.warp(img, boxes)
        loc, v2 = self.infer(crops)
        kp2 = self.to_image(loc, c, side)
        better = v2.mean(1) > vals.mean(1)
        return np.where(better[:, None, None], kp2, kp), np.where(better[:, None], v2, vals)


RACKET_PER_FOREARM = 2.6           # racket 0.665 m / forearm ~0.26 m


def wrist_boxes(kps, scores, k=1.6, push=0.3, score_min=2.0, min_side=40.0, wrists=(9, 10)):
    """Racket boxes from one person's body keypoints (COCO-17 order in the first 17 rows: elbows 7 / 8, wrists 9 / 10;
    raw wholebody-performance scores): for each wrist seen (score >= score_min), a square of side k racket lengths
    (racket = 2.6 forearms; at least min_side px), centred `push` racket lengths beyond the wrist along elbow -> wrist
    (the grip is in the hand; the head can point anywhere, mostly away from the elbow). The box is then padded x1.25 by
    the pose model, like the offline guide boxes. -> (m, 4) boxes, wrist ids (m,)."""
    out, ids = [], []
    for w in wrists:
        e = w - 2
        if scores[w] < score_min:
            continue
        we, ww = np.asarray(kps[e, :2], np.float32), np.asarray(kps[w, :2], np.float32)
        fore = ww - we
        L = float(np.linalg.norm(fore)) if scores[e] >= score_min else 0.0
        if L < 1.0:
            R, d = min_side / max(k, 1e-6), np.zeros(2, np.float32)
        else:
            R, d = L * RACKET_PER_FOREARM, fore / L
        side = max(k * R, min_side)
        c = ww + d * push * R
        out.append([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2])
        ids.append(w)
    return np.asarray(out, np.float32).reshape(-1, 4), np.asarray(ids, np.int64)


def keypoint_box(kp, scale=1.0, min_side=24.0):
    """Tight square box around 5 racket keypoints, scaled about its centre (the pose model pads it x1.25)."""
    lo, hi = kp.min(0), kp.max(0)
    c, half = (lo + hi) / 2, max(float((hi - lo).max()), min_side) / 2 * scale
    return np.array([c[0] - half, c[1] - half, c[0] + half, c[1] + half], np.float32)


# ------------------------------------------------------------------------------------------------ shuttle
TN_H, TN_W = 288, 512


def tn_frame(img):
    """BGR frame -> (3, 288, 512) RGB 0-1 (squashed to TrackNet's input, as the offline script)."""
    x = cv2.resize(img, (TN_W, TN_H), interpolation=cv2.INTER_AREA)
    return x[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def median_background(frames):
    """Median image of sampled frames (TrackNetV3 bg_mode concat)."""
    small = [cv2.resize(f, (TN_W, TN_H), interpolation=cv2.INTER_AREA) for f in frames]
    return np.median(np.stack(small), 0).astype(np.uint8)


def heat_to_xy(hm, thr=0.5):
    """(288, 512) heatmap -> (x, y) of the largest blob's box centre in TrackNet pixels, or None."""
    m = (hm > thr).astype(np.uint8)
    if not m.any():
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
    return (int(x + w / 2), int(y + h / 2))


class TrackNetRunner:
    """One camera's TrackNet window. push(frame, img) -> [(frame, (x, y) image px or None, peak), ...] for the frames
    that are final (no later call covers them). stride 8: one call per 8 frames; stride < 8: each frame averaged over
    every call that covers it (`weights`: 'average' or 'weight' = TrackNetV3's positional ensemble)."""
    def __init__(self, run, stride=8, seq=8, weights="average", thr=0.5, max_gap=2):
        self.run, self.stride, self.seq, self.thr, self.max_gap = run, int(stride), seq, thr, int(max_gap)
        if weights == "weight":
            w = np.array([min(i + 1, seq - i) for i in range(seq)], np.float32)
        else:
            w = np.ones(seq, np.float32)
        self.w = w / w.sum()
        self.bg = None
        self.buf = deque(maxlen=seq)
        self.acc = {}                 # frame -> [heatmap sum, weight sum]
        self.since = 0
        self.img_hw = None
        self.calls = 0
        self.infer_ns = 0

    def set_background(self, bg_small_bgr):
        self.bg = bg_small_bgr[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    def reset(self):
        self.buf.clear()
        self.acc.clear()
        self.since = 0

    def add(self, f, img):
        """Queue a frame -> (frames finalised by a gap, due?). When due, call window() / take() (batched callers)."""
        self.img_hw = img.shape[:2]
        out = []
        # consecutive = the stream's own frame step (2 when a 60 fps video is streamed at 30 fps: TrackNetV3 was
        # trained on 30 fps video); a longer gap (dropped frames) or a restart (a new session) starts a new window
        if self.buf and not (0 < f - self.buf[-1][0] <= self.max_gap):
            out = self._flush_all()
            self.reset()
        self.buf.append((f, tn_frame(img)))
        self.since += 1
        due = len(self.buf) == self.seq and self.since >= self.stride
        if due:
            self.since = 0
        return out, due

    def window(self):
        return np.concatenate([self.bg] + [b[1] for b in self.buf])

    def take(self, hm):
        """Heatmaps (8, 288, 512) of the current window -> the frames now final."""
        self.calls += 1
        for j, (f, _) in enumerate(self.buf):
            a = self.acc.setdefault(f, [np.zeros((TN_H, TN_W), np.float32), 0.0])
            a[0] += hm[j] * self.w[j]
            a[1] += self.w[j]
        first = self.buf[0][0]
        done = sorted(self.acc) if self.stride >= self.seq else sorted(k for k in self.acc if k < first + self.stride)
        return [self._final(k) for k in done]

    def push(self, f, img):
        out, due = self.add(f, img)
        if due:
            t = time.monotonic_ns()
            hm = self.run(self.window()[None])[0][0]
            self.infer_ns += time.monotonic_ns() - t
            out += self.take(hm)
        return out

    def _final(self, k):
        hm, ws = self.acc.pop(k)
        hm = hm / max(ws, 1e-9)
        xy = heat_to_xy(hm, self.thr)
        if xy is None:
            return (k, None, float(hm.max()))
        h, w = self.img_hw
        return (k, (xy[0] * w / TN_W, xy[1] * h / TN_H), float(hm.max()))

    def _flush_all(self):
        return [self._final(k) for k in sorted(self.acc)]

    def offline(self, frames, first=0):
        """Positions (image px or None) for a list of consecutive frames (the end may stay uncovered)."""
        res = {}
        for i, img in enumerate(frames):
            for k, xy, _ in self.push(first + i, img):
                res[k] = xy
        for k, xy, _ in self._flush_all():
            res[k] = xy
        return [res.get(first + i) for i in range(len(frames))]
