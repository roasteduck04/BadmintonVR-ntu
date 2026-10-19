"""v0.8.1 fast live 2D: frame readers, a batched pose engine and the worker loop of the live "fast" engine.

    rtmpose.engine: fast      (rtmlib = the v0.8.0 path: rtmlib PoseTracker, one call per person)

Readers (rtmpose.decoder), all forward-only (frames must not go back) and all able to shrink while reading
(rtmpose.decode_short_side, pixels of the shorter side; null = full resolution):
- cv2    OpenCV VideoCapture (+ cv2.resize INTER_AREA): the v0.8.0 path
- pyav   PyAV (libavcodec, frame threads) + swscale resize in C while converting to BGR
- ffmpeg an ffmpeg process (system ffmpeg, else imageio-ffmpeg) writing raw BGR frames scaled by its own filter threads
- nvdec  ffmpeg with -hwaccel cuda: NVDEC decode + scale_cuda on the GPU (needs an ffmpeg built with CUDA)
- nvcodec PyNvVideoCodec (pip): NVDEC decode, frames copied to the host, resized with cv2
Keypoints always come back in full-resolution source pixels (divided by the reader's scale).

Pose engine (FastPose): the rtmlib models (YOLOX detector + RTMPose / RTMW), but
- the detector runs every rtmpose.det_every frames per camera; between detections each person's box comes from its
  last pose (rtmlib's pose_to_bbox, x1.25) - the same rule as rtmlib PoseTracker(tracking=False);
- every person crop of every camera in a batch goes through the pose model in ONE call (dynamic batch axis; chunks of
  rtmpose.pose_batch), instead of one call per person;
- crops are cut straight from the (possibly shrunk) frame with rtmlib's affine, normalised in float32.
Court filter (rtmpose.court_filter_m, metres; null = off): a detector box is kept only if its bottom centre (the
feet) lands on the court floor within the 3D tracker's limits (pose_v2.COURT_X_MAX / COURT_Z_MAX) plus this margin,
through the camera's calibration - bystanders and spectators never reach the pose model (boxes cut by the image's
bottom edge are kept).
Models: wholebody (133 keypoints) | body (COCO-17 -> slots 0-16) | body_feet (Halpe-26: body + 6 foot points -> slots
0-22); rtmpose.mode picks the size (performance | balanced | lightweight), rtmpose.det overrides the detector.

Worker loop (fast_worker): one process for a list of views. rtmpose.batch_views true = one process for every camera,
frames whose capture times fall in one output step are batched together; false = one process per camera (like
v0.8.0). Decoding runs in one prefetch thread per view (it waits for a frame's capture time in realtime pace), so
decoding overlaps the GPU work. Every frame reports decode / wait / detect / pose times."""
from __future__ import annotations

import os
import queue as queue_mod
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KP = 133
DECODERS = ("cv2", "pyav", "ffmpeg", "nvdec", "nvcodec")
ENGINES = ("rtmlib", "fast")
FAST_MODELS = ("wholebody", "body", "body_feet")
# Halpe-26 foot points -> COCO-WholeBody slots (17 L big toe, 18 L small toe, 19 L heel, 20 R big toe, 21 R small toe,
# 22 R heel); Halpe-26: 20 L big toe, 21 R big toe, 22 L small toe, 23 R small toe, 24 L heel, 25 R heel
HALPE26_TO_WB = {i: i for i in range(17)} | {20: 17, 22: 18, 24: 19, 21: 20, 23: 21, 25: 22}

# Raw SimCC scores of the smaller rtmlib models are on another scale than wholebody's (Halpe-26 / body7: ~0.05-1.05;
# wholebody: ~0.7-8.4) and every score gate downstream uses the wholebody calibration (pose24 scores), so their
# detections were all dropped (no 3D candidates). Quantile maps raw score -> wholebody-performance raw score, fitted on
# the same people in the same frames (body 17 keypoints; tools/live_bench/score_map_fit.py, remote PC, 2026-10-19).
SCORE_TO_WB = {
    "body_feet/balanced": (   # n 4998, r 0.858
        [0.0463, 0.2275, 0.2993, 0.3652, 0.4258, 0.4785, 0.5207, 0.5555, 0.585, 0.614, 0.6369, 0.6566, 0.6776, 0.6971,
         0.7168, 0.734, 0.7528, 0.7717, 0.795, 0.831, 1.023],
        [0.7012, 1.9421, 2.375, 2.7102, 2.9837, 3.2187, 3.4218, 3.5863, 3.7159, 3.8426, 3.9792, 4.0963, 4.2157, 4.34,
         4.4641, 4.5831, 4.7179, 4.8562, 5.0211, 5.3021, 8.419]),
    "body_feet/performance": (   # n 4930, r 0.707
        [0.0695, 0.3499, 0.4455, 0.504, 0.5509, 0.585, 0.6178, 0.647, 0.6752, 0.6957, 0.7128, 0.7293, 0.7446, 0.7606,
         0.7755, 0.7911, 0.8088, 0.8273, 0.8535, 0.884, 1.0537],
        [0.7559, 2.1377, 2.4983, 2.8206, 3.0947, 3.3106, 3.4941, 3.6403, 3.7643, 3.8948, 4.0263, 4.1454, 4.266,
         4.3871, 4.4968, 4.6152, 4.7394, 4.8757, 5.0397, 5.3094, 8.419]),
    "body/balanced": (   # n 4998, r 0.861
        [0.0465, 0.2487, 0.3194, 0.3768, 0.4318, 0.4891, 0.5304, 0.5681, 0.6009, 0.6283, 0.6513, 0.6705, 0.6894,
         0.7088, 0.7268, 0.7447, 0.7624, 0.7814, 0.8048, 0.8391, 0.979],
        [0.7012, 1.9421, 2.3696, 2.6967, 2.9704, 3.215, 3.4227, 3.5873, 3.7159, 3.8406, 3.979, 4.0946, 4.2148, 4.3369,
         4.4641, 4.5833, 4.7185, 4.8543, 5.0218, 5.3021, 8.419]),
    "body/performance": (   # n 4930, r 0.691
        [0.0682, 0.3164, 0.3999, 0.4569, 0.5114, 0.5537, 0.585, 0.6148, 0.642, 0.6642, 0.6877, 0.7085, 0.725, 0.7418,
         0.7587, 0.7762, 0.7941, 0.8151, 0.8395, 0.8753, 1.0645],
        [0.7559, 2.1377, 2.4943, 2.8193, 3.093, 3.3103, 3.4944, 3.6396, 3.7627, 3.8931, 4.0235, 4.1429, 4.2619,
         4.3841, 4.4964, 4.6128, 4.7394, 4.8757, 5.0397, 5.3094, 8.419]),
}


def score_to_wb(model, mode, s):
    """Raw scores of `model`/`mode` on the wholebody-performance raw scale (wholebody: unchanged)."""
    if model == "wholebody":
        return s
    m = SCORE_TO_WB.get(f"{model}/{mode}")
    if m is None:
        raise ValueError(f"no score map for rtmpose.model={model} mode={mode} (have {sorted(SCORE_TO_WB)}); "
                         "fit one with tools/live_bench/score_map_fit.py")
    return np.interp(s, m[0], m[1]).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- readers
def target_size(w, h, short_side):
    """(out_w, out_h, scale) for a frame of w x h shrunk so its shorter side is short_side (never enlarged; even sizes)."""
    if not short_side or short_side >= min(w, h):
        return int(w), int(h), 1.0
    s = float(short_side) / min(w, h)
    ow, oh = max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)
    return ow, oh, ow / float(w)


def probe_video(path):
    """(width, height, fps, codec) via OpenCV (always available)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    four = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = "".join(chr((four >> 8 * i) & 0xFF) for i in range(4)).strip().lower()
    return w, h, fps, ("hevc" if codec in ("hvc1", "hev1", "hevc") else "h264" if codec in ("avc1", "h264") else codec)


class Reader:
    kind = "?"

    def __init__(self, path, first, fps, short_side, threads=0):
        self.path, self.fps, self.threads = path, float(fps), int(threads or 0)
        self.src_w, self.src_h, _, self.codec = probe_video(path)
        self.out_w, self.out_h, self.scale = target_size(self.src_w, self.src_h, short_side)
        self.pos = int(first)            # index of the next frame the reader will return

    def read(self, f):
        raise NotImplementedError

    def close(self):
        pass


class Cv2Reader(Reader):
    kind = "cv2"

    def __init__(self, path, first, fps, short_side, threads=0):
        super().__init__(path, first, fps, short_side, threads)
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))

    def read(self, f):
        while self.pos < f:
            self.cap.grab()
            self.pos += 1
        ok, img = self.cap.read()
        self.pos += 1
        if not ok:
            return None
        if self.scale != 1.0:
            img = self.cv2.resize(img, (self.out_w, self.out_h), interpolation=self.cv2.INTER_AREA)
        return img

    def close(self):
        self.cap.release()


class PyAvReader(Reader):
    kind = "pyav"

    def __init__(self, path, first, fps, short_side, threads=0):
        super().__init__(path, first, fps, short_side, threads)
        import av
        self.c = av.open(path)
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        if self.threads:
            self.s.thread_count = self.threads
        self.tb = float(self.s.time_base)
        start_pts = self.s.start_time or 0
        if first > 0:
            self.c.seek(int(start_pts + (first / self.fps) / self.tb), stream=self.s, backward=True, any_frame=False)
        self.start_pts = start_pts
        self.it = self.c.decode(self.s)
        self.pos = -1
        self._next = None

    def _index(self, fr):
        return int(round((fr.pts - self.start_pts) * self.tb * self.fps)) if fr.pts is not None else self.pos + 1

    def read(self, f):
        while True:
            fr = self._next if self._next is not None else next(self.it, None)
            self._next = None
            if fr is None:
                return None
            i = self._index(fr)
            self.pos = i
            if i < f:
                continue
            if i > f:                    # the wanted frame does not exist (gap): keep this one for the next call
                self._next = fr
                return None
            try:
                img = fr.to_ndarray(format="bgr24", width=self.out_w, height=self.out_h, interpolation="AREA")
            except TypeError:            # older PyAV: no width / height / interpolation on to_ndarray
                img = fr.reformat(width=self.out_w, height=self.out_h, format="bgr24").to_ndarray()
            return img

    def close(self):
        self.c.close()


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class FfmpegReader(Reader):
    kind = "ffmpeg"
    hw = False

    def __init__(self, path, first, fps, short_side, threads=0):
        super().__init__(path, first, fps, short_side, threads)
        exe = ffmpeg_exe()
        if not exe:
            raise RuntimeError("no ffmpeg (install it, or pip install imageio-ffmpeg)")
        self.frame_bytes = self.out_w * self.out_h * 3
        args = [exe, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.threads:
            args += ["-threads", str(self.threads)]
        if self.hw:
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if first > 0:
            args += ["-ss", f"{first / self.fps:.6f}"]
        args += ["-i", path, "-an", "-sn"]
        if self.hw:
            vf = (f"scale_cuda={self.out_w}:{self.out_h}," if self.scale != 1.0 else "") + "hwdownload,format=nv12"
        else:
            vf = f"scale={self.out_w}:{self.out_h}:flags=area" if self.scale != 1.0 else None
        if vf:
            args += ["-vf", vf]
        args += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self.args = args
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes * 2)
        self.pos = int(first)

    def _read_raw(self):
        buf = self.p.stdout.read(self.frame_bytes)
        if buf is None or len(buf) < self.frame_bytes:
            err = self.p.stderr.read().decode(errors="replace").strip() if self.p.poll() is not None else ""
            if err:
                raise RuntimeError(f"{self.kind} reader failed: {err[-400:]}")
            return None
        return buf

    def read(self, f):
        while self.pos < f:
            if self._read_raw() is None:
                return None
            self.pos += 1
        buf = self._read_raw()
        self.pos += 1
        return None if buf is None else np.frombuffer(buf, np.uint8).reshape(self.out_h, self.out_w, 3)

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


class NvdecReader(FfmpegReader):
    kind = "nvdec"
    hw = True


class NvCodecReader(Reader):
    """PyNvVideoCodec SimpleDecoder (NVDEC). Frames come back in host memory as RGB and are resized with cv2."""
    kind = "nvcodec"

    def __init__(self, path, first, fps, short_side, threads=0):
        super().__init__(path, first, fps, short_side, threads)
        import cv2
        import PyNvVideoCodec as nvc
        self.cv2, self.nvc = cv2, nvc
        self.dec = nvc.SimpleDecoder(path, gpu_id=0, use_device_memory=False,
                                     output_color_type=nvc.OutputColorType.RGB)
        self.pos = int(first)

    def read(self, f):
        try:
            fr = self.dec[int(f)]
        except (IndexError, RuntimeError):
            return None
        self.pos = int(f) + 1
        img = np.asarray(fr) if not hasattr(fr, "__dlpack__") else np.from_dlpack(fr)
        img = np.ascontiguousarray(img.reshape(self.src_h, self.src_w, 3)[:, :, ::-1])
        if self.scale != 1.0:
            img = self.cv2.resize(img, (self.out_w, self.out_h), interpolation=self.cv2.INTER_AREA)
        return img


READERS = {"cv2": Cv2Reader, "pyav": PyAvReader, "ffmpeg": FfmpegReader, "nvdec": NvdecReader, "nvcodec": NvCodecReader}


def open_reader(kind, path, first, fps, short_side=None, threads=0):
    if kind not in READERS:
        raise ValueError(f"rtmpose.decoder {kind!r} is not one of {DECODERS}")
    return READERS[kind](path, first, fps, short_side, threads)


# ---------------------------------------------------------------------------------------------------- pose engine
def _solution(model, mode, device, det=None):
    from rtmlib import Body, BodyWithFeet, Wholebody
    cls = {"wholebody": Wholebody, "body": Body, "body_feet": BodyWithFeet}[model]
    kw = {"mode": mode, "backend": "onnxruntime", "device": device}
    if det:
        m = cls.MODE[mode]
        kw.update(det=det, det_input_size=m["det_input_size"] if det == m["det"] else (640, 640),
                  pose=m["pose"], pose_input_size=m["pose_input_size"])
    return cls(**kw)


def kp_map(model):
    """Model keypoint index -> COCO-WholeBody slot."""
    if model == "wholebody":
        return np.arange(KP)
    if model == "body":
        return np.arange(17)
    return np.array([HALPE26_TO_WB.get(i, -1) for i in range(26)])


def providers_of(sol):
    out = {}
    for name in ("det_model", "pose_model"):
        sess = getattr(getattr(sol, name, None), "session", None)
        if sess is not None:
            out[name] = list(sess.get_providers())
    return out


def floor_xz(u, v, K, R, t):
    """Court-floor point (X, Z) under pixel (u, v); court world is Y-down with the floor at Y = 0. None if no hit."""
    ray = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    c = -R.T @ t
    if abs(ray[1]) < 1e-10:
        return None
    s = -c[1] / ray[1]
    if s < 0:
        return None
    p = c + s * ray
    return float(p[0]), float(p[2])


class CourtFilter:
    def __init__(self, cams, margin_m, x_max=None, z_max=None):
        from analysis import pose_v2 as pv
        self.cams = {v: tuple(np.asarray(a, float) for a in c) for v, c in cams.items()}
        self.x = (pv.COURT_X_MAX if x_max is None else x_max) + float(margin_m)
        self.z = (pv.COURT_Z_MAX if z_max is None else z_max) + float(margin_m)

    def on_court(self, view, u, v, img_h=None):
        if img_h is not None and v >= img_h - 2:          # feet cut off by the image edge: cannot tell, keep
            return True
        p = floor_xz(u, v, *self.cams[view])
        return p is not None and abs(p[0]) <= self.x and abs(p[1]) <= self.z

    def keep(self, view, boxes, inv_scale, src_h):
        return [b for b in boxes if self.on_court(view, 0.5 * (b[0] + b[2]) * inv_scale, b[3] * inv_scale, src_h)]


def load_cams(paths):
    import json
    out = {}
    for v, p in paths.items():
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        out[v] = (d["K"], d["R"], d["t"])
    return out


class FastPose:
    def __init__(self, rcfg, device):
        from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        from rtmlib.tools.solution.pose_tracker import pose_to_bbox
        self._cs, self._affine, self._simcc, self._p2b = bbox_xyxy2cs, top_down_affine, get_simcc_maximum, pose_to_bbox
        self.model = rcfg.get("model", "wholebody")
        if self.model not in FAST_MODELS:
            raise ValueError(f"rtmpose.model {self.model!r} is not one of {FAST_MODELS} (fast engine)")
        self.sol = _solution(self.model, rcfg.get("mode", "performance"), device, rcfg.get("det"))
        self.det, self.pose = self.sol.det_model, self.sol.pose_model
        self.providers = providers_of(self.sol)
        self.in_w, self.in_h = (int(x) for x in self.pose.model_input_size)
        self.mean = np.asarray(self.pose.mean, np.float32).reshape(1, 1, 1, 3)
        self.std = np.asarray(self.pose.std, np.float32).reshape(1, 1, 1, 3)
        self.map = kp_map(self.model)
        self.mode = rcfg.get("mode", "performance")
        self.det_every = max(1, int(rcfg.get("det_every", 1)))
        self.batch = max(1, int(rcfg.get("pose_batch", 32)))
        search = rcfg.get("cudnn_search")
        if search and device == "cuda":       # rebuild both sessions with this cuDNN algorithm search mode
            import onnxruntime as ort
            prov = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": str(search).upper()}), "CPUExecutionProvider"]
            self.det.session = ort.InferenceSession(self.det.onnx_model, providers=prov)
            self.pose.session = ort.InferenceSession(self.pose.onnx_model, providers=prov)
            self.providers = providers_of(self.sol)
        self.sess = self.pose.session
        self.pad = bool(rcfg.get("pad_batch", True))
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.state = {}                  # view -> {"n": frames seen, "boxes": last boxes (full-frame px of this reader)}
        cf = rcfg.get("court_filter_m")
        self.court = CourtFilter(load_cams(rcfg["_cam_paths"]), cf) if cf is not None and rcfg.get("_cam_paths") else None
        self.geom = {}                   # view -> (1 / reader scale, source height)
        self.filtered = 0

    def set_view(self, view, scale, src_h):
        self.geom[view] = (1.0 / scale, src_h)

    def boxes(self, view, img):
        st = self.state.setdefault(view, {"n": 0, "boxes": []})
        t = time.monotonic_ns()
        if st["n"] % self.det_every == 0 or len(st["boxes"]) == 0:
            b = self.det(img)
            boxes, detected = [np.asarray(x, np.float32) for x in (b if b is not None else [])], True
            if self.court is not None and view in self.geom:
                n0 = len(boxes)
                boxes = self.court.keep(view, boxes, *self.geom[view])
                self.filtered += n0 - len(boxes)
        else:
            boxes, detected = list(st["boxes"]), False
        st["n"] += 1
        return boxes, time.monotonic_ns() - t, detected

    def crops(self, img, boxes):
        out = []
        for bb in boxes:
            c, s = self._cs(np.asarray(bb, np.float32)[:4], padding=1.25)
            crop, s = self._affine((self.in_w, self.in_h), s, c, img)
            out.append((crop, c, s))
        return out

    def bucket(self, n):
        """Padded batch size for n crops: the next power of two, at most the chunk size. ONNX Runtime prepares every
        new input shape again (measured on an RTX 3060: 55 ms per crop on a new size vs 8-9 ms on a seen size), so a
        few fixed sizes beat exact sizes."""
        if not self.pad:
            return n
        b = 1
        while b < n:
            b *= 2
        return min(b, max(n, self.batch))

    def warmup(self, img_hw=(1080, 1920)):
        """Run the detector once and the pose model once per padded size, so no tuning happens during the run."""
        t = time.monotonic_ns()
        h, w = img_hw
        self.det(np.zeros((h, w, 3), np.uint8))
        sizes, b = [], 1
        while b <= self.batch:
            sizes.append(b)
            b *= 2
        if self.batch not in sizes:
            sizes.append(self.batch)
        for n in (sizes if self.pad else [1]):
            x = np.zeros((n, 3, self.in_h, self.in_w), np.float32)
            self.sess.run(self.out_names, {self.in_name: x})
        return (time.monotonic_ns() - t) / 1e9, sizes

    def infer(self, crops):
        """All crops in chunks of self.batch -> (keypoints (N, K, 2) crop-mapped to the image, scores (N, K)).
        Each chunk is zero-padded to self.bucket(n) crops (the padding's outputs are dropped)."""
        if not crops:
            return np.zeros((0, len(self.map), 2), np.float32), np.zeros((0, len(self.map)), np.float32)
        kps, scs = [], []
        for i in range(0, len(crops), self.batch):
            part = crops[i:i + self.batch]
            x = np.stack([p[0] for p in part]).astype(np.float32)
            x = ((x - self.mean) / self.std).transpose(0, 3, 1, 2)
            nb = self.bucket(len(part))
            if nb > len(part):
                x = np.concatenate([x, np.zeros((nb - len(part),) + x.shape[1:], np.float32)])
            sx, sy = self.sess.run(self.out_names, {self.in_name: np.ascontiguousarray(x)})
            locs, vals = self._simcc(sx, sy)
            k = locs / 2.0                                     # simcc split ratio 2
            for j, p in enumerate(part):
                c, s = p[1], p[2]
                kps.append(k[j] / np.array([self.in_w, self.in_h], np.float32) * s + c - s / 2)
                scs.append(vals[j])
        return np.asarray(kps, np.float32), np.asarray(scs, np.float32)

    def finish(self, view, kps):
        """Next frame's boxes for this view from this frame's poses (rtmlib pose_to_bbox, x1.25)."""
        self.state.setdefault(view, {"n": 0, "boxes": []})["boxes"] = [self._p2b(k) for k in kps]

    def to_raw(self, kps, scs, inv_scale):
        """(N, K, 2) model keypoints in reader pixels -> list of (133, 3) raw arrays in source pixels."""
        out = []
        for k, s in zip(kps, scs):
            r = np.zeros((KP, 3), np.float32)
            ok = self.map >= 0
            r[self.map[ok], :2] = k[ok] * inv_scale
            r[self.map[ok], 2] = score_to_wb(self.model, self.mode, np.asarray(s, np.float32))[ok]
            out.append(r)
        return out


def build_rtmlib(rcfg, device):
    """v0.8.0 path (the 'rtmlib' engine): rtmlib PoseTracker, one pose call per person."""
    from rtmlib import Body, PoseTracker, Wholebody
    solution = Wholebody if rcfg["model"] == "wholebody" else Body
    pose = PoseTracker(solution, det_frequency=int(rcfg["det_every"]), tracking=False, to_openpose=False,
                       mode=rcfg["mode"], backend="onnxruntime", device=device)
    return pose, {n: list(getattr(getattr(pose, n, None), "session").get_providers())
                  for n in ("det_model", "pose_model") if getattr(getattr(pose, n, None), "session", None) is not None}


def check_cuda(device):
    import onnxruntime as ort
    if device != "cuda":
        return
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001 - CUDA from a system install is fine too
            pass
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"onnxruntime has no CUDAExecutionProvider (have {ort.get_available_providers()}); "
                           "pip install -r tools/requirements-pc.txt")


# ---------------------------------------------------------------------------------------------------- worker
def schedule(views, plans, origins, fps, window_s):
    """Batches [(capture offset s of the latest frame, [(view, frame), ...]), ...] in capture-time order. Frames whose
    capture times fall within window_s of the batch's first frame share a batch (window 0 = one frame per batch)."""
    items = sorted((max(0.0, (f - origins[v]) / fps), v, int(f)) for v in views for f in plans[v])
    out, cur, t_first = [], [], None
    for t, v, f in items:
        if cur and t - t_first > window_s + 1e-9:
            out.append((max(x[0] for x in cur), [(x[1], x[2]) for x in cur]))
            cur = []
        if not cur:
            t_first = t
        cur.append((t, v, f))
    if cur:
        out.append((max(x[0] for x in cur), [(x[1], x[2]) for x in cur]))
    return out


SKIP = "skip"


class Prefetch(threading.Thread):
    """Decodes one view's frames in order into a small queue; in realtime pace it waits for each frame's capture time.
    Frames outside holder["keep"][view] (set once the run's plan narrows, e.g. after the warm-up sync) are not decoded:
    the queue gets (frame, SKIP, 0)."""

    def __init__(self, reader, frames, t_cap_ns, depth, stop_ev, view=None, holder=None):
        super().__init__(daemon=True, name=f"decode-{os.path.basename(reader.path)}")
        self.reader, self.frames, self.t_cap_ns, self.stop_ev = reader, frames, t_cap_ns, stop_ev
        self.view, self.holder = view, holder if holder is not None else {}
        self.q = queue_mod.Queue(maxsize=max(1, int(depth)))
        self.error = None

    def run(self):
        try:
            for f in self.frames:
                if self.stop_ev.is_set():
                    break
                keep = self.holder.get("keep")
                if keep is not None and f not in keep.get(self.view, ()):
                    self.q.put((f, SKIP, 0))
                    continue
                tc = self.t_cap_ns(f)
                if tc is not None:
                    now = time.monotonic_ns()
                    if now < tc:
                        time.sleep((tc - now) / 1e9)
                t = time.monotonic_ns()
                img = self.reader.read(f)
                self.q.put((f, img, time.monotonic_ns() - t))
        except Exception as e:  # noqa: BLE001 - reported through the queue
            self.error = f"{type(e).__name__}: {e}"
            self.q.put((None, None, 0))
        finally:
            self.q.put(("end", None, 0))


def fast_worker(views, paths, plans, origins, rcfg, device, fps, t0_val, go_ev, realtime, queues, stop_ev,
                step_s=None, ctl=None):
    """One process for `views`: decode (prefetch threads) -> boxes -> crops -> one batched pose call -> results.
    Messages on queues[view]: ("ready", view, info), ("frame", view, f, dets, timing), ("done", view, None),
    ("error", view, message). `ctl` (queue): {view: frames still needed} narrows the plan - other frames are neither
    decoded nor posed, and nothing is sent for them."""
    def put(v, msg):
        queues[v].put(msg)

    t_load = time.monotonic_ns()
    try:
        check_cuda(device)
        engine = FastPose(rcfg, device)
    except Exception as e:  # noqa: BLE001 - reported to the scheduler, which stops the run
        for v in views:
            put(v, ("error", v, f"{type(e).__name__}: {e}"))
        return
    load_ns = time.monotonic_ns() - t_load
    short = rcfg.get("decode_short_side")
    kind = rcfg.get("decoder", "cv2")
    readers = {}
    try:
        for v in views:
            first = plans[v][0] if plans[v] else 0
            readers[v] = open_reader(kind, paths[v], first, fps, short, rcfg.get("decode_threads", 0))
    except Exception as e:  # noqa: BLE001
        for v in views:
            put(v, ("error", v, f"{kind} reader: {type(e).__name__}: {e}"))
        return
    for v in views:
        engine.set_view(v, readers[v].scale, readers[v].src_h)
    warm_s, warm_sizes = 0.0, []
    if rcfg.get("warmup", True):
        try:
            warm_s, warm_sizes = engine.warmup((readers[views[0]].out_h, readers[views[0]].out_w))
        except Exception as e:  # noqa: BLE001
            for v in views:
                put(v, ("error", v, f"warm-up: {type(e).__name__}: {e}"))
            return
    for v in views:
        put(v, ("ready", v, {"model_load_ns": load_ns, "court_filter": engine.court is not None, "warmup_s": warm_s,
                             "padded_sizes": warm_sizes if engine.pad else None, "providers": engine.providers, "engine": "fast",
                             "decoder": kind, "decode_size": [readers[v].out_w, readers[v].out_h],
                             "batch_views": len(views) > 1, "model": engine.model}))
    while not go_ev.wait(0.05):
        if stop_ev.is_set():
            return
    t0_ns = int(t0_val.value)
    window = (step_s if step_s is not None else 0.0) if len(views) > 1 else 0.0
    batches = schedule(views, plans, origins, fps, window)
    cap_ns = (lambda v: (lambda f: t0_ns + int(max(0.0, f - origins[v]) / fps * 1e9))) if realtime \
        else (lambda v: (lambda f: None))
    holder = {"keep": None}
    pre = {v: Prefetch(readers[v], plans[v], cap_ns(v), rcfg.get("prefetch", 2), stop_ev, v, holder) for v in views}
    for p in pre.values():
        p.start()
    pool = ThreadPoolExecutor(max_workers=max(1, len(views)))
    drop_behind = rcfg.get("drop_behind_s")
    first_call = True
    try:
        for t_last, items in batches:
            if stop_ev.is_set():
                break
            while ctl is not None:
                try:
                    k = ctl.get_nowait()
                except queue_mod.Empty:
                    break
                holder["keep"] = {v: set(int(x) for x in fr) for v, fr in k.items()}
            t_av = t0_ns + int(t_last * 1e9) if realtime else time.monotonic_ns()
            got = {}
            for v, f in items:                         # frames arrive in each view's plan order
                tw = time.monotonic_ns()
                ff, img, dec_ns = pre[v].q.get()
                if ff is None:
                    raise RuntimeError(f"decode {v}: {pre[v].error}")
                if isinstance(img, str) and img == SKIP:
                    continue
                if ff == "end":
                    img = None
                got[(v, f)] = (img, dec_ns, time.monotonic_ns() - tw)
            items = [it for it in items if it in got]
            if not items:
                continue
            if realtime and drop_behind is not None and time.monotonic_ns() - t_av > float(drop_behind) * 1e9:
                for v, f in items:
                    put(v, ("frame", v, f, [], {"t_avail_ns": t_av, "t_done_ns": time.monotonic_ns(), "decode_ns": 0,
                                                "infer_ns": 0, "wait_ns": 0, "dropped": True, "first_call": False}))
                continue
            t1 = time.monotonic_ns()

            def prep(key):
                img = got[key][0]
                if img is None:
                    return key, [], 0, [], False
                boxes, det_ns, detected = engine.boxes(key[0], img)
                return key, boxes, det_ns, engine.crops(img, boxes), detected
            preps = list(pool.map(prep, items))
            t2 = time.monotonic_ns()
            all_crops, spans = [], []
            for key, boxes, det_ns, crops, detected in preps:
                spans.append((key, len(all_crops), len(crops), det_ns, detected))
                all_crops.extend(crops)
            kps, scs = engine.infer(all_crops)
            t3 = time.monotonic_ns()
            for key, a, n, det_ns, detected in spans:
                v, f = key
                engine.finish(v, kps[a:a + n])
                img, dec_ns, wait_ns = got[key]
                inv = 1.0 / readers[v].scale
                dets = engine.to_raw(kps[a:a + n], scs[a:a + n], inv) if img is not None else []
                put(v, ("frame", v, f, dets, {
                    "t_avail_ns": t_av, "t_done_ns": time.monotonic_ns(), "decode_ns": dec_ns, "wait_ns": wait_ns,
                    "infer_ns": t3 - t1, "det_ns": det_ns, "prep_ns": t2 - t1, "pose_ns": t3 - t2,
                    "batch_images": len(items), "batch_crops": len(all_crops), "persons": n, "detected": detected,
                    "dropped": img is None, "first_call": first_call}))
            first_call = False
    except Exception as e:  # noqa: BLE001
        for v in views:
            put(v, ("error", v, f"{type(e).__name__}: {e}"))
    finally:
        pool.shutdown(wait=False)
        for r in readers.values():
            r.close()
        for v in views:
            put(v, ("done", v, None))
