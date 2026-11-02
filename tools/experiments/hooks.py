"""Round-5 experiment hooks (test branch only), attached by tools/experiments/sitecustomize.py when the live modules load.
Each is off unless its config key is set (keys pass through the live config untouched):

- rtmpose.exp_int8=true (or a path): the pose model as an INT8 QDQ model built by tools/experiments/int8_calib.py,
  run by TensorRT with INT8 enabled (FP16 for whatever stays unquantized). Needs rtmpose.ep=trt.
- rtmpose.exp_stagger=N: each camera runs the pose model on every Nth frame only, cameras offset from each other (so at
  every step 1/N of the cameras are fresh); in between, each person's keypoints are carried forward at their last
  velocity (frames from a new detection always run the model). Needs one process per camera (batch_views false).
- output.exp_socket=udp://host:port: frame files go out as UDP datagrams instead of files (manifest and latest.json stay
  files) - what writing costs; Unity does not receive them in this test (tools/experiments/frame_receiver.py does).

Round 6 (racket + shuttle, rs_stream.py): rtmpose.exp_racket=det|wrist|both and/or rtmpose.exp_shuttle=true run the
racket / shuttle models in the 2D processes after the pose model and a 3D process next to the runner; results go to
<run>/racket_shuttle.jsonl, nothing else changes. Settings: rs_stream.settings().
Round 8 (v2, the default rtmpose.exp_rs_mode=process): the racket / shuttle models run in processes of their own next
to the pose processes (exp_rs_group camera = one per camera, all = one for every camera), decode the same frames on the
same clock and send their results straight to the 3D process; exp_rs_mode=inline = round 6 (inside the pose processes).

Always on (a fix, not an option): wholebody at a smaller input (rtmpose.mode=balanced) gets its raw scores mapped onto
the wholebody-performance scale with tools/experiments/score_map_wb.json (score_map_fit.py) - pose2d_fast.score_to_wb
passes wholebody through unchanged, which left every round-5 balanced stream without a single player.
"""
from __future__ import annotations

import json
import os
import socket

import numpy as np

INT8_DIR = os.path.join("~", ".cache", "badmintonvr", "int8")
SCORE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_map_wb.json")


def load_score_maps(path=SCORE_MAP) -> dict:
    """{"wholebody/<mode>": [(lo, hi, x quantiles, y quantiles), ...]} per keypoint group."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {k: [(g["range"][0], g["range"][1], np.asarray(g["x"], np.float32), np.asarray(g["y"], np.float32))
                for g in v["groups"].values()] for k, v in doc.items()}


def map_scores(groups, s):
    """Raw scores (..., 133) of a smaller wholebody input on the wholebody-performance raw scale."""
    s = np.asarray(s, np.float32)
    out = s.copy()
    for lo, hi, x, y in groups:
        out[..., lo:hi] = np.interp(s[..., lo:hi], x, y)
    return out


def int8_path(onnx_model) -> str:
    base = os.path.splitext(os.path.basename(str(onnx_model)))[0]
    return os.path.expanduser(os.path.join(INT8_DIR, f"{base}.int8qdq.onnx"))


def attach(name, module):
    if name == "analysis.live.pose2d_fast":
        _attach_pose2d(module)
        _attach_rs_worker(module)
    elif name == "analysis.live.framewriter":
        _attach_framewriter(module)
        _attach_rs_writer(module)
    elif name == "analysis.live.sources":
        _attach_rs_sources(module)


# ------------------------------------------------------------------------------------------------ racket + shuttle
def _attach_rs_worker(m):
    """2D processes: the racket / shuttle models after the pose model, on the same decoded frame (rs_stream.RS2D)."""
    import rs_stream
    F = m.FastPose
    orig_init, orig_warm, orig_boxes = F.__init__, F.warmup, F.boxes

    def __init__(self, rcfg, device):
        orig_init(self, rcfg, device)
        self._rs = rs_stream.RS2D.maybe(rcfg, device, self.ep)

    def warmup(self, img_hw=(1080, 1920)):
        out = orig_warm(self, img_hw)
        if getattr(self, "_rs", None) is not None:
            s = self._rs.warmup(img_hw)
            print(f"racket/shuttle models warmed up in {s:.1f} s ({self._rs.providers})", flush=True)
        return out

    def boxes(self, view, img):
        if getattr(self, "_rs", None) is not None:
            self._rs.stash(view, img)
        return orig_boxes(self, view, img)

    F.__init__, F.warmup, F.boxes = __init__, warmup, boxes
    orig_session = m._run_session

    def _run_session(engine, readers, k_cal, views, plans, origins, fps, t0_ns, realtime, put, stop_ev, ctl, step_s,
                     rcfg):
        rs = getattr(engine, "_rs", None)
        if rs is None:
            return orig_session(engine, readers, k_cal, views, plans, origins, fps, t0_ns, realtime, put, stop_ev,
                                ctl, step_s, rcfg)
        rs.begin_session()
        inv = {v: k_cal[v] / readers[v].scale for v in views}
        return orig_session(engine, readers, k_cal, views, plans, origins, fps, t0_ns, realtime, rs.wrap_put(put, inv),
                            stop_ev, ctl, step_s, rcfg)

    m._run_session = _run_session              # fast_worker looks it up on the module at call time


def _attach_rs_sources(m):
    """Main process: hand every delivered frame's racket / shuttle 2D to the run's 3D process (inline mode), or start
    the racket / shuttle processes next to the pose processes (v2 process mode: results go straight to the 3D)."""
    import rs_stream
    S = m.VideoRtmlibSource
    orig_try, orig_get = S.try_get, S.get
    orig_start, orig_close, orig_session = S.start, S.close, S.session

    def start(self, plan, origins, ready_timeout_s=600):
        orig_start(self, plan, origins, ready_timeout_s)
        st = rs_stream.settings(self.cfg.get("rtmpose"))
        if st and st["mode"] == "process" and rs_stream.CURRENT is not None:
            self._rs_procs = rs_stream.start_workers(self, plan, origins, rs_stream.CURRENT.q, log=self.log)

    def close(self):
        try:
            orig_close(self)                  # sets stop_ev: the racket / shuttle processes stop too
        finally:
            rs_stream.stop_workers(getattr(self, "_rs_procs", None) or [])
            self._rs_procs = []

    def session(self, plan, origins, realtime=None, timeout_s=120):
        st = rs_stream.settings(self.cfg.get("rtmpose"))
        if st and st["mode"] == "process":
            self.log("  racket / shuttle (process mode) runs in one-shot runs only (live_run.py run): none this stream")
        return orig_session(self, plan, origins, realtime, timeout_s)

    S.start, S.close, S.session = start, close, session

    def try_get(self, view, frame, wait_s=0.002):
        got = orig_try(self, view, frame, wait_s)
        if rs_stream.CURRENT is not None:
            rs_stream.CURRENT.on_2d(view, got)
        return got

    def get(self, view, frame, timeout_s=600):
        got = orig_get(self, view, frame, timeout_s)
        if rs_stream.CURRENT is not None:
            rs_stream.CURRENT.on_2d(view, got)
        return got

    S.try_get, S.get = try_get, get


def _attach_rs_writer(m):
    """Main process: one racket / shuttle 3D process per run (started with the run's FrameWriter, fed each written
    frame, closed with it)."""
    import rs_stream
    F = m.FrameWriter
    orig_init, orig_write, orig_close = F.__init__, F._write, F.close

    def __init__(self, run_dir, cfg, header_doc, *a, **k):
        orig_init(self, run_dir, cfg, header_doc, *a, **k)
        self._rs_sink = rs_stream.RSSink(run_dir, cfg, header_doc) if rs_stream.settings(cfg.get("rtmpose")) else None

    def _write(self, o, t_capture_ns=None):
        out = orig_write(self, o, t_capture_ns)
        if getattr(self, "_rs_sink", None) is not None:
            self._rs_sink.on_frame(o, out[0])
        return out

    def close(self):
        try:
            orig_close(self)
        finally:
            if getattr(self, "_rs_sink", None) is not None:
                self._rs_sink.close()
                self._rs_sink = None

    F.__init__, F._write, F.close = __init__, _write, close


# ------------------------------------------------------------------------------------------------ INT8 + stagger
def _attach_pose2d(m):
    orig_score, maps = m.score_to_wb, load_score_maps()

    def score_to_wb(model, mode, s):
        g = maps.get(f"{model}/{mode}") if model == "wholebody" else None
        return map_scores(g, s) if g else orig_score(model, mode, s)

    m.score_to_wb = score_to_wb               # to_raw and sources.py look it up on the module at call time

    orig_trt_models, orig_trt_session = m.trt_models, m.trt_session

    def trt_models(sol, rcfg, batch, in_hw):
        want = rcfg.get("exp_int8")
        if want:
            q = want if isinstance(want, str) else int8_path(sol.pose_model.onnx_model)
            if not os.path.exists(q):
                raise RuntimeError(f"exp_int8: no INT8 pose model {q} - run tools/experiments/int8_calib.py first")
            sol.pose_model.onnx_model = q
        return orig_trt_models(sol, rcfg, batch, in_hw)

    def trt_session(onnx_path, rcfg, shapes=None):
        if str(onnx_path).endswith(".int8qdq.onnx"):
            return _int8_session(m, onnx_path, rcfg, shapes)
        return orig_trt_session(onnx_path, rcfg, shapes)

    m.trt_models, m.trt_session = trt_models, trt_session

    F = m.FastPose
    orig_init, orig_boxes, orig_infer = F.__init__, F.boxes, F.infer

    def __init__(self, rcfg, device):
        orig_init(self, rcfg, device)
        n = int(rcfg.get("exp_stagger") or 1)
        if n > 1 and rcfg.get("batch_views"):
            print("exp_stagger needs one process per camera (rtmpose.batch_views false): off", flush=True)
            n = 1
        self._exp_stagger = n
        self._exp_order = {v: i for i, v in enumerate(sorted((rcfg.get("_cam_paths") or {}).keys()))}
        self._exp_hist, self._exp_cur = {}, None
        self.exp_predicted = self.exp_fresh = 0

    def boxes(self, view, img):
        out = orig_boxes(self, view, img)
        self._exp_cur = (view, self.state[view]["n"] - 1, bool(out[2]))
        return out

    def infer(self, crops):
        n_st, cur = getattr(self, "_exp_stagger", 1), self._exp_cur
        if n_st > 1 and cur is not None and crops:
            view, n, detected = cur
            h = self._exp_hist.get(view) or []
            if h and n <= h[-1][0]:                       # a new session restarted the frame count
                h = self._exp_hist[view] = []
            off = self._exp_order.get(view, 0) % n_st
            if not detected and (n + off) % n_st != 0 and h and len(h[-1][1]) == len(crops):
                n1, k1, s1 = h[-1]
                vel = 0.0
                if len(h) >= 2 and len(h[-2][1]) == len(k1):
                    n0, k0, _ = h[-2]
                    vel = (k1 - k0) / float(n1 - n0)
                self.exp_predicted += len(crops)
                return (k1 + vel * (n - n1)).astype(np.float32), s1.copy()
        kps, scs = orig_infer(self, crops)
        if n_st > 1 and cur is not None:
            view, n, detected = cur
            h = [] if detected else self._exp_hist.get(view, [])   # a new detection may reorder people
            h.append((n, kps.copy(), scs.copy()))
            self._exp_hist[view] = h[-2:]
        self.exp_fresh += len(crops)
        return kps, scs

    F.__init__, F.boxes, F.infer = __init__, boxes, infer


def _int8_session(m, onnx_path, rcfg, shapes):
    """pose2d_fast.trt_session with INT8 on (explicit QDQ quantization; FP16 for the rest)."""
    import onnxruntime as ort
    cache = os.path.expanduser(os.path.join(rcfg.get("trt_cache") or m.TRT_CACHE, "int8"))
    os.makedirs(cache, exist_ok=True)
    opts = {"trt_fp16_enable": "True", "trt_int8_enable": "True", "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": cache, "trt_timing_cache_enable": "True", "trt_timing_cache_path": cache,
            "trt_max_workspace_size": str(int(rcfg.get("trt_workspace_gb", 4)) << 30)}
    if shapes:
        opts["trt_profile_min_shapes"] = ",".join(m._shape_str(n, s[0]) for n, s in shapes.items())
        opts["trt_profile_opt_shapes"] = ",".join(m._shape_str(n, s[1]) for n, s in shapes.items())
        opts["trt_profile_max_shapes"] = ",".join(m._shape_str(n, s[2]) for n, s in shapes.items())
    prov = [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, providers=prov)
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        raise m.TensorRTUnavailable(f"TensorRT could not build the INT8 model {os.path.basename(onnx_path)}: {msg[:300]}")
    if sess.get_providers()[:1] != ["TensorrtExecutionProvider"]:
        raise m.TensorRTUnavailable(f"onnxruntime could not use TensorRT for the INT8 model (got {sess.get_providers()})")
    return sess


# ------------------------------------------------------------------------------------------------ socket output
_SOCKS = {}          # frame folder -> (socket, address)


def _attach_framewriter(m):
    F = m.FrameWriter
    orig_init, orig_aw = F.__init__, m.atomic_write

    def __init__(self, run_dir, cfg, *a, **k):
        orig_init(self, run_dir, cfg, *a, **k)
        url = (cfg.get("output") or {}).get("exp_socket")
        if url:
            if not str(url).startswith("udp://"):
                raise ValueError(f"output.exp_socket must be udp://host:port, got {url!r}")
            host, port = str(url)[6:].rsplit(":", 1)
            _SOCKS[os.path.normcase(os.path.abspath(self.dir))] = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                                                                   (host, int(port)))

    def atomic_write(path, obj, **kw):
        s = _SOCKS.get(os.path.normcase(os.path.abspath(os.path.dirname(path))))
        if s is not None and os.path.basename(path).startswith("frame_"):
            text = json.dumps(obj, allow_nan=False, separators=(",", ":"))
            s[0].sendto(text.encode("utf-8"), s[1])
            return len(text)
        return orig_aw(path, obj, **kw)

    F.__init__, m.atomic_write = __init__, atomic_write
