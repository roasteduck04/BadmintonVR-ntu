"""2D frame sources for a live run: detections of one camera at one source frame, plus how long they took.

    src = make(cfg); src.start(plan, origins); src.begin(t0_ns); dets, tim = src.get(view, frame); src.close()

`plan` = {view: sorted source frames the run may ask for}; `origins` = {view: source frame (float) captured at t0}
(the section start + the view's offset), so a frame's capture time is t0 + (frame - origin) / source_fps.
start() loads everything (models: can take ~30 s) and returns when ready; begin(t0) starts the clock. Detections are raw RTMPose Wholebody arrays
(133, 3) = x, y, raw SimCC score (the scheduler calibrates them, as offline). Timing dict per frame (ns, monotonic):
t_avail_ns (when the frame "exists" - its capture time in realtime pace), t_done_ns, decode_ns, infer_ns,
wait_ns (worker idle until the frame existed), dropped (bool).

- CachedSource: the offline 2D (data/pose2d/<view>_kp2d.json) replayed frame by frame; no pixels.
- VideoRtmlibSource: one worker process per camera; it decodes its video (grab() past frames it does not need),
  runs rtmlib (same model + settings as tools/cost/pose2d_gpu_timed.py by default) and streams results back through a
  bounded queue (rtmpose.queue_depth frames ahead). In realtime pace a worker waits until a frame's capture time
  (t0 + (frame - first) / source_fps) before reading it, like a camera; with rtmpose.drop_behind_s set it skips frames
  that are already older than that when it gets to them (reported as dropped)."""
from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_mod
import time

import numpy as np

KP = 133


def _raw(kp, score):
    k = np.zeros((KP, 3), np.float32)
    kp = np.asarray(kp, np.float32).reshape(-1, 2)
    sc = np.asarray(score, np.float32).reshape(-1)
    n = min(len(kp), KP)
    k[:n, :2] = kp[:n]
    k[:n, 2] = sc[:n]
    return k


class CachedSource:
    name = "cached"

    def __init__(self, cfg, log=print):
        self.cfg, self.frames, self.t0 = cfg, {}, None
        for v in cfg["views"]:
            t = time.perf_counter()
            with open(cfg["kp2d_paths"][v], "r", encoding="utf-8") as fh:
                self.frames[v] = json.load(fh)["frames"]
            log(f"  cached 2D {v}: {len(self.frames[v])} frames ({time.perf_counter() - t:.1f} s to load)")
        self.fps = float(cfg["source_fps"])
        self.realtime = cfg.get("pace", "realtime") == "realtime"
        self.origins = {}

    def start(self, plan, origins):
        self.origins = {v: float(origins[v]) for v in plan}

    def begin(self, t0_ns):
        self.t0 = int(t0_ns)

    def get(self, view, frame):
        t = time.monotonic_ns()
        fl = self.frames[view]
        raw = fl[frame] if 0 <= frame < len(fl) else []
        dets = [_raw(np.asarray(d["keypoints"], np.float32)[:, :2], np.asarray(d["keypoints"], np.float32)[:, 2])
                for d in raw]
        t_av = (self.t0 + int(max(0.0, frame - self.origins.get(view, frame)) / self.fps * 1e9)
                if self.t0 is not None and self.realtime else t)
        return dets, {"t_avail_ns": t_av, "t_done_ns": time.monotonic_ns(), "decode_ns": 0,
                      "infer_ns": time.monotonic_ns() - t, "wait_ns": 0, "dropped": False}

    def try_get(self, view, frame):
        return self.get(view, frame)

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------- live
def build_model(rcfg, device):
    """rtmlib PoseTracker as the offline 2D script builds it (det_frequency = rtmpose.det_every)."""
    import onnxruntime as ort
    if device == "cuda":
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - CUDA from a system install is fine too
                pass
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"onnxruntime has no CUDAExecutionProvider (have {ort.get_available_providers()}); "
                               "pip install -r tools/requirements-pc.txt")
    from rtmlib import Body, PoseTracker, Wholebody
    solution = Wholebody if rcfg["model"] == "wholebody" else Body
    pose = PoseTracker(solution, det_frequency=int(rcfg["det_every"]), tracking=False, to_openpose=False,
                       mode=rcfg["mode"], backend="onnxruntime", device=device)
    providers = {}
    for name in ("det_model", "pose_model"):
        m = getattr(pose, name, None)
        sess = getattr(m, "session", None)
        if sess is not None:
            providers[name] = list(sess.get_providers())
    return pose, providers


def _worker(view, path, frames, rcfg, device, fps, origin, t0_val, go_ev, realtime, out_q, stop_ev, ctl=None):
    import cv2
    from . import pose2d_fast
    t_load = time.monotonic_ns()
    try:
        pose, providers = build_model(rcfg, device)
    except Exception as e:  # noqa: BLE001 - reported to the scheduler, which stops the run
        out_q.put(("error", view, f"{type(e).__name__}: {e}"))
        return
    out_q.put(("ready", view, {"model_load_ns": time.monotonic_ns() - t_load, "providers": providers}))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        out_q.put(("error", view, f"cannot open {path}"))
        return
    first = frames[0] if frames else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    pos = first
    while not go_ev.wait(0.05):
        if stop_ev.is_set():
            cap.release()
            return
    t0_ns = int(t0_val.value)
    scale = float(rcfg["downscale"])
    drop_behind = rcfg.get("drop_behind_s")
    first_call = True
    keep = None
    for f in frames:
        if stop_ev.is_set():
            break
        while ctl is not None:          # the run narrowed its plan (e.g. after the warm-up sync): skip unneeded frames
            try:
                keep = set(int(x) for x in ctl.get_nowait())
            except queue_mod.Empty:
                break
        if keep is not None and f not in keep:
            continue
        t_av = t0_ns + int(max(0.0, f - origin) / fps * 1e9)
        wait = 0
        if realtime:
            now = time.monotonic_ns()
            if now < t_av:
                time.sleep((t_av - now) / 1e9)
                wait = time.monotonic_ns() - now
        else:
            t_av = time.monotonic_ns()
        t1 = time.monotonic_ns()
        while pos < f:
            cap.grab()
            pos += 1
        if realtime and drop_behind is not None and time.monotonic_ns() - t_av > float(drop_behind) * 1e9:
            cap.grab()
            pos += 1
            out_q.put(("frame", view, f, [], {"t_avail_ns": t_av, "t_done_ns": time.monotonic_ns(), "decode_ns": 0,
                                              "infer_ns": 0, "wait_ns": wait, "dropped": True, "first_call": False}))
            continue
        ok, img = cap.read()
        pos += 1
        t2 = time.monotonic_ns()
        dets = []
        if ok:
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            kps, sc = pose(img)
            if kps is not None and len(kps):
                kps = np.asarray(kps, np.float32) / scale
                sc = pose2d_fast.score_to_wb(rcfg["model"], rcfg["mode"], np.asarray(sc, np.float32))
                dets = [_raw(kps[i], sc[i]) for i in range(len(kps))]
        t3 = time.monotonic_ns()
        out_q.put(("frame", view, f, dets, {"t_avail_ns": t_av, "t_done_ns": t3, "decode_ns": t2 - t1,
                                            "infer_ns": t3 - t2, "wait_ns": wait, "dropped": not ok,
                                            "first_call": first_call}))
        first_call = False
    cap.release()
    out_q.put(("done", view, None))


def _missing():
    return {"t_avail_ns": None, "t_done_ns": time.monotonic_ns(), "decode_ns": 0, "infer_ns": 0, "wait_ns": 0,
            "dropped": True}


class VideoRtmlibSource:
    name = "live"

    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        self.ctx = mp.get_context("spawn")
        self.procs, self.queues, self.stash, self.info, self.done = {}, {}, {}, {}, set()
        self.stop_ev, self.go_ev = self.ctx.Event(), self.ctx.Event()
        self.ctl, self.ctl_views = [], []     # one control queue per worker process + the views it serves
        self.t0_val = self.ctx.Value("q", 0)

    def start(self, plan, origins, ready_timeout_s=600):
        c = self.cfg
        rc = dict(c["rtmpose"])
        realtime = c["pace"] == "realtime"
        if rc.get("engine", "rtmlib") == "fast":
            from . import pose2d_fast
            rc["_cam_paths"] = dict(c["cam_paths"])
            groups = [list(c["views"])] if rc.get("batch_views") else [[v] for v in c["views"]]
            for v in c["views"]:
                # one process feeding every camera's queue: unbounded, else a full queue of one camera blocks the
                # worker while the scheduler waits on another camera (results are small; the process paces itself)
                self.queues[v] = self.ctx.Queue(maxsize=0 if rc.get("batch_views") else max(2, int(rc["queue_depth"])))
                self.stash[v] = {}
            for g in groups:
                cq = self.ctx.Queue()
                self.ctl.append(cq)
                self.ctl_views.append(list(g))
                p = self.ctx.Process(target=pose2d_fast.fast_worker, name="pose2d-" + "+".join(g), daemon=True,
                                     args=(g, {v: c["videos"][v] for v in g}, {v: list(plan[v]) for v in g},
                                           {v: float(origins[v]) for v in g}, rc, c["device"], float(c["source_fps"]),
                                           self.t0_val, self.go_ev, realtime, {v: self.queues[v] for v in g},
                                           self.stop_ev, 1.0 / float(c["fps"]), cq))
                p.start()
                for v in g:
                    self.procs[v] = p
        else:
            for v in c["views"]:
                q = self.ctx.Queue(maxsize=max(2, int(rc["queue_depth"])))
                cq = self.ctx.Queue()
                self.ctl.append(cq)
                self.ctl_views.append(v)
                p = self.ctx.Process(target=_worker, name=f"pose2d-{v}", daemon=True,
                                     args=(v, c["videos"][v], list(plan[v]), rc, c["device"], float(c["source_fps"]),
                                           float(origins[v]), self.t0_val, self.go_ev, realtime, q, self.stop_ev, cq))
                p.start()
                self.procs[v], self.queues[v], self.stash[v] = p, q, {}
        for v in c["views"]:            # wait for every model, so model loading is not counted as frame latency
            kind, _, payload = self._next(v, ready_timeout_s)
            if kind != "ready":
                raise RuntimeError(f"2D worker {v} failed to start: {payload}")
            self.info[v] = payload
            extra = (f", engine fast, decoder {payload['decoder']} {payload['decode_size'][0]}x{payload['decode_size'][1]}, "
                     f"{'one process for all cameras' if payload['batch_views'] else 'one process per camera'}"
                     if payload.get("engine") == "fast" else "")
            self.log(f"  2D worker {v}: model ready in {payload['model_load_ns'] / 1e9:.1f} s, "
                     f"providers {payload['providers']}{extra}")

    def begin(self, t0_ns):
        self.t0_val.value = int(t0_ns)
        self.go_ev.set()

    def narrow(self, keep):
        """The run needs only these frames from now on ({view: frames}): workers skip the others (not decoded / posed)."""
        for cq, vs in zip(self.ctl, self.ctl_views):
            cq.put({v: list(keep.get(v, [])) for v in vs} if isinstance(vs, list) else list(keep.get(vs, [])))

    def _next(self, view, timeout_s):
        try:
            msg = self.queues[view].get(timeout=timeout_s)
        except queue_mod.Empty:
            raise RuntimeError(f"2D worker {view}: nothing for {timeout_s} s") from None
        if msg[0] == "error":
            raise RuntimeError(f"2D worker {view}: {msg[2]}")
        return msg[0], msg[1], msg[2] if msg[0] != "frame" else msg[2:]

    @staticmethod
    def _take(st, frame, keep_back=4):
        """A delivered frame, kept in the stash: at 60 fps two consecutive steps share a source frame (one step's
        interpolation partner is the next step's other frame), and popping it on the first use left the second step
        with an empty 'missing' frame (v0.8.1 fix: a whole side camera was lost whenever that frame was the nearer one).
        Frames are asked for in increasing order per view, so older ones are pruned."""
        got = st[frame]
        for k in [k for k in st if k < frame - keep_back]:
            del st[k]
        return got

    def try_get(self, view, frame, wait_s=0.002):
        """The frame's (dets, timings) if the worker has delivered it (waiting at most wait_s), else None."""
        st = self.stash[view]
        if frame in st:
            return self._take(st, frame)
        if view in self.done:
            return [], _missing()
        try:
            msg = self.queues[view].get(timeout=wait_s)
        except queue_mod.Empty:
            return None
        if msg[0] == "error":
            raise RuntimeError(f"2D worker {view}: {msg[2]}")
        if msg[0] == "frame":
            st[msg[2]] = (msg[3], msg[4])
        elif msg[0] == "done":
            self.done.add(view)
        return self._take(st, frame) if frame in st else None

    def get(self, view, frame, timeout_s=600):
        st = self.stash[view]
        while frame not in st:
            kind, _, payload = self._next(view, timeout_s)
            if kind == "frame":
                f, dets, tim = payload
                st[f] = (dets, tim)
            elif kind == "done":
                self.done.add(view)
                return [], _missing()
        return self._take(st, frame)

    def close(self):
        self.stop_ev.set()
        for v, p in self.procs.items():
            try:                        # drain so a worker blocked on a full queue can exit
                while True:
                    self.queues[v].get_nowait()
            except (queue_mod.Empty, OSError, ValueError):
                pass
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make(cfg, log=print):
    if cfg["backend"] == "cached":
        return CachedSource(cfg, log=log)
    if cfg["backend"] == "live":
        return VideoRtmlibSource(cfg, log=log)
    raise ValueError(f"unknown backend {cfg['backend']!r}")


def plan_frames(schedule) -> dict:
    """{view: sorted unique source frames} the schedule needs."""
    out = {}
    for st in schedule:
        for v, vv in st["views"].items():
            out.setdefault(v, set()).update(int(f) for f in vv["frames"])
    return {v: sorted(s) for v, s in out.items()}
