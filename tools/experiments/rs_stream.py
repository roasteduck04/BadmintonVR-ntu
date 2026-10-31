"""Round-6 experiment (test branch only): racket + shuttle in a live stream, attached by hooks.py (nothing in the live
code is edited; Unity frames and the server are untouched). Off unless a test sets rtmpose.exp_racket / exp_shuttle.

2D (RS2D, inside each 2D process, after the pose model, on the frame that is already decoded):
- racket: boxes from the RacketVision detector (`det`, every exp_racket_det_every frames; wrist boxes in between),
  from the live body keypoints (`wrist`: a box around each wrist, no detector) or both (`both`: detector boxes + wrist
  boxes where no detector box overlaps); then the racket keypoint model on every box of every camera in one call.
- shuttle: TrackNetV3 per camera, a call every exp_shuttle_stride frames (8 = one call per 8 frames); its background
  is the median of the first exp_shuttle_bg_frames frames sampled every 2nd frame (kept for later streams).
- exp_rs_async: the racket / shuttle work in a thread of its own, so the pose results go out at once and the racket /
  shuttle results ride on a later frame's message (the last one or two of a stream are lost).
Results ride on the frame message's timing dict (info["rs"]), which reaches the runner unchanged.

3D (rs3d_main, a process of its own in the runner): each written output frame (step, players' hands from the fitted
body) is matched with the 2D of its source frames:
- racket: the rigid racket fit of the offline variants (racket_rigidfit.RacketFitter: 5 keypoints, all views, grip at
  the player's hand, previous-frame prior) on the candidates whose handle lies near the projected hand; the racket hand
  is locked per player once one hand clearly wins.
- shuttle: `tri` = the nearest frame of every camera triangulated (best pair + inliers); `arc` = a quadratic path
  X(t) fitted to every observation of +-exp_shuttle3d_window frames (sub-frame camera offsets included), evaluated
  at the output frame (falls back to the past / future half when a hit breaks the arc).
Written to <run>/racket_shuttle.jsonl (one line per output frame) + rs_summary.json; rs_report.py scores them."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque

import numpy as np

import rs_live as L

CURRENT = None            # the RSSink of the run in this (main) process


def settings(rcfg) -> dict | None:
    rcfg = rcfg or {}
    racket = str(rcfg.get("exp_racket") or "off").lower()
    racket = None if racket in ("off", "false", "none", "0") else racket
    if racket not in (None, "det", "wrist", "both"):
        raise ValueError(f"rtmpose.exp_racket must be off | det | wrist | both, got {racket!r}")
    shuttle = bool(rcfg.get("exp_shuttle"))
    if not racket and not shuttle:
        return None
    box = rcfg.get("exp_racket_box") or [1.6, 0.3]
    return {"racket": racket, "shuttle": shuttle,
            "racket_ep": rcfg.get("exp_racket_ep"), "shuttle_ep": rcfg.get("exp_shuttle_ep"),
            "det_every": max(1, int(rcfg.get("exp_racket_det_every") or 1)),
            "flip": bool(rcfg.get("exp_racket_flip", True)),
            "refine": bool(rcfg.get("exp_racket_refine", True)),
            "fast_fit": bool(rcfg.get("exp_racket3d_fast", True)),
            "fast_nfev": int(rcfg.get("exp_racket3d_nfev") or 30),
            "solver": str(rcfg.get("exp_racket3d_solver") or "gn"),
            "box_k": float(box[0]), "box_push": float(box[1]),
            "wrist_min": float(rcfg.get("exp_racket_wrist_min", 2.0)),
            "max_cands": int(rcfg.get("exp_racket_max_cands", 8)),
            "stride": int(rcfg.get("exp_shuttle_stride") or 8),
            "weights": str(rcfg.get("exp_shuttle_weights") or "average"),
            "bg_frames": int(rcfg.get("exp_shuttle_bg_frames") or 30),
            "async": bool(rcfg.get("exp_rs_async", False)),
            "racket3d": bool(rcfg.get("exp_racket3d", True)),
            "shuttle3d": str(rcfg.get("exp_shuttle3d") or "arc"),
            "window": int(rcfg.get("exp_shuttle3d_window") or 6),
            "shuttle_px": float(rcfg.get("exp_shuttle3d_px") or 20.0),
            "two_view_deg": float(rcfg.get("exp_shuttle3d_two_view_deg", 8.0)),
            "two_view_px": float(rcfg.get("exp_shuttle3d_two_view_px", 12.0))}


def _r(x, n=1):
    return None if x is None else round(float(x), n)


# ------------------------------------------------------------------------------------------------ 2D (worker)
class RS2D:
    def __init__(self, rcfg, device, ep):
        self.s = st = settings(rcfg)
        cpu = device != "cuda"
        e_r = "cpu" if cpu else (st["racket_ep"] or ep)
        e_s = "cpu" if cpu else (st["shuttle_ep"] or ep)
        self.providers = {}
        self.det = self.pose = self.tn_run = None
        if st["racket"]:
            run, self.providers["racket_pose"] = L.session(
                "racket_pose", e_r, rcfg, {"input": ((1, 3, 256, 256), (16, 3, 256, 256), (64, 3, 256, 256))})
            self.pose = L.RacketPose(run, flip=st["flip"])
            if st["racket"] in ("det", "both"):
                run, self.providers["racket_det"] = L.session(
                    "racket_det", e_r, rcfg, {"input": ((1, 3, 640, 640), (3, 3, 640, 640), (5, 3, 640, 640))})
                self.det = L.RacketDet(run)
        if st["shuttle"]:
            self.tn_run, self.providers["tracknet"] = L.session(
                "tracknet", e_s, rcfg, {"input": ((1, 27, 288, 512), (3, 27, 288, 512), (5, 27, 288, 512))})
        self.ep = {"racket": e_r, "shuttle": e_s}
        self.chunk = 32 if e_r != "cpu" else 4          # racket crops per call (x2 with the flip test)
        self.tn_chunk = 5 if e_s != "cpu" else 1        # TrackNet windows per call
        self.tn, self.bg, self.bg_samples, self.bg_seen = {}, {}, {}, {}
        self.n = {}
        self.imgs, self.lock = {}, threading.Lock()
        self.outbox = {}
        self.aq, self.athread = None, None
        self.max_backlog, self.dropped = 30, 0
        self.first = True

    @classmethod
    def maybe(cls, rcfg, device, ep):
        return cls(rcfg, device, ep) if settings(rcfg) else None

    # -------------------------------------------------------------- warm-up (engines built before the stream)
    def warmup(self, img_hw):
        """Every batch size a stream uses, so TensorRT / cuDNN tune before the stream (CPU: one small call each -
        three CPU processes warming up GPU-sized batches ran a 16 GB laptop out of memory)."""
        h, w = img_hw
        img = np.zeros((h, w, 3), np.uint8)
        t = time.monotonic_ns()
        gpu = self.ep["racket"] != "cpu"
        if self.det is not None:
            for n in ((1, 3, 5) if gpu else (1,)):
                self.det.batch([img] * n)
        if self.pose is not None:
            for n in ((1, 4, 8, 16, self.chunk) if gpu else (1,)):
                self.pose.infer(np.zeros((n, 256, 256, 3), np.uint8))
        if self.tn_run is not None:
            for n in ((1, 3, 5) if self.ep["shuttle"] != "cpu" else (1,)):
                self.tn_run(np.zeros((n, 27, L.TN_H, L.TN_W), np.float32))
        return (time.monotonic_ns() - t) / 1e9

    # -------------------------------------------------------------- the frame of each (view, f) of a step
    def stash(self, view, img):
        with self.lock:
            self.imgs.setdefault(view, deque()).append(img)

    def _take(self, view):
        with self.lock:
            d = self.imgs.get(view)
            return d.popleft() if d else None

    def _pending(self):
        with self.lock:
            return sum(len(d) for d in self.imgs.values())

    def wrap_put(self, put, inv):
        buf = []

        def put2(v, msg):
            if msg[0] != "frame":
                return put(v, msg)
            _, view, f, dets, info = msg
            img = None if info.get("dropped") else self._take(view)
            if img is None:
                extra = self.drain(view)
                return put(v, ("frame", view, f, dets, dict(info, rs=extra) if extra else info))
            buf.append((v, view, f, dets, info, img))
            if self._pending():
                return None
            items = [(view_, f_, img_, dets_, inv[view_]) for (_, view_, f_, dets_, _, img_) in buf]
            res = {}
            if self.s["async"]:
                self.submit(items)
            else:
                res = self.process(items)
            for vq, view_, f_, dets_, info_, _ in buf:
                rs = ([res[(view_, f_)]] if (view_, f_) in res else []) + self.drain(view_)
                put(vq, ("frame", view_, f_, dets_, dict(info_, rs=rs) if rs else info_))
            buf.clear()
            return None
        return put2

    def begin_session(self):
        with self.lock:
            self.imgs.clear()
        for r in self.tn.values():
            r.reset()

    # -------------------------------------------------------------- async
    def submit(self, items):
        if self.athread is None:
            self.aq = queue.Queue()
            self.athread = threading.Thread(target=self._loop, name="rs2d", daemon=True)
            self.athread.start()
        if self.aq.qsize() >= self.max_backlog:          # cannot keep up: drop the oldest waiting step
            try:
                old = self.aq.get_nowait()
                self.dropped += len(old)
            except queue.Empty:
                pass
        self.aq.put(items)

    def _loop(self):
        while True:
            items = self.aq.get()
            if items is None:
                return
            try:
                res = self.process(items)
            except Exception as e:  # noqa: BLE001 - reported on the next message, the pose stream goes on
                res = {(items[0][0], items[0][1]): {"f": items[0][1], "error": f"{type(e).__name__}: {e}"}}
            with self.lock:
                for (view, _), r in res.items():
                    self.outbox.setdefault(view, []).append(r)

    def drain(self, view):
        with self.lock:
            return self.outbox.pop(view, [])

    # -------------------------------------------------------------- the work
    def process(self, items):
        """items [(view, f, img (reader px), dets (calib px), calib px per reader px)] -> {(view, f): result}."""
        st = self.s
        t0 = time.monotonic_ns()
        res = {(v, f): {"f": int(f)} for v, f, *_ in items}
        tim = {"items": len(items)}
        if self.pose is not None:
            self._racket(items, res, tim)
        if self.tn_run is not None:
            self._shuttle(items, res, tim)
        tim["all_ms"] = round((time.monotonic_ns() - t0) / 1e6, 2)
        if self.s["async"]:
            tim["backlog"] = self.aq.qsize() if self.aq is not None else 0
            tim["dropped"] = self.dropped
        if self.first:
            tim["providers"] = self.providers
            tim["ep"] = self.ep
            self.first = False
        res[(items[0][0], items[0][1])]["t"] = tim
        return res

    def _racket(self, items, res, tim):
        st = self.s
        boxes = {}
        det_items = []
        for v, f, img, dets, inv in items:
            n = self.n.get(v, 0)
            self.n[v] = n + 1
            if self.det is not None and n % st["det_every"] == 0:
                det_items.append((v, f, img))
        t = time.monotonic_ns()
        det_out = self.det.batch([im for _, _, im in det_items]) if det_items else []
        tim["det_ms"] = round((time.monotonic_ns() - t) / 1e6, 2) if det_items else 0.0
        tim["det_images"] = len(det_items)
        for (v, f, _), (b, s) in zip(det_items, det_out):
            boxes[(v, f)] = [(bb, "d", -1, -1, float(ss)) for bb, ss in zip(b, s)]
        for v, f, img, dets, inv in items:
            got = boxes.setdefault((v, f), [])
            use_wrist = st["racket"] in ("wrist", "both") or (st["racket"] == "det" and not got and st["det_every"] > 1)
            if not use_wrist:
                continue
            dboxes = [g[0] for g in got]
            for p, d in enumerate(dets):
                kp = np.asarray(d, np.float32)
                wb, wid = L.wrist_boxes(kp[:17, :2] / inv, kp[:17, 2], st["box_k"], st["box_push"], st["wrist_min"])
                for bb, w in zip(wb, wid):
                    if st["racket"] == "both" and any(_iou(bb, db) > 0.3 for db in dboxes):
                        continue
                    got.append((bb, "w", p, int(w), 0.0))
        t = time.monotonic_ns()
        crops, meta = [], []
        for v, f, img, dets, inv in items:
            bl = boxes.get((v, f)) or []
            if not bl:
                continue
            c, cen, side = self.pose.warp(img, np.stack([b[0] for b in bl]))
            crops.append(c)
            meta.append((v, f, inv, bl, cen, side))
        tim["crops"] = int(sum(len(c) for c in crops))
        t1 = time.monotonic_ns()
        if crops:
            loc, val = self._infer(np.concatenate(crops))
        kps, k = [], 0
        for v, f, inv, bl, cen, side in meta:
            n = len(bl)
            kps.append((self.pose.to_image(loc[k:k + n], cen, side), val[k:k + n]))
            k += n
        if st["refine"] and meta:          # second pass on a tight box around each first-pass racket
            crops2, geo = [], []
            for (v, f, inv, bl, cen, side), (kp, _) in zip(meta, kps):
                img = next(im for vv, ff, im, *_ in items if vv == v and ff == f)
                c2, cen2, side2 = self.pose.warp(img, np.stack([L.keypoint_box(q) for q in kp]))
                crops2.append(c2)
                geo.append((cen2, side2))
            loc2, val2 = self._infer(np.concatenate(crops2))
            k = 0
            for i, ((kp, sc), (cen2, side2)) in enumerate(zip(kps, geo)):
                n = len(kp)
                kp2, sc2 = self.pose.to_image(loc2[k:k + n], cen2, side2), val2[k:k + n]
                k += n
                better = sc2.mean(1) > sc.mean(1)
                kps[i] = (np.where(better[:, None, None], kp2, kp), np.where(better[:, None], sc2, sc))
        t2 = time.monotonic_ns()
        for (v, f, inv, bl, cen, side), (kp, sc) in zip(meta, kps):
            kp = kp * inv
            cands = []
            for j, (bb, src, p, w, ds) in enumerate(bl):
                cands.append({"kp": [[_r(kp[j, i, 0]), _r(kp[j, i, 1]), _r(sc[j, i], 3)] for i in range(5)],
                              "src": src, "p": p, "w": w, "ds": _r(ds, 3), "m": float(np.mean(sc[j]))})
            cands.sort(key=lambda c: -c["m"])
            for c in cands:
                c["m"] = _r(c["m"], 3)
            res[(v, f)]["racket"] = cands[:st["max_cands"]]
        tim["warp_ms"] = round((t1 - t) / 1e6, 2)
        tim["pose_ms"] = round((t2 - t1) / 1e6, 2)

    def _infer(self, crops):
        locs, vals = [], []
        for a in range(0, len(crops), self.chunk):
            lo, va = self.pose.infer(crops[a:a + self.chunk])
            locs.append(lo)
            vals.append(va)
        return np.concatenate(locs), np.concatenate(vals)

    def _shuttle(self, items, res, tim):
        st = self.s
        due = []
        t = time.monotonic_ns()
        for v, f, img, dets, inv in items:
            r = self.tn.get(v)
            if r is None:
                r = self.tn[v] = L.TrackNetRunner(self.tn_run, stride=st["stride"], weights=st["weights"])
            if v not in self.bg:
                seen = self.bg_seen.get(v, 0)
                self.bg_seen[v] = seen + 1
                if seen % 2 == 0:
                    self.bg_samples.setdefault(v, []).append(
                        L.cv2.resize(img, (L.TN_W, L.TN_H), interpolation=L.cv2.INTER_AREA))
                if len(self.bg_samples.get(v, [])) >= st["bg_frames"]:
                    self.bg[v] = np.median(np.stack(self.bg_samples.pop(v)), 0).astype(np.uint8)
                    r.set_background(self.bg[v])
                continue
            out, is_due = r.add(f, img)
            if out:
                res[(v, f)].setdefault("sh", []).extend(_sh(out, inv))
            if is_due:
                due.append((v, f, r, inv))
        t1 = time.monotonic_ns()
        for a in range(0, len(due), self.tn_chunk):
            part = due[a:a + self.tn_chunk]
            hm = self.tn_run(np.stack([r.window() for _, _, r, _ in part]))[0]
            for j, (v, f, r, inv) in enumerate(part):
                res[(v, f)].setdefault("sh", []).extend(_sh(r.take(hm[j]), inv))
        t2 = time.monotonic_ns()
        tim["tn_prep_ms"] = round((t1 - t) / 1e6, 2)
        tim["tn_ms"] = round((t2 - t1) / 1e6, 2)
        tim["tn_calls"] = len(due)


def _sh(out, inv):
    return [[int(k), None if xy is None else _r(xy[0] * inv), None if xy is None else _r(xy[1] * inv), _r(pk, 3)]
            for k, xy, pk in out]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ------------------------------------------------------------------------------------------------ sink (main process)
class RSSink:
    """Created with each run's FrameWriter: starts the 3D process, forwards the 2D results and the written frames."""
    def __init__(self, run_dir, cfg, header_doc):
        import multiprocessing as mp
        from analysis.live import sync
        global CURRENT
        st = settings(cfg.get("rtmpose"))
        steps = {int(s["step"]): {v: [list(map(int, vv["frames"])), float(vv["alpha"])] for v, vv in s["views"].items()}
                 for s in sync.schedule(cfg)}
        cams = {c["name"]: [c["K"], c["R"], c["t"]] for c in header_doc["meta"]["cameras"] if c["name"] in cfg["views"]}
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.proc = ctx.Process(target=rs3d_main, name="rs3d", daemon=True,
                                args=(run_dir, st, cams, steps, {v: float(o) for v, o in cfg["offsets"].items()},
                                      float(cfg["source_fps"]), self.q))
        self.proc.start()
        self.seen = {}
        CURRENT = self

    def on_2d(self, view, got):
        if got is None:
            return
        tim = got[1]
        rs = tim.get("rs") if isinstance(tim, dict) else None
        if not rs:
            return
        seen = self.seen.setdefault(view, deque(maxlen=64))
        if id(tim) in seen:
            return
        seen.append(id(tim))
        self.q.put(("2d", view, rs))

    def on_frame(self, o, obj):
        players = {}
        for pid, p in o["players"].items():
            J = np.asarray(p["J"], np.float64)
            C = np.asarray(p.get("C", np.ones(len(J))), np.float64)
            players[pid] = {"valid": bool(p["valid"]), "quality": int(p["quality"]),
                            "hands": J[[20, 21, 22, 23]].tolist(), "hc": C[[20, 21, 22, 23]].tolist(),
                            "pelvis": J[0].tolist()}
        self.q.put(("frame", int(obj["step"]), int(obj["ref_frame"]), obj.get("t_capture_ns"),
                    obj.get("t_written_ns"), players))

    def close(self, timeout_s=60):
        global CURRENT
        self.q.put(("end",))
        self.proc.join(timeout_s)
        if self.proc.is_alive():
            self.proc.terminate()
        if CURRENT is self:
            CURRENT = None


# ------------------------------------------------------------------------------------------------ 3D (own process)
def rs3d_main(run_dir, st, cams_raw, steps, offsets, fps, q):
    import gzip
    from analysis import racket_rigidfit as rf
    rec = gzip.open(os.path.join(run_dir, "rs_inputs.jsonl.gz"), "wt", encoding="utf-8")
    rec.write(json.dumps({"header": {"st": st, "cams": cams_raw, "steps": steps, "offsets": offsets, "fps": fps}}) + "\n")
    R3 = RS3D(run_dir, st, {v: rf.Camera(*c) for v, c in cams_raw.items()}, steps, offsets, fps, record=rec)
    try:
        R3.loop(q)
    finally:
        R3.finish()
        rec.close()


def replay(inputs, out_dir, overrides=None, log=print):
    """Run the 3D again on a run's recorded inputs (rs_inputs.jsonl.gz) with other 3D settings -> out_dir/
    racket_shuttle.jsonl + rs_summary.json (2D and arrival order as recorded; no waiting, so every frame gets the
    2D the stream had received by then)."""
    import gzip
    from analysis import racket_rigidfit as rf
    with gzip.open(inputs, "rt", encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh]
    h = lines[0]["header"]
    st = dict(h["st"], **(overrides or {}))
    steps = {int(k): v for k, v in h["steps"].items()}
    os.makedirs(out_dir, exist_ok=True)
    R3 = RS3D(out_dir, st, {v: rf.Camera(*c) for v, c in h["cams"].items()}, steps, h["offsets"], h["fps"])
    t = time.perf_counter()
    for m in lines[1:]:
        if m[0] == "2d":
            R3._add_2d(m[1], m[2])
        elif m[0] == "frame":
            R3.pending.append((0.0, m[1:]))
        R3.process(force=False, now=0.0)
    R3.process(force=True)
    R3.finish()
    log(f"replayed {sum(1 for m in lines[1:] if m[0] == 'frame')} frames in {time.perf_counter() - t:.1f} s -> {out_dir}")
    return out_dir


class RS3D:
    def __init__(self, run_dir, st, cams, steps, offsets, fps, record=None):
        from analysis import racket_rigidfit as rf
        self.record = record
        self.rf, self.st, self.cams, self.steps, self.off, self.fps = rf, st, cams, steps, offsets, fps
        self.fitter = rf.RacketFitter(cams, rf.RacketTemplate(), rf.FitConfig())
        self.hc = rf.RacketTemplate().head_center()
        self.rk, self.sh = {}, {}
        self.sh_max = {v: -1 for v in cams}
        self.rk_max = {v: -1 for v in cams}
        self.pending = deque()
        self.last = {}               # pid -> (step, FitResult)
        self.hand_votes = {}         # pid -> [left, right]
        self.hand = {}               # pid -> hand used last frame (0 left, 1 right)
        self.retry_at = {}           # (pid, hand) -> step before which no full racket fit is tried again
        self.arc_prev = None         # (ref, [X0, V, A]) of the last shuttle arc (warm start for the next frame)
        self.camK = {v: c.K for v, c in cams.items()}
        self.out = open(os.path.join(run_dir, "racket_shuttle.jsonl"), "w", encoding="utf-8", buffering=1)
        self.run_dir = run_dir
        self.stats = {"frames": 0, "racket": {}, "shuttle": 0, "racket_ms": [], "shuttle_ms": [], "lag_ms": [],
                      "after_pose_ms": [], "worker": [], "racket2d": {}, "shuttle2d": {}, "errors": []}

    # -------------------------------------------------------------- input
    def loop(self, q):
        while True:
            try:
                msg = q.get(timeout=0.05)
            except queue.Empty:
                self.process(force=False)
                continue
            if msg[0] == "end":
                self.process(force=True)
                return
            if self.record is not None:
                self.record.write(json.dumps(list(msg), separators=(",", ":"), default=str) + "\n")
            if msg[0] == "2d":
                self._add_2d(msg[1], msg[2])
            elif msg[0] == "frame":
                self.pending.append((time.monotonic(), msg[1:]))
            self.process(force=False)

    def _add_2d(self, view, entries):
        for e in entries:
            if "error" in e:
                self.stats["errors"].append(e["error"])
            if "t" in e:
                self.stats["worker"].append(e["t"])
            f = e.get("f")
            if "racket" in e:
                self.rk[(view, f)] = [np.asarray(c["kp"], np.float64) for c in e["racket"]]
                self.rk_max[view] = max(self.rk_max.get(view, -1), f)
                d = self.stats["racket2d"].setdefault(view, [0, 0])
                d[0] += 1
                d[1] += bool(e["racket"])
            for k, x, y, pk in e.get("sh") or []:
                self.sh[(view, k)] = None if x is None else (x, y, pk)
                self.sh_max[view] = max(self.sh_max.get(view, -1), k)
                d = self.stats["shuttle2d"].setdefault(view, [0, 0])
                d[0] += 1
                d[1] += x is not None

    # -------------------------------------------------------------- output frames
    def _ready(self, step, wait_s):
        """The 2D this frame needs has arrived (racket: its source frames; shuttle: +window frames beyond), or the
        frame has waited 1.5 s (exp_rs_async sends the 2D a step or more after the pose)."""
        if wait_s > 1.5:
            return True
        need = self.st["window"] + 1
        for v, (frames, alpha) in self.steps.get(step, {}).items():
            if self.st["shuttle"] and self.sh_max.get(v, -1) < max(frames) + need:
                return False
            if self.st["racket"] and self.st["racket3d"] and self.rk_max.get(v, -1) < self._nearest(frames, alpha):
                return False                # (2D arrives in frame order per camera: a later frame = this one is done)
        return True

    def process(self, force, now=None):
        while self.pending:
            t_in, (step, ref, t_cap, t_written, players) = self.pending[0]
            wait = (time.monotonic() if now is None else now) - t_in
            if not force and not self._ready(step, wait):
                return
            self.pending.popleft()
            self._frame(step, ref, t_cap, t_written, players)

    def _frame(self, step, ref, t_cap, t_written, players):
        views = self.steps.get(step, {})
        line = {"step": step, "ref_frame": ref, "t_capture_ns": t_cap, "t_pose_ns": t_written}
        if self.st["racket"] and self.st["racket3d"]:
            t = time.perf_counter()
            line["racket"] = {pid: self._racket(pid, step, p, views) for pid, p in sorted(players.items())}
            ms = (time.perf_counter() - t) * 1000
            line["racket_ms"] = round(ms, 2)
            self.stats["racket_ms"].append(ms)
        if self.st["shuttle"]:
            t = time.perf_counter()
            line["shuttle"] = self._shuttle(ref, views)
            ms = (time.perf_counter() - t) * 1000
            line["shuttle_ms"] = round(ms, 2)
            self.stats["shuttle_ms"].append(ms)
            self.stats["shuttle"] += line["shuttle"] is not None
        now = time.monotonic_ns()
        line["t_rs_ns"] = now
        if t_cap is not None:
            self.stats["lag_ms"].append((now - t_cap) / 1e6)
        if t_written is not None:
            self.stats["after_pose_ms"].append((now - t_written) / 1e6)
        self.stats["frames"] += 1
        self.out.write(json.dumps(line, allow_nan=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _nearest(frames, alpha):
        return frames[0] if len(frames) < 2 or alpha < 0.5 else frames[1]

    def _racket(self, pid, step, p, views):
        rs = self.stats["racket"].setdefault(pid, {"valid": 0, "fit": 0, "left": 0, "right": 0})
        if not p["valid"]:
            return None
        rs["valid"] += 1
        hands = np.asarray(p["hands"], np.float64)       # left wrist, right wrist, left hand, right hand
        hc = np.asarray(p["hc"], np.float64)
        H = [hands[2] if hc[2] > 0.05 else hands[0], hands[3] if hc[3] > 0.05 else hands[1]]
        ok = [bool(np.isfinite(h).all()) for h in H]
        by_hand = [{}, {}]
        dist = [0.0, 0.0]               # per hand: sum over views of the nearest handle's distance (metres, capped)
        seen = 0
        for v, (frames, alpha) in views.items():
            cands = self.rk.get((v, self._nearest(frames, alpha)))
            if not cands or v not in self.cams:
                continue
            cam = self.cams[v]
            uv = [cam.project(h)[0] if o else None for h, o in zip(H, ok)]
            ppm = [cam.px_per_m(h) if o else 1.0 for h, o in zip(H, ok)]
            best = [0.45, 0.45]
            for c in cands:
                if float(np.mean(c[:, 2])) < 0.2:
                    continue
                d = [float(np.linalg.norm(c[2, :2] - u)) / m if u is not None else np.inf for u, m in zip(uv, ppm)]
                for kk in (0, 1):
                    if d[kk] < 0.45:
                        by_hand[kk].setdefault(v, []).append((d[kk], c))
                        best[kk] = min(best[kk], d[kk])
            dist[0] += best[0]
            dist[1] += best[1]
            seen += 1
        if not seen or (not by_hand[0] and not by_hand[1]):
            return None
        # the hand whose projection has racket handles nearest in every view, chosen each frame (the live pose can
        # swap a far player's arms, so a hand locked by its label goes wrong); hysteresis: keep last frame's hand
        # unless the other one is clearly nearer
        k = int(np.argmin(dist))
        prev_k = self.hand.get(pid)
        if prev_k is not None and k != prev_k and dist[k] > 0.7 * dist[prev_k]:
            k = prev_k
        self.hand[pid] = k
        votes = self.hand_votes.setdefault(pid, [0, 0])
        votes[k] += 1
        prev = self.last.get(pid)
        prior = prev[1].x if prev is not None and step - prev[0] <= 3 else None
        r = None
        for kk in (k, 1 - k):                    # the chosen hand, then the other one if that fit fails
            cand_by_view = {v: [c for _, c in sorted(lst, key=lambda x: x[0])] for v, lst in by_hand[kk].items()}
            if not cand_by_view:
                continue
            r = self._fit_hand((pid, kk), step, cand_by_view, H[kk], prior if kk == k else None)
            if r is not None:
                k = kk
                self.hand[pid] = k
                break
        if r is None:
            return None
        self.last[pid] = (step, r)
        rs["fit"] += 1
        rs["left" if k == 0 else "right"] += 1
        Rm = r.R
        head = r.grip + Rm @ self.hc
        return {"grip": [round(float(x), 4) for x in r.grip], "head": [round(float(x), 4) for x in head],
                "axis": [round(float(x), 4) for x in Rm[:, 2]], "across": [round(float(x), 4) for x in Rm[:, 0]],
                "hand": "left" if k == 0 else "right", "views": len(r.views), "err": round(r.err, 4),
                "conf": round(self.rf.confidence(r), 3)}

    def _fit_hand(self, key, step, cand_by_view, hand, prior):
        """key = (player, hand index): a failed full fit is not retried for that hand for 3 steps."""
        fast = self.st.get("fast_fit", True)
        r = self._fast_fit(cand_by_view, hand, prior) if prior is not None and fast else None
        if r is None:
            if fast and step < self.retry_at.get(key, -1):
                return None
            r = self._full_fit(cand_by_view, hand, prior) if fast else self.fitter.fit(cand_by_view, hand=hand,
                                                                                      prior=prior)
            if r is None or r.err > self.fitter.cfg.frame_accept:
                self.retry_at[key] = step + 3
                return None
        return r if r.err <= self.fitter.cfg.frame_accept else None

    def _fast_fit(self, cand_by_view, hand, prior):
        """Live fast path: one solve from the previous frame's pose (the offline fitter tries many starts each frame,
        ~10x the time); each view keeps the candidate nearest the result. None (-> the full fit) when it is not
        clearly good: median view error >= 0.08 racket lengths or a view above the offline rejection level."""
        f = self.fitter
        obs = {v: cs[0] for v, cs in cand_by_view.items() if f._usable(cs[0])}
        if not obs:
            return None
        x, cost = self._solve(prior.copy(), obs, hand, prior, 8)
        return self._finish(x, cost, obs, cand_by_view, hand, prior, 0.08)

    def _full_fit(self, cand_by_view, hand, prior):
        """The offline fitter's starts (previous pose, triangulated shaft x 4 rolls, else directions around the hand),
        at most 8 of them, short solves, then the same finish as the fast path (offline: up to 32+ starts, longer)."""
        f = self.fitter
        obs = {v: cs[0] for v, cs in cand_by_view.items() if f._usable(cs[0])}
        if not obs or (hand is None and len(obs) < 2):
            return None
        starts = f._starts(obs, hand, prior, ())
        if len(starts) > 8:
            starts = starts[:1] + starts[1::max(1, (len(starts) - 1) // 7)][:7]
        best = None
        for x0 in starts:
            pr = f._prior_aligned(prior, self.rf.rotmat(x0[3:])) if prior is not None else None
            x, cost = self._solve(x0, obs, hand, pr, 6)
            if best is None or cost < best[1]:
                best = (x, cost, pr)
        x, cost = self._solve(best[0], obs, hand, best[2], 12)
        return self._finish(x, cost, obs, cand_by_view, hand, best[2], f.cfg.frame_accept)

    def _solve(self, x, obs, hand, prior, iters):
        """exp_racket3d_solver gn (default: _gn) | scipy (the offline fitter's least_squares, ~10x slower)."""
        if self.st.get("solver", "gn") == "scipy":
            return self.fitter._solve(x, obs, hand, prior, max_nfev=self.st.get("fast_nfev", 30) * 2)
        x1, c1 = self._gn(x, obs, hand, prior, iters)
        sw = {v: self.fitter.view_error(x1, v, o)[1] for v, o in obs.items()}
        sw0 = {v: self.fitter.view_error(x, v, o)[1] for v, o in obs.items()}
        if sw != sw0:                                     # left / right labels flipped in a view: once more
            x1, c1 = self._gn(x1, obs, hand, prior, iters)
        return x1, c1

    def _gn(self, x, obs, hand, prior, iters=8):
        """The offline fit's cost (racket_rigidfit.RacketFitter._residual_fn: score-weighted keypoint reprojection /
        sigma_px with a Cauchy loss, grip-to-hand and previous-pose priors, left / right swap per view) minimised by
        damped Gauss-Newton with analytic derivatives (rotation as a left increment) - scipy's least_squares with
        numerical derivatives took ~10 ms per solve here."""
        f, cfg, rf = self.fitter, self.fitter.cfg, self.rf
        flip = np.array([1.0, -1.0, 1.0])
        g, R = np.array(x[:3], float), rf.rotmat(x[3:])
        V = []
        for v, o in obs.items():
            o = o[rf._SWAP] if f.view_error(x, v, o)[1] else o
            c = self.cams[v]
            w = np.sqrt(np.where(o[:, 2] >= cfg.min_score, o[:, 2], 0.0)) / cfg.sigma_px
            V.append((c.K, c.R @ np.diag(flip), c.t, o[:, :2], w))
        if prior is not None:
            pg, pRt = prior[:3], rf.rotmat(prior[3:]).T
        cost = np.inf
        lam = 1e-3
        for it in range(iters):
            Rk = f.kp @ R.T                                   # (5,3) keypoints relative to the grip
            P = g + Rk
            rows, res = [], []
            for K, A, t, uvo, w in V:
                Xc = P @ A.T + t
                z = Xc[:, 2]
                ok = z > 0.05
                zz = np.where(ok, z, 1.0)
                uvw = Xc @ K.T
                uv = uvw[:, :2] / zz[:, None]
                e = (uv - uvo) * w[:, None]
                # d uv / d Xc (rows u, v) for a general K, then d uv / d world point and d / d rotation increment
                zero = np.zeros_like(zz)
                D = np.stack([np.stack([K[0, 0] / zz, K[0, 1] / zz, -(uv[:, 0] - K[0, 2]) / zz], 1),
                              np.stack([zero, K[1, 1] / zz, -(uv[:, 1] - K[1, 2]) / zz], 1)], 1)    # (5,2,3)
                B = D @ A
                S = np.zeros((5, 3, 3))
                S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -Rk[:, 2], Rk[:, 1], -Rk[:, 0]
                S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = Rk[:, 2], -Rk[:, 1], Rk[:, 0]
                Jv = np.concatenate([B, -B @ S], 2) * w[:, None, None]                              # (5,2,6)
                use = ok & (w > 0)
                rows.append(Jv[use].reshape(-1, 6))
                res.append(e[use].reshape(-1))
            if not sum(len(q) for q in res):
                return np.concatenate([g, rf.rotvec(R)]), np.inf
            J = np.concatenate(rows)                                  # (2n, 6)
            r = np.concatenate(res)
            s = r ** 2
            sw = 1.0 / np.sqrt(1.0 + s)                               # Cauchy IRLS weights (f_scale 1)
            J, r = J * sw[:, None], r * sw
            c_new = 0.5 * float(np.log1p(s).sum())
            extra_J, extra_r = [], []
            if hand is not None:
                extra_J.append(np.hstack([np.eye(3), np.zeros((3, 3))]) / cfg.hand_sigma)
                extra_r.append((g - hand) / cfg.hand_sigma)
            if prior is not None:
                extra_J.append(np.hstack([np.eye(3), np.zeros((3, 3))]) / cfg.prior_pos_sigma)
                extra_r.append((g - pg) / cfg.prior_pos_sigma)
                extra_J.append(np.hstack([np.zeros((3, 3)), np.eye(3)]) / cfg.prior_rot_sigma)
                extra_r.append(rf.rotvec(R @ pRt) / cfg.prior_rot_sigma)
            if extra_J:
                J = np.concatenate([J] + extra_J)
                r = np.concatenate([r] + extra_r)
                c_new += 0.5 * float(sum((q ** 2).sum() for q in extra_r))
            cost = c_new
            H = J.T @ J
            d = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), J.T @ r)
            g = g + d[:3]
            R = rf.rotmat(d[3:]) @ R
            if np.abs(d).max() < 2e-4:                      # 0.2 mm / 0.01 deg
                break
        return np.concatenate([g, rf.rotvec(R)]), cost

    def _finish(self, x, cost, obs, cand_by_view, hand, prior, accept):
        f = self.fitter
        changed = False
        for v, cs in cand_by_view.items():
            errs = [(f.view_error(x, v, c)[0], i) for i, c in enumerate(cs) if f._usable(c)]
            if errs:
                e, i = min(errs)
                if e < (f.view_error(x, v, obs[v])[0] if v in obs else np.inf) * 0.7:
                    obs[v] = cs[i]
                    changed = True
        if changed:
            x, cost = self._solve(x, obs, hand, prior, 6)
        per = {v: f.view_error(x, v, o)[0] for v, o in obs.items()}
        while len(per) > 1 and max(per.values()) > f.cfg.view_reject:      # drop the worst view, solve again
            worst = max(per, key=per.get)
            del obs[worst]
            x, cost = self._solve(x, obs, hand, prior, 6)
            per = {v: f.view_error(x, v, o)[0] for v, o in obs.items()}
        if not per or np.median(list(per.values())) >= accept or max(per.values()) > f.cfg.view_reject:
            return None
        views = list(per)
        score = float(np.mean([obs[v][obs[v][:, 2] >= f.cfg.min_score, 2].mean() for v in views]))
        return self.rf.FitResult(x=x, cost=cost, err=float(np.mean(list(per.values()))),
                                 per_view={v: round(e, 4) for v, e in per.items()}, views=views,
                                 choice={v: 0 for v in views}, score=score)

    # -------------------------------------------------------------- shuttle
    def _obs(self, ref, lo, hi):
        """[(view, t rel. to ref frame, uv)] of the shuttle in every view for ref-relative times lo..hi."""
        out = []
        for v in self.cams:
            o = self.off.get(v, 0.0)
            for k in range(int(np.floor(ref + o + lo)), int(np.ceil(ref + o + hi)) + 1):
                got = self.sh.get((v, k))
                t = k - o - ref
                if got is not None and lo - 1e-6 <= t <= hi + 1e-6:
                    out.append((v, t, np.array(got[:2], np.float64)))
        return out

    def _tri(self, obs):
        """Best-supported point from the observations nearest t=0 per view (pairs + inliers)."""
        from analysis.triangulate import triangulate_point
        best_t = {}
        for v, t, uv in obs:
            if v not in best_t or abs(t) < abs(best_t[v][0]):
                best_t[v] = (t, uv)
        vs = [v for v in best_t if abs(best_t[v][0]) <= 0.75]
        if len(vs) < 2:
            return None
        thr = self.st["shuttle_px"]
        best = None
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                X = self._dlt([vs[i], vs[j]], [best_t[vs[i]][1], best_t[vs[j]][1]], triangulate_point)
                if X is None:
                    continue
                inl = [v for v in vs if self._err(v, X, best_t[v][1]) < thr]
                if len(inl) >= 2 and (best is None or len(inl) > len(best[1])):
                    best = (X, inl)
        if best is None:
            return None
        X = self._dlt(best[1], [best_t[v][1] for v in best[1]], triangulate_point)
        if X is None:
            return None
        e = [self._err(v, X, best_t[v][1]) for v in best[1]]
        return X, None, best[1], float(np.median(e)), len(best[1])

    def _dlt(self, vs, uvs, triangulate_point):
        try:
            X = triangulate_point([self.cams[v].P() for v in vs], uvs) * np.array([1.0, -1.0, 1.0])
        except (ValueError, np.linalg.LinAlgError):
            return None
        return X if np.isfinite(X).all() and -2.0 < X[1] < 15.0 else None

    def _err(self, v, X, uv):
        p, z = self.cams[v].project(X)
        return float(np.linalg.norm(p - uv)) if z > 0.1 else 1e9

    def _trusted(self, res):
        """Seen by 3+ cameras, or by 2 whose rays meet at a wide enough angle with a tight fit (two nearly parallel
        rays leave the depth loose: most 1-2 m misses in the offline-2D replay were such 2-camera points)."""
        X, _, vs, e = res[0], res[1], res[2], res[3]
        if len(vs) >= 3:
            return True
        if e > self.st.get("two_view_px", 12.0):
            return False
        need = self.st.get("two_view_deg", 8.0)
        if not need:
            return True
        d = []
        for v in vs[:2]:
            c = self.cams[v]
            C = (-c.R.T @ c.t) * np.array([1.0, -1.0, 1.0])        # camera centre in the analysis world
            u = C - X
            d.append(u / max(np.linalg.norm(u), 1e-9))
        return float(np.degrees(np.arccos(np.clip(np.dot(d[0], d[1]), -1, 1)))) >= need

    def _arc(self, obs, X0):
        """Quadratic path through the observations (robust), -> (X(0), V(0) per frame, views, median px, n obs)."""
        vs = sorted({v for v, _, _ in obs})
        if len(vs) < 2 or len(obs) < 5:
            return None
        Ks = np.stack([self.cams[v].K for v, _, _ in obs])
        Rs = np.stack([self.cams[v].R for v, _, _ in obs])
        tcs = np.stack([self.cams[v].t for v, _, _ in obs])
        ts = np.array([t for _, t, _ in obs])
        uv = np.stack([u for _, _, u in obs])
        thr = self.st["shuttle_px"]
        flip = np.array([1.0, -1.0, 1.0])

        x0 = np.array(X0 if len(X0) == 9 else np.concatenate([X0, np.zeros(6)]), float)
        A = np.einsum("nij,j->nij", Rs, flip)                     # camera rotation with the Y flip folded in
        Tm = np.stack([np.ones_like(ts), ts, 0.5 * ts ** 2], 1)    # X(t) = X0 + V t + A t^2 / 2
        p = x0.copy()
        lam = 1e-3
        for it in range(12):                    # Gauss-Newton, analytic derivatives, soft_l1 IRLS (scale thr)
            X = Tm[:, 0:1] * p[:3] + Tm[:, 1:2] * p[3:6] + Tm[:, 2:3] * p[6:9]
            Xc = np.einsum("nij,nj->ni", A, X) + tcs
            z = Xc[:, 2]
            zz = np.where(z > 0.05, z, 1.0)
            uvw = np.einsum("nij,nj->ni", Ks, Xc)
            uvp = uvw[:, :2] / zz[:, None]
            e = uvp - uv
            zero = np.zeros_like(zz)
            D = np.stack([np.stack([Ks[:, 0, 0] / zz, Ks[:, 0, 1] / zz, -(uvp[:, 0] - Ks[:, 0, 2]) / zz], 1),
                          np.stack([zero, Ks[:, 1, 1] / zz, -(uvp[:, 1] - Ks[:, 1, 2]) / zz], 1)], 1)
            B = D @ A                                                # (n,2,3) d uv / d X
            J = np.concatenate([B * Tm[:, 0, None, None], B * Tm[:, 1, None, None], B * Tm[:, 2, None, None]], 2)
            J[z <= 0.05] = 0.0
            e[z <= 0.05] = 0.0
            r = e.reshape(-1)
            J = J.reshape(-1, 9)
            sw = (1.0 + (r / thr) ** 2) ** -0.25                    # sqrt of soft_l1's rho'
            J, r = J * sw[:, None], r * sw
            H = J.T @ J
            try:
                d = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), J.T @ r)
            except np.linalg.LinAlgError:
                return None
            p = p + d
            if np.abs(d[:3]).max() < 1e-5:
                break
        X = Tm[:, 0:1] * p[:3] + Tm[:, 1:2] * p[3:6] + Tm[:, 2:3] * p[6:9]
        Xc = np.einsum("nij,nj->ni", A, X) + tcs
        uvw = np.einsum("nij,nj->ni", Ks, Xc)
        z = uvw[:, 2]
        fun_r = (uvw[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None] - uv).ravel()
        fun_r[np.repeat(Xc[:, 2] <= 0.05, 2)] = 1e6

        class _R:
            pass
        r = _R()
        r.x, r.fun = p, fun_r
        e = np.linalg.norm(r.fun.reshape(-1, 2), axis=1)
        inl = e < thr
        used = sorted({obs[i][0] for i in np.nonzero(inl)[0]})
        if len(used) < 2 or inl.sum() < 4:
            return None
        return r.x[:3], r.x[3:6], used, float(np.median(e[inl])), int(inl.sum()), r.x

    def _shuttle(self, ref, views):
        W, thr = self.st["window"], self.st["shuttle_px"]
        obs = self._obs(ref, -W, W)
        if not obs:
            self.arc_prev = None
            return None
        res, mode, tri = None, None, False
        if self.st["shuttle3d"] == "arc":
            seeds = []
            if self.arc_prev is not None and 0 < ref - self.arc_prev[0] <= 3:     # last frame's arc, moved on
                dt, q = ref - self.arc_prev[0], self.arc_prev[1]
                seeds.append(np.concatenate([q[:3] + q[3:6] * dt + 0.5 * q[6:9] * dt * dt, q[3:6] + q[6:9] * dt,
                                             q[6:9]]))
            seeds.append(None)                                                  # None = a triangulated point
            for seed in seeds:
                if seed is None:
                    tri = self._tri(obs)
                    seed = tri[0] if tri else None
                    if seed is None:
                        for lo, hi in ((-1.5, 1.5), (-3, 3)):
                            t2 = self._tri(self._obs(ref, lo, hi))
                            if t2 is not None:
                                seed = t2[0]
                                break
                    if seed is None:
                        break
                for lo, hi, m in ((-W, W, "arc"), (-W, 0, "arc_past"), (0, W, "arc_future")):
                    a = self._arc(obs if (lo, hi) == (-W, W) else self._obs(ref, lo, hi), seed)
                    if a is not None and (res is None or a[3] < res[3]):
                        res, mode = a, m
                    if res is not None and res[3] < thr * 0.6:
                        break
                if res is not None and res[3] < thr * 0.6:
                    break
        if res is None:
            if tri is False:
                tri = self._tri(obs)
            if tri:
                res, mode = tri + (None,), "tri"
        if res is not None and not self._trusted(res):
            res = None
        self.arc_prev = (ref, res[5]) if res is not None and res[5] is not None else None
        if res is None:
            return None
        X, V, vs, e, n, _ = res
        out = {"pos": [round(float(x), 4) for x in X], "views": len(vs), "err_px": round(e, 2), "n_obs": n,
               "mode": mode}
        if V is not None:
            out["speed_kmh"] = round(float(np.linalg.norm(V) * self.fps * 3.6), 1)
        return out

    # -------------------------------------------------------------- end
    def finish(self):
        self.out.close()
        s = self.stats

        def stat(x):
            x = np.asarray(x, float)
            return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()),
                    "n": int(len(x))} if len(x) else None
        wk = {}
        for k in ("all_ms", "det_ms", "warp_ms", "pose_ms", "tn_ms", "tn_prep_ms", "crops", "tn_calls", "items",
                  "backlog", "dropped"):
            wk[k] = stat([t[k] for t in s["worker"] if k in t])
        prov = next((t.get("providers") for t in s["worker"] if t.get("providers")), None)
        ep = next((t.get("ep") for t in s["worker"] if t.get("ep")), None)
        summ = {"settings": self.st, "frames": s["frames"], "racket": s["racket"], "hand_votes": self.hand_votes,
                "shuttle_frames": s["shuttle"], "racket_ms": stat(s["racket_ms"]), "shuttle_ms": stat(s["shuttle_ms"]),
                "lag_ms": stat(s["lag_ms"]), "after_pose_ms": stat(s["after_pose_ms"]), "worker": wk,
                "providers": prov, "ep": ep,
                "racket2d_frames_with_cands": {v: (d[1] / d[0] if d[0] else None) for v, d in s["racket2d"].items()},
                "shuttle2d_found": {v: (d[1] / d[0] if d[0] else None) for v, d in s["shuttle2d"].items()},
                "errors": s["errors"][:5]}
        with open(os.path.join(self.run_dir, "rs_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=1, default=str)
