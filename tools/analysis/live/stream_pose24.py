"""StreamPose24 (spec § 6): the pose3d node's logic as a plain class.

push(frameset) per time-step: calibrate dets -> online singles tracker (nearest source frame of each view) ->
interpolated per-view rows -> raw triangulation + causal EMA (fallback). Steps < warm-up are output at once from the
fallback (quality 1); at warm-up end the body model is estimated and locked. Every WindowStepper window is launched
when its last step has arrived AND the previous window is resolved: fixed-lag lrswap on [history + window] ->
window obs -> init (previous solution where there is one, else _fill_init of the window's raw triangulation) ->
solve_window per player (inline, or in a process pool). A window is "late" when its solve finishes more than
budget_s after its last step arrived (budget None = asfast: never late); late players' frames come from the
fallback (quality 2) and the late solution is dropped (stream.late_policy drop), waited for (use), or - v0.8.3
"behind" - waited for only while the stream is behind anyway (the window's last step reached the core more than
budget_s after its capture time: dropping then saves no delay, it only loses the fit and the next window's warm start). Output frames are emitted once each, in step order.
v0.8.4 live engine: warm_state (export_warm_state() of an earlier warm-up: body models, ankle plane) starts the core
already warm - no warm-up steps, the tracker re-acquires P1 / P2 by court half from the first step, the first window
is fitted with the saved body models. stop_at(m) ends a stream early (only steps < m are published at flush).
Gap handling (spec 6.2, causal): a step whose body has been unseen for more than max_gap steps (fit.MAX_GAP = 20 frames
at 60 fps, scaled to the step rate) is invalid (NaN, quality NONE). The leading edge of a gap, up to max_gap steps, is
still held/interpolated by the fit and output with quality FITTED but flagged interp=True (confidence 'filled')."""
from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from analysis import pose_v2 as pv
from analysis.live import fallback, lrswap_lag, outputs, stream_fit, sync
from analysis.live.tracker_singles import SinglesTracker
from analysis.pose24 import fit, scores
from analysis.pose_fit import _fill_init


@dataclass
class _Window:
    k: int
    start: int
    stop: int
    commit: int
    t_ready_ns: int
    behind: bool = False
    lo: dict = field(default_factory=dict)
    jobs: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    ev: dict = field(default_factory=dict)


def model_to_dict(m) -> dict:
    return {"lengths": _plain(m.lengths), "rest": np.asarray(m.rest, float).tolist(), "sgn": float(m.sgn)}


def model_from_dict(d):
    lengths = d["lengths"]
    return stream_fit.BodyModel(lengths, fit.edge_arrays(lengths), np.asarray(d["rest"], float), float(d["sgn"]))


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class StreamPose24:
    def __init__(self, cfg, cams, budget_s=None, workers=0, score_params=None, events=None,
                 clock=time.monotonic_ns, warm_state=None):
        self.views = sorted(cfg["views"])
        self.cams = {v: cams[v] for v in self.views}
        self.P = pv.proj_mats(self.cams, self.views)
        self.ref_frames = list(cfg["ref_frames"])
        self.n = len(self.ref_frames)
        self.fps = float(cfg["fps"])
        self.kind = cfg["skeleton"]
        self.warm = 0 if warm_state else int(cfg["warmup_steps"])
        self.warmed = False
        self.max_gap = int(round(fit.MAX_GAP * self.fps / float(cfg["source_fps"])))   # MAX_GAP: source frames
        self.hist = int(cfg["history"])
        self.budget_ns = None if budget_s is None else int(float(budget_s) * 1e9)
        self.params = score_params or scores.load_params()
        self.events = events or (lambda e: None)
        self.clock = clock
        self.cfg = cfg
        self.min_body_kp = int(cfg["stream"]["min_body_kp"])
        self.fit_every = max(1, int(cfg["stream"].get("fit_every") or 1))
        self.max_nfev = cfg["stream"].get("max_iter")
        self.tol = cfg["stream"].get("solver_tol")
        self.lsmr_maxiter = cfg["stream"].get("lsmr_maxiter")
        self.late_policy = cfg["stream"].get("late_policy") or "drop"
        if self.late_policy not in ("drop", "use", "behind"):
            raise ValueError(f"stream.late_policy must be drop, use or behind, got {self.late_policy!r}")
        self.partner_px = float(cfg["sync"]["partner_max_px"])
        ankle = (warm_state or {}).get("ankle_h_m") if warm_state else cfg.get("ankle_h_m")
        self.tracker = SinglesTracker(self.cams, self.views, self.fps, cfg["tracker"], ankle_h=ankle,
                                      wcfg=cfg.get("warmup"), warm_steps=self.warm)
        self.n_tracker_events = 0
        self.late_init = {}
        self.pids = list(self.tracker.pids)
        nan = lambda *s: np.full(s, np.nan)
        self.obs = {p: {v: np.full((self.n, 133, 3), np.nan, np.float32) for v in self.views} for p in self.pids}
        self.Xr = {p: nan(self.n, 35, 3) for p in self.pids}
        self.Xfb = {p: nan(self.n, 35, 3) for p in self.pids}
        self.Xfit = {p: nan(self.n, 35, 3) for p in self.pids}
        self.seen = {p: np.zeros(self.n, bool) for p in self.pids}
        self.ema = {p: fallback.EmaSmoother(cfg["fallback"]["alpha"], cfg["fallback_max_hold"]) for p in self.pids}
        self.model = {}
        if warm_state:
            for pid, m in (warm_state.get("models") or {}).items():
                if pid in self.pids and m:
                    self.model[pid] = model_from_dict(m)
            self.warmed = True
        self.stepper = stream_fit.WindowStepper(self.warm, self.n, cfg["window"], cfg["overlap"])
        self.pending = self.stepper.next_window()
        self.inflight, self.ready = [], {}
        self.t_recv, self.ref, self.t_cap = {}, {}, {}
        self.next_pub, self.last_step, self.k = 0, -1, 0
        self.pool = None
        if workers:
            from . import cpulimits
            nfit = (cfg.get("cpu_budget") or {}).get("fit")
            with cpulimits.child_env(nfit):
                self.pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=mp.get_context("spawn"),
                                                initializer=cpulimits.apply if nfit else None,
                                                initargs=(nfit,) if nfit else ())
            for f in [self.pool.submit(stream_fit.ping) for _ in range(int(workers))]:
                f.result()

    # ------------------------------------------------------------------ per step
    def push(self, fs):
        i = int(fs["step"])
        if i in self.t_recv or i < self.next_pub or not 0 <= i < self.n:
            return self.poll()
        t_recv = int(fs.get("t_recv_ns") or self.clock())
        self.t_recv[i], self.ref[i] = t_recv, int(fs["ref_frame"])
        if fs.get("t_capture_ns") is not None:
            self.t_cap[i] = int(fs["t_capture_ns"])
        self.last_step = max(self.last_step, i)
        t0 = time.perf_counter_ns()
        frames = {}
        for v in self.views:
            vv = fs["views"].get(v) or {"alpha": 0.0, "frames": []}
            frames[v] = (float(vv.get("alpha", 0.0)),
                         [sync.calibrate_dets(fr["dets"], self.params)[0] for fr in vv["frames"]])
        t1 = time.perf_counter_ns()
        near = {v: (fl[sync.nearest_index(len(fl), a)] if fl else []) for v, (a, fl) in frames.items()}
        assign = self.tracker.step(i, near)
        t2 = time.perf_counter_ns()
        for e in self.tracker.events[self.n_tracker_events:]:
            self.events(dict(e, node="pose3d", ev="tracker"))
        self.n_tracker_events = len(self.tracker.events)
        for pid, dets in assign.items():
            for v, idx in dets.items():
                a, fl = frames[v]
                det = near[v][idx]
                if len(fl) == 2:
                    ni = sync.nearest_index(2, a)
                    j = sync.partner(det, fl[1 - ni], self.partner_px)
                    other = fl[1 - ni][j] if j is not None else None
                    A, B = (det, other) if ni == 0 else (other, det)
                    self.obs[pid][v][i] = sync.interp_rows(A, B, a)
                else:
                    self.obs[pid][v][i] = det
        for pid in self.pids:
            uv, w = stream_fit.window_obs(self.obs[pid], self.views, i, i + 1)
            Xr = stream_fit.triangulate35(self.cams, self.views, uv, w)[0][0]
            self.Xr[pid][i] = Xr
            self.seen[pid][i] = bool((w[0, :17] >= pv.W_MIN).any(axis=-1).sum() >= self.min_body_kp)
            self.Xfb[pid][i] = self.ema[pid].push(Xr)
        t3 = time.perf_counter_ns()
        self.events({"node": "pose3d", "ev": "step", "step": i, "t_recv_ns": t_recv, "calib_ns": t1 - t0,
                     "track_ns": t2 - t1, "tri_ns": t3 - t2,
                     "n_cand": len(getattr(self.tracker, "last_candidates", []) or []),
                     "assigned": {p: len(d) for p, d in assign.items()}})
        if i < self.warm:
            self.ready[i] = self._fallback_frames(i, i + 1, outputs.WARMUP)[0]
        if not self.warmed and i >= self.warm - 1:
            self._finish_warmup()
        return self.poll()

    def _frame(self, i):
        return {"step": i, "ref_frame": self.ref.get(i, self.ref_frames[i]), "t_recv_ns": self.t_recv.get(i),
                "players": {}}

    def _fallback_frames(self, a, b, quality):
        frames = [self._frame(i) for i in range(a, b)]
        for pid in self.pids:
            uv, w = stream_fit.window_obs(self.obs[pid], self.views, a, b)
            r = outputs.regress_frames(self.cams, self.views, self.Xfb[pid][a:b], uv, w, self.kind)
            for t, fr in enumerate(frames):
                fr["players"][pid] = outputs.player_frame(r, t, quality)
        for fr in frames:
            fr["quality"] = outputs.frame_quality(fr["players"])
        return frames

    def _finish_warmup(self):
        self.warmed = True
        if self.tracker.ankle_auto:
            self.events(dict(self.tracker.estimate_ankle(), node="pose3d", ev="ankle"))
        span = {p: {v: self.obs[p][v][:self.warm] for v in self.views} for p in self.pids}
        swaps = lrswap_lag.window_swaps(self.cams, self.views, span, slice(0, self.warm))
        for pid in self.pids:
            uv, w = stream_fit.window_obs(self.obs[pid], self.views, 0, self.warm, swaps[pid])
            self.model[pid] = stream_fit.warmup_model(self.cams, self.views, uv, w, self.cfg["stream"])
        self.events({"node": "pose3d", "ev": "warmup", "models": {p: self.model[p] is not None for p in self.pids}})

    # ------------------------------------------------------------------ windows
    def _late_model(self, pid, upto):
        """A player without a warm-up body model (not seen during warm-up): build one from the last warm-up-length
        steps once it has been seen in at least half of them."""
        a = max(0, upto - self.warm)
        if self.seen[pid][a:upto].sum() < max(1, self.warm // 2):
            return None
        span = {p: {v: self.obs[p][v][a:upto] for v in self.views} for p in (pid,)}
        swaps = lrswap_lag.window_swaps(self.cams, self.views, span, slice(0, upto - a))
        uv, w = stream_fit.window_obs(self.obs[pid], self.views, a, upto, swaps[pid])
        m = stream_fit.warmup_model(self.cams, self.views, uv, w, self.cfg["stream"])
        if m is not None:
            self.model[pid] = m
            self.late_init[pid] = int(upto)
            self.events({"node": "pose3d", "ev": "late_model", "pid": pid, "step": int(upto)})
        return m

    def _launch_ready(self):
        while (self.pending is not None and self.last_step >= self.pending[2] - 1 and not self.inflight
               and self.warmed):
            self._launch(*self.pending)
            self.pending = self.stepper.next_window()

    def _launch(self, lo, start, stop, commit):
        win = _Window(self.k, start, stop, commit, self.t_recv.get(stop - 1, self.clock()))
        if self.budget_ns is not None and (stop - 1) in self.t_cap:
            win.behind = win.t_ready_ns - self.t_cap[stop - 1] > self.budget_ns
        self.k += 1
        t0 = time.perf_counter_ns()
        h0 = max(0, start - self.hist)
        span = {p: {v: self.obs[p][v][h0:stop] for v in self.views} for p in self.pids}
        swaps = lrswap_lag.window_swaps(self.cams, self.views, span, slice(0, stop - h0))
        t1 = time.perf_counter_ns()
        for pid in self.pids:
            lo_p = lo if lo < start and np.isfinite(self.Xfit[pid][lo:start]).all() else start
            win.lo[pid] = lo_p
            uv, w = stream_fit.window_obs(self.obs[pid], self.views, lo_p, stop, swaps[pid][:, lo_p - h0:])
            win.inputs[pid] = (uv, w)
            m = self.model.get(pid)
            if m is None:
                m = self._late_model(pid, start)
            if m is None:
                win.status[pid] = "none"
                continue
            Xr, _, _ = stream_fit.triangulate35(self.cams, self.views, uv, w)
            ok = np.isfinite(Xr).all(-1)
            if not (ok[:, 11] | ok[:, 12]).any():
                win.status[pid] = "none"
                continue
            if win.k % self.fit_every:
                win.status[pid] = "skipped"
                continue
            init = _fill_init(Xr, ok, m.rest)
            prev = self.Xfit[pid][lo_p:stop]
            have = np.isfinite(prev).all(axis=(1, 2))
            init[have] = prev[have]
            free = np.zeros(stop - lo_p, bool)
            free[start - lo_p:] = True
            job = {"P": self.P, "uv": uv, "w": w, "init": init, "free": free, "edges": m.edges, "anchor": Xr,
                   "contact": fit.detect_contact(Xr, self.fps), "sgn": m.sgn, "fps": self.fps,
                   "max_nfev": self.max_nfev, "tol": self.tol, "lsmr_maxiter": self.lsmr_maxiter}
            win.jobs[pid] = self.pool.submit(stream_fit.solve_window, job) if self.pool else stream_fit.solve_window(job)
        win.ev = {"lrswap_ns": t1 - t0, "prep_ns": time.perf_counter_ns() - t1}
        self.inflight.append(win)

    def poll(self, block=False):
        now = self.clock()
        for win in self.inflight:
            for pid, job in win.jobs.items():
                if pid in win.status:
                    continue
                keep = self.late_policy == "use" or (self.late_policy == "behind" and win.behind)
                wait = block and (self.budget_ns is None or keep)
                res = job if isinstance(job, dict) else (job.result() if (job.done() or wait) else None)
                if res is not None:
                    late = self.budget_ns is not None and res["t_done_ns"] - win.t_ready_ns > self.budget_ns
                    win.status[pid] = ("late_used" if keep else "late") if late else "fitted"
                    win.result[pid] = res
                elif self.budget_ns is not None and not keep and now - win.t_ready_ns > self.budget_ns:
                    win.status[pid] = "late"
        while self.inflight and len(self.inflight[0].status) == len(self.pids):
            self._resolve(self.inflight.pop(0))
        self._launch_ready()
        if self.inflight and len(self.inflight[0].status) == len(self.pids):
            return self.poll(block)
        out = []
        while self.next_pub in self.ready:
            out.append(self.ready.pop(self.next_pub))
            self.next_pub += 1
        return out

    def _unsupported(self, pid, a, b, upto):
        """Steps in [a, b) with no body seen whose run of unseen steps (known so far, < upto) is longer than the
        offline fit's MAX_GAP, or reaches back to step 0: the offline fit leaves these NaN (support_segments)."""
        seen = self.seen[pid][:upto]
        out = np.zeros(b - a, bool)
        for i in range(a, b):
            if seen[i]:
                continue
            lo = i
            while lo > 0 and not seen[lo - 1]:
                lo -= 1
            hi = i
            while hi + 1 < upto and not seen[hi + 1]:
                hi += 1
            out[i - a] = lo == 0 or hi - lo + 1 > self.max_gap
        return out

    def _resolve(self, win):
        t0 = time.perf_counter_ns()
        a, b = win.start, win.commit
        frames = [self._frame(i) for i in range(a, b)]
        for pid in self.pids:
            if win.status[pid] in ("fitted", "late_used"):
                lo = win.lo[pid]
                self.Xfit[pid][lo:win.stop] = win.result[pid]["X"]
                uv, w = win.inputs[pid]
                Xm = self.Xfit[pid][a:b].copy()
                Xm[self._unsupported(pid, a, b, win.stop)] = np.nan
                r = outputs.regress_frames(self.cams, self.views, Xm, uv[a - lo:b - lo], w[a - lo:b - lo], self.kind)
                q = outputs.FITTED
            else:
                uv, w = stream_fit.window_obs(self.obs[pid], self.views, a, b)
                r = outputs.regress_frames(self.cams, self.views, self.Xfb[pid][a:b], uv, w, self.kind)
                q = outputs.LATE
            for t, fr in enumerate(frames):
                fr["players"][pid] = outputs.player_frame(r, t, q)
        for fr in frames:
            fr["quality"] = outputs.frame_quality(fr["players"])
            self.ready[fr["step"]] = fr
        self.events(dict(win.ev, node="pose3d", ev="window", k=win.k, start=win.start, stop=win.stop,
                         commit_stop=win.commit, lo=dict(win.lo), t_ready_ns=win.t_ready_ns,
                         solve_ns={p: r["solve_ns"] for p, r in win.result.items()},
                         nfev={p: r.get("nfev") for p, r in win.result.items()},
                         t_done_ns={p: r["t_done_ns"] for p, r in win.result.items()}, status=dict(win.status),
                         regress_ns=time.perf_counter_ns() - t0, t_resolved_ns=self.clock()))

    # ------------------------------------------------------------------ v0.8.4 warm state / early stop
    def export_warm_state(self) -> dict:
        """What a later stream can reuse instead of its own warm-up: each player's body model and the ankle plane."""
        return {"models": {pid: (model_to_dict(m) if m is not None else None) for pid, m in self.model.items()},
                "ankle_h_m": float(self.tracker.ankle_h), "views": list(self.views), "fps": self.fps}

    def stop_at(self, m):
        """End the stream after step m - 1: steps >= m are never published (a stopped live stream)."""
        m = max(int(m), self.next_pub)
        if m >= self.n:
            return
        self.n = m
        self.stepper.n = m
        if self.pending is not None:
            lo, start, stop, commit = self.pending
            self.pending = None if start >= m else (lo, start, min(stop, m), min(commit, m) if stop < m else m)

    # ------------------------------------------------------------------ end of stream
    def flush(self, sleep_s=0.002):
        out = []
        while self.inflight or (self.pending is not None and self.last_step >= self.pending[2] - 1
                                and self.warmed):
            out += self.poll(block=True)
            if self.inflight:
                time.sleep(sleep_s)
        for i in range(self.next_pub, self.n):
            if i in self.ready:
                continue
            if i in self.t_recv:
                self.ready[i] = self._fallback_frames(i, i + 1, outputs.LATE)[0]
            else:
                fr = self._frame(i)
                fr["players"] = {p: {"J": np.full((len(self._names()), 3), np.nan), "C": np.zeros(len(self._names())),
                                     "aux": np.full((14, 3), np.nan), "auxc": np.zeros(14), "interp": False,
                                     "valid": False, "quality": outputs.NONE, "forward_sign": 1.0} for p in self.pids}
                fr["quality"] = outputs.NONE
                self.ready[i] = fr
        out += self.poll()
        return out

    def _names(self):
        from analysis import skeleton24 as sk
        return sk.TYPES[self.kind][0]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
