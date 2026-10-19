"""One simulated-live run, frame by frame (no ROS):

    schedule (output steps at cfg fps) -> per step: wait until its frames exist (realtime) -> 2D from the source
    -> StreamPose24.push -> every output frame written at once (FrameWriter) -> at the end: flush, final doc, summary.

Clock: t0 = the section start's capture time. Step i's capture time = t0 + (ref_i - ref_0) / source_fps. In
`realtime` pace a step is pushed no earlier than the capture time of its last source frame (as live cameras would
deliver it) and the 3D core is polled while waiting, so fits finish and frames are written in the background; in
`asfast` pace nothing waits and the capture time is the moment the step's 2D was ready (latency = processing only).

Sync `auto`: the warm-up runs with zero offsets; right after the last warm-up step the per-view offsets are
estimated (pose_v2.estimate_offsets on the warm-up observations, at step rate) and the rest of the schedule is rebuilt.

Events (timing.jsonl): node "runner": "frames" (per step: capture / availability / 2D timings per view), "out"
(per written frame), "sync", "phase"; node "pose3d": the core's "step" / "window" / "warmup"."""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from analysis import pose_v2 as pv
from analysis.live import config as live_config
from analysis.live import docwriter, framewriter, machine, samplers, sources, sync
from analysis.live.stream_pose24 import StreamPose24
from analysis.live.timing import TimingLog
from analysis.pose24 import build

LEAD_S = 0.2           # t0 is this far after the source is ready


def plan_for(cfg, schedule):
    """Source frames each view may need. Known offsets: exactly the schedule's frames. Auto sync: every frame in the
    section +- the offset search range (the offsets are only known after warm-up)."""
    if cfg["offsets_source"] != "auto":
        return sources.plan_frames(schedule)
    m = int(np.ceil(float(cfg["sync"]["estimate"]["max_offset_s"]) * cfg["source_fps"])) + 1
    lo, hi = cfg["ref_frames"][0] - m, cfg["ref_frames"][-1] + m + 1
    n = {v: int(cfg["source"][v]["frame_count"]) for v in cfg["views"]}
    return {v: list(range(max(0, lo), min(n[v], hi + 1))) for v in cfg["views"]}


def estimate_offsets(core, cfg, upto, log=print) -> dict:
    """Offsets (source frames) from the core's step-rate observations of steps [0, upto)."""
    est = cfg["sync"]["estimate"]
    stride = int(cfg["stride"])
    max_steps = max(1, int(round(float(est["max_offset_s"]) * cfg["fps"])))
    dense = {pid: {v: core.obs[pid][v][:upto].astype(float) for v in core.views} for pid in core.pids}
    offs, errs = pv.estimate_offsets(core.cams, dense, core.views, cfg["time_ref"], max_offset=max_steps,
                                     step=int(est["step_frames"]), log=lambda *a: None)
    res = {"offsets": {v: float(offs.get(v, 0.0)) * stride for v in core.views},
           "errors_px": {v: (None if not np.isfinite(errs.get(v, np.inf)) else float(errs[v])) for v in core.views},
           "max_offset_steps": max_steps, "steps_used": int(upto)}
    failed = [v for v in core.views if res["errors_px"][v] is None]
    res["failed_views"] = failed
    for v in failed:
        res["offsets"][v] = 0.0
    log(f"  sync (auto, {upto} warm-up steps): " + ", ".join(
        f"{v} {res['offsets'][v]:+.2f} fr ({res['errors_px'][v]:.1f} px)" if res["errors_px"][v] is not None
        else f"{v} failed -> 0" for v in core.views))
    return res


class Runner:
    def __init__(self, cfg, run_dir, log=print, source=None, cams=None, clock=time.monotonic_ns, sleep=time.sleep):
        self.cfg, self.run_dir, self.log, self.cams = cfg, run_dir, log, cams
        self.source, self.clock, self.sleep = source, clock, sleep
        framewriter.refuse_inside_assets(run_dir)

    # ------------------------------------------------------------------ helpers
    def _written(self, obj, dt):
        self.tl.write({"ev": "out", "step": obj["step"], "quality": obj["quality"], "t_capture_ns": obj["t_capture_ns"],
                       "t_written_ns": obj["t_written_ns"], "write_ns": dt})

    def _emit(self, outs, cap):
        for o in outs:
            self.writer.submit(o, cap.get(o["step"]))
            self.outs.append(o)

    def _wait_until(self, t_ns, cap):
        while True:
            now = self.clock()
            if now >= t_ns:
                return
            self._emit(self.core.poll(), cap)
            self.sleep(min(0.002, (t_ns - now) / 1e9))

    def _collect(self, st, cap):
        """2D of every view's source frames for one step, polling the core while the source is busy."""
        views, per_view, t_av, t_done = {}, {}, [], []
        for v, vv in st["views"].items():
            frames, pv_t = [], []
            for f in vv["frames"]:
                got = None
                while got is None:
                    got = self.src.try_get(v, f)
                    if got is None:
                        self._emit(self.core.poll(), cap)
                dets, tim = got
                frames.append({"src_frame": int(f), "dets": dets})
                pv_t.append(tim)
                if tim.get("t_avail_ns") is not None:
                    t_av.append(tim["t_avail_ns"])
                t_done.append(tim["t_done_ns"])
            views[v] = {"alpha": vv["alpha"], "frames": frames}
            per_view[v] = {"decode_ms": sum(t["decode_ns"] for t in pv_t) / 1e6,
                           "infer_ms": sum(t["infer_ns"] for t in pv_t) / 1e6,
                           "wait_ms": sum(t["wait_ns"] for t in pv_t) / 1e6,
                           "dropped": sum(bool(t["dropped"]) for t in pv_t), "n_frames": len(pv_t),
                           "n_dets": sum(len(fr["dets"]) for fr in frames),
                           "first_call": any(t.get("first_call") for t in pv_t)}
        return views, per_view, (max(t_av) if t_av else None), max(t_done)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        c, log = self.cfg, self.log
        os.makedirs(self.run_dir, exist_ok=True)
        live_config.dump(c, os.path.join(self.run_dir, "config.resolved.yaml"))
        mach = machine.report()
        with open(os.path.join(self.run_dir, "machine.json"), "w", encoding="utf-8") as fh:
            json.dump(mach, fh, indent=1)
        log(f"  machine: {machine.short(mach)}")
        self.tl = TimingLog(os.path.join(self.run_dir, "timing.jsonl"), "runner")
        t_setup = time.perf_counter()
        cams = self.cams or pv.load_cams(c["cam_paths"])
        schedule = sync.schedule(c)
        n = len(schedule)
        self.src = self.source or sources.make(c, log=log)
        budget = c["fit_budget_s"] if c["pace"] == "realtime" else None
        self.core = StreamPose24(c, cams, budget_s=budget, workers=c["workers"], events=self.tl.write,
                                 clock=self.clock)
        sides = self.core.tracker.sides
        image_size = {v: [c["source"][v]["width"], c["source"][v]["height"]] for v in c["views"]}
        self.writer = framewriter.FrameWriter(self.run_dir, c, docwriter.header(c, cams, sides, image_size), sides,
                                              write_frames=c["output"]["frames"],
                                              background=bool(c["output"].get("background_writer", True)),
                                              on_written=self._written)
        self.outs, cap = [], {}
        plan = plan_for(c, schedule)
        ref0 = c["ref_frames"][0]
        origins = {v: ref0 + float(c["offsets"][v]) for v in c["views"]}
        t_src = time.perf_counter()
        self.src.start(plan, origins)
        src_ready_s = time.perf_counter() - t_src
        setup_s = time.perf_counter() - t_setup
        sync_info = None
        gpu_csv, cpu_csv = os.path.join(self.run_dir, "gpu_samples.csv"), os.path.join(self.run_dir, "cpu_samples.csv")
        realtime = c["pace"] == "realtime"
        sfps = float(c["source_fps"])
        with samplers.GpuSampler(gpu_csv, c["samplers"]["gpu_ms"]) as gs, \
                samplers.CpuSampler(cpu_csv, c["samplers"]["cpu_ms"]) as cs:
            t0 = self.clock() + (int(LEAD_S * 1e9) if c["pace"] == "realtime" else 0)
            self.src.begin(t0)
            self.tl.write({"ev": "phase", "phase": "start", "t0_ns": t0, "setup_s": setup_s,
                           "source_ready_s": src_ready_s})
            log(f"  {n} steps at {c['fps']:g} fps ({c['pace']}, backend {c['backend']}), "
                f"setup {setup_s:.1f} s (source ready {src_ready_s:.1f} s)")
            t_loop = time.perf_counter()
            i = 0
            while i < n:
                st = schedule[i]
                t_capture = t0 + int((st["ref_frame"] - ref0) / sfps * 1e9)
                if realtime:
                    last = max(max(vv["frames"]) - origins[v] for v, vv in st["views"].items())
                    self._wait_until(t0 + int(max(0.0, last) / sfps * 1e9), cap)
                views, per_view, t_av, t_2d = self._collect(st, cap)
                if not realtime:
                    t_capture = t_2d
                cap[i] = t_capture
                t_push = self.clock()
                self.tl.write({"ev": "frames", "step": i, "ref_frame": st["ref_frame"], "t_capture_ns": t_capture,
                               "t_avail_ns": t_av, "t_2d_done_ns": t_2d, "t_push_ns": t_push, "views": per_view})
                self._emit(self.core.push({"step": i, "ref_frame": st["ref_frame"], "t_recv_ns": t_push,
                                           "views": views}), cap)
                if c["offsets_source"] == "auto" and sync_info is None and i == c["warmup_steps"] - 1:
                    sync_info = estimate_offsets(self.core, c, i + 1, log)
                    c["offsets"] = sync_info["offsets"]
                    c["offsets_estimated"] = sync_info
                    schedule = schedule[:i + 1] + sync.schedule(c)[i + 1:]
                    self.tl.write(dict(sync_info, ev="sync"))
                    if c["sync"].get("narrow_after_warmup", True) and hasattr(self.src, "narrow"):
                        keep = sources.plan_frames(schedule[i + 1:])
                        self.src.narrow(keep)
                        n_before = sum(sum(1 for f in plan[v] if f >= min(keep[v], default=10 ** 9)) for v in keep)
                        self.tl.write({"ev": "narrow", "frames_after": {v: len(fr) for v, fr in keep.items()}})
                        log(f"  2D plan narrowed after the sync: {sum(len(f) for f in keep.values())} frames still "
                            f"needed (was {n_before} from here on)")
                i += 1
                if i % max(1, int(c["fps"])) == 0:
                    log(f"    step {i}/{n}  written {len(self.outs)}  "
                        f"({(time.perf_counter() - t_loop):.1f} s)")
            self._emit(self.core.flush(), cap)
            self.writer.close()
            loop_s = time.perf_counter() - t_loop
            self.core.close()
            self.src.close()
            self.tl.write({"ev": "phase", "phase": "end", "loop_s": loop_s})
        res = {"steps": n, "written": len(self.outs), "loop_s": loop_s, "setup_s": setup_s,
               "source_ready_s": src_ready_s, "writer": self.writer.stats(), "sync": sync_info,
               "offsets": dict(c["offsets"]), "gpu_sampler_error": gs.error, "cpu_sampler_error": cs.error,
               "source_info": getattr(self.src, "info", {})}
        if c["output"]["final_doc"]:
            t = time.perf_counter()
            path = docwriter.write(c, self.outs, cams, sides, self.run_dir, image_size=image_size)
            res["final_doc"] = os.path.basename(path)
            res["final_doc_s"] = time.perf_counter() - t
            log(f"  wrote {path}")
            if c["output"]["copy_to_analysis_dir"]:
                dst = build.output_path(c["paths"]["analysis_dir"], c["out_clip"])
                shutil.copyfile(path, dst)
                res["copied_to"] = dst
        self.tl.close()
        with open(os.path.join(self.run_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        return res
