"""Timing log of a live run (timing.jsonl) and its per-step derivation. All times are time.monotonic_ns().

Per step (runner "frames" + core "step" + its window + runner "out"):
  capture   when the step's reference frame was captured (realtime: on the simulated camera clock; asfast: when its
            2D was ready, so latency = processing only)
  wait_2d   last source frame available -> its 2D ready (ms)       infer / decode / wait: 2D worker times (ms, max view)
  step3d    score calibration + tracker + triangulation (ms)        calib / track / tri separately
  fit_share the step's window cost (lrswap + prep + max player solve + regress) / frames it commits (ms)
  hold      pushed into the core -> written (ms): the fixed-lag wait for the window + its fit
  e2e       capture -> written (ms)"""
from __future__ import annotations

import glob
import json
import os
import threading

import numpy as np

MS = 1e6
STAGES = [("decode", "2D: decode (max view)"), ("infer", "2D: model (max view)"), ("wait_2d", "frame captured → collected (2D + backlog)"),
          ("calib", "3D: score calibration"), ("track", "3D: tracker"), ("tri", "3D: triangulate + EMA"),
          ("fit_share", "3D: window fit ÷ frames"), ("hold", "pushed → written (fixed lag + fit)"),
          ("write", "write frame file"), ("e2e", "end-to-end (capture → written)")]


class TimingLog:
    def __init__(self, path, node):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.node = node
        self.fh = open(path, "a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, ev):
        ev = dict(ev)
        ev.setdefault("node", self.node)
        line = json.dumps(ev, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n"
        with self.lock:
            self.fh.write(line)

    def close(self):
        self.fh.close()


def load(run_dir) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "timing*.jsonl"))):
        with open(p, "r", encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def _d(a, b):
    return None if a is None or b is None else (a - b) / MS


def step_rows(events) -> list:
    frames, core, outs, wins = {}, {}, {}, []
    for e in events:
        ev, node = e.get("ev"), e.get("node")
        if ev == "window":
            wins.append(e)
        elif ev == "frames":
            frames[e["step"]] = e
        elif ev == "step" and node == "pose3d":
            core[e["step"]] = e
        elif ev == "out":
            outs[e["step"]] = e
    wins.sort(key=lambda w: w["start"])
    rows = []
    for s in sorted(set(frames) | set(outs)):
        f, c, o = frames.get(s), core.get(s), outs.get(s)
        w = next((x for x in wins if x["start"] <= s < x["commit_stop"]), None)
        r = {"step": s, "quality": o.get("quality") if o else None, "decode": None, "infer": None, "wait_2d": None,
             "cam_wait": None, "dropped": 0, "n_dets": None, "first_call": False, "calib": None, "track": None,
             "tri": None, "fit_share": None, "win_k": None, "win_status": None, "hold": None, "write": None,
             "e2e": None, "t_capture": None, "t_written": None}
        if f:
            vs = list((f.get("views") or {}).values())
            if vs:
                r.update(decode=max(v["decode_ms"] for v in vs), infer=max(v["infer_ms"] for v in vs),
                         cam_wait=max(v["wait_ms"] for v in vs), dropped=sum(v["dropped"] for v in vs),
                         n_dets=sum(v["n_dets"] for v in vs), first_call=any(v["first_call"] for v in vs))
            r["wait_2d"] = _d(f.get("t_2d_done_ns"), f.get("t_avail_ns"))
            r["t_capture"] = f.get("t_capture_ns")
        if c:
            r.update(calib=c["calib_ns"] / MS, track=c["track_ns"] / MS, tri=c["tri_ns"] / MS)
        if w:
            n = w["commit_stop"] - w["start"]
            solve = max(w["solve_ns"].values()) if w["solve_ns"] else 0
            r.update(fit_share=(w["lrswap_ns"] + w["prep_ns"] + solve + w["regress_ns"]) / MS / n, win_k=w["k"],
                     win_status=w["status"])
        if o:
            r["t_written"] = o["t_written_ns"]
            r["write"] = o["write_ns"] / MS
            r["e2e"] = _d(o["t_written_ns"], o.get("t_capture_ns"))
            if f:
                r["hold"] = _d(o["t_written_ns"], f.get("t_push_ns"))
        rows.append(r)
    return rows


def pct(x, qs=(50, 90, 99)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    out = {f"p{q}": float(np.percentile(x, q)) for q in qs}
    out.update(mean=float(x.mean()), max=float(x.max()), n=int(len(x)))
    return out


def windows(events) -> list:
    return sorted((e for e in events if e.get("ev") == "window"), key=lambda w: w["start"])


def summary(rows, events, cfg) -> dict:
    ph = {e["phase"]: e for e in events if e.get("ev") == "phase"}
    caps = [r["t_capture"] for r in rows if r["t_capture"] is not None]
    wr = [r["t_written"] for r in rows if r["t_written"] is not None]
    span = (max(wr) - min(caps)) / 1e9 if caps and wr else None
    section_s = len(cfg["ref_frames"]) / float(cfg["fps"])
    q = {}
    for r in rows:
        if r["quality"] is not None:
            q[r["quality"]] = q.get(r["quality"], 0) + 1
    ws = windows(events)
    st = {}
    for w in ws:
        for pid, s in w["status"].items():
            st.setdefault(pid, {}).setdefault(s, 0)
            st[pid][s] += 1
    solve = {}
    for w in ws:
        for pid, ns in w["solve_ns"].items():
            solve.setdefault(pid, []).append(ns / MS)
    first_written = min(wr) if wr else None
    t0 = ph.get("start", {}).get("t0_ns")
    out = {"steps": len(rows), "written": len(wr), "section_s": section_s,
           "wall_s": span, "realtime_factor": (section_s / span) if span else None,
           "achieved_output_fps": (len(wr) / span) if span else None, "target_fps": float(cfg["fps"]),
           "loop_s": ph.get("end", {}).get("loop_s"), "setup_s": ph.get("start", {}).get("setup_s"),
           "source_ready_s": ph.get("start", {}).get("source_ready_s"),
           "first_frame_s": ((first_written - t0) / 1e9) if first_written is not None and t0 is not None else None,
           "quality_counts": q, "windows": len(ws), "window_status": st,
           "solve_ms": {pid: pct(v) for pid, v in solve.items()},
           "stages": {k: pct([r[k] for r in rows]) for k, _ in STAGES},
           "dropped_2d_frames": int(sum(r["dropped"] for r in rows)),
           "first_2d_call_ms": next((r["infer"] for r in rows if r["first_call"]), None)}
    fitted = [r for r in rows if r["quality"] == 0]
    out["e2e_fitted"] = pct([r["e2e"] for r in fitted])
    return out
