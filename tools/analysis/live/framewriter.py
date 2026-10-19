"""Per-frame output of a live run (one JSON file per output step, readable while the run is going).

<run>/unity_live_frames/manifest.json   written once: the analysis doc header (schema, clip, court, skeleton,
                           cameras, players) + live info (source / output fps, views, offsets, started); frames: []
<run>/unity_live_frames/frame_<step:06d>.json
    {"v": 1, "step", "ref_frame", "frame_id", "time", "quality", "t_capture_ns", "t_written_ns", "latency_ms",
     "players": [{"id", "side", "valid", "quality", "quality_name", "warmup", "late", "interpolated",
                  "forward_sign", "frame": <exactly one analysis-doc player frame, incl. causal facing>}]}
<run>/unity_live_frames/latest.json     {"last_step", "n_written", "frame", "t_written_ns"} rewritten after every frame
(v0.8.1 and older: <run>/live/manifest.json, <run>/live/frames/<step:06d>.json, <run>/live/latest.json; readers take both
through frames_dir() / frame_files().)

Every file is written to <name>.tmp then os.replace()d (retried while Windows reports the target busy), so a
reader never sees half a file; readers skip *.tmp. NaN never reaches the files (Unity's JsonUtility rejects it).
Facing uses past frames only (circular mean of the last `facing_window` headings); the handedness sign is the
median of the player's warm-up frames, fixed once warm-up ends."""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time

import numpy as np

from analysis import facing as facingmod
from analysis.live import outputs
from analysis.pose24 import build
from analysis.pose24 import doc as docmod

QNAMES = {outputs.FITTED: "fitted", outputs.WARMUP: "warmup", outputs.LATE: "late", outputs.NONE: "none"}
HEADING_KEYS = ("heading", "shoulder_heading", "head_heading")


def atomic_write(path, obj, retries=50, wait_s=0.01):
    text = json.dumps(obj, allow_nan=False, separators=(",", ":"))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    for k in range(retries):
        try:
            os.replace(tmp, path)
            return len(text)
        except PermissionError:          # Windows: a reader has the target open
            if k == retries - 1:
                raise
            time.sleep(wait_s)
    return len(text)


def refuse_inside_assets(path):
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if "Assets" in parts:
        raise ValueError(f"run folder {path} is inside a Unity Assets folder (Unity would import every frame file)")


class CausalFacing:
    def __init__(self, window, face, conf_min):
        self.window, self.face, self.conf_min = int(window), face, conf_min
        self.hist = {k: [] for k in HEADING_KEYS}

    def push(self, joints_flat, sign):
        raw = facingmod.compute_facing(joints_flat, sign=sign, face=self.face, conf_min=self.conf_min)
        if raw is None:
            return None
        out = dict(raw)
        for k in HEADING_KEYS:
            h = self.hist[k]
            h.append(raw.get(k))
            del h[:-self.window]
            vals = [x for x in h if x is not None]
            if k in raw and vals:
                out[k] = round(math.atan2(sum(math.sin(x) for x in vals), sum(math.cos(x) for x in vals)), 4)
        return out


class SignLock:
    """Handedness per player: running median over warm-up frames, then fixed."""

    def __init__(self):
        self.vals, self.fixed = [], None

    def push(self, sign, warmup):
        if self.fixed is not None:
            return self.fixed
        if sign is not None:
            self.vals.append(float(sign))
        cur = -1.0 if self.vals and float(np.median(self.vals)) < 0 else 1.0
        if not warmup and self.vals:
            self.fixed = cur
        return cur



FRAMES_DIR = "unity_live_frames"          # the folder Unity follows: manifest.json, latest.json, frame_000123.json
FRAME_NAME = "frame_{step:06d}.json"
OLD_DIR, OLD_FRAMES = "live", "frames"    # v0.8.0 / v0.8.1 layout: live/manifest.json, live/frames/000123.json


def frames_dir(run_dir):
    """The run's frame folder (new layout first, then the v0.8.0 / v0.8.1 one); None when the run has none."""
    for d in (os.path.join(run_dir, FRAMES_DIR), os.path.join(run_dir, OLD_DIR)):
        if os.path.exists(os.path.join(d, "manifest.json")):
            return d
    return None


def frame_files(run_dir):
    """Every frame file of a run, in step order (either layout); [] when there is none."""
    d = frames_dir(run_dir)
    if d is None:
        return []
    if os.path.basename(d) == OLD_DIR:
        f = os.path.join(d, OLD_FRAMES)
        return [os.path.join(f, n) for n in sorted(os.listdir(f)) if n.endswith(".json")] if os.path.isdir(f) else []
    return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.startswith("frame_") and n.endswith(".json")]


class FrameWriter:
    def __init__(self, run_dir, cfg, header_doc, sides, write_frames=True, background=False, on_written=None):
        refuse_inside_assets(run_dir)
        self.cfg, self.sides = cfg, dict(sides)
        self.dir = os.path.join(run_dir, FRAMES_DIR)
        self.fdir = self.dir
        os.makedirs(self.fdir, exist_ok=True)
        self.write_frames = bool(write_frames)
        self.sfps = float(cfg["source_fps"])
        self.kind = cfg["skeleton"]
        face = build.FACE_IDX_28 if self.kind == "smpl-28" else None
        self.facing = {p: CausalFacing(cfg["facing_window"], face, build.FACING_CONF_MIN) for p in self.sides}
        self.signs = {p: SignLock() for p in self.sides}
        self.n_written, self.bytes, self.write_ns = 0, 0, []
        man = dict(header_doc)
        man["players"] = [{"id": p, "side": self.sides[p], "frames": []} for p in sorted(self.sides)]
        man["live"] = {"format": "live-frames/1", "source_fps": self.sfps, "output_fps": float(cfg["fps"]),
                       "stride": int(cfg["stride"]), "views": list(cfg["views"]), "time_ref": cfg["time_ref"],
                       "offsets": dict(cfg["offsets"]), "offsets_source": cfg.get("offsets_source"),
                       "ref_start": int(cfg["ref_frames"][0]), "steps": len(cfg["ref_frames"]),
                       "warmup_steps": int(cfg["warmup_steps"]), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                              time.gmtime()),
                       "frame_name": FRAME_NAME}
        atomic_write(os.path.join(self.dir, "manifest.json"), docmod.json_safe(man))
        self.on_written = on_written
        self.q, self.thread, self.error = None, None, None
        if background:
            self.q = queue.Queue()
            self.thread = threading.Thread(target=self._loop, name="frame-writer", daemon=True)
            self.thread.start()

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as e:  # noqa: BLE001 - surfaced by close()
                self.error = e

    def submit(self, o, t_capture_ns=None):
        """Write now (foreground) or hand the frame to the writer thread (background)."""
        if self.q is None:
            return self._write(o, t_capture_ns)
        self.q.put((o, t_capture_ns))
        return None

    def close(self):
        if self.thread is not None:
            self.q.put(None)
            self.thread.join()
            self.thread = None
        if self.error is not None:
            raise self.error

    def frame_obj(self, o, t_capture_ns=None):
        r = int(o["ref_frame"])
        players = []
        for pid in sorted(o["players"]):
            p = o["players"][pid]
            warm = p["quality"] == outputs.WARMUP
            sign = self.signs.setdefault(pid, SignLock()).push(p.get("forward_sign") if p["valid"] else None, warm)
            fr = docmod.frames_from_world24(np.asarray(p["J"])[None], np.asarray(p["C"])[None],
                                            np.asarray(p["aux"])[None], np.asarray(p["auxc"])[None],
                                            np.array([bool(p["interp"])]), self.sfps)[0]
            fr["frame_id"], fr["time"] = r, round(r / self.sfps, 4)
            fac = self.facing.get(pid)
            fr["facing"] = fac.push(fr.get("joints_flat"), sign) if fac and fr.get("joints_flat") else None
            players.append({"id": pid, "side": self.sides.get(pid), "valid": bool(p["valid"]),
                            "quality": int(p["quality"]), "quality_name": QNAMES.get(int(p["quality"]), "?"),
                            "warmup": warm, "late": p["quality"] == outputs.LATE, "interpolated": bool(p["interp"]),
                            "forward_sign": sign, "reproj_px": p.get("reproj_px", {}), "frame": fr})
        return {"v": 1, "step": int(o["step"]), "ref_frame": r, "frame_id": r, "time": round(r / self.sfps, 4),
                "quality": int(o["quality"]), "t_capture_ns": t_capture_ns, "players": players}

    def write(self, o, t_capture_ns=None):
        return self._write(o, t_capture_ns)

    def _write(self, o, t_capture_ns=None):
        t0 = time.perf_counter_ns()
        obj = self.frame_obj(o, t_capture_ns)
        t_w = time.monotonic_ns()
        obj["t_written_ns"] = t_w
        obj["latency_ms"] = None if t_capture_ns is None else round((t_w - t_capture_ns) / 1e6, 2)
        obj = docmod.json_safe(obj)
        name = FRAME_NAME.format(step=obj["step"])
        if self.write_frames:
            self.bytes += atomic_write(os.path.join(self.dir, name), obj)
            self.n_written += 1
            atomic_write(os.path.join(self.dir, "latest.json"),
                         {"last_step": obj["step"], "n_written": self.n_written, "frame": name, "t_written_ns": t_w})
        dt = time.perf_counter_ns() - t0
        self.write_ns.append(dt)
        if self.on_written is not None:
            self.on_written(obj, dt)
        return obj, dt

    def stats(self) -> dict:
        w = np.asarray(self.write_ns, float) / 1e6
        return {"frames": self.n_written, "bytes": self.bytes,
                "bytes_per_frame": self.bytes / self.n_written if self.n_written else None,
                "write_ms": ({"median": float(np.median(w)), "p95": float(np.percentile(w, 95)), "max": float(w.max())}
                             if len(w) else None)}
