"""One pipeline job at a time, run as a subprocess of an existing CLI (stoppable, log tailed, progress parsed).

A stream job also reports its phases (start, models, warm-up, streaming, saving, report) from the run's log lines, and
its frame count straight from <run>/unity_live_frames/latest.json (rewritten after every frame), so the page's bar moves
with every frame instead of every 60 steps the log prints.

The server never imports the heavy pipeline into its own process: a stream / 2D / build job is `python tools/...`
started here, its stdout read line by line on a thread. Stop kills the whole process tree (the live run starts 2D
worker processes), on Windows with taskkill /T, elsewhere through the job's own process group.
"""
from __future__ import annotations

import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

STEP_RE = re.compile(r"step (\d+)/(\d+)\s+written (\d+)")
DONE_RE = re.compile(r"run done in ([\d.]+) s: (\d+)/(\d+) frames")
BUILD_STEP_RE = re.compile(r"^STEP (\d+)/(\d+) (.*)")          # tools/pipeline_server/offline_build.py
BUILD_DONE_RE = re.compile(r"^build done in (\d+) s")
ERR_RE = re.compile(r"^(\w+\.)*\w*(Error|Exception)\b.*:|^SystemExit|^error:")
RUNDIR_RE = re.compile(r"^(\S+ )?run folder ")
MODEL_RE = re.compile(r"2D worker (\S+): model ready in ([\d.]+) s")
CACHED_RE = re.compile(r"cached 2D (\S+): (\d+) frames")
SETUP_RE = re.compile(r"(\d+) steps at ([\d.]+) fps .*setup ([\d.]+) s")
SYNC_RE = re.compile(r"sync \(auto, (\d+) warm-up steps\): (.*)")
REPORT_RE = re.compile(r"^(\S+ )?report \S")

# (id, label) of a stream job's phases, in order
STREAM_PHASES = [("start", "Starting"), ("models", "Loading the models"), ("warmup", "Warm-up: picking the players"),
                 ("stream", "Streaming frames to Unity"), ("save", "Saving the analysis file"),
                 ("report", "Writing the report")]


class Job:
    def __init__(self, kind, argv, cwd, run_dir=None, title="", on_line=None, env=None, meta=None):
        """meta (stream jobs): {"views": cameras, "warmup_steps": frames of the warm-up, "backend": live | cached}."""
        self.kind, self.argv, self.cwd, self.run_dir, self.title = kind, list(argv), cwd, run_dir, title
        self.on_line = on_line
        self.env = env
        self.meta = dict(meta or {})
        self.phases = [{"id": i, "label": lab, "state": "pending", "t0": None, "t1": None, "detail": ""}
                       for i, lab in STREAM_PHASES] if kind == "stream" else []
        self._ready = []
        if self.phases:
            self._phase("start")
        self.lines = collections.deque(maxlen=400)
        self.state = "starting"          # starting | running | done | failed | stopped
        self.started = time.time()
        self.ended = None
        self.returncode = None
        self.progress = {"step": 0, "steps": None, "written": 0}
        self.error = None
        self.proc = None
        self._thread = None
        self._stop_requested = False
        self.stdin_pipe = False

    def start(self):
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **(self.env or {}))
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if self.stdin_pipe else subprocess.DEVNULL, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1, env=env, **kw)
        self.state = "running"
        self._thread = threading.Thread(target=self._read, name=f"job-{self.kind}", daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("ENGINE "):
                self._engine_line(line[7:])
                continue
            self.lines.append(line)
            m = STEP_RE.search(line)
            if m:
                self.progress = {"step": int(m.group(1)), "steps": int(m.group(2)), "written": int(m.group(3))}
            m = BUILD_STEP_RE.match(line)
            if m:
                self.progress = {"step": int(m.group(1)) - 1, "steps": int(m.group(2)), "written": int(m.group(1)) - 1,
                                 "label": m.group(3)}
            m = BUILD_DONE_RE.match(line)
            if m:
                n = self.progress.get("steps") or 1
                self.progress = {"step": n, "steps": n, "written": n, "label": "done", "seconds": float(m.group(1))}
            m = DONE_RE.search(line)
            if m:
                self.progress = {"step": int(m.group(3)), "steps": int(m.group(3)), "written": int(m.group(2)),
                                 "seconds": float(m.group(1))}
            if self.phases:
                self._phase_line(line)
            if ERR_RE.match(line):
                self.error = line            # the last exception line printed (the one that ended the run)
            if self.on_line is not None:
                try:
                    self.on_line(self, line)
                except Exception:  # noqa: BLE001 - a hook must never stop the log reader
                    pass
        self.returncode = self.proc.wait()
        self.ended = time.time()
        if self._stop_requested:
            self.state = "stopped"
        else:
            self.state = "done" if self.returncode == 0 else "failed"
            if self.state == "failed" and self.error is None:
                self.error = next((ln for ln in reversed(self.lines) if ln.strip()), f"exit code {self.returncode}")
        for ph in self.phases:
            if ph["state"] == "active":
                ph["state"] = {"done": "done", "failed": "failed"}.get(self.state, "stopped")
                ph["t1"] = self.ended

    # ------------------------------------------------------------------ stream phases
    def _phase(self, pid, detail=None):
        """Make phase `pid` the active one (every earlier phase done); no-op if it is active or already passed."""
        ids = [p["id"] for p in self.phases]
        k = ids.index(pid)
        if self.phases[k]["state"] in ("active", "done"):
            if detail is not None:
                self.phases[k]["detail"] = detail
            return
        now = time.time()
        for p in self.phases[:k]:
            if p["state"] != "done":
                p["state"] = "done"
                p["t0"] = p["t0"] or now
                p["t1"] = now
        ph = self.phases[k]
        ph.update(state="active", t0=now)
        if detail is not None:
            ph["detail"] = detail

    def _phase_line(self, line):
        n_views = self.meta.get("views")
        if RUNDIR_RE.match(line):
            what = "2D models on the GPU" if self.meta.get("backend") == "live" else "the stored 2D files"
            self._phase("models", f"{what}" + (f": 0 / {n_views} cameras" if n_views else ""))
            return
        m = MODEL_RE.search(line) or CACHED_RE.search(line)
        if m:
            self._ready.append(m.group(1))
            what = "model ready" if MODEL_RE.search(line) else "stored 2D loaded"
            self._phase("models", f"{what}: {len(self._ready)}" + (f" / {n_views}" if n_views else "") + " cameras")
            return
        m = SETUP_RE.search(line)
        if m:
            self.progress = {"step": 0, "steps": int(m.group(1)), "written": 0}
            prev = self.phases[1]["detail"]
            setup = f"setup {float(m.group(3)):.1f} s (not counted as delay)"
            self.phases[1]["detail"] = f"{prev.split(': ', 1)[-1]} · {setup}" if self._ready else setup
            wu = int(self.meta.get("warmup_steps") or 0)
            self._phase("warmup" if wu else "stream",
                        f"first {wu} frames: players picked without seeds" if wu else None)
            return
        m = SYNC_RE.search(line)
        if m:
            self._phase("warmup", f"camera offsets estimated: {m.group(2)}")
            return
        if DONE_RE.search(line):
            self._phase("report", "timing, quality, plots")
            return
        if REPORT_RE.match(line):
            for p in self.phases:
                if p["state"] != "done":
                    p.update(state="done", t0=p["t0"] or time.time(), t1=time.time())

    def _engine_line(self, payload):
        """`ENGINE {json}` status lines (v0.8.4 live engine); plain jobs never print them."""

    def live_written(self):
        """Frames written so far, from the run's latest.json (None before the first frame)."""
        if not self.run_dir:
            return None
        for sub in ("unity_live_frames", os.path.join("live")):
            p = os.path.join(self.run_dir, sub, "latest.json")
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return int(json.load(fh).get("n_written"))
            except (OSError, ValueError, TypeError):
                continue
        return None

    def _update_from_frames(self):
        """Move the warm-up -> streaming -> saving phases on with the frame count."""
        n = self.live_written()
        if n is None:
            return
        if n > self.progress.get("written", 0):
            self.progress = {**self.progress, "written": n}
        if not self.running or not self.phases:
            return
        wu, steps = int(self.meta.get("warmup_steps") or 0), self.progress.get("steps")
        if self.phases[2]["state"] == "active" and n >= wu:
            self._phase("stream", "every frame in order; falls behind instead of skipping")
        if steps and n >= steps and self.phases[3]["state"] in ("active", "done"):
            self._phase("save", "both players' whole stream in one analysis file")

    @property
    def running(self):
        return self.state in ("starting", "running")

    def stop(self, timeout_s=10.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        self._stop_requested = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if self._thread is not None:
            self._thread.join(timeout_s)

    def wait(self, timeout_s=None):
        if self._thread is not None:
            self._thread.join(timeout_s)
        return self.state

    def status(self) -> dict:
        if self.kind == "stream":
            self._update_from_frames()
        return {"kind": self.kind, "title": self.title, "state": self.state, "started": self.started,
                "ended": self.ended, "elapsed_s": round((self.ended or time.time()) - self.started, 1),
                "now": time.time(), "returncode": self.returncode, "progress": dict(self.progress),
                "error": self.error, "run_dir": self.run_dir, "log": list(self.lines)[-60:], "argv": self.argv,
                "phases": [dict(p) for p in self.phases], "meta": dict(self.meta)}


ENGINE_WARM_PHASES = [("models", "Loading the 2D models"), ("players", "Warm-up: picking the players"),
                      ("ready", "Ready to stream")]
ENGINE_STREAM_PHASES = [("stream", "Streaming frames to Unity"), ("save", "Saving the analysis file"),
                        ("report", "Writing the report")]


class EngineJob(Job):
    """v0.8.4: the live engine (`live_run.py engine`) as one long-lived job. Commands go to its stdin (send); its
    ENGINE status lines set `engine` {state, warm, run_dir, error, ...}; each warm-up / stream gets its own phases and,
    for a stream, run_dir / progress (frame count from the run's latest.json)."""

    def __init__(self, argv, cwd, clip, preset, title="live engine", commands=None):
        """commands = the file the engine follows (live_run.py engine --commands); no stdin: on Windows a thread
        blocked on a stdin pipe stalls every child process the engine starts."""
        super().__init__("engine", argv, cwd, title=title)
        self.commands = commands
        self.clip, self.preset = clip, preset
        self.engine = {"state": "starting", "clip": clip, "preset": preset, "warm": None, "run_dir": None,
                       "error": None, "streams": 0}
        self.send_lock = threading.Lock()

    def send(self, obj):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("the live engine is not running")
        with self.send_lock:
            if self.commands:
                with open(self.commands, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(obj) + "\n")
            else:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()

    def _set_phases(self, spec):
        self.phases = [{"id": i, "label": lab, "state": "pending", "t0": None, "t1": None, "detail": ""}
                       for i, lab in spec]

    def _engine_line(self, payload):
        try:
            ev = json.loads(payload)
        except ValueError:
            return
        e = ev.get("ev")
        self.lines.append(f"[engine] {e}: " + ", ".join(f"{k}={v}" for k, v in ev.items() if k not in ("ev", "trace",
                                                                                                    "info", "warm")))
        if e == "state":
            st = ev.get("state")
            self.engine["state"] = st
            if st == "loading":
                self._set_phases(ENGINE_WARM_PHASES)
                self._phase("models", "models into GPU memory, warm-up passes (TensorRT engines on the first run)")
            elif st == "warming":
                if not self.phases or self.phases[0]["id"] != "models":
                    self._set_phases(ENGINE_WARM_PHASES)
                    self.phases[0].update(state="done", detail="already loaded (reused)", t0=time.time(), t1=time.time())
                self._phase("players", "the first frames of the part: P1 / P2, body sizes, floor, offsets")
            elif st == "ready" and ev.get("warm"):
                self.engine["warm"] = ev["warm"]
                self._phase("ready", f"players {ev['warm'].get('players')} in {ev['warm'].get('seconds')} s")
                for ph in self.phases:
                    ph["state"], ph["t1"] = "done", ph["t1"] or time.time()
            elif st == "off":
                self.engine["warm"] = None
        elif e == "models":
            self.engine["models_s"] = ev.get("seconds")
            if self.phases and self.phases[0]["id"] == "models":
                self.phases[0]["detail"] = f"loaded in {ev.get('seconds')} s"
        elif e == "stream_start":
            self.run_dir = ev.get("run_dir")
            self.engine["run_dir"] = self.run_dir
            self.engine["streams"] += 1
            self.meta.update(run=os.path.basename(str(self.run_dir).rstrip("/\\")), warmup_steps=0)
            self.progress = {"step": 0, "steps": int(ev.get("steps") or 0), "written": 0}
            self._set_phases(ENGINE_STREAM_PHASES)
            self._phase("stream", "warm start: every frame in order from the first one")
            self.stream_started = time.time()
        elif e == "stream_done":
            self.progress = {**self.progress, "written": int(ev.get("written") or 0), "steps": int(ev.get("steps") or 0)}
            for ph in self.phases:
                if ph["state"] != "done":
                    ph.update(state="done", t0=ph["t0"] or time.time(), t1=time.time())
            self.engine["last_stream"] = {"run_dir": ev.get("run_dir"), "written": ev.get("written"),
                                          "stopped": ev.get("stopped")}
        elif e == "error":
            self.engine["error"] = {"message": ev.get("message"), "cmd": ev.get("cmd"), "t": time.time()}
            self.error = ev.get("message")
            for ph in self.phases:
                if ph["state"] == "active":
                    ph.update(state="failed", t1=time.time())

    def _phase_line(self, line):
        if not self.phases or self.phases[0]["id"] != "stream":
            return
        if DONE_RE.search(line):
            self._phase("report", "timing, quality, plots")

    def _update_from_frames(self):
        n = self.live_written()
        if n is None or not self.phases or self.phases[0]["id"] != "stream":
            return
        if n > self.progress.get("written", 0):
            self.progress = {**self.progress, "written": n}
        steps = self.progress.get("steps")
        if steps and n >= steps and self.phases[0]["state"] == "active":
            self._phase("save", "both players' whole stream in one analysis file")

    def status(self) -> dict:
        self._update_from_frames()
        out = super().status()
        out["engine"] = dict(self.engine)
        return out

    def stop(self, timeout_s=10.0):
        try:
            self.send({"cmd": "quit"})
            self.proc.wait(timeout_s)
        except Exception:  # noqa: BLE001 - not responding: kill it
            pass
        super().stop(timeout_s)


class JobManager:
    """One job at a time (the GPU and the CPU are shared by every pipeline step)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None
        self.history = collections.deque(maxlen=20)

    def start(self, job: Job) -> Job:
        with self.lock:
            if self.current is not None and self.current.running:
                raise RuntimeError(f"a {self.current.kind} job is running ({self.current.title}); stop it first")
            if self.current is not None:
                self.history.appendleft(self.current.status())
            self.current = job.start()
            return job

    def stop(self):
        job = self.current
        if job is not None and job.running:
            job.stop()
        return job


def python_cmd(*args) -> list:
    return [sys.executable, *args]
