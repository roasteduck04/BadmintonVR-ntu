"""One pipeline job at a time, run as a subprocess of an existing CLI (stoppable, log tailed, progress parsed).

The server never imports the heavy pipeline into its own process: a stream / 2D / build job is `python tools/...`
started here, its stdout read line by line on a thread. Stop kills the whole process tree (the live run starts 2D
worker processes), on Windows with taskkill /T, elsewhere through the job's own process group.
"""
from __future__ import annotations

import collections
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
ERR_RE = re.compile(r"^(\w+\.)*\w*(Error|Exception).*:|^SystemExit|^error:")


class Job:
    def __init__(self, kind, argv, cwd, run_dir=None, title="", on_line=None, env=None):
        self.kind, self.argv, self.cwd, self.run_dir, self.title = kind, list(argv), cwd, run_dir, title
        self.on_line = on_line
        self.env = env
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

    def start(self):
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        env = dict(os.environ, PYTHONUNBUFFERED="1", **(self.env or {}))
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                     bufsize=1, env=env, **kw)
        self.state = "running"
        self._thread = threading.Thread(target=self._read, name=f"job-{self.kind}", daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
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
        return {"kind": self.kind, "title": self.title, "state": self.state, "started": self.started,
                "ended": self.ended, "elapsed_s": round((self.ended or time.time()) - self.started, 1),
                "returncode": self.returncode, "progress": dict(self.progress), "error": self.error,
                "run_dir": self.run_dir, "log": list(self.lines)[-60:], "argv": self.argv}


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
