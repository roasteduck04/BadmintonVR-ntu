"""Background resource samplers for a live run.

GpuSampler: `nvidia-smi --query-gpu ... -lms N` streamed into gpu_samples.csv (works in WSL, where the per-process
table is always empty). CpuSampler: a thread recording system CPU % and this process tree's RSS into cpu_samples.csv
(needs psutil; without it the file says so). Both are context managers and never fail the run."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time

GPU_FIELDS = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw"


class GpuSampler:
    def __init__(self, path, interval_ms=250):
        self.path, self.interval_ms = path, int(interval_ms)
        self.proc, self.fh, self.error = None, None, None

    def __enter__(self):
        exe = shutil.which("nvidia-smi")
        if not exe:
            self.error = "nvidia-smi not found"
            return self
        self.fh = open(self.path, "w", encoding="utf-8", newline="")
        self.fh.write(GPU_FIELDS.replace(".", "_") + "\n")
        self.fh.flush()
        try:
            self.proc = subprocess.Popen([exe, f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits",
                                          f"-lms={self.interval_ms}"], stdout=self.fh, stderr=subprocess.DEVNULL)
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.fh is not None:
            self.fh.close()
        return False


class CpuSampler:
    def __init__(self, path, interval_ms=500):
        self.path, self.interval = path, int(interval_ms) / 1000.0
        self.stop_ev, self.thread, self.error = threading.Event(), None, None

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            self.error = "psutil not installed"
            return self
        self.thread = threading.Thread(target=self._loop, args=(psutil,), daemon=True)
        self.thread.start()
        return self

    def _loop(self, psutil):
        me = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "cpu_percent", "rss_mb", "children"])
            t0 = time.monotonic()
            while not self.stop_ev.wait(self.interval):
                try:
                    kids = me.children(recursive=True)
                    rss = me.memory_info().rss + sum(k.memory_info().rss for k in kids if k.is_running())
                except psutil.Error:
                    kids, rss = [], 0
                w.writerow([round(time.monotonic() - t0, 3), psutil.cpu_percent(interval=None),
                            round(rss / 2 ** 20, 1), len(kids)])
                fh.flush()

    def __exit__(self, *exc):
        self.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return False


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(str(r.get(key, "")).strip()))
        except ValueError:
            continue
    return out


def summarise_gpu(path) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    u, m = _floats(rows, "utilization_gpu"), _floats(rows, "memory_used")
    if not u:
        return {"samples": 0}
    return {"samples": len(u), "util_mean": sum(u) / len(u), "util_max": max(u),
            "util_busy_frac": sum(x > 20 for x in u) / len(u), "mem_used_max_mb": max(m) if m else None,
            "mem_used_min_mb": min(m) if m else None}


def summarise_cpu(path) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    c, r = _floats(rows, "cpu_percent"), _floats(rows, "rss_mb")
    if not c:
        return {"samples": 0}
    return {"samples": len(c), "cpu_mean": sum(c) / len(c), "cpu_max": max(c), "rss_max_mb": max(r) if r else None}
