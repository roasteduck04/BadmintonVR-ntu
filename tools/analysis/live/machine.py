"""Machine report for a live run: what ran where. Every probe is best-effort (a missing tool or package is recorded
as None / an error string, never an exception), so the report works on the laptop, in WSL and on a bare venv."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import socket
import subprocess
import sys

PACKAGES = ("numpy", "scipy", "opencv-python", "opencv-python-headless", "onnxruntime", "onnxruntime-gpu", "rtmlib",
            "psutil", "pyyaml", "matplotlib")


def _run(cmd, timeout=10):
    try:
        # stdin=DEVNULL: under the live engine stdin is the server's command pipe; Git for Windows waited on it and,
        # killed at the timeout, its child kept the output pipe open - the run hung for good (v0.8.4)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str | None:
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or None


def memory_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            m = re.search(r"MemTotal:\s+(\d+)", fh.read())
            return round(int(m.group(1)) / 2 ** 20, 1) if m else None
    return None


def physical_cores() -> int | None:
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def gpus() -> list:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if not out:
        return []
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            res.append({"name": parts[0], "memory_mb": int(float(parts[1])), "driver": parts[2]})
    head = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)
    for g in res:
        g["cuda"] = m.group(1) if m else None
    return res


def onnxruntime_info() -> dict:
    try:
        import onnxruntime as ort
        return {"version": ort.__version__, "providers": list(ort.get_available_providers())}
    except Exception as e:  # noqa: BLE001 - a broken install is itself worth reporting
        return {"version": None, "error": f"{type(e).__name__}: {e}"}


def packages() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            continue
    return out


def git_state(cwd=".") -> dict:
    head = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    if not head:
        return {"commit": None}
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    status = _run(["git", "-C", cwd, "status", "--porcelain", "--untracked-files=no"]) or ""
    changed = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {"commit": head.strip(), "branch": branch.strip(), "dirty": bool(changed), "changed_files": changed[:50]}


def report(cwd=".") -> dict:
    rel = platform.release()
    return {"host": socket.gethostname(), "os": platform.platform(), "wsl": "microsoft" in rel.lower(),
            "python": sys.version.split()[0], "python_exe": sys.executable, "cpu": cpu_model(),
            "cpu_logical": os.cpu_count(), "cpu_physical": physical_cores(), "ram_gb": memory_gb(), "gpus": gpus(),
            "onnxruntime": onnxruntime_info(), "packages": packages(), "git": git_state(cwd)}


def short(m) -> str:
    """One line for logs: host, CPU, RAM, GPU."""
    g = m["gpus"][0]["name"] if m.get("gpus") else "no NVIDIA GPU"
    return f"{m['host']}: {m.get('cpu')} ({m.get('cpu_logical')} threads), {m.get('ram_gb')} GB RAM, {g}"
