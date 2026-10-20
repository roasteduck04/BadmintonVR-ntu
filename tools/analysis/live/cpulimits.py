"""v0.8.3 CPU thread budget of a live run: every process gets a fixed number of math-library / OpenCV / decoder threads.

Without it each process's BLAS / OpenMP / OpenCV / PyAV picks every core: on the 48-thread lab PC the 2D step took
74 ms per frame inside a stream vs 21 ms alone (docs/reports/v0.8.3-lab-tests.md), the CPU at 100%, the GPU half idle.

    cpu:
      limits: false            # true = apply the budget below
      main_threads: auto       # the run's main process (tracker, triangulation, writer)
      pose2d_threads: auto     # each 2D process (pre/post-processing; one per camera, or one for all)
      decode_threads: auto     # PyAV / OpenCV decoder threads per camera video
      fit_threads: auto        # each body-fit worker process

auto splits os.cpu_count() so the total stays under the machine: per camera decode + 2D threads = cores / (4 x cameras)
each, per fit worker cores / (4 x workers), main cores / 8 (each clamped to 1..8). 48 threads, 3 cameras, 2 fit workers:
decode 4, 2D 4, fit 6, main 6 (= 42). Children are spawned, so the budget is set in their environment before they
start (OMP_NUM_THREADS & co., read when the libraries load) and again with threadpoolctl once they run."""
from __future__ import annotations

import contextlib
import os

ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
KEYS = ("main_threads", "pose2d_threads", "decode_threads", "fit_threads")


def _clamp(x):
    return max(1, min(8, int(x)))


def budget(cfg, n_cpu=None) -> dict | None:
    """{main, pose2d, decode, fit} threads for this run, or None when cpu.limits is off."""
    c = cfg.get("cpu") or {}
    if not c.get("limits"):
        return None
    n = int(n_cpu or os.cpu_count() or 4)
    cams = max(1, len(cfg.get("views") or []) if not (cfg.get("rtmpose") or {}).get("batch_views") else 1)
    workers = max(1, int(cfg.get("workers") or (cfg.get("stream") or {}).get("workers") or 1))
    auto = {"main_threads": _clamp(n // 8), "pose2d_threads": _clamp(n // (4 * cams)),
            "decode_threads": _clamp(n // (4 * max(1, len(cfg.get("views") or [])))),
            "fit_threads": _clamp(n // (4 * workers))}
    out = {}
    for k in KEYS:
        v = c.get(k, "auto")
        out[k.replace("_threads", "")] = auto[k] if v in (None, "auto") else max(1, int(v))
    out["cpus"] = n
    return out


def apply(n):
    """Cap this process at n threads for BLAS / OpenMP (threadpoolctl) and OpenCV. Safe to call more than once."""
    if not n:
        return
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(n))
    except Exception:  # noqa: BLE001 - threadpoolctl missing or a library it cannot reach: the env vars still hold
        pass
    try:
        import cv2
        cv2.setNumThreads(int(n))
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def child_env(n):
    """While starting spawned children: their thread variables = n (restored afterwards)."""
    if not n:
        yield
        return
    old = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for k in ENV_KEYS:
            os.environ[k] = str(int(n))
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def describe(b) -> str:
    if not b:
        return "CPU threads not limited"
    return (f"CPU threads per process (of {b['cpus']}): main {b['main']}, 2D {b['pose2d']}, decode {b['decode']} per "
            f"camera, fit {b['fit']} per worker")
