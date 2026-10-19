# tools/analysis/live/stream_fit.py
"""Streaming fit pieces (spec § 6.4). The window problem is pose24.fit.WindowProblem24 unchanged; WindowStepper
reproduces fit._fit_segment's window schedule (window W, overlap O, 2 fixed frames before each later window's
start; the window reaching the stream end commits everything). The body model (lengths, rest pose, handedness) is
estimated once on the warm-up, as fit_player's pass 1 does on the whole clip, then locked."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import fit, kpset, lrswap

DEFAULT_MAX_NFEV = 40          # WindowProblem24.solve's default (the offline fit)


def window_obs(obs_pid, views, a, b, swaps=None):
    D = np.stack([np.asarray(obs_pid[v][a:b], np.float32).copy() for v in views], axis=2)   # (T,133,V,3)
    if swaps is not None:
        for k in range(len(views)):
            m = np.asarray(swaps[k], bool)
            if m.any():
                D[m, :, k] = D[m][:, lrswap.PERM, k]
    return kpset.obs35(D[..., :2].astype(np.float64), np.nan_to_num(D[..., 2]).astype(np.float64))


def triangulate35(cams, views, uv, w):
    T, K, V, _ = uv.shape
    X, conf, _, inl = pv.triangulate_batch({v: cams[v] for v in views}, list(views), uv.reshape(-1, V, 2),
                                           w.reshape(-1, V))
    return X.reshape(T, K, 3), conf.reshape(T, K), inl.reshape(T, K, V)


@dataclass
class BodyModel:
    lengths: dict
    edges: tuple
    rest: np.ndarray
    sgn: float


def warmup_model(cams, views, uv, w, scfg):
    """Body model from the warm-up observations; scfg = the run config's `stream` section (min_hip_frames,
    good_conf / good_inliers for multi-view (>= 3) and two-view rigs)."""
    Xr, conf, inl = triangulate35(cams, views, uv, w)
    ok = np.isfinite(Xr).all(-1)
    if (ok[:, 11] | ok[:, 12]).sum() < int(scfg["min_hip_frames"]):
        return None
    sgn = kpset.forward_sign(Xr, default=kpset.world_sign({v: cams[v] for v in views}))
    V = len(views)
    rig = "multi" if V >= 3 else "two_view"
    good = (conf > float(scfg["good_conf"][rig])) & (inl.sum(-1) >= int(scfg["good_inliers"][rig]))
    lengths = fit.estimate_lengths(Xr, good)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hip = np.nanmean(np.where(ok[:, [11, 12], None], Xr[:, [11, 12]], np.nan), axis=1)
        rest = np.nanmedian(Xr - hip[:, None], axis=0)
    rest = np.where(np.isfinite(rest), rest, kpset.rest_offsets_down(sgn))
    return BodyModel(lengths, fit.edge_arrays(lengths), rest, float(sgn))


class WindowStepper:
    def __init__(self, first, n_total, window, overlap):
        self.first, self.n, self.window, self.overlap = int(first), int(n_total), int(window), int(overlap)
        self.start = self.first

    def next_window(self):
        if self.start >= self.n:
            return None
        stop = min(self.start + self.window, self.n)
        lo = max(self.first, self.start - 2)
        commit = stop if stop == self.n else stop - self.overlap
        out = (lo, self.start, stop, commit)
        self.start = self.n if stop == self.n else stop - self.overlap
        return out


def solve_window(job) -> dict:
    t0 = time.perf_counter_ns()
    prob = fit.WindowProblem24(job["P"], job["uv"], job["w"], job["init"], job["free"], job["edges"], job["anchor"],
                               job["contact"], job["sgn"], job["fps"], use_limits=job.get("use_limits", True))
    info = {}
    X = prob.solve(max_nfev=int(job.get("max_nfev") or DEFAULT_MAX_NFEV), tol=job.get("tol"),
                   lsmr_maxiter=job.get("lsmr_maxiter"), info=info)
    return {"X": X, "solve_ns": time.perf_counter_ns() - t0, "t_done_ns": time.monotonic_ns(), "nfev": info.get("nfev"),
            "status": info.get("status")}


def ping() -> int:
    return 1
