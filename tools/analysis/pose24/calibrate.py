"""pose24 calibrate-scores (spec § 5.1): refit the per-group SimCC -> probability logistic from held-out views.

For every cached (frame, keypoint, view) of a pose24 build: triangulate from the other views, project into
the view; "correct" = residual < TARGET_FRAC x the person's bbox height there (bbox > 10 px, as eval). Raw scores are the
detector's own, read from the kp2d files through the cache's det_index (review 1 L5); without kp2d files they are
recovered from the cached calibrated ones with the params the build used (saturates above ~6.9). A group is kept at its old params when there are
< 500 samples, the raw scores barely vary, or every sample has the same label."""
from __future__ import annotations

import datetime
import json
import warnings

import numpy as np
from scipy.optimize import minimize

from analysis import pose_v2 as pv
from analysis.pose24 import scores, track
from analysis.pose24 import doc as docmod

TARGET_FRAC = 0.1
MIN_BBOX_PX = 10.0        # as eval.heldout_errors


def kp2d_raw_scores(cache, data_dir="data"):
    """{pid: (F,133,V) float32} the detector's own raw SimCC scores of the detections the build used (det_index of
    tracker frame f = raw kp2d frame f + shifts[view]), NaN where none; None when a view's kp2d file is missing.
    Review 1 L5: inverting the cached float32 probabilities (scores.to_raw) saturates raw scores above ~6.9."""
    import json
    import os
    raw = {}
    if not cache.get("views"):
        return None
    for v in cache["views"]:
        path = os.path.join(data_dir, "pose2d", f"{v}_kp2d.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            raw[v] = [[np.asarray(det["keypoints"], np.float32)[:, 2] for det in fr] for fr in json.load(fh)["frames"]]
    shifts = cache.get("shifts") or {}
    out = {}
    for pid in cache["obs"]:
        D = np.asarray(cache["extra"][pid]["det_index"])
        F = len(cache["obs"][pid][cache["views"][0]])
        R = np.full((F, 133, len(cache["views"])), np.nan, np.float32)
        for k, v in enumerate(cache["views"]):
            s = int(shifts.get(v, 0))
            for f in np.nonzero(D[k, :F] >= 0)[0]:
                g = f + s
                if 0 <= g < len(raw[v]) and D[k, f] < len(raw[v][g]):
                    R[f, :, k] = raw[v][g][D[k, f]]
        out[pid] = R
    return out


def heldout_samples(bundle, params, step=3, raw=None):
    """(raw scores, correct?, group) of every held-out (frame, keypoint, view). raw = kp2d_raw_scores(...) (the
    detector's scores); None: recovered from the cached calibrated scores (saturates above ~6.9, L5)."""
    views = bundle["views"]
    V = len(views)
    gid = np.empty(133, int)
    for gi, (a, b) in enumerate(scores.GROUPS.values()):
        gid[a:b] = gi
    raws, ys, groups = [], [], []
    if V < 3:
        return np.array([]), np.array([], bool), np.array([], int)
    for pid in bundle["obs"]:
        uv, w = pv.obs_arrays(bundle, pid, idx=list(range(133)))
        uv, w = uv[::step], w[::step]
        F = len(w)
        rs = raw[pid][::step] if raw is not None else None
        for k in range(V):
            rest = [i for i in range(V) if i != k]
            rv = [views[i] for i in rest]
            X, _, _, _ = pv.triangulate_batch({v: bundle["cams"][v] for v in rv}, rv,
                                              uv[:, :, rest].reshape(-1, len(rest), 2), w[:, :, rest].reshape(-1, len(rest)))
            pr, depth = pv.project(pv.proj_mats(bundle["cams"], [views[k]]), X)
            r = np.linalg.norm(pr[:, 0] - uv[:, :, k].reshape(-1, 2), axis=-1).reshape(F, 133)
            body = np.where(w[:, :17, k] >= 0.3, uv[:, :17, k, 1], np.nan)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bbox = np.nanmax(body, axis=1) - np.nanmin(body, axis=1)
            bbox = np.where(bbox > MIN_BBOX_PX, bbox, np.nan)    # 1-2 body keypoints give no body height
            rk = rs[:, :, k] if rs is not None else scores.to_raw(w[:, :, k], params)
            ok = (np.isfinite(r) & (w[:, :, k] > 0) & np.isfinite(bbox)[:, None] & (depth[:, 0].reshape(F, 133) > 0)
                  & np.isfinite(rk))
            raws.append(rk[ok])
            ys.append((r < TARGET_FRAC * bbox[:, None])[ok])
            groups.append(np.broadcast_to(gid, (F, 133))[ok])
    return np.concatenate(raws), np.concatenate(ys), np.concatenate(groups)


def fit_logistic(raw, y, start=(pv.SCORE_MID, pv.SCORE_WIDTH)):
    y = np.asarray(y, float)

    def nll(p):
        z = (raw - p[0]) / np.exp(p[1])
        return np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
    r = minimize(nll, [start[0], np.log(start[1])], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000})
    return float(r.x[0]), float(np.exp(r.x[1]))


def reliability_table(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            rows.append({"bin": [float(lo), float(hi)], "predicted": float(p[m].mean()),
                         "observed": float(np.mean(y[m])), "n": int(m.sum())})
    return rows


def calibrate_scores(clips, cache_dir=pv.CACHE_DIR, out_path=scores.CALIB_PATH, log=print, data_dir="data") -> dict:
    """data_dir: where data/pose2d kp2d files live (raw scores via det_index); without them the raw scores are
    recovered by inverting the cached calibrated ones (saturates above ~6.9; logged)."""
    raws, ys, gs, used = [], [], [], None
    for c in clips:
        b = docmod.load_debug_cache(track.cache_path(c, cache_dir))
        p = b.get("score_params") or scores.load_params(path=None)
        if used is not None and p["groups"] != used["groups"]:
            # raw scores are recovered and reliability_before is scored with one set of params
            raise ValueError(f"{c} was built with other score params than {clips[0]}: rebuild them with the same "
                             "score_calibration.json first")
        used = p
        rs = kp2d_raw_scores(b, data_dir)
        r, y, g = heldout_samples(b, used, raw=rs)
        raws.append(r), ys.append(y), gs.append(g)
        log(f"    {c}: {len(r)} held-out samples"
            + ("" if rs is not None else " (no kp2d files: raw scores inverted from the cache, saturate above ~6.9)"))
    raw, y, g = np.concatenate(raws), np.concatenate(ys), np.concatenate(gs)
    out = {"groups": {}, "target": f"held-out-view residual < {TARGET_FRAC} x bbox height",
           "fitted_on": list(clips), "date": datetime.date.today().isoformat()}
    for gi, name in enumerate(scores.GROUPS):
        m = g == gi
        old = used["groups"][name]
        if m.sum() < 500 or np.std(raw[m]) < 0.05 or y[m].all() or not y[m].any():
            out["groups"][name] = dict(old, n=int(m.sum()), kept=True)
            log(f"    {name}: kept mid {old['mid']} width {old['width']} ({int(m.sum())} samples, no usable spread)")
            continue
        mid, width = fit_logistic(raw[m], y[m], (old["mid"], old["width"]))
        p_old = 1 / (1 + np.exp(-(raw[m] - old["mid"]) / old["width"]))
        p_new = 1 / (1 + np.exp(-(raw[m] - mid) / width))
        out["groups"][name] = {"mid": round(mid, 4), "width": round(width, 4), "n": int(m.sum()), "kept": False,
                               "reliability_before": reliability_table(p_old, y[m]),
                               "reliability_after": reliability_table(p_new, y[m])}
        log(f"    {name}: mid {old['mid']} -> {mid:.3f}, width {old['width']} -> {width:.3f} ({int(m.sum())} samples)")
        for row in out["groups"][name]["reliability_after"]:
            log(f"        p {row['bin'][0]:.1f}-{row['bin'][1]:.1f}: predicted {row['predicted']:.2f} observed {row['observed']:.2f} (n {row['n']})")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(docmod.json_safe(out), indent=1, allow_nan=False))
    log(f"    wrote {out_path} — rebuild the clips for it to take effect")
    return out
