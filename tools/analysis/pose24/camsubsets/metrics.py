"""Per-row metrics for the camera-subset study (spec § 5), on arrays (never on 75 docs)."""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np

from analysis import eval_pose_mv as epm
from analysis.pose24 import eval as ev
from analysis.pose24 import kpset

LIMB_PAIRS = (("left_hip", "left_knee"), ("right_hip", "right_knee"), ("left_knee", "left_ankle"),
              ("right_knee", "right_ankle"), ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
              ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"))


HELDOUT_MIN_SAMPLES = 200


def _has(full_cache, pid, view):
    return view in full_cache["views"] and pid in full_cache["obs"]


def _det_mask(full_cache, pid, view, n):
    """(n,) bool: frames where the full build assigned `pid` a detection in `view` (det_index >= 0); all True when
    the cache has no det_index for the player (old caches)."""
    det = ((full_cache.get("extra") or {}).get(pid) or {}).get("det_index")
    if det is None:
        return np.ones(n, bool)
    d = np.asarray(det)[full_cache["views"].index(view)]
    out = np.zeros(n, bool)
    out[:min(n, len(d))] = d[:n] >= 0
    return out


def heldout_px(X35_up, full_cache, pid, view):
    """Held-out px error of X35_up projected into `view`, only on frames where the full build has a detection of
    the player there (R21); None with fewer than HELDOUT_MIN_SAMPLES keypoint samples."""
    if not _has(full_cache, pid, view):
        return None
    r, bbox = ev.heldout_errors(np.asarray(X35_up, float)[:, ev.LIMB_FIT], full_cache, pid, view)
    m = _det_mask(full_cache, pid, view, len(r))
    out = ev._row([(np.where(m[:, None], r, np.nan), np.where(m, bbox, np.nan))])
    return out if out["px"] is not None and out["px"]["n"] >= HELDOUT_MIN_SAMPLES else None


def vs_ref(J, J_ref, names, swing=None):
    out = ev.compare_joints(J, J_ref, names)
    if swing is not None and swing.any():
        out["swing"] = ev.compare_joints(J, J_ref, names, mask=swing)["angle_mae_deg"]
    return out


def _lr_perm(names):
    perm = list(range(len(names)))
    for a, n in enumerate(names):
        if n.startswith("left_") and "right_" + n[5:] in names:
            b = names.index("right_" + n[5:])
            perm[a], perm[b] = b, a
    return perm


def lr_swap_frames(J, J_ref, names, margin_m=0.05):
    n = min(len(J), len(J_ref))
    J, R = J[:n], J_ref[:n]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d0 = np.nanmean(np.linalg.norm(J - R, axis=-1), axis=1)
        d1 = np.nanmean(np.linalg.norm(J[:, _lr_perm(names)] - R, axis=-1), axis=1)
    return int(np.sum(np.isfinite(d0) & np.isfinite(d1) & (d1 < d0 - margin_m)))


def coverage_stats(J, inlier, min_views=2):
    """(coverage, measured, full_body) of one player's result (R16): coverage = fraction of frames with a finite pelvis
    (joint 0), full_body = frames with every joint finite, measured = a coverage frame with >= `min_views` inlier
    views on any keypoint (inlier (F, 35, V); 2 for a multi-view fit, 1 for mono)."""
    J = np.asarray(J, float)
    if J.ndim != 3 or not len(J):
        return 0.0, 0.0, 0.0
    pel = np.isfinite(J[:, 0]).all(-1)
    inl = np.asarray(inlier, bool)
    n = min(len(inl), len(pel))
    meas = np.zeros(len(pel), bool)
    if n and inl.ndim == 3:
        meas[:n] = pel[:n] & (inl[:n].sum(-1) >= min_views).any(axis=1)
    return float(pel.mean()), float(meas.mean()), float(np.isfinite(J).all(-1).all(-1).mean())


def used_frac(inlier, views):
    body = np.asarray(inlier)[:, :17].any(axis=1)                 # (F, V)
    return {v: round(float(body[:, k].mean()), 4) for k, v in enumerate(views)}


def ray_angles_deg(cams, views, pelvis_down):
    centres = {v: -cams[v][1].T @ cams[v][2] for v in views}
    X = np.asarray(pelvis_down, float)
    out = {}
    for a, b in combinations(sorted(views), 2):
        ra, rb = X - centres[a], X - centres[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.sum(ra * rb, -1) / (np.linalg.norm(ra, axis=-1) * np.linalg.norm(rb, axis=-1))
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        out[f"{a}+{b}"] = float(np.nanmedian(ang)) if np.isfinite(ang).any() else None
    return out


def bbox_height_px(full_cache, pid, view):
    if not _has(full_cache, pid, view):
        return None
    uv, w = kpset.player_obs(full_cache, pid)
    bbox = ev.bbox_heights(uv, w, full_cache["views"].index(view))
    bbox = np.where(_det_mask(full_cache, pid, view, len(bbox)), bbox, np.nan)
    return float(np.nanmedian(bbox)) if np.isfinite(bbox).any() else None


def quality(J, C, sigma, fps, names):
    i = {n: k for k, n in enumerate(names)}
    Xl = np.where((C[..., None] > ev.C_MIN), J, np.nan)[:, [i[n] for n in epm.LIMBS]]
    row = epm.motion_stats(Xl, fps)
    lens = [np.linalg.norm(J[:, i[a]] - J[:, i[b]], axis=-1) for a, b in LIMB_PAIRS]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cv = [np.nanstd(x) / np.nanmean(x) for x in lens if np.isfinite(x).sum() > 10]
        row["limb_cv"] = float(np.mean(cv)) if cv else None
        s = np.where(C > ev.C_MIN, sigma, np.nan)
        row["mean_sigma_mm"] = float(np.nanmean(s)) if np.isfinite(s).any() else None
    return row


def swing_windows(J_ref, fps, names, rallies=None, q=95, pad=10):
    """(F,) bool: frames within `pad` of a wrist speed above the q-th percentile inside the rallies (D3)."""
    F = len(J_ref)
    out = np.zeros(F, bool)
    if F < 2:
        return out
    sp = np.stack([np.r_[np.nan, np.linalg.norm(np.diff(J_ref[:, names.index(w)], axis=0), axis=-1) * fps]
                   for w in ("left_wrist", "right_wrist")], 1)
    s = np.max(np.where(np.isfinite(sp), sp, -np.inf), axis=1)
    s[~np.isfinite(s)] = np.nan
    inr = np.zeros(F, bool)
    for r in (rallies or [{"start": 0, "end": F - 1}]):
        lo, hi = max(0, int(r["start"])), min(F - 1, int(r["end"]))
        if lo <= hi:
            inr[lo:hi + 1] = True
    if not np.isfinite(s[inr]).any():
        return out
    with np.errstate(invalid="ignore"):
        hot = inr & (s > np.nanpercentile(s[inr], q))
    for f in np.nonzero(hot)[0]:
        out[max(0, f - pad):f + pad + 1] = True
    return out


def identity_proxy(pel_sub, pel_ref_by_pid, pid_ref, margin_m=0.5):
    own = np.linalg.norm((pel_sub - pel_ref_by_pid[pid_ref])[:, [0, 2]], axis=-1)
    others = [np.linalg.norm((pel_sub - p)[:, [0, 2]], axis=-1) for q, p in pel_ref_by_pid.items() if q != pid_ref]
    if not others:
        return {"wrong_frames": 0, "switches": 0}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wrong = np.nanmin(np.stack(others), axis=0) < own - margin_m
    wrong &= np.isfinite(own)
    switches = int(np.sum(wrong[1:] & ~wrong[:-1]) + wrong[0]) if len(wrong) else 0
    return {"wrong_frames": int(wrong.sum()), "switches": switches}
