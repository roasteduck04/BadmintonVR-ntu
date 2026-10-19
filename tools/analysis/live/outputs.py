"""Fit / fallback keypoints -> SMPL-28 joints + confidence for live output frames: the offline
build.regress_player (regress + confidence) on a window's frames."""
from __future__ import annotations

import warnings

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import build, confidence, fit

FITTED, WARMUP, LATE, NONE = 0, 1, 2, 3


def regress_frames(cams, views, X, uv, w, kind):
    P = pv.proj_mats(cams, views)
    r, inlier = fit._residuals(P, X, uv, w)
    out = build.regress_player({"cams": cams, "views": list(views)},
                               {"X": X, "w": w, "resid": r, "inlier": inlier}, kind)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seen = np.asarray(w) >= pv.W_MIN
        reproj = np.nanmedian(np.where(seen, r, np.nan), axis=1)          # (T, V) px, NaN: view saw nothing
    return {"J": out["J"], "C": out["C"], "aux": out["aux"], "auxc": confidence.to_conf(out["aux_sigma"]),
            "filled": np.asarray(out["filled"][:, 0], bool),
            "forward_sign": -1.0 if out["info"]["mirrored"] else 1.0, "reproj": reproj, "views": list(views)}


def player_frame(r, t, quality):
    valid = bool(np.isfinite(r["J"][t, 0]).all())
    rp = r.get("reproj")
    reproj = ({v: (None if not np.isfinite(rp[t, k]) else round(float(rp[t, k]), 2)) for k, v in enumerate(r["views"])}
              if rp is not None else {})
    return {"J": r["J"][t], "C": r["C"][t], "aux": r["aux"][t], "auxc": r["auxc"][t], "interp": bool(r["filled"][t]),
            "valid": valid, "quality": int(quality) if valid else NONE, "forward_sign": float(r["forward_sign"]),
            "reproj_px": reproj}


def frame_quality(players) -> int:
    q = [p["quality"] for p in players.values() if p["valid"]]
    return max(q) if q else NONE
