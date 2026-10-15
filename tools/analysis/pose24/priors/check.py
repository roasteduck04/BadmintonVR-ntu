"""v0.7.5 monoprior: sanity gate of a learned 1-camera prior against the full multi-camera rig (Task 3).

The full build's fitted keypoints (debug cache extra[pid]["X"], Y-up -> Y-down court world) are moved into the
prior's camera frame with the calibration (X_cam = R @ X + t, as pose_v2.project / mono.ray_plane), made
root-relative to the hip midpoint (kpset left_hip, right_hip: the prior's root), and compared joint by joint with
the prior scaled by mono._depth_rows' leg-length ratio (oracle lengths). Frames used: prior conf >=
mono.PRIOR_CONF_MIN (the frames the fit would take depth rows from) and a finite rig value.

  rig_rel_cam   (F,35,3) Y-down world -> (F,n,3) camera-frame root-relative
  compare       one player's prior vs rig -> per-joint Pearson r of x, y, z, median |dz| (cm), n; group medians
  verdict       PASS (median r_z over legs + arms > GATE_R) | FLIP (< -GATE_R) | BLOCKED
  run           all players of a clip view -> result + <view>_<method>.check.json next to the prior
  format_table  ASCII table (cp932 console)"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.pose24 import kpset, mono
from analysis.pose24.priors import motionbert as mb

GATE_R = 0.5
ROOT = (11, 12)                                     # left_hip, right_hip
GROUPS = {"legs": (11, 12, 13, 14, 15, 16), "arms": (5, 6, 7, 8, 9, 10)}
GROUPS["legs_arms"] = GROUPS["legs"] + GROUPS["arms"]


def rig_rel_cam(X35_down, cam, kp):
    """(F,35,3) Y-down world keypoints -> (F,len(kp),3) camera-frame (OpenCV axes) coordinates relative to the hip
    midpoint; NaN where a joint or either hip is missing."""
    K, R, t = (np.asarray(x, float) for x in cam)
    X = np.asarray(X35_down, float)
    root = X[:, list(ROOT)].mean(1)
    return (X[:, list(kp)] - root[:, None]) @ R.T                     # t cancels in the difference


def _r(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def joint_stats(P, G, ok, kp):
    """Scaled prior P and rig G (F,n,3), usable mask ok (F,n) -> {name: {r_x, r_y, r_z, med_abs_dz_cm, n}} and
    group medians {group: {median_r_z, median_r_x, median_r_y, median_abs_dz_cm}}."""
    joints = {}
    for i, k in enumerate(kp):
        m = ok[:, i]
        p, g = P[m, i], G[m, i]
        joints[kpset.NAMES[k]] = {"r_x": _r(p[:, 0], g[:, 0]), "r_y": _r(p[:, 1], g[:, 1]), "r_z": _r(p[:, 2], g[:, 2]),
                                  "med_abs_dz_cm": float(np.median(np.abs(p[:, 2] - g[:, 2])) * 100) if m.any() else None,
                                  "n": int(m.sum())}
    groups = {}
    for gname, ks in GROUPS.items():
        js = [joints[kpset.NAMES[k]] for k in ks if kpset.NAMES[k] in joints]
        groups[gname] = {f"median_{q}": _med([j[q] for j in js]) for q in ("r_z", "r_x", "r_y", "med_abs_dz_cm")}
        groups[gname]["median_abs_dz_cm"] = groups[gname].pop("median_med_abs_dz_cm")
    return joints, groups


def pairs(prior, X35_down, cam, lengths, conf_min=mono.PRIOR_CONF_MIN):
    """-> (scale, P (F,n,3) prior * scale, G (F,n,3) rig, ok (F,n)). scale: mono._depth_rows' leg-length ratio."""
    kp = np.asarray(prior["kp"], int)
    rel, conf = np.asarray(prior["rel_cam"], float), np.asarray(prior["conf"], float)
    F = len(conf)
    X = np.asarray(X35_down, float)
    if len(X) != F:
        raise ValueError(f"prior has {F} frames, the rig {len(X)}")
    s = mono._depth_rows(prior, lengths, None, F)["scale"]
    G = rig_rel_cam(X, cam, kp)
    P = rel * s
    ok = (conf[:, None] >= conf_min) & np.isfinite(G).all(-1) & np.isfinite(P).all(-1)
    return s, P, G, ok


def compare(prior, X35_down, cam, lengths, conf_min=mono.PRIOR_CONF_MIN):
    """One player's prior vs the rig -> {"scale", "joints", "groups"} (joint_stats)."""
    s, P, G, ok = pairs(prior, X35_down, cam, lengths, conf_min)
    joints, groups = joint_stats(P, G, ok, np.asarray(prior["kp"], int))
    return {"scale": s, "joints": joints, "groups": groups}


def verdict(res):
    r = res["groups"]["legs_arms"]["median_r_z"]
    if r is None:
        return "BLOCKED"
    return "PASS" if r > GATE_R else "FLIP" if r < -GATE_R else "BLOCKED"


def check_path(clip, view, method, prior_dir=None):
    return os.path.join(prior_dir or mb.DEFAULT_OUT_DIR, clip, f"{view}_{method}.check.json")


def run(clip, view, method, cache, prior_dir=None, conf_min=mono.PRIOR_CONF_MIN, write=True):
    """Check data/smpl_mono/<clip>/<view>_<method>.npz against `cache` (the clip's pose24 debug cache): per player and
    pooled over players (frames concatenated); writes <view>_<method>.check.json beside the prior."""
    prior_dir = os.fspath(prior_dir) if prior_dir is not None else None
    path = os.path.join(prior_dir or mb.DEFAULT_OUT_DIR, clip, f"{view}_{method}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no prior at {path}")
    if view not in cache["cams"]:
        raise ValueError(f"view {view!r} has no camera in the cache")
    cam = cache["cams"][view]
    players, Ps, Gs, oks, kp0, meta = {}, [], [], [], None, None
    for pid in sorted((cache.get("extra") or {})):
        X = (cache["extra"][pid] or {}).get("X")
        prior = mono.load_prior(path, pid)
        if X is None or prior is None:
            continue
        meta = prior["meta"]
        Xd = np.asarray(X, float) * np.array([1.0, -1.0, 1.0])
        s, P, G, ok = pairs(prior, Xd, cam, mono.lengths_for("oracle", cache, pid), conf_min)
        kp = np.asarray(prior["kp"], int)
        joints, groups = joint_stats(P, G, ok, kp)
        players[pid] = {"scale": s, "frames_conf_ok": int((prior["conf"] >= conf_min).sum()),
                        "joints": joints, "groups": groups}
        players[pid]["verdict"] = verdict(players[pid])
        if kp0 is None or np.array_equal(kp0, kp):
            kp0 = kp
            Ps.append(P), Gs.append(G), oks.append(ok)
    if not players:
        raise ValueError(f"no player has both a prior and a fitted X ({path})")
    joints, groups = joint_stats(np.concatenate(Ps), np.concatenate(Gs), np.concatenate(oks), kp0)
    pooled = {"joints": joints, "groups": groups}
    res = {"clip": clip, "view": view, "method": method, "prior": path, "prior_meta": meta,
           "cache_build_id": cache.get("build_id"), "conf_min": conf_min, "gate_r": GATE_R,
           "groups_def": {g: [kpset.NAMES[k] for k in ks] for g, ks in GROUPS.items()},
           "players": players, "pooled": pooled, "verdict": verdict(pooled)}
    out = check_path(clip, view, method, prior_dir)
    if write:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, allow_nan=False)
        os.replace(tmp, out)
    res["path"] = out
    return res


def _f(v, fmt):
    return "   -" if v is None else format(v, fmt)


def format_table(res):
    """ASCII per-joint table (pooled, then per-player group medians) + verdict."""
    lines = [f"check {res.get('clip')} {res.get('view')} {res.get('method')}  (conf >= {res.get('conf_min')}, "
             f"pooled over players)",
             f"{'joint':<16}{'r_z':>7}{'r_x':>7}{'r_y':>7}{'|dz|cm':>8}{'n':>7}"]
    for name, j in res["pooled"]["joints"].items():
        lines.append(f"{name:<16}{_f(j['r_z'], '7.3f')}{_f(j['r_x'], '7.3f')}{_f(j['r_y'], '7.3f')}"
                     f"{_f(j['med_abs_dz_cm'], '8.1f')}{j['n']:>7d}")
    lines.append(f"{'group (median)':<16}{'r_z':>7}{'r_x':>7}{'r_y':>7}{'|dz|cm':>8}")
    for who, d in [("all", res["pooled"])] + sorted(res["players"].items()):
        for g, v in d["groups"].items():
            lines.append(f"{(who + ' ' + g):<16}{_f(v['median_r_z'], '7.3f')}{_f(v['median_r_x'], '7.3f')}"
                         f"{_f(v['median_r_y'], '7.3f')}{_f(v['median_abs_dz_cm'], '8.1f')}")
    for pid, d in sorted(res["players"].items()):
        lines.append(f"{pid}: scale {d['scale']:.4f}, {d['frames_conf_ok']} frames conf ok, verdict {d['verdict']}")
    lines.append(f"VERDICT {res['verdict']} (median r_z legs+arms vs gate {GATE_R})")
    return "\n".join(lines)
