"""pose24 evaluation without ground truth (spec § 10).

    PYTHONPATH=tools python -m analysis.pose24 eval --clip singles_05x --lovo
    PYTHONPATH=tools python -m analysis.pose24 eval --clip singles_05x_2cam --reference singles_05x

  heldout       project the fitted keypoints into cameras this build didn't use (from the reference build's
                cache — same timeline via --timeline-of): median / p90 px and px / bbox height
  lovo          full builds (--lovo): refit without camera v, project into v — every camera incl. the sync ref
  vs_reference  W-MPJPE (court frame), PA-MPJPE (observed joints), pelvis XZ cm, joint-angle MAE / p90
  motion        per player, same code for pose24 and posefit (eval_pose_mv definitions on the 12 limb joints):
                coverage, measured, reproj_px, jerk, bone_cv, ankle skate, lag_frames; + contact skate
                (heels/toes below 3 cm) and trunk lateral lean range (p95 - p5; H4 check)
  reliability   predicted sigma (mm) vs held-out px per quintile + Spearman rho
Writes data/pose_eval/pose24/<clip>.json.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
from scipy.stats import spearmanr

from analysis import eval_pose_mv as epm
from analysis import pose_v2 as pv
from analysis import skeleton24 as sk
from analysis.pose24 import angles, fit, kpset, track
from analysis.pose24 import build as pb
from analysis.pose24 import doc as docmod

OUT_DIR = os.path.join("data", "pose_eval", "pose24")
UP = np.array([0.0, 1.0, 0.0])
LIMB_FIT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]            # fit keypoints in epm.LIMBS order
LIMB_SMPL = [sk.SMPL28_NAMES.index(n) for n in epm.LIMBS]
PA_EXCLUDE = {"spine1", "spine2", "spine3", "left_collar", "right_collar", "pelvis"}
CONTACT_NAMES = ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]
C_MIN = 1e-6                     # use every joint the output shows (conf 0 = filled placeholder)


def _stats(e):
    e = np.asarray(e, float).ravel()
    e = e[np.isfinite(e)]
    if not len(e):
        return None
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "n": int(len(e))}


def load_build(clip, analysis_dir, cache_dir):
    with open(pb.output_path(analysis_dir, clip), "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc, docmod.load_debug_cache(track.cache_path(clip, cache_dir))


def named_world(doc, pid, names):
    """(F, n, 3) Y-up court-world positions of named joints, looked up in joint_names then aux_names."""
    jn, an = doc["skeleton"]["joint_names"], doc["skeleton"].get("aux_names") or []
    p = next(p for p in doc["players"] if p["id"] == pid)
    X = np.full((len(p["frames"]), len(names), 3), np.nan)
    for f, fr in enumerate(p["frames"]):
        pos = fr.get("position")
        if pos is None:
            continue
        off = np.array([pos[0], 0.0, pos[1]])
        a = np.asarray(fr["joints_flat"], float).reshape(-1, 4) if fr.get("joints_flat") else None
        x = np.asarray(fr["aux_flat"], float).reshape(-1, 4) if fr.get("aux_flat") else None
        for i, n in enumerate(names):
            src, k = (a, jn.index(n)) if n in jn else ((x, an.index(n)) if n in an else (None, None))
            if src is not None and src[k, 3] > C_MIN:
                X[f, i] = src[k, :3] + off
    return X


def procrustes_error(A, B):
    """Mean joint distance after the best similarity transform of A onto B (both (n,3))."""
    mA, mB = A.mean(0), B.mean(0)
    A0, B0 = A - mA, B - mB
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max((A0 ** 2).sum(), 1e-12)
    return float(np.linalg.norm(s * A0 @ R - B0, axis=1).mean())


def _doc_sign(X, i):
    left = X[:, i["left_hip"]] - X[:, i["right_hip"]]
    up = X[:, i["neck"]] - X[:, i["pelvis"]]
    toe = "left_foot" if "left_foot" in i else ("left_big_toe" if "left_big_toe" in i else None)
    if toe is None:
        return 1.0
    d = np.sum(np.cross(left, up) * (X[:, i[toe]] - X[:, i["left_ankle"]]), -1)
    d = d[np.isfinite(d)]
    return 1.0 if not len(d) or np.median(np.sign(d)) >= 0 else -1.0


def joint_angles(X, names):
    """Named-joint angles (deg): knee / elbow flexion, shoulder abduction (upper arm vs pelvis->neck axis),
    trunk forward / lateral lean (pelvis->neck vs vertical)."""
    i = {n: k for k, n in enumerate(names)}

    def g(n):
        return X[:, i[n]]
    out = {}
    for s in ("left", "right"):
        out[f"{s}_knee_flexion"] = angles.angle_deg(g(f"{s}_knee") - g(f"{s}_hip"), g(f"{s}_ankle") - g(f"{s}_knee"))
        out[f"{s}_elbow_flexion"] = angles.angle_deg(g(f"{s}_elbow") - g(f"{s}_shoulder"), g(f"{s}_wrist") - g(f"{s}_elbow"))
        out[f"{s}_shoulder_abduction"] = angles.angle_deg(g(f"{s}_elbow") - g(f"{s}_shoulder"), g("pelvis") - g("neck"))
    fwd, lat = angles.trunk_lean(g("neck") - g("pelvis"), g("left_hip") - g("right_hip"), UP, _doc_sign(X, i))
    out["trunk_forward_lean"], out["trunk_lateral_lean"] = fwd, lat
    return out


ANGLE_NAMES = ["pelvis", "neck", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def _angle_joints(doc):
    names = ANGLE_NAMES + [n for n in ("left_foot", "left_big_toe") if n in doc["skeleton"]["joint_names"]
                           or n in (doc["skeleton"].get("aux_names") or [])][:1]
    return names


def heldout_errors(X_up, ref_bundle, ref_pid, view, k_idx=LIMB_FIT):
    """Pixel error of Y-up world keypoints X_up (F,k,3) projected into `view` vs the reference cache's
    synced 2D there (score >= 0.5), and the person's bbox height in that view (F,)."""
    uv, w = kpset.player_obs(ref_bundle, ref_pid)
    k = ref_bundle["views"].index(view)
    n = min(len(X_up), len(uv))
    Xd = X_up[:n] * np.array([1.0, -1.0, 1.0])
    pr, depth = pv.project(pv.proj_mats(ref_bundle["cams"], [view]), Xd)
    obs = uv[:n][:, k_idx, k]
    r = np.linalg.norm(pr[:, :, 0] - obs, axis=-1)
    r = np.where((w[:n][:, k_idx, k] >= 0.5) & (depth[:, :, 0] > 0), r, np.nan)
    return r, bbox_heights(uv[:n], w[:n], k)


def bbox_heights(uv, w, k):
    """(F,) body bbox height (px) in view index k from 35-keypoint obs (score >= 0.3; NaN when <= 10 px)."""
    body = np.where(w[:, :17, k, None] >= 0.3, uv[:, :17, k], np.nan)[..., 1]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bbox = np.nanmax(body, axis=1) - np.nanmin(body, axis=1)
    return np.where(bbox > 10, bbox, np.nan)


def _row(errs):
    r = np.concatenate([e[0].ravel() for e in errs]) if errs else np.array([])
    rel = np.concatenate([(e[0] / e[1][:, None]).ravel() for e in errs]) if errs else np.array([])
    return {"px": _stats(r), "per_bbox": _stats(rel)}


def reliability(sig, err, bins=5):
    sig, err = np.asarray(sig, float).ravel(), np.asarray(err, float).ravel()
    m = np.isfinite(sig) & np.isfinite(err)
    if m.sum() < 50:
        return None
    q = np.quantile(sig[m], np.linspace(0, 1, bins + 1))
    rows = []
    for lo, hi in zip(q[:-1], q[1:]):
        b = m & (sig >= lo) & (sig <= hi)
        rows.append({"sigma_mm": [float(lo), float(hi)], "median_err_px": float(np.median(err[b])), "n": int(b.sum())})
    return {"bins": rows, "spearman": float(spearmanr(sig[m], err[m]).correlation)}


def _contact_skate(Xf, fps):
    sp = []
    for j in range(Xf.shape[1]):
        low = Xf[:-1, j, 1] < 0.03
        v = np.linalg.norm(np.diff(Xf[:, j, [0, 2]], axis=0), axis=-1) * fps
        m = low & (Xf[1:, j, 1] < 0.03) & np.isfinite(v)
        sp.extend(v[m].tolist())
    return float(np.median(sp)) if sp else None


def motion_row(doc, bundle, pid):
    fps = float(doc["meta"]["fps"])
    X, _ = epm.doc_world(doc, pid, c_min=C_MIN)
    row = epm.motion_stats(X, fps)
    e = epm.reproj_errors(bundle, pid, X)
    row["reproj_px"] = float(np.nanmedian(e)) if np.isfinite(e).any() else None
    row["lag_frames"] = epm.lag_frames(X, epm.raw_reference(bundle, pid))
    p = next(p for p in doc["players"] if p["id"] == pid)
    n = len(p["frames"])
    row["coverage"] = sum(fr.get("joints_flat") is not None for fr in p["frames"]) / n
    row["measured"] = sum(fr.get("joints_flat") is not None and not fr.get("interpolated") for fr in p["frames"]) / n
    row["contact_skate_mps"] = _contact_skate(named_world(doc, pid, CONTACT_NAMES), fps)
    names = _angle_joints(doc)
    lat = joint_angles(named_world(doc, pid, names), names)["trunk_lateral_lean"]
    lat = lat[np.isfinite(lat)]
    row["trunk_lateral_lean_range_deg"] = float(np.percentile(lat, 95) - np.percentile(lat, 5)) if len(lat) > 20 else None
    return row


def _compare_arrays(A, B, names, mask=None):
    """Raw per-frame errors of Y-up (F,n,3) A vs B: w / root / pelvis XZ / pelvis height (m), PA (m, frames with
    >= 6 finite joints), per-angle abs error (deg)."""
    n = min(len(A), len(B)) if mask is None else min(len(A), len(B), len(mask))
    A, B = A[:n], B[:n]
    if mask is not None:
        m = np.asarray(mask, bool)[:n]
        A, B = A[m], B[m]
    i = {nm: k for k, nm in enumerate(names)}
    pa_idx = [k for k, nm in enumerate(names) if nm not in PA_EXCLUDE]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.linalg.norm(A - B, axis=-1)
        w = np.nanmean(d, axis=1)
        Ar, Br = A - A[:, i["pelvis"], None], B - B[:, i["pelvis"], None]
        root = np.nanmean(np.linalg.norm(Ar - Br, axis=-1), axis=1)
    pel = np.linalg.norm((A[:, i["pelvis"]] - B[:, i["pelvis"]])[:, [0, 2]], axis=-1)
    pel_h = np.abs(A[:, i["pelvis"], 1] - B[:, i["pelvis"], 1])
    pa = []
    for f in range(len(A)):
        m = np.isfinite(A[f, pa_idx]).all(1) & np.isfinite(B[f, pa_idx]).all(1)
        if m.sum() >= 6:
            pa.append(procrustes_error(A[f, pa_idx][m], B[f, pa_idx][m]))
    ang = {}
    an = [x for x in ANGLE_NAMES + ["left_foot"] if x in i]
    if all(x in i for x in ANGLE_NAMES):
        ja, jb = joint_angles(A[:, [i[x] for x in an]], an), joint_angles(B[:, [i[x] for x in an]], an)
        ang = {k: np.abs(ja[k] - jb[k]) for k in ja}
    return {"w": w, "root": root, "pa": np.array(pa), "pel": pel, "pel_h": pel_h, "ang": ang}


def _angle_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mae": float(v.mean()), "p90": float(np.percentile(v, 90))} if len(v) else None


def compare_joints(A, B, names, mask=None):
    """A vs B on Y-up (F,n,3) named joints; mask (F,) bool limits the frames used."""
    r = _compare_arrays(A, B, names, mask)
    cm = lambda x: _stats(np.asarray(x) * 100)
    return {"w_mpjpe_cm": cm(r["w"]), "root_mpjpe_cm": cm(r["root"]), "pa_mpjpe_cm": cm(r["pa"]),
            "pelvis_xz_cm": cm(r["pel"]), "pelvis_h_cm": cm(r["pel_h"]),
            "angle_mae_deg": {k: _angle_stats(v) for k, v in r["ang"].items()}}


def vs_reference(doc_b, doc_r, pid_map):
    names = [n for n in sk.SMPL28_NAMES if n in doc_b["skeleton"]["joint_names"] and n in doc_r["skeleton"]["joint_names"]]
    w, pa, pel, ang = [], [], [], {}
    for pid, rpid in pid_map.items():
        r = _compare_arrays(named_world(doc_b, pid, names), named_world(doc_r, rpid, names), names)
        w.append(r["w"])
        pel.append(r["pel"])
        pa.append(r["pa"])
        for k, v in r["ang"].items():
            ang.setdefault(k, []).append(v)
    cm = lambda xs: _stats(np.concatenate(xs) * 100) if xs else None
    return {"w_mpjpe_cm": cm(w), "pa_mpjpe_cm": _stats(np.concatenate(pa) * 100) if pa else None,
            "pelvis_xz_cm": cm(pel),
            "angle_mae_deg": {k: {"mae": float(np.nanmean(np.concatenate(v))), "p90": float(np.nanpercentile(np.concatenate(v), 90))}
                              for k, v in ang.items()}}


def _posefit_rows(clip, doc, analysis_dir, cache_dir, log):
    """{pose24 player id: posefit motion row}. The posefit doc and its pose_v2 cache (<clip>.npz, the format
    load_debug_cache reads) carry their own player ids, so players are matched by position (review L3)."""
    path = os.path.join(analysis_dir, f"{clip}_posefit.analysis.json")
    cpath = os.path.join(cache_dir, f"{clip}.npz")
    if not os.path.exists(path) or not os.path.exists(cpath):
        log(f"    posefit row skipped: {os.path.basename(path)} or its pose_v2 cache missing")
        return None
    with open(path, "r", encoding="utf-8") as fh:
        pf = json.load(fh)
    b2 = docmod.load_debug_cache(cpath)
    to_pf = docmod.match_players(pf, doc)
    for p in doc["players"]:
        if p["id"] not in to_pf:
            log(f"    posefit row: no posefit player within {docmod.MATCH_MAX_M} m of {p['id']}")
    return {pid: motion_row(pf, b2, fid) for pid, fid in to_pf.items() if fid in b2["pids"]}


def evaluate(clip, reference=None, lovo=False, posefit=True, analysis_dir=pb.config.DEFAULT_ANALYSIS_DIR,
             cache_dir=pv.CACHE_DIR, out_dir=OUT_DIR, fit_opts=None, log=print) -> dict:
    doc, cache = load_build(clip, analysis_dir, cache_dir)
    res = {"clip": clip, "views": cache["views"], "runtime_s": doc["meta"]["pose24"]["runtime_s"], "players": {}}
    for p in doc["players"]:
        res["players"][p["id"]] = {"pose24": motion_row(doc, cache, p["id"])}
    rel_s, rel_e = [], []
    if reference:
        rdoc, rcache = load_build(reference, analysis_dir, cache_dir)
        pid_map = docmod.match_players(rdoc, doc)
        held = [v for v in rcache["views"] if v not in cache["views"]]
        res["heldout"] = {}
        for v in held:
            errs = []
            for pid, rpid in pid_map.items():
                X = cache["extra"][pid]["X"][:, LIMB_FIT].astype(float)
                errs.append(heldout_errors(X, rcache, rpid, v))
                rel_s.append(cache["extra"][pid]["sigma_mm"][:len(errs[-1][0]), LIMB_SMPL])
                rel_e.append(errs[-1][0])
            res["heldout"][v] = _row(errs)
        res["vs_reference"] = vs_reference(doc, rdoc, pid_map)
        res["reference"] = reference
    if lovo:
        res["lovo"] = {}
        for v in cache["views"]:
            rest = [u for u in cache["views"] if u != v]
            sub = dict(cache, views=rest, cams={u: cache["cams"][u] for u in rest})
            errs = []
            for pid in cache["obs"]:
                fr = fit.fit_player(sub, pid, log=lambda *a: None, **(fit_opts or {}))
                X = fr["X"][:, LIMB_FIT] * np.array([1.0, -1.0, 1.0])
                errs.append(heldout_errors(X, cache, pid, v))
                rel_s.append(cache["extra"][pid]["sigma_mm"][:, LIMB_SMPL])
                rel_e.append(errs[-1][0])
            res["lovo"][v] = _row(errs)
            log(f"    lovo {v}: {res['lovo'][v]['px']}")
    if rel_s:
        res["reliability"] = reliability(np.concatenate([s.ravel() for s in rel_s]),
                                         np.concatenate([e.ravel() for e in rel_e]))
        if res["reliability"] is not None:    # LOVO pairs the full-build sigma (which used view v) with refits
            res["reliability"]["source"] = " + ".join(
                (["heldout errors vs build sigma"] if reference else [])
                + (["lovo refit errors vs full-build sigma"] if lovo else []))
    if posefit:
        rows = _posefit_rows(clip, doc, analysis_dir, cache_dir, log)
        if rows:
            for pid, row in rows.items():
                res["players"].setdefault(pid, {})["posefit"] = row
    os.makedirs(out_dir, exist_ok=True)
    docmod.write_json(os.path.join(out_dir, f"{clip}.json"), docmod.json_safe(res))
    print_table(res, log)
    return res


def print_table(res, log=print):
    keys = ["coverage", "measured", "reproj_px", "jerk", "bone_cv", "skate", "contact_skate_mps", "lag_frames",
            "trunk_lateral_lean_range_deg"]
    log(f"[pose24 eval] {res['clip']} ({', '.join(res['views'])})")
    for pid, rows in res["players"].items():
        for name, row in rows.items():
            log(f"  {pid:3s} {name:8s} " + "  ".join(f"{k} {row.get(k):.3g}" if isinstance(row.get(k), float)
                                                      else f"{k} -" for k in keys))
    for section in ("heldout", "lovo"):
        for v, row in (res.get(section) or {}).items():
            px = row["px"] or {}
            log(f"  {section:7s} {v:16s} median {px.get('median', float('nan')):.1f}px  p90 {px.get('p90', float('nan')):.1f}px")
    if res.get("vs_reference"):
        vr = res["vs_reference"]
        log(f"  vs {res['reference']}: W-MPJPE {vr['w_mpjpe_cm']}  PA-MPJPE {vr['pa_mpjpe_cm']}  pelvis {vr['pelvis_xz_cm']}")
