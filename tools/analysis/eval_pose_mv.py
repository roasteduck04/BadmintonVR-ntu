"""Compare pose variants without ground truth: image agreement, held-out cameras, motion quality.

    PYTHONPATH=tools python -m analysis.eval_pose_mv --clip singles_05x
    PYTHONPATH=tools python -m analysis.eval_pose_mv --all --lovo --out data/pose_eval

For each clip it scores the base doc and every pose variant next to it (`_posev2`, `_posefit`)
on the 12 limb joints all skeletons share (shoulders, elbows, wrists, hips, knees, ankles):

  reproj_px    median distance of the 3D joint, projected into every camera, to that camera's
               confident RTMPose keypoint (in-sample; synced 2D from the pose_v2 cache)
  lovo_px      held-out camera: rebuild without camera v, project into v (the honest one;
               `--lovo`, skips the reference camera, rebuilds base with build_real)
  jerk         RMS third difference (mm/frame^3) — lower is smoother
  bone_cv      std/mean of limb lengths over the clip (%) — 0 for rigid bones
  skate        median horizontal speed (m/s) of ankles while within 8 cm of their low point
  lag_frames   delay the smoothing adds: wrists/elbows/ankles vs the raw (unfiltered) pose v2
               triangulation of the same clip, 3D velocity cross-correlation (+ = lags)
  coverage / measured  frames with pose / frames not flagged interpolated
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis import pose_v2 as pv

LIMBS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
WB_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BONES = [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
         ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
         ("left_hip", "left_knee"), ("right_hip", "right_knee"),
         ("left_knee", "left_ankle"), ("right_knee", "right_ankle")]
VARIANTS = ["", "_posev2", "_posefit"]


def doc_world(doc, pid, names=LIMBS, c_min=0.3):
    """(F, len(names), 3) court world (Y-up) joints for one player, NaN where missing, and the
    per-frame interpolated flag."""
    jn = doc["skeleton"]["joint_names"]
    idx = [jn.index(n) for n in names]
    p = next(p for p in doc["players"] if p["id"] == pid)
    F = len(p["frames"])
    X = np.full((F, len(names), 3), np.nan)
    interp = np.zeros(F, bool)
    for f, fr in enumerate(p["frames"]):
        jf, pos = fr.get("joints_flat"), fr.get("position")
        if jf is None or pos is None:
            continue
        a = np.asarray(jf, float).reshape(-1, 4)[idx]
        w = a[:, :3] + np.array([pos[0], 0.0, pos[1]])
        w[a[:, 3] < c_min] = np.nan
        X[f] = w
        interp[f] = bool(fr.get("interpolated", False))
    return X, interp


def reproj_errors(bundle, pid, Xw, views=None, w_min=0.5):
    """Per-(frame, joint, view) pixel error of Y-up world joints vs the cached synced 2D."""
    views = views or bundle["views"]
    k = [bundle["views"].index(v) for v in views]
    uv, w = pv.obs_arrays(bundle, pid, idx=WB_IDX)
    uv, w = uv[:, :, k], w[:, :, k]
    n = min(len(Xw), len(uv))
    Xc = Xw[:n].copy()
    Xc[..., 1] *= -1.0
    pr, depth = pv.project(pv.proj_mats(bundle["cams"], views), Xc)
    r = np.linalg.norm(pr - uv[:n], axis=-1)
    ok = (w[:n] >= w_min) & np.isfinite(r) & (depth > 0)
    return np.where(ok, r, np.nan)


def motion_stats(X, fps):
    out = {}
    j3 = np.diff(X, n=3, axis=0)
    jerk = np.linalg.norm(j3, axis=-1)
    out["jerk"] = float(np.sqrt(np.nanmean(jerk ** 2)) * 1000) if np.isfinite(jerk).any() else None
    cvs = []
    for a, b in BONES:
        d = np.linalg.norm(X[:, LIMBS.index(a)] - X[:, LIMBS.index(b)], axis=-1)
        d = d[np.isfinite(d)]
        if len(d) > 20:
            cvs.append(np.std(d) / np.mean(d))
    out["bone_cv"] = float(np.mean(cvs) * 100) if cvs else None
    sk = []
    for j in (LIMBS.index("left_ankle"), LIMBS.index("right_ankle")):
        y = X[:, j, 1]
        if np.isfinite(y).sum() < 30:
            continue
        low = np.nanpercentile(y, 5)
        v = np.linalg.norm(np.diff(X[:, j, [0, 2]], axis=0), axis=-1) * fps
        m = (y[1:] < low + 0.08) & np.isfinite(v)
        sk.extend(v[m].tolist())
    out["skate"] = float(np.median(sk)) if sk else None
    return out


LAG_JOINTS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_ankle", "right_ankle"]


def raw_reference(bundle, pid):
    """Unfiltered pose v2 triangulation of the limb joints (zero-lag reference)."""
    from analysis import schema
    W, C, _ = pv.triangulate_player(bundle, pid)
    J, _ = pv.to_joints58(W, C)
    return J[:, [schema.JOINT_NAMES.index(n) for n in LIMBS]]


def lag_frames(Xw, Xref, max_lag=6):
    """Sub-frame lag of Xw's joint velocities behind Xref's (mean over LAG_JOINTS)."""
    n = min(len(Xw), len(Xref))
    lags = []
    for name in LAG_JOINTS:
        j = LIMBS.index(name)
        a = np.diff(Xref[:n, j], axis=0)
        b = np.diff(Xw[:n, j], axis=0)
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (np.linalg.norm(a, axis=1) < 0.5)
        if ok.sum() < 100:
            continue
        best = []
        for L in range(-max_lag, max_lag + 1):
            s = slice(max(0, -L), len(a) - max(0, L))
            s2 = slice(max(0, L), len(a) - max(0, -L))
            m = ok[s] & ok[s2]
            best.append(np.sum(a[s][m] * b[s2][m]) / np.sqrt(np.sum(a[s][m] ** 2) * np.sum(b[s2][m] ** 2) + 1e-12))
        best = np.array(best)
        i = int(np.argmax(best))
        frac = 0.0
        if 0 < i < len(best) - 1:
            y0, y1, y2 = best[i - 1:i + 2]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        lags.append(i - max_lag + frac)
    return float(np.mean(lags)) if lags else None


# ---------------------------------------------------------------------------
# held-out camera rebuilds
# ---------------------------------------------------------------------------

def lovo_v2(bundle, pid, hold):
    views = [v for v in bundle["views"] if v != hold]
    sub = dict(bundle, views=views, cams={v: bundle["cams"][v] for v in views})
    W, C, _ = pv.triangulate_player(sub, pid)
    J, Cj = pv.to_joints58(W, C)
    J, Cj, _ = pv.postprocess(J, Cj, bundle["fps"])
    from analysis import schema
    idx = [schema.JOINT_NAMES.index(n) for n in LIMBS]
    X = J[:, idx]
    X[Cj[:, idx] < 0.3] = np.nan
    return X


def lovo_fit(bundle, pid, hold, v2_frames):
    from analysis import pose_fit as pf
    views = [v for v in bundle["views"] if v != hold]
    sub = dict(bundle, views=views, cams={v: bundle["cams"][v] for v in views})
    X, support = pf.fit_player(sub, pid, v2_frames, log=lambda *a: None)
    Xw = X[:, WB_IDX].copy()
    Xw[..., 1] *= -1.0
    return Xw


def lovo_base(cfg, hold, pids):
    from analysis.build_real import assemble_multi
    views = [v for v in cfg["views"] if v != hold]
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        doc = assemble_multi({v: cfg["cam_paths"][v] for v in views}, cfg["seeds_path"],
                             {v: cfg["kp2d_paths"][v] for v in views}, cfg["out_clip"], cfg["mode"], 60.0)
    return {pid: doc_world(doc, pid)[0] for pid in pids if any(p["id"] == pid for p in doc["players"])}


def lovo_base_row(base_doc) -> str:
    """Row the base held-out numbers belong to: lovo_base rebuilds with the *current* build_real (schema 5.0,
    58 joints); a frozen older base doc (schema 4.0, 28 joints) is another producer, so its in-sample row must
    not carry them (review 2 L3)."""
    from analysis import schema
    same = (str(base_doc.get("schema_version")) == schema.SCHEMA_VERSION
            and len((base_doc.get("skeleton") or {}).get("joint_names") or []) == schema.NUM_JOINTS)
    return "base" if same else "base_rebuilt"


# ---------------------------------------------------------------------------

def evaluate_clip(out_clip, analysis_dir=pv.DEFAULT_ANALYSIS_DIR, lovo=False, log=print):
    bundle = pv.load_cache(out_clip)
    fps = bundle["fps"]
    docs = {}
    for suf in VARIANTS:
        p = os.path.join(analysis_dir, f"{out_clip}{suf}.analysis.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                docs[suf or "base"] = json.load(fh)
    res = {"clip": out_clip, "views": bundle["views"], "methods": {}}
    refs = {pid: raw_reference(bundle, pid) for pid in bundle["pids"]}
    for name, doc in docs.items():
        per = {}
        for pid in bundle["pids"]:
            if not any(p["id"] == pid for p in doc["players"]):
                continue
            X, interp = doc_world(doc, pid)
            have = np.isfinite(X[:, :, 0]).any(axis=1)
            r = reproj_errors(bundle, pid, X)
            st = {"coverage": float(have.mean() * 100), "measured": float((have & ~interp).mean() * 100),
                  "reproj_px": float(np.nanmedian(r)) if np.isfinite(r).any() else None,
                  "lag_frames": lag_frames(X, refs[pid])}
            st.update(motion_stats(X, fps))
            per[pid] = st
        res["methods"][name] = per
    if lovo:
        cfg = pv.clip_config(pv_clip(out_clip), analysis_dir, views=bundle["views"], out_clip=out_clip)
        v2doc = docs.get("_posev2")
        holds = bundle["views"][1:]
        for name in [n for n in ("base", "_posev2", "_posefit") if n in docs]:
            errs = {pid: [] for pid in bundle["pids"]}
            for hold in holds:
                log(f"    lovo {name} without {hold}")
                if name == "base":
                    Xs = lovo_base(cfg, hold, bundle["pids"])
                for pid in bundle["pids"]:
                    if name == "base":
                        X = Xs.get(pid)
                    elif name == "_posev2":
                        X = lovo_v2(bundle, pid, hold)
                    else:
                        v2f = next(p for p in v2doc["players"] if p["id"] == pid)["frames"]
                        X = lovo_fit(bundle, pid, hold, v2f)
                    if X is None:
                        continue
                    r = reproj_errors(bundle, pid, X, views=[hold])
                    errs[pid].append(r[np.isfinite(r)])
            row = lovo_base_row(docs["base"]) if name == "base" else name
            if row != name:
                res.setdefault("notes", []).append(
                    f"{row}: held-out rebuilds by the current build_real; the base row is the frozen "
                    f"schema {docs['base'].get('schema_version')} doc (another producer)")
            for pid, e in errs.items():
                e = np.concatenate(e) if e else np.array([])
                if pid in res["methods"][name]:
                    st = res["methods"].setdefault(row, {}).setdefault(pid, {})
                    st["lovo_px"] = float(np.median(e)) if len(e) > 50 else None
                    st["lovo_p90_px"] = float(np.percentile(e, 90)) if len(e) > 50 else None
    return res


def pv_clip(out_clip):
    if out_clip in pv.CLIPS:
        return out_clip
    prefixes = [c for c in pv.CLIPS if out_clip.startswith(c + "_")]
    if not prefixes:
        raise KeyError(out_clip)
    return max(prefixes, key=len)   # doubles_2sides_4angle_2cam -> doubles_2sides_4angle, not doubles_2sides


def print_table(res):
    cols = ["coverage", "measured", "reproj_px", "lovo_px", "lovo_p90_px", "jerk", "bone_cv", "skate", "lag_frames"]
    print(f"\n{res['clip']}  ({len(res['views'])} views)")
    print(f"  {'method':9s} {'player':6s} " + " ".join(f"{c:>11s}" for c in cols))
    for name, per in res["methods"].items():
        for pid, st in per.items():
            vals = [st.get(c) for c in cols]
            print(f"  {name:9s} {pid:6s} " + " ".join(f"{v:11.2f}" if isinstance(v, float) else f"{'-':>11s}" for v in vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lovo", action="store_true", help="held-out camera rebuilds (slow)")
    ap.add_argument("--analysis-dir", default=pv.DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--out", default=None, help="directory for <clip>.pose_eval.json")
    args = ap.parse_args(argv)
    clips = (list(pv.CLIPS) + ["doubles_2sides_4angle_2cam"]) if args.all else (args.clip or [])
    for c in clips:
        if not os.path.exists(os.path.join(pv.CACHE_DIR, f"{c}.npz")):
            print(f"{c}: no pose_v2 cache — skipped")
            continue
        res = evaluate_clip(c, args.analysis_dir, lovo=args.lovo)
        print_table(res)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"{c}.pose_eval.json"), "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
