"""Pose fit — multi-view skeleton fitting over time, written as pose variant "posefit".

    PYTHONPATH=tools python -m analysis.pose_fit build --all
    PYTHONPATH=tools python -m analysis.pose_fit build --clip doubles_2sides_4angle_2cam

Instead of triangulating each joint independently, the 23 wholebody body+foot keypoints of a
player are solved jointly per window of frames (scipy least_squares, sparse analytic Jacobian):

    sum  robust( sqrt(w) * reprojection / 8 px )        every view, calibrated 2D score w
  + sum  (|Xa - Xb| - L_ab) / 1.5 cm                     clip-constant, left/right-symmetric bones
  + sum  (X[t-1] - 2 X[t] + X[t+1]) / sigma_acc(joint)   smooth motion (per joint group)
  + sum  max(0, -height(foot)) / 1 cm                    feet never below the floor
  + sum  (X - X_init) / 0.5 m                            weak anchor for unseen joints

so a frame seen by a single camera still gets a physically consistent pose, and outlier
keypoints are down-weighted by the robust loss instead of flipping a whole joint. Needs the
observation cache and doc written by `analysis.pose_v2` (run that first); hands and face are
taken from pose v2 and re-anchored to the fitted wrists/head.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

from analysis import pose_v2 as pv
from analysis import schema
from analysis.conventions import analysis_sign
from analysis.facing import fill_facing, frames_sign

VARIANT = "posefit"
K_IDX = list(range(23))
SIGMA_PX = 8.0
SIGMA_BONE = 0.015
SIGMA_FLOOR = 0.01
SIGMA_ANCHOR = 0.5
F_SCALE = 2.0
WINDOW = 60
OVERLAP = 10
GOOD_PX = 20.0

# keypoint edges (wholebody indices) — pairs listed as (left, right) share one length
_EDGES = [((5, 7), (6, 8)), ((7, 9), (8, 10)), ((11, 13), (12, 14)), ((13, 15), (14, 16)),
          ((5, 11), (6, 12)), ((15, 17), (16, 20)), ((15, 18), (16, 21)), ((15, 19), (16, 22)),
          ((0, 1), (0, 2)), ((1, 3), (2, 4)), ((5, 6), None), ((11, 12), None), ((3, 4), None)]
# per-keypoint acceleration sigma (m per frame^2 at 60 fps)
_ACC = np.array([0.02, 0.02, 0.02, 0.02, 0.02,        # nose, eyes, ears
                 0.012, 0.012, 0.03, 0.03, 0.06, 0.06,  # shoulders, elbows, wrists
                 0.01, 0.01, 0.025, 0.025, 0.035, 0.035,  # hips, knees, ankles
                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])    # toes, heels
_FEET = [15, 16, 17, 18, 19, 20, 21, 22]


def edge_list():
    e, groups = [], []
    for g, (a, b) in enumerate(_EDGES):
        for pair in (a, b):
            if pair is not None:
                e.append(pair)
                groups.append(g)
    return np.array(e), np.array(groups)


def bone_lengths(X, ok):
    """Median, left/right-averaged length per edge group from raw triangulations."""
    E, G = edge_list()
    L = np.full(len(_EDGES), np.nan)
    for g in range(len(_EDGES)):
        ds = []
        for a, b in E[G == g]:
            m = ok[:, a] & ok[:, b]
            if m.sum() >= 10:
                ds.append(np.median(np.linalg.norm(X[m, a] - X[m, b], axis=1)))
        if ds:
            L[g] = float(np.mean(ds))
    return L


class WindowProblem:
    """One window of T frames; variables are the free frames' keypoints (Y-down court world)."""

    def __init__(self, P, uv, w, X_init, free, L, fps=60.0):
        self.P, self.T, self.K = P, X_init.shape[0], X_init.shape[1]
        self.X0 = X_init.copy()
        self.free = free                                  # (T,) bool
        self.fidx = -np.ones(self.T, int)
        self.fidx[free] = np.arange(free.sum())
        self.nv = int(free.sum()) * self.K * 3
        o = (w >= pv.W_MIN) & np.isfinite(uv).all(-1) & free[:, None, None]
        self.ot, self.ok, self.ov = np.nonzero(o)
        self.ouv = uv[self.ot, self.ok, self.ov]
        self.osw = np.sqrt(w[self.ot, self.ok, self.ov]) / SIGMA_PX
        E, G = edge_list()
        keep = np.isfinite(L[G])
        self.E, self.EL = E[keep], L[G][keep]
        self.acc = _ACC * (60.0 / fps) ** 2
        tri = np.arange(1, self.T - 1)
        self.at = tri[free[tri - 1] | free[tri] | free[tri + 1]]

    def var(self, t, k):
        return (self.fidx[t] * self.K + k) * 3

    def unpack(self, x):
        X = self.X0.copy()
        X[self.free] = x.reshape(-1, self.K, 3)
        return X

    def residuals(self, x):
        X = self.unpack(x)
        out = []
        Xo = X[self.ot, self.ok]
        p = np.einsum("nij,nj->ni", self.P[self.ov], np.c_[Xo, np.ones(len(Xo))])
        z = np.where(np.abs(p[:, 2]) < 1e-6, 1e-6, p[:, 2])
        out.append(((p[:, :2] / z[:, None] - self.ouv) * self.osw[:, None]).ravel())
        Xf = X[self.free]
        d = Xf[:, self.E[:, 0]] - Xf[:, self.E[:, 1]]
        out.append(((np.linalg.norm(d, axis=-1) - self.EL) / SIGMA_BONE).ravel())
        a = X[self.at - 1] - 2 * X[self.at] + X[self.at + 1]
        out.append((a / self.acc[None, :, None]).ravel())
        out.append((np.maximum(0.0, Xf[:, _FEET, 1]) / SIGMA_FLOOR).ravel())
        out.append(((Xf - self.X0[self.free]) / SIGMA_ANCHOR).ravel())
        return np.concatenate(out)

    def jacobian(self, x):
        X = self.unpack(x)
        rows, cols, vals = [], [], []
        r0 = 0
        # reprojection
        Xo = X[self.ot, self.ok]
        Pv = self.P[self.ov]
        p = np.einsum("nij,nj->ni", Pv, np.c_[Xo, np.ones(len(Xo))])
        z = np.where(np.abs(p[:, 2]) < 1e-6, 1e-6, p[:, 2])
        u = p[:, :2] / z[:, None]
        vbase = self.var(self.ot, self.ok)
        n = len(self.ot)
        for c in range(2):
            J = (Pv[:, c, :3] - u[:, c:c + 1] * Pv[:, 2, :3]) / z[:, None] * self.osw[:, None]
            rr = r0 + np.arange(n) * 2 + c
            for i in range(3):
                rows.append(rr); cols.append(vbase + i); vals.append(J[:, i])
        r0 += 2 * n
        # bones
        Xf = X[self.free]
        tf = np.nonzero(self.free)[0]
        d = Xf[:, self.E[:, 0]] - Xf[:, self.E[:, 1]]
        nrm = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        g = d / nrm / SIGMA_BONE                                 # (Tf, E, 3)
        rr = r0 + np.arange(len(tf) * len(self.E)).reshape(len(tf), len(self.E))
        for side, sign in ((0, 1.0), (1, -1.0)):
            vb = self.var(tf[:, None], self.E[None, :, side])
            for i in range(3):
                rows.append(rr.ravel()); cols.append((vb + i).ravel()); vals.append((sign * g[..., i]).ravel())
        r0 += rr.size
        # acceleration
        na = len(self.at)
        base = r0 + (np.arange(na)[:, None, None] * self.K * 3 + np.arange(self.K)[None, :, None] * 3
                     + np.arange(3)[None, None, :])
        for off, coef in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            t = self.at + off
            m = self.free[t]
            if not m.any():
                continue
            vb = self.var(t[m][:, None], np.arange(self.K)[None, :])[..., None] + np.arange(3)
            val = np.broadcast_to(coef / self.acc[None, :, None], (m.sum(), self.K, 3))
            rows.append(base[m].ravel()); cols.append(vb.ravel()); vals.append(val.ravel())
        r0 += na * self.K * 3
        # floor hinge on Y (index 1)
        nf = len(_FEET)
        yf = Xf[:, _FEET, 1]
        rr = r0 + np.arange(len(tf) * nf).reshape(len(tf), nf)
        vb = self.var(tf[:, None], np.array(_FEET)[None, :]) + 1
        rows.append(rr.ravel()); cols.append(vb.ravel()); vals.append(np.where(yf > 0, 1.0 / SIGMA_FLOOR, 0.0).ravel())
        r0 += rr.size
        # anchor
        rows.append(r0 + np.arange(self.nv)); cols.append(np.arange(self.nv))
        vals.append(np.full(self.nv, 1.0 / SIGMA_ANCHOR))
        r0 += self.nv
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(r0, self.nv))

    def solve(self, max_nfev=40):
        x0 = self.X0[self.free].ravel()
        res = least_squares(self.residuals, x0, jac=self.jacobian, method="trf", loss="soft_l1",
                            f_scale=F_SCALE, tr_solver="lsmr", max_nfev=max_nfev, x_scale=1.0)
        return self.unpack(res.x)


def _fill_init(X, ok, rest_offset):
    """Per-keypoint linear interpolation over the segment; never-seen keypoints from the
    clip's median offset to the hip centre."""
    T = len(X)
    out = X.copy()
    t = np.arange(T)
    for k in range(X.shape[1]):
        m = ok[:, k]
        if m.sum() >= 1:
            for c in range(3):
                out[:, k, c] = np.interp(t, t[m], X[m, k, c])
    missing = ~ok.any(axis=0)
    if missing.any():
        hip = np.nanmean(out[:, [11, 12]], axis=1)
        out[:, missing] = hip[:, None] + rest_offset[None, missing]
    return out


def fit_player(bundle, pid, v2_frames, log=print):
    """Returns world keypoints X (F,23,3) Y-down, support (F,23) supporting-view weight, seg mask."""
    views = bundle["views"]
    P = pv.proj_mats(bundle["cams"], views)
    uv, w = pv.obs_arrays(bundle, pid, idx=K_IDX)
    F = uv.shape[0]
    Xr, _, _, _ = pv.triangulate_batch(bundle["cams"], views, uv.reshape(-1, len(views), 2),
                                       w.reshape(-1, len(views)))
    Xr = Xr.reshape(F, len(K_IDX), 3)
    ok = np.isfinite(Xr).all(-1)
    L = bone_lengths(Xr, ok)
    hip = np.nanmean(np.where(ok[:, [11, 12], None], Xr[:, [11, 12]], np.nan), axis=1)
    rest = np.nanmedian(Xr - hip[:, None], axis=0)
    rest = np.nan_to_num(rest)
    active = np.array([fr.get("joints_flat") is not None for fr in v2_frames]) | (ok.sum(1) >= 6)
    X = np.full((F, len(K_IDX), 3), np.nan)
    edges = np.diff(np.r_[0, active.astype(int), 0])
    for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        if e - s < 3 or not (ok[s:e, 11] | ok[s:e, 12]).any():
            continue
        Xi = _fill_init(Xr[s:e], ok[s:e], rest)
        seg = Xi.copy()
        start = 0
        while start < e - s:
            stop = min(start + WINDOW, e - s)
            lo = max(0, start - 2)
            free = np.zeros(stop - lo, bool)
            free[start - lo:] = True
            prob = WindowProblem(P, uv[s + lo:s + stop], w[s + lo:s + stop], seg[lo:stop], free, L,
                                 bundle["fps"])
            seg[lo:stop] = prob.solve()
            if stop == e - s:
                break
            start = stop - OVERLAP
        X[s:e] = seg
    pr, depth = pv.project(P, X)
    r = np.linalg.norm(pr - uv, axis=-1)
    good = (w >= pv.W_MIN) & (r < GOOD_PX) & (depth > 0)
    support = (w * good).sum(axis=-1)
    return X, support


def assemble(X, support, v2_doc_player, fps):
    """Fitted keypoints -> 58-joint world frames; hands/face from pose v2 re-anchored."""
    F = len(X)
    W = np.zeros((F, 133, 3))
    C = np.zeros((F, 133))
    ok = np.isfinite(X).all(-1)
    W[:, K_IDX] = np.nan_to_num(X)
    conf = np.clip(0.35 + 0.6 * np.minimum(support, 2.0) / 2.0, 0, 1)
    C[:, K_IDX] = np.where(ok, conf, 0.0)
    W[..., 1] *= -1.0
    J, Cj = pv.to_joints58(W, C)
    names = schema.JOINT_NAMES
    lw, rw, head = names.index("left_wrist"), names.index("right_wrist"), names.index("head")
    for f, fr in enumerate(v2_doc_player["frames"]):
        jf, pos = fr.get("joints_flat"), fr.get("position")
        if jf is None or not np.isfinite(J[f, 0]).all():
            continue
        a = np.array(jf).reshape(-1, 4)
        v2w = a[:, :3] + np.array([pos[0], 0.0, pos[1]])
        for lo, hi, anchor in ((22, 37, lw), (37, 52, rw), (52, 56, head)):
            if np.isfinite(J[f, anchor]).all():
                J[f, lo:hi] = v2w[lo:hi] + (J[f, anchor] - v2w[anchor])
                Cj[f, lo:hi] = a[lo:hi, 3]
    prior_only = support[:, [5, 6, 11, 12]].sum(axis=1) < 1.0
    L = pv.median_bone_lengths(J, Cj, schema.PARENTS)
    J = pv.enforce_bones(J, L, schema.PARENTS)
    return pv.frames_from_world(J, Cj, prior_only)


def build(out_clip, analysis_dir=pv.DEFAULT_ANALYSIS_DIR, log=print):
    bundle = pv.load_cache(out_clip)
    v2_path = os.path.join(analysis_dir, f"{out_clip}_{pv.VARIANT}.analysis.json")
    with open(v2_path, "r", encoding="utf-8") as fh:
        v2 = json.load(fh)
    base_path = os.path.join(analysis_dir, f"{out_clip}.analysis.json")
    with open(base_path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    log(f"[posefit] {out_clip}: views {', '.join(bundle['views'])}")
    fps, n = bundle["fps"], bundle["frame_count"]
    v2p = {p["id"]: p for p in v2["players"]}
    players = []
    for pid in bundle["pids"]:
        if pid not in v2p:
            continue
        X, support = fit_player(bundle, pid, v2p[pid]["frames"], log=log)
        frames = assemble(X, support, v2p[pid], fps)
        n_ok = sum(fr["joints_flat"] is not None for fr in frames)
        n_meas = sum(fr["joints_flat"] is not None and not fr["interpolated"] for fr in frames)
        log(f"    {pid}: pose {100*n_ok/n:.1f}% (camera-supported {100*n_meas/n:.1f}%)")
        players.append((pid, frames))
    meta = dict(v2["meta"]["pose_variant"], name=VARIANT,
                fit={"sigma_px": SIGMA_PX, "sigma_bone": SIGMA_BONE, "window": WINDOW, "f_scale": F_SCALE})
    doc = pv.make_doc(out_clip, fps, n, players, bundle, "posefit_multiview_skeleton", meta)
    pv.carry_over(base, doc, log=log)
    for p in doc["players"]:
        fill_facing(p["frames"], fps, sign=frames_sign(p["frames"], analysis_sign(bundle["cams"])))
    errs = schema.validate_analysis(doc)
    if errs:
        raise ValueError("posefit doc invalid:\n  " + "\n  ".join(errs[:10]))
    out = os.path.join(analysis_dir, f"{out_clip}_{VARIANT}.analysis.json")
    text = json.dumps(doc, allow_nan=False)      # Unity's JsonUtility can't parse NaN
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"    wrote {out}")
    pv.write_racket_sidecars(analysis_dir, out_clip, f"{out_clip}_{VARIANT}", base, doc, log=log)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clip", action="append", help="output clip id with a pose_v2 cache")
    b.add_argument("--all", action="store_true")
    b.add_argument("--analysis-dir", default=pv.DEFAULT_ANALYSIS_DIR)
    args = ap.parse_args(argv)
    clips = (list(pv.CLIPS) + ["doubles_2sides_4angle_2cam"]) if args.all else (args.clip or [])
    for c in clips:
        if not os.path.exists(os.path.join(pv.CACHE_DIR, f"{c}.npz")):
            print(f"[posefit] {c}: no pose_v2 cache, run `analysis.pose_v2 build` first — skipped")
            continue
        build(c, args.analysis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
