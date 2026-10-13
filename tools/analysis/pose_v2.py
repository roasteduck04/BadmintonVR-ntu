"""Pose v2 — robust multi-view triangulation, written as a pose variant next to the base clip.

    PYTHONPATH=tools python -m analysis.pose_v2 build --all
    PYTHONPATH=tools python -m analysis.pose_v2 build --clip doubles_2sides_4angle \
        --views backright_41,frontleft_41 --out-clip doubles_2sides_4angle_2cam

Same inputs as `build_real.py multi` (RTMPose wholebody 2D + court calibration + seeds), but:
  * RTMPose SimCC scores (raw 0-10) are mapped to a calibrated probability (fit against
    held-out-view residuals), so every confidence gate actually gates;
  * cameras are synced *before* tracking (track -> sync -> re-track on shifted streams);
  * the tracker can re-acquire a LOST player, damps velocity while coasting, and places
    detections without visible ankles from the hip height instead of the body centroid;
  * views whose calibration disagrees with the others (held-out residual > 40 px) are dropped;
  * joints (body, feet, face, hands) are triangulated in batch: normalised-coordinate
    weighted DLT, consensus seed pair with cheirality + ray-angle checks, Huber
    Gauss-Newton refinement on reprojection error;
  * no causal filters: spike + bone-length outlier rejection, short-gap fill, zero-phase
    Butterworth per joint group, strict median bone lengths; interpolated frames are flagged.

Output: `<clip>_posev2.analysis.json` (58-joint schema v5) in the analysis dir, with the base
clip's racket/shuttle/rallies carried over (racket re-expressed against the new pelvis) and
`<clip>_posev2.racket_<variant>.json` for every racket sidecar the base clip has. The tracked,
synced 2D observations are cached in data/pose_cache/<clip>.npz for `analysis.pose_fit`.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, filtfilt

from analysis import appearance, schema, wholebody
from analysis import player_tracker as pt
from analysis.conventions import analysis_sign
from analysis.facing import fill_facing, frames_sign

DEFAULT_ANALYSIS_DIR = os.path.join("Assets", "StreamingAssets", "analysis")
CACHE_DIR = os.path.join("data", "pose_cache")
VARIANT = "posev2"

# clip -> (camera suffix, mode, seeds clip). Views default to the base doc's cameras.
CLIPS = {
    "singles_1x": ("11", "singles", "singles_1x"),
    "singles_05x": ("051", "singles", "singles_05x"),
    "doubles_1side": ("21", "doubles", "doubles_1side"),
    "doubles_2sides": ("31", "doubles", "doubles_2sides"),
    "doubles_2sides_4angle": ("41", "doubles", "doubles_2sides_4angle"),
}
RACKET_SIDECARS = ("rigid", "bootstrap", "guided", "imu")

# --- score calibration: P(held-out residual < 25 px | raw SimCC score), fit 2026-09-24 on
# singles_05x + doubles_2sides (205k keypoints): 2.0 -> 0.07, 2.5 -> 0.3, 3.0 -> 0.7, 3.5 -> 0.92
SCORE_MID = 2.75
SCORE_WIDTH = 0.30

SCORE_GATE = 0.35       # tracker: mean calibrated body score of a detection
W_MIN = 0.3             # triangulation: minimum calibrated keypoint score
INLIER_PX = 25.0
MAX_ERR_PX = 25.0
ERR_SCALE = 30.0
MIN_RAY_DEG = 2.0
HUBER_PX = 8.0
VIEW_BAD_PX = 40.0
REACQ_MAX = 6.0
COAST_DAMP = 0.9
COURT_X_MAX = 3.6       # doubles sideline 3.05 m + lunge margin (base tracker: 4.0)
COURT_Z_MAX = 8.0       # baseline 6.7 m + 1.3 m behind (7.7 cut players retrieving deep)
APPEAR_WEIGHT = 3.0     # metres of assignment cost per unit Hellinger colour distance
APPEAR_ALPHA = 0.1
APPEAR_CLEAR = 1.5      # update a colour model only when no other player is this close
HIP_HEIGHT = 0.9
SHOULDER_HEIGHT = 1.4
MAX_GAP = 20

# wholebody indices triangulated: body+feet, jaw, nose tip, eye rings, both hands
TRI_IDX = list(range(23)) + [31, 53] + list(range(59, 71)) + list(range(91, 133))

# zero-phase low-pass cutoff (Hz) per 58-joint name group
_CUTOFF = {"core": 6.0, "knee_elbow": 8.0, "foot": 10.0, "hand": 12.0}
_SPIKE_M = {"core": 0.15, "knee_elbow": 0.25, "foot": 0.25, "hand": 0.35}


def norm_score(raw):
    """Raw RTMPose SimCC score -> calibrated probability in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(raw, dtype=np.float64) - SCORE_MID) / SCORE_WIDTH))


def joint_group(name: str) -> str:
    if any(k in name for k in ("wrist", "thumb", "index", "middle", "ring", "pinky")):
        return "hand"
    if any(k in name for k in ("ankle", "toe", "heel")):
        return "foot"
    if any(k in name for k in ("knee", "elbow")):
        return "knee_elbow"
    return "core"


# ---------------------------------------------------------------------------
# cameras + projection
# ---------------------------------------------------------------------------

def load_cams(cam_paths: dict) -> dict:
    """{view: path} -> {view: (K, R, t)}."""
    out = {}
    for v, p in cam_paths.items():
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        out[v] = (np.array(d["K"], float), np.array(d["R"], float), np.array(d["t"], float).reshape(3))
    return out


def proj_mats(cams: dict, views: list) -> np.ndarray:
    return np.stack([cams[v][0] @ np.hstack([cams[v][1], cams[v][2][:, None]]) for v in views])


def project(P: np.ndarray, X: np.ndarray):
    """P (V,3,4), X (...,3) -> uv (...,V,2), depth (...,V)."""
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    p = np.einsum("vij,...j->...vi", P, Xh)
    depth = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / depth[..., None]
    return uv, depth


def _dlt(Pn: np.ndarray, xn: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted DLT in normalised camera coordinates. Pn (V,3,4) = [R|t]; xn (N,V,2); w (N,V)."""
    r1 = xn[..., 0:1] * Pn[None, :, 2, :] - Pn[None, :, 0, :]
    r2 = xn[..., 1:2] * Pn[None, :, 2, :] - Pn[None, :, 1, :]
    A = np.concatenate([r1, r2], axis=1) * np.concatenate([w, w], axis=1)[..., None]
    A = np.nan_to_num(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return Xh[:, :3] / Xh[:, 3:4]


def _gauss_newton(P, uv, w, X, iters=3):
    """Huber-weighted GN on pixel reprojection error. w (N,V) = 0 for unused views."""
    for _ in range(iters):
        Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        p = np.einsum("vij,nj->nvi", P, Xh)
        z = p[..., 2]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        pred = p[..., :2] / z[..., None]
        res = np.nan_to_num(pred - uv)
        r = np.linalg.norm(res, axis=-1)
        hub = np.where(r <= HUBER_PX, 1.0, HUBER_PX / np.maximum(r, 1e-9))
        ww = w * hub
        Ju = (P[None, :, 0, :3] - pred[..., 0:1] * P[None, :, 2, :3]) / z[..., None]
        Jv = (P[None, :, 1, :3] - pred[..., 1:2] * P[None, :, 2, :3]) / z[..., None]
        JTJ = (np.einsum("nv,nvi,nvj->nij", ww, Ju, Ju) + np.einsum("nv,nvi,nvj->nij", ww, Jv, Jv))
        JTr = (np.einsum("nv,nvi,nv->ni", ww, Ju, res[..., 0]) + np.einsum("nv,nvi,nv->ni", ww, Jv, res[..., 1]))
        JTJ += np.eye(3)[None] * 1e-6
        ok = np.isfinite(JTJ).all(axis=(1, 2)) & np.isfinite(JTr).all(axis=1)
        dX = np.zeros_like(X)
        dX[ok] = np.linalg.solve(JTJ[ok], JTr[ok][..., None])[..., 0]
        X = X - dX
    return X


def triangulate_batch(cams: dict, views: list, uv: np.ndarray, w: np.ndarray,
                      w_min=W_MIN, inlier_px=INLIER_PX):
    """Robust multi-view triangulation of N points at once.

    uv (N,V,2) pixels, w (N,V) calibrated scores (NaN/0 = unseen).
    Returns X (N,3) (NaN = failed), conf (N,), err (N,) px, inliers (N,V) bool.
    """
    N, V = w.shape
    P = proj_mats(cams, views)
    Rt = np.stack([np.hstack([cams[v][1], cams[v][2][:, None]]) for v in views])
    Kinv = np.stack([np.linalg.inv(cams[v][0]) for v in views])
    C = np.stack([-cams[v][1].T @ cams[v][2] for v in views])
    uvh = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], axis=-1)
    xn = np.einsum("vij,nvj->nvi", Kinv, uvh)[..., :2]
    w = np.nan_to_num(w)
    valid = (w >= w_min) & np.isfinite(uv).all(axis=-1)

    best_score = np.full(N, -np.inf)
    best_inl = np.zeros((N, V), bool)
    for a in range(V):
        for b in range(a + 1, V):
            m = valid[:, a] & valid[:, b]
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            X = _dlt(Rt[[a, b]], xn[idx][:, [a, b]], w[idx][:, [a, b]])
            pr, depth = project(P, X)
            r = np.linalg.norm(pr - uv[idx], axis=-1)
            inl = valid[idx] & (depth > 0) & (r < inlier_px)
            ok = (depth[:, a] > 0) & (depth[:, b] > 0) & np.isfinite(X).all(axis=1)
            da, db = X - C[a], X - C[b]
            cosang = np.einsum("ni,ni->n", da, db) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1) + 1e-12)
            narrow = cosang > np.cos(np.radians(MIN_RAY_DEG))
            score = (w[idx] * inl).sum(axis=1) - 1e-3 * np.where(inl, r, 0).sum(axis=1) / inlier_px
            score = np.where(narrow, score - 0.5, score)
            score = np.where(ok & inl[:, a] & inl[:, b], score, -np.inf)
            better = score > best_score[idx]
            best_score[idx[better]] = score[better]
            best_inl[idx[better]] = inl[better]

    X = np.full((N, 3), np.nan)
    conf = np.zeros(N)
    err = np.full(N, np.nan)
    has = best_inl.sum(axis=1) >= 2
    if not has.any():
        return X, conf, err, best_inl
    idx = np.nonzero(has)[0]
    inl = best_inl[idx]
    Xi = _dlt(Rt, np.nan_to_num(xn[idx]), w[idx] * inl)
    Xi = _gauss_newton(P, np.nan_to_num(uv[idx]), w[idx] * inl, Xi)
    # one re-check of the inlier set around the refined point
    pr, depth = project(P, Xi)
    r = np.linalg.norm(pr - uv[idx], axis=-1)
    inl2 = valid[idx] & (depth > 0) & (r < inlier_px)
    changed = (inl2 != inl).any(axis=1) & (inl2.sum(axis=1) >= 2)
    if changed.any():
        c = np.nonzero(changed)[0]
        Xi[c] = _gauss_newton(P, np.nan_to_num(uv[idx[c]]), w[idx[c]] * inl2[c], Xi[c], iters=2)
        inl[c] = inl2[c]
        pr, depth = project(P, Xi)
        r = np.linalg.norm(pr - uv[idx], axis=-1)
    front = np.where(inl, depth > 0, True).all(axis=1)
    wi = w[idx] * inl
    e = (np.where(inl, r, 0) * wi).sum(axis=1) / np.maximum(wi.sum(axis=1), 1e-9)
    n_inl = inl.sum(axis=1)
    good = front & (n_inl >= 2) & np.isfinite(Xi).all(axis=1) & (e <= MAX_ERR_PX)
    wmean = wi.sum(axis=1) / np.maximum(n_inl, 1)
    c = wmean * np.minimum(1.0, n_inl / 3.0) * np.exp(-e / ERR_SCALE)
    X[idx[good]] = Xi[good]
    conf[idx[good]] = np.clip(c[good], 0.0, 1.0)
    err[idx[good]] = e[good]
    out_inl = np.zeros((N, V), bool)
    out_inl[idx[good]] = inl[good]
    return X, conf, err, out_inl


# ---------------------------------------------------------------------------
# 2D loading + tracking
# ---------------------------------------------------------------------------

def load_kp2d(path: str):
    """kp2d.json -> (frames, fps, idx): frames[f] = list of (133,3) float32 arrays, scores
    calibrated, detections with mean calibrated body score < SCORE_GATE dropped; idx[f] = the
    kept detections' original indices (for appearance lookup)."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    frames, idx = [], []
    for fr in doc["frames"]:
        dets, ids = [], []
        for d, det in enumerate(fr):
            k = np.asarray(det["keypoints"], dtype=np.float32)
            k[:, 2] = norm_score(k[:, 2])
            if k[:17, 2].mean() >= SCORE_GATE:
                dets.append(k)
                ids.append(d)
        frames.append(dets)
        idx.append(ids)
    return frames, float(doc.get("fps", 60.0)), idx


def _shift(frames: list, k: int) -> list:
    if k > 0:
        return frames[k:]
    if k < 0:
        return [[] for _ in range(-k)] + frames
    return frames


def _plane_point(u, v, K, R, t, height):
    """Intersect the pixel ray with the horizontal plane `height` m above the floor
    (court world is Y-down: the plane is Y = -height). Returns (X, Z) or None."""
    ray = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    c = -R.T @ t
    if abs(ray[1]) < 1e-10:
        return None
    s = (-height - c[1]) / ray[1]
    if s < 0:
        return None
    p = c + s * ray
    return float(p[0]), float(p[2])


def court_pos(kps, K, R, t, s_min=0.3):
    """Ground position of a detection: ankles on the floor, else hips at hip height,
    else shoulders at shoulder height."""
    for (a, b), h in (((15, 16), 0.0), ((11, 12), HIP_HEIGHT), ((5, 6), SHOULDER_HEIGHT)):
        pts = [kps[j, :2] for j in (a, b) if kps[j, 2] >= s_min]
        if pts:
            u, v = np.mean(pts, axis=0)
            return _plane_point(u, v, K, R, t, h)
    return None


class TrackerV2:
    """PlayerTracker with pre-synced streams, per-view clothing-colour models, tighter court
    bounds, LOST re-acquisition and coast damping.

    kp2d: {view: frames} from `load_kp2d`; shifts: {view: int} so that stream v's frame f+k
    is the reference frame f; kp_idx/appear: detection indices + `appearance.load_clip`."""

    def __init__(self, cams: dict, kp2d: dict, seeds_views: dict, mode: str, fps: float,
                 shifts: dict | None = None, kp_idx: dict | None = None, appear: dict | None = None):
        self.mode = mode
        self.players = ["P1", "P2"] if mode == "singles" else ["P1", "P2", "P3", "P4"]
        self.views = sorted(kp2d)
        self.cams = cams
        shifts = shifts or {}
        self.shifts = {v: int(shifts.get(v, 0)) for v in self.views}
        self.kp2d = {v: _shift(kp2d[v], self.shifts[v]) for v in self.views}
        self.kp_idx = {v: _shift(kp_idx[v], self.shifts[v]) for v in self.views} if kp_idx else None
        self.appear = appear or {}
        self.app_model = {}
        self.frame_count = min(len(self.kp2d[v]) for v in self.views)
        self.fps = fps
        self.states: dict[str, pt.PlayerState] = {}
        self.sides: dict[str, str] = {}
        for pid in self.players:
            cand = []
            for v in self.views:
                seed = seeds_views.get(v, {}).get("seeds", {}).get(pid)
                if seed is None:
                    continue
                K, R, t = cams[v]
                p = pt.pixel_to_ground(seed[0], seed[1], K, R, t)
                if p is not None:
                    cand.append(p)
            if not cand:
                continue
            x, z = np.median(np.array(cand), axis=0)
            self.sides[pid] = "near" if z < 0 else "far"
            self.states[pid] = pt.PlayerState(pid, x=float(x), z=float(z), side=self.sides[pid], fps=fps)
        self.reacquired = 0

    def _dets(self, f):
        out = {}
        for v in self.views:
            K, R, t = self.cams[v]
            dets = []
            frame = self.kp2d[v][f] if f < len(self.kp2d[v]) else []
            ids = self.kp_idx[v][f] if (self.kp_idx and f < len(self.kp_idx[v])) else [None] * len(frame)
            app = self.appear.get(v, {})
            for k, di in zip(frame, ids):
                cp = court_pos(k, K, R, t)
                if cp is None or abs(cp[0]) > COURT_X_MAX or abs(cp[1]) > COURT_Z_MAX:
                    continue
                dets.append({"pos": np.array(cp), "kps": k, "sig": pt.BodySignature.from_keypoints(k),
                             "completeness": float((k[:17, 2] >= 0.3).mean()), "pid": None,
                             "app": app.get((f + self.shifts[v], di)) if di is not None else None})
            out[v] = dets
        return out

    def _cost(self, d, ps, pos_max, v=None, same_half=False):
        pos_cost = float(np.linalg.norm(d["pos"] - ps.pos))
        if same_half and np.sign(d["pos"][1]) != (-1.0 if ps.side == "near" else 1.0):
            return 1e6
        body = ps.signature.distance(d["sig"]) if (d["sig"] is not None and ps.signature is not None) else 0.0
        total = pos_cost + pt.BODY_WEIGHT * body
        if total > pos_max:
            return 1e6
        model = self.app_model.get((ps.pid, v))
        if d.get("app") is not None and model is not None:
            total += APPEAR_WEIGHT * appearance.hellinger(d["app"], model)
        return total

    def _update_appearance(self, f, dets):
        pos = {p: self.states[p].pos for p in self.states if self.states[p].state != "LOST"}
        for v in self.views:
            for d in dets[v]:
                pid = d["pid"]
                if pid is None or d.get("app") is None or d["completeness"] < 0.5:
                    continue
                if any(np.linalg.norm(d["pos"] - q) < APPEAR_CLEAR for p, q in pos.items() if p != pid):
                    continue
                key = (pid, v)
                m = self.app_model.get(key)
                self.app_model[key] = d["app"].copy() if m is None else (1 - APPEAR_ALPHA) * m + APPEAR_ALPHA * d["app"]

    def run(self) -> dict:
        pids = list(self.states)
        out = {pid: {v: [None] * self.frame_count for v in self.views} for pid in self.players}
        self.track = {pid: np.full((self.frame_count, 2), np.nan) for pid in pids}
        self.events = []
        for f in range(self.frame_count):
            for pid in pids:
                ps = self.states[pid]
                if ps.state != "LOST":
                    ps.predict()
            dets = self._dets(f)
            for v in self.views:
                dv = dets[v]
                active = [p for p in pids if self.states[p].state != "LOST"]
                if dv and active:
                    cost = np.array([[self._cost(d, self.states[p], self.states[p].threshold(), v) for p in active]
                                     for d in dv])
                    for r, c in zip(*linear_sum_assignment(cost)):
                        if cost[r, c] < 1e6:
                            dv[r]["pid"] = active[c]
                    asg = [d for d in dv if d["pid"]]
                    for i in range(len(asg)):
                        for j in range(i + 1, len(asg)):
                            if np.linalg.norm(asg[i]["pos"] - asg[j]["pos"]) < pt.OCCLUSION_PROXIMITY:
                                for d in (asg[i], asg[j]):
                                    if d["completeness"] < 0.5:
                                        d["pid"] = None
                lost = [p for p in pids if self.states[p].state == "LOST"]
                free = [d for d in dv if d["pid"] is None]
                if lost and free:
                    cost = np.array([[self._cost(d, self.states[p], REACQ_MAX, v, same_half=True) for p in lost]
                                     for d in free])
                    for r, c in zip(*linear_sum_assignment(cost)):
                        if cost[r, c] < 1e6:
                            free[r]["pid"] = lost[c]
            self._update_appearance(f, dets)
            pos, wts, kps = {}, {}, {}
            for v in self.views:
                for d in dets[v]:
                    if d["pid"] is None:
                        continue
                    pos.setdefault(d["pid"], []).append(d["pos"])
                    wts.setdefault(d["pid"], []).append(max(d["completeness"], 0.05))
                    kps.setdefault(d["pid"], {})[v] = d["kps"]
            for pid in pids:
                ps = self.states[pid]
                if pid in pos:
                    m = np.average(np.array(pos[pid]), axis=0, weights=wts[pid])
                    if ps.state == "LOST":
                        self.events.append((f, pid, "reacquired", ps.pos.round(2).tolist(), m.round(2).tolist()))
                        ps._x = np.array([m[0], m[1], 0.0, 0.0])
                        ps._P = np.diag([pt.MEASURE_NOISE ** 2] * 2 + [1.0, 1.0])
                        ps.state, ps.coast_count = "ACTIVE", 0
                        self.reacquired += 1
                    else:
                        ps.update(m)
                    sample = next(iter(kps[pid].values()))
                    if ps.signature is None:
                        ps.signature = pt.BodySignature.from_keypoints(sample)
                    else:
                        ps.signature.update(sample)
                    for v, k in kps[pid].items():
                        out[pid][v][f] = k
                else:
                    was = ps.state
                    ps.coast()
                    ps._x[2:] *= COAST_DAMP
                    if ps.state == "LOST" and was != "LOST":
                        self.events.append((f, pid, "lost", ps.pos.round(2).tolist(), None))
                self.track[pid][f] = ps.pos
        return out


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------

def dense(frames: list, n: int, n_kpt=133) -> np.ndarray:
    """list of (133,3) arrays / None -> (n,133,3) float32, NaN where unseen."""
    D = np.full((n, n_kpt, 3), np.nan, np.float32)
    for f in range(min(n, len(frames))):
        if frames[f] is not None:
            D[f] = frames[f]
    return D


def sample_at(D: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Linear interpolation of a dense track at fractional frames (score = min of neighbours)."""
    f0 = np.floor(f).astype(int)
    a = (f - f0)[:, None]
    n = len(D)
    ok0 = (f0 >= 0) & (f0 < n)
    ok1 = (f0 + 1 >= 0) & (f0 + 1 < n)
    A = np.full((len(f),) + D.shape[1:], np.nan, np.float32)
    B = A.copy()
    A[ok0] = D[f0[ok0]]
    B[ok1] = D[f0[ok1] + 1]
    B = np.where((a[..., None] < 1e-6), A, B)
    out = A.copy()
    out[..., :2] = (1 - a[..., None]) * A[..., :2] + a[..., None] * B[..., :2]
    out[..., 2] = np.minimum(A[..., 2], B[..., 2])
    return out


def _pair_error(Pa, Pb, A, B, s_min=0.5):
    """Median two-view reprojection error of matched keypoints A, B (N,3)."""
    m = (A[:, 2] >= s_min) & (B[:, 2] >= s_min) & np.isfinite(A[:, 0]) & np.isfinite(B[:, 0])
    if m.sum() < 20:
        return np.inf
    rows = []
    for P, X in ((Pa, A[m]), (Pb, B[m])):
        rows += [X[:, 0:1] * P[2] - P[0], X[:, 1:2] * P[2] - P[1]]
    M = np.stack(rows, axis=1)
    M /= np.linalg.norm(M, axis=2, keepdims=True) + 1e-12
    _, _, Vt = np.linalg.svd(M)
    Xh = Vt[:, -1, :]
    e = []
    for P, X in ((Pa, A[m]), (Pb, B[m])):
        p = Xh @ P.T
        e.append(np.hypot(p[:, 0] / p[:, 2] - X[:, 0], p[:, 1] / p[:, 2] - X[:, 1]))
    return float(np.median((e[0] + e[1]) / 2))


SYNC_JOINTS = [5, 6, 11, 12, 13, 14, 15, 16]


def estimate_offsets(cams, dense_by_pid: dict, views: list, ref: str, max_offset=30, step=5, log=print):
    """{view: fractional offset} so that view frame f+off matches reference frame f."""
    offsets, errors = {ref: 0.0}, {ref: 0.0}
    for v in views:
        if v == ref:
            continue
        Pa, Pb = proj_mats(cams, [ref])[0], proj_mats(cams, [v])[0]
        best = (np.inf, 0.0, None)
        for pid, D in dense_by_pid.items():
            Dr, Dv = D.get(ref), D.get(v)
            if Dr is None or Dv is None:
                continue
            n = min(len(Dr), len(Dv))
            fr = np.arange(max_offset, n - max_offset - 1, step)
            if len(fr) < 10:
                continue
            A = Dr[fr][:, SYNC_JOINTS].reshape(-1, 3)
            for off in range(-max_offset, max_offset + 1):
                B = Dv[fr + off][:, SYNC_JOINTS].reshape(-1, 3)
                e = _pair_error(Pa, Pb, A, B)
                if e < best[0]:
                    best = (e, float(off), pid)
        err, off, pid = best
        if pid is not None:
            Dr, Dv = dense_by_pid[pid][ref], dense_by_pid[pid][v]
            n = min(len(Dr), len(Dv))
            fr = np.arange(max_offset, n - max_offset - 1, step)
            A = Dr[fr][:, SYNC_JOINTS].reshape(-1, 3)
            for cand in np.round(np.arange(off - 1.0, off + 1.0001, 0.1), 2):
                B = sample_at(Dv, fr + cand)[:, SYNC_JOINTS].reshape(-1, 3)
                e = _pair_error(Pa, Pb, A, B)
                if e < err:
                    err, off = e, float(cand)
        offsets[v], errors[v] = off, err
        log(f"    sync {v} vs {ref}: {off:+.1f} frames ({err:.1f}px, {pid})")
    return offsets, errors


# ---------------------------------------------------------------------------
# observations -> world joints
# ---------------------------------------------------------------------------

def track_and_sync(clip_cfg: dict, log=print) -> dict:
    """Load 2D, sync, track, drop bad views. Returns an observation bundle."""
    views = list(clip_cfg["views"])
    cams_all = load_cams({v: clip_cfg["cam_paths"][v] for v in views})
    kp2d, kp_idx, fps = {}, {}, 60.0
    for v in views:
        kp2d[v], fps, kp_idx[v] = load_kp2d(clip_cfg["kp2d_paths"][v])
    appear = appearance.load_clip(clip_cfg["clip"])
    log(f"    appearance features: {', '.join(v for v in views if v in appear) or 'none (run analysis.appearance)'}")
    with open(clip_cfg["seeds_path"], "r", encoding="utf-8") as fh:
        seeds_views = json.load(fh).get("views", {})
    mode = clip_cfg["mode"]
    excluded = {}
    # the timeline stays pinned to the base doc's reference view (review 2 L4): if it is dropped, the new
    # reference's offsets are re-expressed on the old timeline, since the base racket / shuttle / rallies are
    # copied by frame index
    time_ref = clip_cfg.get("time_ref") if clip_cfg.get("time_ref") in views else views[0]
    offsets = {}
    while True:
        ref = time_ref if time_ref in views else views[0]
        pin = 0.0 if ref == time_ref else float(offsets[ref])
        cams = {v: cams_all[v] for v in views}
        tr = TrackerV2(cams, {v: kp2d[v] for v in views}, seeds_views, mode, fps, kp_idx=kp_idx, appear=appear)
        raw = tr.run()
        n0 = tr.frame_count
        dense0 = {pid: {v: dense(raw[pid][v], n0) for v in views} for pid in raw}
        offsets, sync_err = estimate_offsets(cams, dense0, views, ref, log=log)
        offsets = {v: o + pin for v, o in offsets.items()}
        shifts = {v: int(np.round(o)) for v, o in offsets.items()}
        tr = TrackerV2(cams, {v: kp2d[v] for v in views}, seeds_views, mode, fps, shifts=shifts,
                       kp_idx=kp_idx, appear=appear)
        matched = tr.run()
        n = int(clip_cfg.get("frame_count") or tr.frame_count)
        obs = {}
        for pid in matched:
            obs[pid] = {}
            for v in views:
                D = dense(matched[pid][v], n)
                r = offsets[v] - shifts[v]
                obs[pid][v] = sample_at(D, np.arange(n) + r) if abs(r) > 1e-6 else D
        if len(views) < 3:
            break
        health = view_health(cams, views, obs)
        log("    view held-out residual: " + ", ".join(f"{v} {e:.1f}px" for v, e in health.items()))
        bad = [v for v, e in health.items() if e > VIEW_BAD_PX]
        if not bad or len(views) - len(bad) < 2:
            break
        for v in bad:
            excluded[v] = round(float(health[v]), 1)
            log(f"    dropping {v}: calibration disagrees with the other views ({health[v]:.0f}px)")
        views = [v for v in views if v not in bad]
    log(f"    tracker re-acquisitions: {tr.reacquired}")
    return {"views": views, "cams": {v: cams_all[v] for v in views}, "obs": obs, "fps": fps,
            "frame_count": n, "offsets": offsets, "sync_err": sync_err, "excluded": excluded,
            "sides": dict(tr.sides), "mode": mode, "time_ref_view": time_ref}


def obs_arrays(bundle: dict, pid: str, idx=TRI_IDX):
    """(F,J,V,2) pixels and (F,J,V) calibrated scores for one player."""
    views = bundle["views"]
    D = np.stack([bundle["obs"][pid][v][:, idx] for v in views], axis=2)  # (F,J,V,3)
    return D[..., :2].astype(np.float64), np.nan_to_num(D[..., 2]).astype(np.float64)


def view_health(cams, views, obs, step=5, body=tuple(range(17))) -> dict:
    """Median held-out-view residual (px) of confident body keypoints, per view."""
    out = {}
    for k, v in enumerate(views):
        rest = [u for u in views if u != v]
        errs = []
        for pid in obs:
            D = np.stack([obs[pid][u][::step][:, list(body)] for u in views], axis=2)
            uv, w = D[..., :2].reshape(-1, len(views), 2), np.nan_to_num(D[..., 2]).reshape(-1, len(views))
            ri = [views.index(u) for u in rest]
            X, _, _, _ = triangulate_batch({u: cams[u] for u in rest}, rest, uv[:, ri], w[:, ri])
            pr, depth = project(proj_mats(cams, [v]), X)
            r = np.linalg.norm(pr[:, 0] - uv[:, k], axis=-1)
            m = np.isfinite(r) & (w[:, k] >= 0.5) & (depth[:, 0] > 0)
            errs.append(r[m])
        e = np.concatenate(errs) if errs else np.array([])
        out[v] = float(np.median(e)) if len(e) > 50 else float("nan")
    return out


def triangulate_player(bundle: dict, pid: str):
    """World keypoints (Y-up, metres) for one player: X (F,133,3), conf (F,133), err (F,133)."""
    uv, w = obs_arrays(bundle, pid)
    F, J, V, _ = uv.shape
    X, conf, err, _ = triangulate_batch(bundle["cams"], bundle["views"], uv.reshape(F * J, V, 2),
                                        w.reshape(F * J, V))
    W = np.zeros((F, 133, 3))
    Cf = np.zeros((F, 133))
    E = np.full((F, 133), np.nan)
    Xr = X.reshape(F, J, 3)
    ok = np.isfinite(Xr).all(axis=-1)
    W[:, TRI_IDX] = np.where(ok[..., None], Xr, 0.0)
    Cf[:, TRI_IDX] = np.where(ok, conf.reshape(F, J), 0.0)
    E[:, TRI_IDX] = err.reshape(F, J)
    W[..., 1] *= -1.0   # court world is Y-down; analysis is Y-up
    return W, Cf, E


def to_joints58(W133, C133):
    F = len(W133)
    J = np.full((F, schema.NUM_JOINTS, 3), np.nan)
    C = np.zeros((F, schema.NUM_JOINTS))
    for f in range(F):
        j, c = wholebody.wholebody133_to_smpl58(W133[f], C133[f])
        J[f], C[f] = j, c
    J[C < 0.05] = np.nan
    return J, C


# ---------------------------------------------------------------------------
# temporal processing (all zero-phase / non-causal)
# ---------------------------------------------------------------------------

def _bfs_bones(parents):
    kids = {j: [] for j in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            kids[p].append(j)
    order, q = [], deque(j for j, p in enumerate(parents) if p < 0)
    while q:
        n = q.popleft()
        order.append(n)
        q.extend(kids[n])
    return [j for j in order if parents[j] >= 0]


def median_bone_lengths(J, C, parents, c_min=0.3):
    L = {}
    for j, p in enumerate(parents):
        if p < 0:
            continue
        m = (C[:, j] >= c_min) & (C[:, p] >= c_min) & np.isfinite(J[:, j, 0]) & np.isfinite(J[:, p, 0])
        if m.sum() >= 10:
            L[j] = float(np.median(np.linalg.norm(J[m, j] - J[m, p], axis=1)))
    return L


# joints wholebody133_to_smpl58 derives from others (mid-points / interpolations): never an independent measurement
DERIVED_JOINTS = ("pelvis", "neck", "spine1", "spine2", "spine3", "left_collar", "right_collar")


def reject_outliers(J, C, parents, groups, fps=60.0, derived=None):
    """NaN out bone-length outliers (>35% off the clip median) and single-frame spikes. A bone with a derived
    end (`DERIVED_JOINTS`, e.g. the pelvis = hip mid-point whose confidence is min(hips)) blames its measured end;
    bones between two derived joints are skipped (review 2 L13)."""
    J = J.copy()
    if derived is None:
        derived = {schema.JOINT_NAMES.index(n) for n in DERIVED_JOINTS} if J.shape[1] == schema.NUM_JOINTS else set()
    L = median_bone_lengths(J, C, parents)
    for j, p in enumerate(parents):
        if j not in L or L[j] < 0.02:
            continue
        if j in derived and p in derived:
            continue
        d = np.linalg.norm(J[:, j] - J[:, p], axis=1)
        bad = np.isfinite(d) & (np.abs(d - L[j]) > 0.35 * L[j])
        if p in derived:
            worse_child = np.ones(len(J), bool)
        elif j in derived:
            worse_child = np.zeros(len(J), bool)
        else:
            worse_child = C[:, j] <= C[:, p]
        J[bad & worse_child, j] = np.nan
        J[bad & ~worse_child, p] = np.nan
    scale = (60.0 / fps) ** 2
    for j in range(J.shape[1]):
        x = J[:, j]
        ok = np.isfinite(x[:, 0])
        idx = np.nonzero(ok)[0]
        if len(idx) < 5:
            continue
        xs = x[idx]
        d = np.full(len(idx), 0.0)
        d[1:-1] = np.linalg.norm(xs[1:-1] - 0.5 * (xs[:-2] + xs[2:]), axis=1)
        thr = _SPIKE_M[groups[j]] * scale
        nb = np.maximum(np.r_[0, d[:-1]], np.r_[d[1:], 0])
        spike = (d > thr) & (d > 1.5 * nb)
        J[idx[spike], j] = np.nan
    return J


def fill_gaps(J, C, max_gap=MAX_GAP):
    """Linear fill of per-joint gaps <= max_gap frames. Returns (J, C, filled mask)."""
    J, C = J.copy(), C.copy()
    F, NJ, _ = J.shape
    filled = np.zeros((F, NJ), bool)
    for j in range(NJ):
        ok = np.isfinite(J[:, j, 0])
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            continue
        gaps = np.nonzero(np.diff(idx) > 1)[0]
        for g in gaps:
            a, b = idx[g], idx[g + 1]
            if b - a - 1 > max_gap:
                continue
            t = (np.arange(a + 1, b) - a) / (b - a)
            J[a + 1:b, j] = J[a, j] + t[:, None] * (J[b, j] - J[a, j])
            C[a + 1:b, j] = 0.8 * min(C[a, j], C[b, j])
            filled[a + 1:b, j] = True
    return J, C, filled


def zero_phase(J, groups, fps=60.0):
    """Butterworth (2nd order, forward-backward) per joint on contiguous valid runs."""
    J = J.copy()
    coef = {g: butter(2, min(fc / (fps / 2.0), 0.99)) for g, fc in _CUTOFF.items()}
    for j in range(J.shape[1]):
        b, a = coef[groups[j]]
        ok = np.isfinite(J[:, j, 0])
        edges = np.diff(np.r_[0, ok.astype(int), 0])
        for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            if e - s >= 4:
                J[s:e, j] = filtfilt(b, a, J[s:e, j], axis=0, padlen=min(9, e - s - 1))
    return J


def enforce_bones(J, L, parents):
    """Strict median bone lengths, root-to-leaf, keeping each bone's direction."""
    J = J.copy()
    for j in _bfs_bones(parents):
        if j not in L:
            continue
        p = parents[j]
        d = J[:, j] - J[:, p]
        n = np.linalg.norm(d, axis=1)
        ok = np.isfinite(n) & (n > 1e-6)
        J[ok, j] = J[ok, p] + d[ok] * (L[j] / n[ok])[:, None]
    return J


def postprocess(J, C, fps=60.0, court=(5.0, 8.0)):
    """Raw world joints (F,58,3) -> (J, C, frame_interpolated) after outlier rejection,
    gap fill, zero-phase smoothing and strict bone lengths."""
    parents = schema.PARENTS
    groups = [joint_group(n) for n in schema.JOINT_NAMES]
    J = J.copy()
    out = ~np.isfinite(J[:, 0, 0]) | (np.abs(J[:, 0, 0]) > court[0]) | (np.abs(J[:, 0, 2]) > court[1])
    J[out] = np.nan
    J[(J[..., 1] < -0.3) | (J[..., 1] > 3.2)] = np.nan
    C = np.where(np.isfinite(J[..., 0]), C, 0.0)
    L = median_bone_lengths(J, C, parents)
    J = reject_outliers(J, C, parents, groups, fps)
    C = np.where(np.isfinite(J[..., 0]), C, 0.0)
    J, C, filled = fill_gaps(J, C)
    J = zero_phase(J, groups, fps)
    J = enforce_bones(J, L, parents)
    return J, C, filled[:, 0]


def frames_from_world(J, C, interp):
    """World joints -> analysis frames (body-local joints_flat + court position)."""
    frames = []
    for f in range(len(J)):
        if not np.isfinite(J[f, 0]).all():
            frames.append({"joints_flat": None, "position": None, "interpolated": False})
            continue
        pel = J[f, 0]
        loc = J[f] - np.array([pel[0], 0.0, pel[2]])
        miss = ~np.isfinite(loc).all(axis=1)
        loc[miss] = [0.0, pel[1], 0.0]
        c = np.where(miss, 0.0, C[f])
        flat = np.empty(schema.NUM_JOINTS * 4)
        flat[0::4], flat[1::4], flat[2::4], flat[3::4] = loc[:, 0], loc[:, 1], loc[:, 2], c
        frames.append({"joints_flat": [round(float(x), 5) for x in flat],
                       "position": [round(float(pel[0]), 4), round(float(pel[2]), 4)],
                       "interpolated": bool(interp[f])})
    return frames


# ---------------------------------------------------------------------------
# document + sidecars
# ---------------------------------------------------------------------------

def _shift_racket(r, dx, dz):
    if not r:
        return None
    r = dict(r)
    for k in ("grip", "head", "side"):
        if r.get(k) is not None:
            r[k] = [round(r[k][0] + dx, 5), r[k][1], round(r[k][2] + dz, 5)]
    return r


def _pos_delta(base_frames, new_frames, f):
    b = base_frames[f].get("position") if f < len(base_frames) else None
    n = new_frames[f].get("position")
    if b is None or n is None:
        return None
    return b[0] - n[0], b[1] - n[1]


CARRY_GATE_M = 0.5     # carry-over: max court distance (m) between a base and a variant pelvis in one frame


def doc_time_ref(doc: dict):
    """The view a doc's frame index is synced to: meta.pose_variant.time_ref_view (pose24, posev2), else the
    sorted-first camera (build_real and the old posev2 docs reference views[0] of the sorted views); None if the
    doc names no camera."""
    meta = doc.get("meta") or {}
    ref = (meta.get("pose_variant") or {}).get("time_ref_view")
    if ref:
        return ref
    names = sorted(c["name"] for c in meta.get("cameras") or [] if c.get("name"))
    return names[0] if names else None


def same_timeline(base: dict, time_ref, log=print) -> bool:
    """False (with a log line) when the base doc is synced to another reference view than this build: then its
    racket / shuttle / rallies must not be copied by frame index (review 1 L13). Unknown references are trusted."""
    ref = doc_time_ref(base)
    if ref is None or time_ref is None or ref == time_ref:
        return True
    log(f"    WARNING: base doc is synced to {ref} but this build to {time_ref} (reference view dropped?): "
        f"racket / shuttle / rallies not carried")
    return False


def _ident(x):
    return x


def match_frames(base: dict, doc: dict, gate_m: float = CARRY_GATE_M) -> dict:
    """Identity-safe base -> variant player match, per frame: {doc id: [base id or None per doc frame]}.

    In every frame the base and doc players that have a position (pelvis court XZ) are paired by a Hungarian
    assignment on their court distance; pairs further apart than gate_m stay unmatched. A base tracker identity
    swap (the doubles_2sides P3 / P4 swap) or a base player the tracker lost is therefore never carried onto
    another body: matching by player id or by one clip-constant map re-attached rackets metres from the hand."""
    out = {p["id"]: [None] * len(p["frames"]) for p in doc["players"]}
    n = max((len(p["frames"]) for p in doc["players"]), default=0)
    for f in range(n):
        bi, bx, di, dx = [], [], [], []
        for src, ids, xs in ((base["players"], bi, bx), (doc["players"], di, dx)):
            for p in src:
                pos = p["frames"][f].get("position") if f < len(p["frames"]) else None
                if pos is not None and len(pos) == 2 and all(v is not None for v in pos):
                    ids.append(p["id"])
                    xs.append(pos)
        if not bi or not di:
            continue
        cost = np.linalg.norm(np.asarray(dx, float)[:, None, :] - np.asarray(bx, float)[None, :, :], axis=-1)
        cost = np.where(np.isfinite(cost), cost, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= gate_m:
                out[di[i]][f] = bi[j]
    return out


def _match_counts(ids):
    """[base id or None] -> {base id: frames, ..., "unmatched": frames} in order of first appearance."""
    out = {}
    for b in ids:
        k = "unmatched" if b is None else b
        out[k] = out.get(k, 0) + 1
    return out


def carry_rackets(base: dict, doc: dict, match: dict | None = None, clean=_ident, log=print, gate_m=CARRY_GATE_M):
    """Per-frame racket (re-expressed against the new pelvis), grip and stroke from the base player matched to
    each doc player in that frame (`match_frames`); unmatched frames get None. Returns the match."""
    match = match if match is not None else match_frames(base, doc, gate_m)
    bp = {p["id"]: p for p in base["players"]}
    for p in doc["players"]:
        ids = match.get(p["id"]) or [None] * len(p["frames"])
        for f, fr in enumerate(p["frames"]):
            b = bp.get(ids[f])
            dd = _pos_delta(b["frames"], p["frames"], f) if b is not None else None
            if dd is None:
                fr["racket"] = fr["grip"] = fr["stroke"] = None
                continue
            bf = b["frames"][f]
            fr["racket"] = clean(_shift_racket(bf.get("racket"), *dd))
            fr["grip"] = clean(bf.get("grip"))
            fr["stroke"] = clean(bf.get("stroke"))
        counts = _match_counts(ids)
        nf = max(len(ids), 1)
        log(f"    {p['id']} <- base " + ", ".join(f"{k} {100 * v / nf:.0f}%" for k, v in counts.items())
            + f" (per-frame pelvis match, gate {gate_m} m)")
    return match


def carry_over(base: dict, doc: dict, log=print, gate_m=CARRY_GATE_M):
    """Copy shuttle/rallies from the base doc; racket / grip / stroke per frame from the base player at the same
    place (identity-safe, `carry_rackets`). Returns the per-frame match. Nothing is carried (all-None match) when
    the base doc is synced to another reference view than the doc (`same_timeline`)."""
    if not same_timeline(base, doc_time_ref(doc), log=log):
        return carry_rackets(base, doc, match={p["id"]: [None] * len(p["frames"]) for p in doc["players"]},
                             log=lambda *a: None)
    for k in ("shuttle", "shuttle_legacy", "rallies"):
        if k in base:
            doc[k] = base[k]
    for key in ("racket", "shuttle"):
        doc["meta"]["producers"][key] = base["meta"].get("producers", {}).get(key)
    return carry_rackets(base, doc, log=log, gate_m=gate_m)


def write_racket_sidecars(analysis_dir, base_clip, out_clip, base, doc, log=print, match=None,
                          gate_m=CARRY_GATE_M):
    """Re-express every `<base_clip>.racket_<v>.json` against the new doc's pelvis: one sidecar player per doc
    player, in doc order (as Unity's RacketVariantDoc requires), each frame taken from the base player matched
    to it in that frame (`match_frames`, or `match` from carry_over); unmatched frames are None. which_hand is
    the most-matched base player's; `carry_over` records the per-player match counts. Without `match`, nothing is
    written when the base doc is on another timeline (`same_timeline`)."""
    if match is None:
        if not same_timeline(base, doc_time_ref(doc), log=log):
            return
        match = match_frames(base, doc, gate_m)
    bp = {p["id"]: p for p in base["players"]}
    for var in RACKET_SIDECARS:
        src = os.path.join(analysis_dir, f"{base_clip}.racket_{var}.json")
        if not os.path.exists(src):
            continue
        with open(src, "r", encoding="utf-8") as fh:
            sc = json.load(fh)
        tracks = {sp["id"]: sp for sp in sc["players"]}
        players, counts_all = [], {}
        for p in doc["players"]:
            ids = match.get(p["id"]) or [None] * len(p["frames"])
            frames, used = [], {}
            for f in range(len(p["frames"])):
                bid = ids[f]
                sp, b = tracks.get(bid), bp.get(bid)
                r = sp["frames"][f] if (sp is not None and f < len(sp["frames"])) else None
                dd = _pos_delta(b["frames"], p["frames"], f) if (r and b is not None) else None
                frames.append(_shift_racket(r, *dd) if dd is not None else None)
                if bid is not None and sp is not None:
                    used[bid] = used.get(bid, 0) + 1
            major = max(used, key=used.get) if used else None      # ties: first seen (dicts keep order)
            players.append({"id": p["id"], "which_hand": tracks[major].get("which_hand") if major else None,
                            "frames": frames})
            counts_all[p["id"]] = _match_counts(ids)
        sc["players"] = players
        sc["carry_over"] = {"method": "per-frame pelvis match", "gate_m": gate_m, "source_clip": base_clip,
                            "players": counts_all}
        sc["clip_id"] = out_clip
        sc["frame_count"] = doc["meta"]["frame_count"]
        dst = os.path.join(analysis_dir, f"{out_clip}.racket_{var}.json")
        text = json.dumps(sc, allow_nan=False)   # Unity's JsonUtility can't parse NaN
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text)
        log(f"    wrote {dst}")


def make_doc(clip_id, fps, n, players, bundle, producer, extra_meta):
    from analysis.build_real import build_real_doc_multi
    pdata = [{"id": pid, "side": bundle["sides"].get(pid, "near"), "pose_frames": fr} for pid, fr in players]
    cam_meta = []
    for v in bundle["views"]:
        K, R, t = bundle["cams"][v]
        cam_meta.append({"name": v, "K": K.tolist(), "R": R.tolist(), "t": t.tolist()})
    doc = build_real_doc_multi(clip_id, fps, n, pdata, cam_meta=cam_meta, mode=bundle["mode"])
    for p, (_, fr) in zip(doc["players"], players):
        for out_f, in_f in zip(p["frames"], fr):
            out_f["interpolated"] = in_f.get("interpolated", False)
    doc["skeleton"]["type"] = schema.SKELETON_TYPE
    doc["meta"]["producers"]["pose"] = producer
    doc["meta"]["producers"]["position"] = producer
    doc["meta"]["pose_variant"] = extra_meta
    return doc


def clip_config(clip, analysis_dir=DEFAULT_ANALYSIS_DIR, views=None, out_clip=None):
    base_clip = clip
    suffix, mode, seeds = CLIPS[clip]
    out_clip = out_clip or clip
    base_path = os.path.join(analysis_dir, f"{out_clip}.analysis.json")
    if not os.path.exists(base_path):
        base_path = os.path.join(analysis_dir, f"{base_clip}.analysis.json")
    with open(base_path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    if views is None:
        views = [c["name"] for c in base["meta"].get("cameras", [])] or \
                [f"{v}_{suffix}" for v in ("back", "backleft", "backright", "frontleft", "frontright")]
    views = sorted(views)
    time_ref = doc_time_ref(base)
    return {"clip": clip, "out_clip": out_clip, "base_path": base_path, "base": base, "mode": mode,
            "views": views, "time_ref": time_ref if time_ref in views else views[0],
            "frame_count": int(base["meta"]["frame_count"]),
            "cam_paths": {v: os.path.join("data", "calib", f"{v}_cam.json") for v in views},
            "kp2d_paths": {v: os.path.join("data", "pose2d", f"{v}_kp2d.json") for v in views},
            "seeds_path": os.path.join("data", "calib", f"{seeds}_seeds.json")}


def save_cache(bundle, out_clip):
    os.makedirs(CACHE_DIR, exist_ok=True)
    arrs = {}
    for pid in bundle["obs"]:
        uv, w = obs_arrays(bundle, pid, idx=list(range(133)))
        arrs[f"{pid}_uv"] = uv.astype(np.float32)
        arrs[f"{pid}_w"] = w.astype(np.float32)
    meta = {k: bundle[k] for k in ("views", "fps", "frame_count", "offsets", "excluded", "sides", "mode")}
    meta["time_ref_view"] = bundle.get("time_ref_view")
    meta["cams"] = {v: [np.asarray(x).tolist() for x in bundle["cams"][v]] for v in bundle["views"]}
    meta["pids"] = list(bundle["obs"])
    path = os.path.join(CACHE_DIR, f"{out_clip}.npz")
    np.savez_compressed(path, meta=json.dumps(meta), **arrs)
    return path


def load_cache(out_clip):
    z = np.load(os.path.join(CACHE_DIR, f"{out_clip}.npz"))
    meta = json.loads(str(z["meta"]))
    cams = {v: tuple(np.array(x) for x in meta["cams"][v]) for v in meta["views"]}
    obs = {}
    for pid in meta["pids"]:
        uv, w = z[f"{pid}_uv"], z[f"{pid}_w"]
        obs[pid] = {v: np.concatenate([uv[:, :, k], w[:, :, k, None]], axis=-1)
                    for k, v in enumerate(meta["views"])}
    return dict(meta, cams=cams, obs=obs)


def build(clip, analysis_dir=DEFAULT_ANALYSIS_DIR, views=None, out_clip=None, log=print):
    cfg = clip_config(clip, analysis_dir, views, out_clip)
    out_clip = cfg["out_clip"]
    log(f"[posev2] {out_clip}: {cfg['mode']}, views {', '.join(cfg['views'])}")
    bundle = track_and_sync(cfg, log=log)
    log(f"    cached {save_cache(bundle, out_clip)}")
    fps, n = bundle["fps"], bundle["frame_count"]
    players = []
    for pid in bundle["obs"]:
        W, C133, _ = triangulate_player(bundle, pid)
        J, C = to_joints58(W, C133)
        J, C, interp = postprocess(J, C, fps)
        frames = frames_from_world(J, C, interp)
        n_ok = sum(fr["joints_flat"] is not None for fr in frames)
        n_meas = sum(fr["joints_flat"] is not None and not fr["interpolated"] for fr in frames)
        hands = C[:, 22:52]
        log(f"    {pid}: pose {100*n_ok/n:.1f}% (measured {100*n_meas/n:.1f}%), "
            f"hand joints conf>0.3 {100*np.mean(hands > 0.3):.0f}%")
        players.append((pid, frames))
    meta = {"name": VARIANT, "views": bundle["views"], "excluded_views": bundle["excluded"],
            "time_ref_view": bundle["time_ref_view"],
            "offsets": {v: round(o, 2) for v, o in bundle["offsets"].items()},
            "score_calibration": {"mid": SCORE_MID, "width": SCORE_WIDTH}, "base": os.path.basename(cfg["base_path"])}
    doc = make_doc(out_clip, fps, n, players, bundle, "posev2_robust_triangulation", meta)
    carry_over(cfg["base"], doc, log=log)
    for p in doc["players"]:
        fill_facing(p["frames"], fps, sign=frames_sign(p["frames"], analysis_sign(bundle["cams"])))
    errs = schema.validate_analysis(doc)
    if errs:
        raise ValueError("posev2 doc invalid:\n  " + "\n  ".join(errs[:10]))
    out = os.path.join(analysis_dir, f"{out_clip}_{VARIANT}.analysis.json")
    text = json.dumps(doc, allow_nan=False)      # Unity's JsonUtility can't parse NaN
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"    wrote {out}")
    write_racket_sidecars(analysis_dir, out_clip, f"{out_clip}_{VARIANT}", cfg["base"], doc, log=log)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clip", action="append")
    b.add_argument("--all", action="store_true")
    b.add_argument("--views", default=None, help="comma-separated view subset")
    b.add_argument("--out-clip", default=None, help="output clip id (e.g. for a view subset)")
    b.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    args = ap.parse_args(argv)
    clips = list(CLIPS) if args.all else (args.clip or [])
    views = args.views.split(",") if args.views else None
    for c in clips:
        build(c, args.analysis_dir, views=views, out_clip=args.out_clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
