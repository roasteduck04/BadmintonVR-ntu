"""pose24 canonical estimator — multi-view robust skeleton fit with biomechanical priors (spec § 4;
review H2, H4, M4-M7).

Per player, overlapping 60-frame windows are solved with scipy least_squares (soft_l1, sparse Jacobian)
over the 3D positions of the 35 fit keypoints (kpset), in the court world (Y-down, metres):

  reprojection   sqrt(w) * px residual / 8 px           every view, calibrated score w
  limb lengths   (|Xa - Xb| - L) / 1 cm                 per subject, left = right (fixed within a pass)
  pelvis, head   hip width / 0.5 cm; head points / 1.5 cm
  trunk (soft)   shoulder width, same-side shoulder-hip / 6 cm — own mean per side, no symmetry (H4)
  acceleration   2nd difference / sigma(keypoint)       wrists + knuckles 0.15 m/frame^2 (smash-safe)
  joint limits   hinge outside angles.LIMITS / 5 deg    (numeric local Jacobian)
  floor          ankles >= 7 cm, heels/toes >= 1.5 cm   / 1 cm
  foot contact   horizontal velocity of a planted heel/toe / 0.5 cm per frame
  anchor         to the robust triangulation where it exists / 0.3 m
  depth prior    optional (monocular fit): (X_k - hip_mid) . camera axis - d, / sigma  (after the limits;
                 unit axis, sigma > 0, rows with a non-finite d dropped)
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

from analysis import pose_v2 as pv
from analysis.pose_fit import _fill_init
from analysis.pose24 import angles, kpset

SIGMA_PX = 8.0
F_SCALE = 2.0
SIGMA_LIMB = 0.010
SIGMA_PELVIS = 0.005
SIGMA_HEAD = 0.015
SIGMA_TRUNK = 0.06
SIGMA_FLOOR = 0.01
SIGMA_CONTACT = 0.005
SIGMA_ANCHOR = 0.3
SIGMA_LIMIT_DEG = 5.0
FD_H = 1e-4
UP_DOWN = np.array([0.0, -1.0, 0.0])        # "up" in the Y-down court world

# acceleration sigma per fit keypoint (m per frame^2 at 60 fps; scaled by (60/fps)^2)
ACC = np.array([0.02, 0.02, 0.02, 0.02, 0.02,                   # nose, eyes, ears
                0.012, 0.012, 0.05, 0.05, 0.15, 0.15,            # shoulders, elbows, wrists
                0.01, 0.01, 0.025, 0.025, 0.035, 0.035,          # hips, knees, ankles
                0.04, 0.04, 0.04, 0.04, 0.04, 0.04,              # toes, heels
                0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  # finger MCPs
                0.02, 0.02, 0.02, 0.02])                         # jaw, eye centres, nose tip
FLOOR_MIN = {15: 0.07, 16: 0.07, 17: 0.015, 18: 0.015, 19: 0.015, 20: 0.015, 21: 0.015, 22: 0.015}

# (group, keypoint pairs sharing one length, sigma)
EDGES = [
    ("upper_arm", [(5, 7), (6, 8)], SIGMA_LIMB), ("forearm", [(7, 9), (8, 10)], SIGMA_LIMB),
    ("thigh", [(11, 13), (12, 14)], SIGMA_LIMB), ("shank", [(13, 15), (14, 16)], SIGMA_LIMB),
    ("ankle_big_toe", [(15, 17), (16, 20)], SIGMA_LIMB), ("ankle_small_toe", [(15, 18), (16, 21)], SIGMA_LIMB),
    ("ankle_heel", [(15, 19), (16, 22)], SIGMA_LIMB),
    ("wrist_index_mcp", [(9, 23), (10, 27)], SIGMA_LIMB), ("wrist_middle_mcp", [(9, 24), (10, 28)], SIGMA_LIMB),
    ("wrist_ring_mcp", [(9, 25), (10, 29)], SIGMA_LIMB), ("wrist_pinky_mcp", [(9, 26), (10, 30)], SIGMA_LIMB),
    ("mcp_spread", [(23, 26), (27, 30)], SIGMA_LIMB),
    ("hip_width", [(11, 12)], SIGMA_PELVIS),
    ("nose_eye", [(0, 1), (0, 2)], SIGMA_HEAD), ("eye_ear", [(1, 3), (2, 4)], SIGMA_HEAD),
    ("ear_ear", [(3, 4)], SIGMA_HEAD), ("ear_jaw", [(3, 31), (4, 31)], SIGMA_HEAD),
    ("nose_jaw", [(0, 31)], SIGMA_HEAD), ("nose_nose_tip", [(0, 34)], SIGMA_HEAD),
    ("eye_eye_centre", [(1, 32), (2, 33)], SIGMA_HEAD),
]
# soft trunk edges: one length each (no left/right symmetry — lateral flexion changes them)
SOFT = {"shoulder_width": (5, 6), "left_shoulder_hip": (5, 11), "right_shoulder_hip": (6, 12)}


def edge_arrays(lengths):
    """lengths {"edges": {group: m}, "soft": {name: m}} -> E (n,2), EL (n,), ES (n,) sigma."""
    E, EL, ES = [], [], []
    for name, pairs, sigma in EDGES:
        L = lengths["edges"].get(name)
        if L is not None and np.isfinite(L):
            for a, b in pairs:
                E.append((a, b)), EL.append(L), ES.append(sigma)
    for name, (a, b) in SOFT.items():
        L = lengths["soft"].get(name)
        if L is not None and np.isfinite(L):
            E.append((a, b)), EL.append(L), ES.append(SIGMA_TRUNK)
    return np.array(E, int).reshape(-1, 2), np.array(EL, float), np.array(ES, float)


class WindowProblem24:
    """One window of T frames; variables = the free frames' 35 keypoints (Y-down court world)."""

    def __init__(self, P, uv, w, X_init, free, edges, anchor, contact, sgn, fps=60.0, use_limits=True, depth=None):
        self.P, self.T, self.K = P, X_init.shape[0], X_init.shape[1]
        self.X0 = X_init.copy()
        self.free = free
        self.fidx = -np.ones(self.T, int)
        self.fidx[free] = np.arange(free.sum())
        self.nv = int(free.sum()) * self.K * 3
        self.tf = np.nonzero(free)[0]
        o = (w >= pv.W_MIN) & np.isfinite(uv).all(-1) & free[:, None, None]
        self.ot, self.ok, self.ov = np.nonzero(o)
        self.ouv = uv[self.ot, self.ok, self.ov]
        self.osw = np.sqrt(w[self.ot, self.ok, self.ov]) / SIGMA_PX
        self.E, self.EL, self.ES = edges
        self.acc = ACC * (60.0 / fps) ** 2
        tri = np.arange(1, self.T - 1)
        self.at = tri[free[tri - 1] | free[tri] | free[tri + 1]]
        self.floor_k = np.array(sorted(FLOOR_MIN))
        self.floor_h = np.array([FLOOR_MIN[k] for k in self.floor_k])
        c = contact[:-1] & contact[1:] & (free[:-1] | free[1:])[:, None]
        ct, ci = np.nonzero(c)
        self.ct, self.ck = ct, np.array(kpset.CONTACT, int)[ci]
        am = np.isfinite(anchor).all(-1) & free[:, None]
        self.an_t, self.an_k = np.nonzero(am)
        self.an_x = anchor[self.an_t, self.an_k]
        self.sgn = sgn
        self.limits = angles.LIMITS if use_limits else []
        self.n_limit_res = len(self.limits) * len(self.tf)
        if depth is None:
            self.dt, self.dk, self.dd = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
            self.daxis, self.dsig = np.zeros(3), 1.0
        else:
            dt, dk, dd = np.asarray(depth["t"], int), np.asarray(depth["k"], int), np.asarray(depth["d"], float)
            axis, sig = np.asarray(depth["axis"], float), float(depth["sigma"])
            if not sig > 0:
                raise ValueError(f"depth prior sigma must be > 0 m, got {sig!r}")
            n = np.linalg.norm(axis)
            if not (np.isfinite(n) and n > 0):
                raise ValueError(f"depth prior axis must be a finite non-zero vector, got {axis!r}")
            keep = free[dt] & (dk != 11) & (dk != 12) & np.isfinite(dd)
            self.dt, self.dk, self.dd = dt[keep], dk[keep], dd[keep]
            self.daxis, self.dsig = axis / n, sig
        self.nd = len(self.dt)

    def var(self, t, k):
        return (self.fidx[t] * self.K + k) * 3

    def unpack(self, x):
        X = self.X0.copy()
        X[self.free] = x.reshape(-1, self.K, 3)
        return X

    def _limit_values(self, Xf):
        return [fn(Xf, self.sgn, UP_DOWN) for _, fn, _, _, _ in self.limits]

    def _project_obs(self, X):
        """Projection of every observation's keypoint: (p (n,3) homogeneous, z (n,) safe depth, weight (n,) =
        osw in front of the camera, 0 behind it — a clamped depth there would flip the pixel's sign)."""
        Xo = X[self.ot, self.ok]
        p = np.einsum("nij,nj->ni", self.P[self.ov], np.c_[Xo, np.ones(len(Xo))])
        front = p[:, 2] > 1e-6
        return p, np.where(front, p[:, 2], 1.0), np.where(front, self.osw, 0.0)

    def residuals(self, x):
        X = self.unpack(x)
        out = []
        p, z, sw = self._project_obs(X)
        out.append(((p[:, :2] / z[:, None] - self.ouv) * sw[:, None]).ravel())
        Xf = X[self.tf]
        d = Xf[:, self.E[:, 0]] - Xf[:, self.E[:, 1]]
        out.append(((np.linalg.norm(d, axis=-1) - self.EL) / self.ES).ravel())
        a = X[self.at - 1] - 2 * X[self.at] + X[self.at + 1]
        out.append((a / self.acc[None, :, None]).ravel())
        out.append((np.maximum(0.0, Xf[:, self.floor_k, 1] + self.floor_h) / SIGMA_FLOOR).ravel())
        dv = X[self.ct + 1, self.ck][:, [0, 2]] - X[self.ct, self.ck][:, [0, 2]]
        out.append((dv / SIGMA_CONTACT).ravel())
        out.append(((X[self.an_t, self.an_k] - self.an_x) / SIGMA_ANCHOR).ravel())
        for (_, _, lo, hi, _), v in zip(self.limits, self._limit_values(Xf)):
            r = np.maximum(0.0, v - hi) + np.maximum(0.0, lo - v)
            out.append(np.nan_to_num(r) / SIGMA_LIMIT_DEG)
        if self.nd:
            hm = 0.5 * (X[self.dt, 11] + X[self.dt, 12])
            out.append(((X[self.dt, self.dk] - hm) @ self.daxis - self.dd) / self.dsig)
        return np.concatenate(out)

    def jacobian(self, x):
        X = self.unpack(x)
        rows, cols, vals = [], [], []
        r0 = 0
        # reprojection
        Pv = self.P[self.ov]
        p, z, sw = self._project_obs(X)
        u = p[:, :2] / z[:, None]
        vbase = self.var(self.ot, self.ok)
        n = len(self.ot)
        for c in range(2):
            Jc = (Pv[:, c, :3] - u[:, c:c + 1] * Pv[:, 2, :3]) / z[:, None] * sw[:, None]
            rr = r0 + np.arange(n) * 2 + c
            for i in range(3):
                rows.append(rr), cols.append(vbase + i), vals.append(Jc[:, i])
        r0 += 2 * n
        # edges
        Xf = X[self.tf]
        d = Xf[:, self.E[:, 0]] - Xf[:, self.E[:, 1]]
        nrm = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        g = d / nrm / self.ES[None, :, None]
        rr = r0 + np.arange(len(self.tf) * len(self.E)).reshape(len(self.tf), len(self.E))
        for side, sign in ((0, 1.0), (1, -1.0)):
            vb = self.var(self.tf[:, None], self.E[None, :, side])
            for i in range(3):
                rows.append(rr.ravel()), cols.append((vb + i).ravel()), vals.append((sign * g[..., i]).ravel())
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
            rows.append(base[m].ravel()), cols.append(vb.ravel()), vals.append(val.ravel())
        r0 += na * self.K * 3
        # floor (Y index 1; height = -Y)
        nf = len(self.floor_k)
        yf = Xf[:, self.floor_k, 1] + self.floor_h
        rr = r0 + np.arange(len(self.tf) * nf).reshape(len(self.tf), nf)
        vb = self.var(self.tf[:, None], self.floor_k[None, :]) + 1
        rows.append(rr.ravel()), cols.append(vb.ravel())
        vals.append(np.where(yf > 0, 1.0 / SIGMA_FLOOR, 0.0).ravel())
        r0 += rr.size
        # contact: X and Z velocity of planted keypoints
        nc = len(self.ct)
        for j, c in enumerate((0, 2)):
            rr = r0 + np.arange(nc) * 2 + j
            for dt, sign in ((1, 1.0), (0, -1.0)):
                t = self.ct + dt
                m = self.free[t]
                rows.append(rr[m]), cols.append(self.var(t[m], self.ck[m]) + c)
                vals.append(np.full(int(m.sum()), sign / SIGMA_CONTACT))
        r0 += 2 * nc
        # anchor
        nan_ = len(self.an_t)
        for i in range(3):
            rows.append(r0 + np.arange(nan_) * 3 + i), cols.append(self.var(self.an_t, self.an_k) + i)
            vals.append(np.full(nan_, 1.0 / SIGMA_ANCHOR))
        r0 += 3 * nan_
        # joint limits: numeric derivatives of each angle wrt the keypoints it uses
        rr = r0 + np.arange(len(self.tf))
        for (_, fn, lo, hi, kps), v in zip(self.limits, self._limit_values(Xf)):
            act = np.where(v > hi, 1.0, np.where(v < lo, -1.0, 0.0))
            act[~np.isfinite(v)] = 0.0
            if act.any():
                for k in kps:
                    for c in range(3):
                        Xp, Xm = Xf.copy(), Xf.copy()
                        Xp[:, k, c] += FD_H
                        Xm[:, k, c] -= FD_H
                        dv = (fn(Xp, self.sgn, UP_DOWN) - fn(Xm, self.sgn, UP_DOWN)) / (2 * FD_H)
                        val = np.nan_to_num(act * dv / SIGMA_LIMIT_DEG)
                        nz = val != 0
                        rows.append(rr[nz]), cols.append(self.var(self.tf[nz], k) + c), vals.append(val[nz])
            rr = rr + len(self.tf)
        r0 += self.n_limit_res
        # depth prior (kept rows are free frames, non-hip keypoints)
        if self.nd:
            rr = r0 + np.arange(self.nd)
            for k, coef in ((self.dk, 1.0), (11, -0.5), (12, -0.5)):
                vb = self.var(self.dt, k)
                for c in range(3):
                    rows.append(rr), cols.append(vb + c), vals.append(np.full(self.nd, coef * self.daxis[c] / self.dsig))
            r0 += self.nd
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(r0, self.nv))

    def solve(self, max_nfev=40):
        x0 = self.X0[self.free].ravel()
        res = least_squares(self.residuals, x0, jac=self.jacobian, method="trf", loss="soft_l1",
                            f_scale=F_SCALE, tr_solver="lsmr", max_nfev=max_nfev, x_scale=1.0)
        return self.unpack(res.x)


def _median_len(X, good, a, b, min_n=10):
    m = good[:, a] & good[:, b] & np.isfinite(X[:, a, 0]) & np.isfinite(X[:, b, 0])
    if m.sum() < min_n:
        return None
    d = np.linalg.norm(X[m, a] - X[m, b], axis=-1)
    med = float(np.median(d))
    return med, float(np.median(np.abs(d - med))), int(m.sum())


STATURE_PRIOR = (1.70, 0.10)       # ASSUMPTION every player, sex unknown (no player data yet)
SEGMENT_RATIO = {"thigh": 0.245, "shank": 0.246, "upper_arm": 0.186, "forearm": 0.146, "foot": 0.152}  # Winter 2009 / de Leva 1996
WINDOW, OVERLAP = 60, 10
MAX_GAP = 20
GOOD_PX = 20.0
CONTACT_H, CONTACT_V, CONTACT_MIN_RUN = 0.03, 0.3, 3


def settings() -> dict:
    """The fit constants, recorded in every doc's meta.pose24.fit."""
    return {"sigma_px": SIGMA_PX, "f_scale": F_SCALE, "sigma_limb_m": SIGMA_LIMB, "sigma_pelvis_m": SIGMA_PELVIS,
            "sigma_head_m": SIGMA_HEAD, "sigma_trunk_m": SIGMA_TRUNK, "sigma_floor_m": SIGMA_FLOOR,
            "sigma_contact_m_per_frame": SIGMA_CONTACT, "sigma_anchor_m": SIGMA_ANCHOR,
            "sigma_limit_deg": SIGMA_LIMIT_DEG, "acc_m_per_frame2_at_60fps": ACC.tolist(),
            "floor_min_m": {kpset.NAMES[k]: h for k, h in FLOOR_MIN.items()},
            "limits_deg": {n: [lo, hi] for n, _, lo, hi, _ in angles.LIMITS},
            "window": WINDOW, "overlap": OVERLAP, "max_gap": MAX_GAP, "good_px": GOOD_PX,
            "contact": {"height_m": CONTACT_H, "speed_mps": CONTACT_V, "min_run": CONTACT_MIN_RUN},
            "stature_prior_m": list(STATURE_PRIOR), "segment_ratios": SEGMENT_RATIO}


def estimate_lengths(X, good, prior=STATURE_PRIOR, min_n=10):
    """Per-subject lengths from frames where both endpoints are `good` (median; left/right averaged for
    EDGES, own value for SOFT). Limb segments are shrunk towards ratio x stature, where stature combines
    the segment-implied value with the prior; with hundreds of frames the prior barely moves them (weak)."""
    edges, spread, soft = {}, {}, {}
    for name, pairs, _ in EDGES:
        st = [s for s in (_median_len(X, good, a, b, min_n) for a, b in pairs) if s]
        if st:
            edges[name] = float(np.mean([s[0] for s in st]))
            spread[name] = (float(np.mean([s[1] for s in st])), sum(s[2] for s in st))
    for name, (a, b) in SOFT.items():
        s = _median_len(X, good, a, b, min_n)
        if s:
            soft[name] = s[0]
    foot = [s for s in (_median_len(X, good, 19, 17, min_n), _median_len(X, good, 22, 20, min_n)) if s]
    seg = {k: edges.get(k) for k in ("thigh", "shank", "upper_arm", "forearm")}
    seg["foot"] = float(np.mean([s[0] for s in foot])) if foot else None
    implied = {k: v / SEGMENT_RATIO[k] for k, v in seg.items() if v}
    mu0, sd0 = prior
    h_meas = None
    h_post = mu0
    if implied:
        h = np.array(list(implied.values()))
        h_meas = float(np.median(h))
        sd_meas = 1.4826 * float(np.median(np.abs(h - h_meas))) / np.sqrt(len(h)) + 0.02
        h_post = (h_meas / sd_meas ** 2 + mu0 / sd0 ** 2) / (1 / sd_meas ** 2 + 1 / sd0 ** 2)
    for k in ("thigh", "shank", "upper_arm", "forearm"):
        if k in edges:
            mad, n = spread[k]
            sd_s = 1.4826 * mad / np.sqrt(max(n / 10.0, 1.0)) + 0.002
            sd_p = SEGMENT_RATIO[k] * sd0
            edges[k] = float((edges[k] / sd_s ** 2 + SEGMENT_RATIO[k] * h_post / sd_p ** 2)
                             / (1 / sd_s ** 2 + 1 / sd_p ** 2))
    return {"edges": edges, "soft": soft,
            "stature": {"prior_m": list(prior), "measured_m": h_meas, "posterior_m": float(h_post),
                        "implied_m": implied}}


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(int), 0])
    return np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]


def _clean_runs(m, min_run):
    m = m.copy()
    for a, b in zip(*_runs(~m)):          # close short gaps inside a contact
        if 0 < a and b < len(m) and b - a < min_run:
            m[a:b] = True
    for a, b in zip(*_runs(m)):           # drop short contacts
        if b - a < min_run:
            m[a:b] = False
    return m


def detect_contact(X, fps, h_max=CONTACT_H, v_max=CONTACT_V, min_run=CONTACT_MIN_RUN):
    """(F, 6) bool: heel/toe keypoints (kpset.CONTACT) planted — below h_max and slower than v_max
    (both neighbouring frame speeds), runs of >= min_run frames. X is Y-down."""
    P = X[:, kpset.CONTACT]
    h = -P[..., 1]
    d = np.linalg.norm(np.diff(P[..., [0, 2]], axis=0), axis=-1) * fps
    pad = np.full((1, P.shape[1]), np.nan)
    v = np.fmax(np.vstack([pad, d]), np.vstack([d, pad]))
    with np.errstate(invalid="ignore"):
        c = (h < h_max) & (v < v_max)
    return np.stack([_clean_runs(c[:, j], min_run) for j in range(c.shape[1])], axis=1)


def support_segments(w, min_kp=6, max_gap=MAX_GAP, min_len=3):
    """[(start, end)] frame ranges to fit: frames where >= min_kp body keypoints are seen in >= 1 view,
    merged across gaps of <= max_gap unseen frames."""
    seen = ((w[:, :17] >= pv.W_MIN).any(axis=-1)).sum(axis=-1) >= min_kp
    idx = np.nonzero(seen)[0]
    if not len(idx):
        return []
    cuts = np.nonzero(np.diff(idx) > max_gap + 1)[0]
    starts, ends = np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]] + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def _residuals(P, X, uv, w):
    pr, depth = pv.project(P, X)
    r = np.linalg.norm(pr - uv, axis=-1)
    with np.errstate(invalid="ignore"):
        inlier = (w >= pv.W_MIN) & (r < GOOD_PX) & (depth > 0)
    return r, inlier


def _p99_speed(Xw, fps):
    s = np.linalg.norm(np.diff(Xw, axis=0), axis=-1) * fps
    s = s[np.isfinite(s)]
    return float(np.percentile(s, 99)) if len(s) else None


def _fit_segment(P, uv, w, init, edges, anchor, contact, sgn, fps, use_limits, depth=None):
    """depth (optional): WindowProblem24 depth rows over segment frames; sliced + re-indexed per window."""
    seg = init.copy()
    n, start = len(seg), 0
    while start < n:
        stop = min(start + WINDOW, n)
        lo = max(0, start - 2)
        free = np.zeros(stop - lo, bool)
        free[start - lo:] = True
        dwin = None
        if depth is not None:
            dt = np.asarray(depth["t"], int)
            m = (dt >= lo) & (dt < stop)
            dwin = {"axis": depth["axis"], "sigma": depth["sigma"], "t": dt[m] - lo,
                    "k": np.asarray(depth["k"])[m], "d": np.asarray(depth["d"])[m]}
        prob = WindowProblem24(P, uv[lo:stop], w[lo:stop], seg[lo:stop], free, edges, anchor[lo:stop],
                               contact[lo:stop], sgn, fps, use_limits=use_limits, depth=dwin)
        seg[lo:stop] = prob.solve()
        if stop == n:
            break
        start = stop - OVERLAP
    return seg


def fit_player(bundle, pid, passes=2, use_contact=True, use_limits=True, log=print) -> dict:
    """Fit one player's 35 keypoints over the clip. Pass 1 starts from the robust triangulation (gaps
    interpolated); later passes re-estimate limb lengths and foot contacts from the previous fit."""
    views = bundle["views"]
    V = len(views)
    P = pv.proj_mats(bundle["cams"], views)
    uv, w = kpset.player_obs(bundle, pid)
    F, K = w.shape[:2]
    fps = float(bundle["fps"])
    Xr, conf, _, inl = pv.triangulate_batch(bundle["cams"], views, uv.reshape(-1, V, 2), w.reshape(-1, V))
    Xr, conf, inl = Xr.reshape(F, K, 3), conf.reshape(F, K), inl.reshape(F, K, V)
    ok = np.isfinite(Xr).all(-1)
    world = kpset.world_sign(bundle["cams"])
    sgn = kpset.forward_sign(Xr, default=world)
    if sgn != world:
        log(f"    WARNING {pid}: body handedness {sgn:+.0f} disagrees with the calibration's {world:+.0f} "
            "(mostly L/R-swapped detections?)")
    min_views = 3 if V >= 3 else 2
    good = (conf > (0.7 if V >= 3 else 0.45)) & (inl.sum(-1) >= min_views)
    lengths = estimate_lengths(Xr, good)
    contact = detect_contact(Xr, fps) if use_contact else np.zeros((F, len(kpset.CONTACT)), bool)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hip = np.nanmean(np.where(ok[:, [11, 12], None], Xr[:, [11, 12]], np.nan), axis=1)
        rest = np.nanmedian(Xr - hip[:, None], axis=0)
    rest = np.where(np.isfinite(rest), rest, kpset.rest_offsets_down(sgn))
    segments = [(s, e) for s, e in support_segments(w) if (ok[s:e, 11] | ok[s:e, 12]).any()]
    if not segments:
        seen = ((w[:, :17] >= pv.W_MIN).sum(axis=1) >= 6).sum(axis=0)          # frames per view with a body
        log(f"    WARNING {pid}: no frame where two views see the hips - no 3D pose (frames seen per view: "
            + ", ".join(f"{v} {int(n)}" for v, n in zip(views, seen)) + "); out of frame in the others?")
    X = np.full((F, K, 3), np.nan)
    for p in range(passes):
        edges = edge_arrays(lengths)
        for s, e in segments:
            init = _fill_init(Xr[s:e], ok[s:e], rest) if p == 0 else X[s:e].copy()
            X[s:e] = _fit_segment(P, uv[s:e], w[s:e], init, edges, Xr[s:e], contact[s:e], sgn, fps, use_limits)
        if p < passes - 1:
            _, inlier = _residuals(P, X, uv, w)
            lengths = estimate_lengths(X, inlier.sum(-1) >= min_views)
            if use_contact:
                contact = detect_contact(X, fps)
    r, inlier = _residuals(P, X, uv, w)
    peak = {"raw": _p99_speed(Xr[:, [9, 10]], fps), "fit": _p99_speed(X[:, [9, 10]], fps)}
    log(f"    {pid}: fit {len(segments)} segment(s), p99 wrist speed raw {peak['raw']} / fit {peak['fit']} m/s, "
        f"contact frames {int(contact.any(axis=1).sum())}")
    return {"X": X, "uv": uv, "w": w, "resid": r, "inlier": inlier, "raw": Xr, "contact": contact,
            "lengths": lengths, "sgn": sgn, "segments": segments, "peak_wrist_mps": peak}
