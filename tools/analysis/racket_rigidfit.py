"""Rigid racket model fitted to multi-view RacketVision keypoints (racket variant "rigid").

One 6DoF pose per frame is fitted to all five keypoints in all views at once, with the
left/right labels allowed to swap per view (they are image-side labels on a symmetric
head) and the grip softly anchored at the player's hand, so a single view is enough.
World frame = analysis world (court metres, Y-up); cameras are the Y-down PnP calibrations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

KP_NAMES = ("top", "bottom", "handle", "left", "right")
TOP, BOTTOM, HANDLE, LEFT, RIGHT = range(5)
_SWAP = np.array([TOP, BOTTOM, HANDLE, RIGHT, LEFT])
_RZ_PI = np.diag([-1.0, -1.0, 1.0])
_Y_FLIP = np.array([1.0, -1.0, 1.0])


@dataclass
class RacketTemplate:
    length: float = 0.665       # handle (butt) to top (tip)
    head_length: float = 0.25   # top to bottom (throat)
    head_width: float = 0.21    # left to right
    width_z: float = 0.545      # butt to the left/right line
    grip_z: float = 0.075       # butt to the hand

    def keypoints(self) -> np.ndarray:
        """Local keypoints (5,3): origin at the grip, shaft +z, across +x (left -> right)."""
        g, hw = self.grip_z, self.head_width / 2.0
        return np.array([
            [0.0, 0.0, self.length - g],
            [0.0, 0.0, self.length - self.head_length - g],
            [0.0, 0.0, -g],
            [-hw, 0.0, self.width_z - g],
            [hw, 0.0, self.width_z - g],
        ])

    def head_center(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.length - self.head_length / 2.0 - self.grip_z])

    def as_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


@dataclass
class FitConfig:
    sigma_px: float = 8.0          # keypoint noise scale (Cauchy f_scale is 1 sigma)
    min_score: float = 0.1         # keypoints below this are ignored
    min_kps: int = 3               # a view needs this many usable keypoints
    hand_sigma: float = 0.07       # grip-to-hand prior, metres
    prior_pos_sigma: float = 0.25  # previous-frame grip prior, metres
    prior_rot_sigma: float = 0.8   # previous-frame rotation prior, radians
    view_reject: float = 0.22      # drop a view whose error exceeds this fraction of racket length
    frame_accept: float = 0.14     # reject a frame whose mean error exceeds this
    tau: float = 0.035             # zero-phase spring smoothing, seconds
    gap_max: int = 10              # frames of slerp gap filling
    spike_deg: float = 45.0        # single-frame shaft spike threshold
    spike_m: float = 0.20          # single-frame grip spike threshold


class Camera:
    def __init__(self, K, R, t):
        self.K = np.asarray(K, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)

    def to_cam(self, X):
        return (np.asarray(X, dtype=np.float64) * _Y_FLIP) @ self.R.T + self.t

    def project(self, X):
        """World points (...,3) -> (pixels (...,2), depth (...))."""
        Xc = self.to_cam(X)
        z = Xc[..., 2]
        uvw = Xc @ self.K.T
        return uvw[..., :2] / np.where(np.abs(uvw[..., 2:3]) < 1e-9, 1e-9, uvw[..., 2:3]), z

    def px_per_m(self, X) -> float:
        z = float(self.to_cam(X)[2])
        return float(self.K[0, 0]) / max(z, 0.1)

    def P(self):
        """3x4 projection in the camera's own (Y-down) world."""
        return self.K @ np.hstack([self.R, self.t[:, None]])


@dataclass
class FitResult:
    x: np.ndarray            # [grip xyz, rotvec]
    cost: float
    err: float               # mean normalised keypoint error over used views
    per_view: dict           # view -> normalised error
    views: list
    choice: dict             # view -> candidate index used
    score: float             # mean keypoint score over used views

    @property
    def R(self):
        return Rotation.from_rotvec(self.x[3:]).as_matrix()

    @property
    def grip(self):
        return self.x[:3]


def _skew(k):
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def rotmat(rv) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3) + _skew(rv)
    K = _skew(np.asarray(rv) / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rotvec(R) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    if th > np.pi - 1e-4:
        return Rotation.from_matrix(R).as_rotvec()
    return w * (th / (2.0 * np.sin(th)))


def rotation_from_shaft(shaft, roll=0.0) -> np.ndarray:
    """Rotation whose local +z is `shaft`, rolled by `roll` radians about it."""
    z = np.asarray(shaft, dtype=np.float64)
    z = z / max(np.linalg.norm(z), 1e-12)
    ref = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    return np.column_stack([x, y, z])


def align_symmetry(R, R_ref) -> np.ndarray:
    """Pick R or R.Rz(pi) (the same physical racket) closest to R_ref."""
    R2 = R @ _RZ_PI
    return R if np.trace(R_ref.T @ R) >= np.trace(R_ref.T @ R2) else R2


def _fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)])


class RacketFitter:
    def __init__(self, cams: dict, template: RacketTemplate, cfg: FitConfig | None = None):
        self.cams = cams
        self.tpl = template
        self.cfg = cfg or FitConfig()
        self.kp = template.keypoints()

    def points(self, x):
        return x[:3] + self.kp @ rotmat(x[3:]).T

    def _usable(self, obs):
        return int((obs[:, 2] >= self.cfg.min_score).sum()) >= self.cfg.min_kps

    def view_error(self, x, view, obs):
        """(normalised error, swapped?) of one view's keypoints against pose x."""
        uv, _ = self.cams[view].project(self.points(x))
        w = np.where(obs[:, 2] >= self.cfg.min_score, obs[:, 2], 0.0)
        if w.sum() <= 0:
            return np.inf, False
        scale = max(float(np.linalg.norm(uv[TOP] - uv[HANDLE])), 30.0)
        d0 = np.linalg.norm(uv - obs[:, :2], axis=1)
        d1 = np.linalg.norm(uv - obs[_SWAP, :2], axis=1)
        e0 = float((w * d0).sum() / w.sum())
        e1 = float((w[_SWAP] * d1).sum() / w.sum())
        return (e1 / scale, True) if e1 < e0 else (e0 / scale, False)

    def _residual_fn(self, obs_list, hand, prior):
        cfg = self.cfg
        Ks = np.stack([self.cams[v].K for v, _ in obs_list])
        Rs = np.stack([self.cams[v].R for v, _ in obs_list])
        ts = np.stack([self.cams[v].t for v, _ in obs_list])
        uv_o = np.stack([o[:, :2] for _, o in obs_list])
        w = np.stack([np.sqrt(np.where(o[:, 2] >= cfg.min_score, o[:, 2], 0.0)) for _, o in obs_list])
        w = (w / cfg.sigma_px)[..., None]
        if prior is not None:
            p_g, p_Rt = prior[:3], rotmat(prior[3:]).T

        def fun(x):
            R = rotmat(x[3:])
            P = (x[:3] + self.kp @ R.T) * _Y_FLIP
            Xc = np.einsum("vij,kj->vki", Rs, P) + ts[:, None, :]
            uvw = np.einsum("vij,vkj->vki", Ks, Xc)
            z = uvw[..., 2:3]
            r = (uvw[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z) - uv_o) * w
            r[Xc[..., 2] <= 0.05] = 50.0
            out = [r.ravel()]
            if hand is not None:
                out.append((x[:3] - hand) / cfg.hand_sigma)
            if prior is not None:
                out.append((x[:3] - p_g) / cfg.prior_pos_sigma)
                out.append(rotvec(R @ p_Rt) / cfg.prior_rot_sigma)
            return np.concatenate(out)
        return fun

    def _solve(self, x0, obs_by_view, hand, prior, max_nfev):
        swap = {v: self.view_error(x0, v, o)[1] for v, o in obs_by_view.items()}
        res = None
        for _ in range(2):
            obs_list = [(v, o[_SWAP] if swap[v] else o) for v, o in obs_by_view.items()]
            res = least_squares(self._residual_fn(obs_list, hand, prior), x0, loss="cauchy",
                                f_scale=1.0, max_nfev=max_nfev,
                                x_scale=np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]))
            new = {v: self.view_error(res.x, v, o)[1] for v, o in obs_by_view.items()}
            if new == swap:
                break
            swap, x0 = new, res.x
        return res.x, float(res.cost)

    def _prior_aligned(self, prior, R0):
        Rp = align_symmetry(Rotation.from_rotvec(prior[3:]).as_matrix(), R0)
        return np.concatenate([prior[:3], Rotation.from_matrix(Rp).as_rotvec()])

    def _starts(self, obs_by_view, hand, prior, extra):
        starts = [prior.copy()] if prior is not None else []
        starts += [s for s in extra if s is not None]
        tri = self._triangulated_shaft(obs_by_view)
        if tri is not None:
            handle, top = tri
            shaft = top - handle
            if np.linalg.norm(shaft) > 0.2:
                g = hand if hand is not None else handle + shaft / np.linalg.norm(shaft) * self.tpl.grip_z
                for roll in np.deg2rad([0, 45, 90, 135]):
                    starts.append(np.concatenate([g, Rotation.from_matrix(rotation_from_shaft(shaft, roll)).as_rotvec()]))
        if not starts and hand is not None:
            for d in _fibonacci_dirs(16):
                for roll in (0.0, np.pi / 2):
                    starts.append(np.concatenate([hand, Rotation.from_matrix(rotation_from_shaft(d, roll)).as_rotvec()]))
        return starts

    def _triangulated_shaft(self, obs_by_view):
        from analysis.triangulate import triangulate_point
        Ps, h_uv, t_uv = [], [], []
        for v, o in obs_by_view.items():
            if o[HANDLE, 2] >= 0.3 and o[TOP, 2] >= 0.3:
                Ps.append(self.cams[v].P())
                h_uv.append(o[HANDLE, :2])
                t_uv.append(o[TOP, :2])
        if len(Ps) < 2:
            return None
        try:
            h = triangulate_point(Ps, h_uv) * _Y_FLIP
            t = triangulate_point(Ps, t_uv) * _Y_FLIP
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not (np.isfinite(h).all() and np.isfinite(t).all()):
            return None
        return h, t

    def fit(self, cand_by_view: dict, hand=None, prior=None, extra_starts=()):
        """cand_by_view: {view: [obs (5,3), ...]} with the preferred candidate first."""
        choice = {v: 0 for v, cs in cand_by_view.items() if cs and self._usable(cs[0])}
        if not choice or (hand is None and len(choice) < 2):
            return None
        obs = {v: cand_by_view[v][0] for v in choice}
        starts = self._starts(obs, hand, prior, extra_starts)
        if not starts:
            return None

        best = None
        for x0 in starts:
            pr = self._prior_aligned(prior, rotmat(x0[3:])) if prior is not None else None
            x, cost = self._solve(x0, obs, hand, pr, max_nfev=25 if len(starts) > 6 else 60)
            if best is None or cost < best[1]:
                best = (x, cost, pr)
            if prior is not None and x0 is starts[0] and \
                    np.median([self.view_error(x, v, o)[0] for v, o in obs.items()]) < 0.08:
                break
        x, cost, pr = best
        x, cost = self._solve(x, obs, hand, pr, max_nfev=150)

        changed = False
        for v, cs in cand_by_view.items():
            if len(cs) < 2:
                continue
            errs = [(self.view_error(x, v, c)[0], i) for i, c in enumerate(cs) if self._usable(c)]
            if errs:
                e, i = min(errs)
                cur = self.view_error(x, v, obs[v])[0] if v in obs else np.inf
                if e < cur * 0.7:
                    obs[v], choice[v] = cs[i], i
                    changed = True
        if changed:
            x, cost = self._solve(x, obs, hand, pr, max_nfev=150)

        while len(obs) > 1:
            per = {v: self.view_error(x, v, o)[0] for v, o in obs.items()}
            worst = max(per, key=per.get)
            others = np.median([e for v, e in per.items() if v != worst])
            if per[worst] <= max(self.cfg.view_reject, 2.5 * others):
                break
            del obs[worst], choice[worst]
            if hand is None and len(obs) < 2:
                return None
            x, cost = self._solve(x, obs, hand, pr, max_nfev=150)

        per = {v: self.view_error(x, v, o)[0] for v, o in obs.items()}
        views = [v for v, e in per.items() if e <= self.cfg.view_reject]
        if not views:
            return None
        err = float(np.mean([per[v] for v in views]))
        score = float(np.mean([obs[v][obs[v][:, 2] >= self.cfg.min_score, 2].mean() for v in views]))
        return FitResult(x=x, cost=cost, err=err, per_view={v: round(per[v], 4) for v in per},
                         views=views, choice={v: choice[v] for v in views}, score=score)


def confidence(res: FitResult) -> float:
    view_factor = (0.0, 0.55, 0.8, 0.95, 1.0)[min(len(res.views), 4)]
    return float(np.clip(res.score * view_factor * np.exp(-(res.err / 0.15) ** 2), 0.0, 1.0))


def fit_track(fitter: RacketFitter, frames: list, hands: list, cfg: FitConfig | None = None,
              progress=None) -> list:
    """Forward pass with a previous-frame prior, then a backward repair pass."""
    cfg = cfg or fitter.cfg
    n = len(frames)
    out = [None] * n
    last, last_f = None, -99
    for f in range(n):
        if progress and f % 200 == 0:
            progress(f, n)
        prior = last.x if last is not None and f - last_f <= 3 else None
        r = fitter.fit(frames[f], hands[f], prior=prior) if frames[f] else None
        if r is not None and r.err <= cfg.frame_accept:
            out[f] = r
            last, last_f = r, f
    nxt, nxt_f = None, n + 99
    for f in range(n - 1, -1, -1):
        if out[f] is not None and out[f].err <= cfg.frame_accept * 0.6:
            nxt, nxt_f = out[f], f
            continue
        if not frames[f] or nxt is None or nxt_f - f > 3:
            if out[f] is not None:
                nxt, nxt_f = out[f], f
            continue
        r = fitter.fit(frames[f], hands[f], prior=nxt.x, extra_starts=(nxt.x,))
        if r is not None and r.err <= cfg.frame_accept and (out[f] is None or r.cost < out[f].cost):
            out[f] = r
        if out[f] is not None:
            nxt, nxt_f = out[f], f
    return out


def _runs(valid):
    runs, start = [], None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def finalize_track(results: list, fps: float, cfg: FitConfig | None = None) -> dict:
    """Spike removal, symmetry/hemisphere alignment, slerp gap fill, zero-phase smoothing."""
    from racket_smoothing import spring_smooth

    cfg = cfg or FitConfig()
    n = len(results)
    valid = np.array([r is not None for r in results])
    g = np.zeros((n, 3))
    Rm = np.tile(np.eye(3), (n, 1, 1))
    conf = np.zeros(n)
    nviews = np.zeros(n, dtype=int)
    err = np.full(n, np.nan)
    for i, r in enumerate(results):
        if r is not None:
            g[i], Rm[i], conf[i], nviews[i], err[i] = r.grip, r.R, confidence(r), len(r.views), r.err

    prev = None
    for i in range(n):
        if valid[i]:
            if prev is not None:
                Rm[i] = align_symmetry(Rm[i], Rm[prev])
            prev = i

    idx = np.flatnonzero(valid)
    for k in range(1, len(idx) - 1):
        a, i, b = idx[k - 1], idx[k], idx[k + 1]
        if b - a > 4:
            continue
        sa, si, sb = Rm[a][:, 2], Rm[i][:, 2], Rm[b][:, 2]
        mid = sa + sb
        mid /= max(np.linalg.norm(mid), 1e-9)
        ang = np.degrees(np.arccos(np.clip(si @ mid, -1, 1)))
        ang_ab = np.degrees(np.arccos(np.clip(sa @ sb, -1, 1)))
        jump = np.linalg.norm(g[i] - (g[a] + g[b]) / 2)
        if (ang > cfg.spike_deg and ang_ab < cfg.spike_deg) or \
                (jump > cfg.spike_m and np.linalg.norm(g[a] - g[b]) < cfg.spike_m):
            valid[i] = False

    status = np.array(["missing"] * n, dtype=object)
    status[valid & (nviews >= 2)] = "detected"
    status[valid & (nviews == 1)] = "partial"

    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap < 1 or gap > cfg.gap_max:
            continue
        Rb = align_symmetry(Rm[b], Rm[a])
        slerp = Slerp([a, b], Rotation.from_matrix(np.stack([Rm[a], Rb])))
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            g[f] = g[a] * (1 - t) + g[b] * t
            Rm[f] = slerp([f]).as_matrix()[0]
            conf[f] = 0.7 * min(conf[a], conf[b])
            status[f] = "interpolated"
        valid[a + 1:b] = True

    prev = None
    for i in range(n):
        if valid[i]:
            if prev is not None:
                Rm[i] = align_symmetry(Rm[i], Rm[prev])
            prev = i

    q = Rotation.from_matrix(Rm).as_quat()
    for a, b in _runs(valid):
        for i in range(a + 1, b):
            if q[i] @ q[i - 1] < 0:
                q[i] = -q[i]
    g_s = spring_smooth(g, fps, cfg.tau, valid)
    q_s = spring_smooth(q, fps, cfg.tau, valid)
    q_s /= np.maximum(np.linalg.norm(q_s, axis=1, keepdims=True), 1e-12)
    R_s = Rotation.from_quat(q_s).as_matrix()
    g[valid], Rm[valid] = g_s[valid], R_s[valid]

    return {"grip": g, "R": Rm, "valid": valid, "status": status, "confidence": conf,
            "views": nviews, "err": err}


def estimate_template(cams: dict, samples: list, default: RacketTemplate | None = None) -> tuple:
    """Median racket dimensions from frames where all five keypoints triangulate cleanly.

    samples: list of {view: obs (5,3)} (one player-frame each). Returns (template, n_used).
    """
    from analysis.triangulate import triangulate_joint

    base = default or RacketTemplate()
    Ps = {v: c.P() for v, c in cams.items()}
    L, H, W, Z = [], [], [], []
    for obs in samples:
        pts = []
        for k in range(5):
            X, _c, e = triangulate_joint(Ps, {v: o[k] for v, o in obs.items()}, 0.6)
            if X is None or e > 12.0:
                break
            pts.append(X)
        if len(pts) < 5:
            continue
        top, bottom, handle, left, right = pts
        axis = top - handle
        length = np.linalg.norm(axis)
        if length < 0.3:
            continue
        axis /= length
        L.append(length)
        H.append(np.linalg.norm(top - bottom))
        W.append(np.linalg.norm(left - right))
        Z.append(float(((left + right) / 2 - handle) @ axis))
    if len(L) < 10:
        return base, len(L)

    def clamp(v, ref, lo=0.8, hi=1.2):
        return float(np.clip(v, ref * lo, ref * hi))

    tpl = RacketTemplate(
        length=clamp(np.median(L), base.length, 0.9, 1.1),
        head_length=clamp(np.median(H), base.head_length),
        head_width=clamp(np.median(W), base.head_width),
        width_z=clamp(np.median(Z), base.width_z, 0.85, 1.1),
        grip_z=base.grip_z,
    )
    return tpl, len(L)
