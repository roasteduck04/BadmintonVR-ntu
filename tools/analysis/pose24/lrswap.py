"""Left/right keypoint swap detection per view (spec § 2 step 4; review M2).

RTMPose often flips left/right on back views; with 2 views nothing rejects it. Per player, per view,
per frame there are two hypotheses: as detected / L<->R swapped (body 17 + feet 6 + both hand blocks +
the two face eye rings, whose centres feed the fit; the rest of the face 68 is not permuted).
Both hypotheses are scored on the same keypoints (slot j and its mirror slot both visible). A frame is evidence
only when (a) the reference is trustworthy (>= 3 views: consensus keypoints from >= 2 inlier views with a low
reprojection error), (b) at least one hypothesis is plausible (median residual < PLAUSIBLE_PX) and (c) the two
hypotheses differ by a clear margin; otherwise it costs 0 for both. Real data (final review): a consensus built
on a view that tracks the wrong person fits neither hypothesis of a good view, and a barely-seen player gives
a few near-tied frames —neither may flip a view. A small per-frame prior on "swapped" makes no-evidence
stretches fall back to "as detected".
  >= 3 views: emission = median px distance of view v's shoulders..ankles to the consensus
              triangulation of the other views (as detected vs swapped). With 3 views the consensus is a
              2-view triangulation, which can't reject a second flipped view.
  2 views:    view 1 is tested relative to view 0: 2-view reprojection residual + left/right limb
              asymmetry of the 2-view triangulation. (Both views swapped together is a consistent mirror
              and can't be detected — logged as a limitation.)
Viterbi with a switch cost (and the swap prior) picks a temporally consistent hypothesis; swapped frames are
corrected.
"""
from __future__ import annotations

import warnings

import numpy as np

from analysis import pose_v2 as pv

SWITCH_COST = 4.0
EMIT_PX = 10.0
ASYM_M = 0.05
CONS_ERR_PX = 15.0               # consensus keypoint: reprojection error over its inlier views (px)
PLAUSIBLE_PX = pv.INLIER_PX      # a hypothesis is plausible when its median residual is below this (px)
MARGIN = 0.5                     # clear per-frame difference between the hypotheses (cost units: 5 px)
SWAP_PRIOR = 0.2                 # per-frame cost of the "swapped" state (no evidence -> as detected)
CHECK = list(range(5, 17))        # shoulders .. ankles (133 indices)
_PAIRS = ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]
          + [(91 + i, 112 + i) for i in range(21)]
          + [(59, 68), (60, 67), (61, 66), (62, 65), (63, 70), (64, 69)])   # iBUG eye rings 36-41 <-> 42-47
PERM = np.arange(133)
for _a, _b in _PAIRS:
    PERM[_a], PERM[_b] = _b, _a
_LIMB_PAIRS = [((5, 7), (6, 8)), ((7, 9), (8, 10)), ((11, 13), (12, 14)), ((13, 15), (14, 16))]
_PCHECK = list(PERM[CHECK])


def _both(c0, c1, plausible=None, margin=MARGIN):
    """(F, 2) costs; a frame counts as evidence only when both hypotheses are finite, the better one is
    plausible (`plausible` (F,) bool, None = no check) and they differ by >= `margin`; else 0 for both."""
    ev = np.isfinite(c0) & np.isfinite(c1)
    with np.errstate(invalid="ignore"):
        ev &= np.abs(c0 - c1) >= margin
    if plausible is not None:
        ev &= plausible
    return np.stack([np.where(ev, c0, 0.0), np.where(ev, c1, 0.0)], axis=1)


def viterbi2(cost, switch=SWITCH_COST, prior=0.0) -> np.ndarray:
    """Two-state Viterbi: cost (F, 2) per frame and state; switching states costs `switch`; state 1
    ("swapped") costs `prior` more per frame."""
    F = len(cost)
    if F == 0:
        return np.zeros(0, bool)
    cost = np.asarray(cost, float) + np.array([0.0, prior])
    D = np.zeros((F, 2))
    back = np.zeros((F, 2), int)
    D[0] = cost[0]
    for f in range(1, F):
        for s in (0, 1):
            stay, move = D[f - 1, s], D[f - 1, 1 - s] + switch
            back[f, s] = s if stay <= move else 1 - s
            D[f, s] = min(stay, move) + cost[f, s]
    path = np.zeros(F, int)
    path[-1] = int(np.argmin(D[-1]))
    for f in range(F - 1, 0, -1):
        path[f - 1] = back[f, path[f]]
    return path.astype(bool)


def _median_rows(r, m, min_n=4):
    r = np.where(m, np.minimum(r, 200.0), np.nan)
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(r, axis=1)
    return np.where(cnt >= min_n, med, np.nan)


def _consensus_costs(cams, views, uv, w, k):
    """(F, 2) costs of view k (as detected, swapped) against the other views' consensus."""
    F, V = uv.shape[0], uv.shape[2]
    rest = [i for i in range(V) if i != k]
    rv = [views[i] for i in rest]
    X, _, err, inl = pv.triangulate_batch({v: cams[v] for v in rv}, rv,
                                          uv[:, CHECK][:, :, rest].reshape(-1, len(rest), 2),
                                          w[:, CHECK][:, :, rest].reshape(-1, len(rest)))
    with np.errstate(invalid="ignore"):
        trusted = ((inl.sum(axis=1) >= 2) & (err < CONS_ERR_PX)).reshape(F, len(CHECK))
    pr, depth = pv.project(pv.proj_mats(cams, [views[k]]), X)
    pr = pr[:, 0].reshape(F, len(CHECK), 2)
    r0 = np.linalg.norm(pr - uv[:, CHECK, k], axis=-1)
    r1 = np.linalg.norm(pr - uv[:, _PCHECK, k], axis=-1)
    m = (trusted & np.isfinite(r0) & np.isfinite(r1) & (depth[:, 0].reshape(F, -1) > 0)
         & (w[:, CHECK, k] >= pv.W_MIN) & (w[:, _PCHECK, k] >= pv.W_MIN))
    m0, m1 = _median_rows(r0, m), _median_rows(r1, m)
    with np.errstate(invalid="ignore"):
        plausible = np.fmin(m0, m1) < PLAUSIBLE_PX
    return _both(m0 / EMIT_PX, m1 / EMIT_PX, plausible)


def _two_view_residual(cams, views, uv2, w2):
    """uv2 (N,2,2), w2 (N,2): plain weighted DLT (no gating) -> X (N,3), mean px residual (N,)."""
    Rt = np.stack([np.hstack([cams[v][1], cams[v][2][:, None]]) for v in views])
    Kinv = np.stack([np.linalg.inv(cams[v][0]) for v in views])
    xn = np.einsum("vij,nvj->nvi", Kinv, np.concatenate([uv2, np.ones(uv2.shape[:-1] + (1,))], -1))[..., :2]
    ok = (w2 >= pv.W_MIN).all(-1) & np.isfinite(uv2).all((-1, -2))
    X = np.full((len(uv2), 3), np.nan)
    if ok.any():
        X[ok] = pv._dlt(Rt, xn[ok], w2[ok])
    pr, _ = pv.project(pv.proj_mats(cams, views), X)
    return X, np.linalg.norm(pr - uv2, axis=-1).mean(axis=-1)


def _pair_costs(cams, views, uv, w):
    """(F, 2) costs of view 1 (as detected, swapped) relative to view 0."""
    F = uv.shape[0]
    vis = (w[:, CHECK, 0] >= pv.W_MIN) & (w[:, CHECK, 1] >= pv.W_MIN) & (w[:, _PCHECK, 1] >= pv.W_MIN)
    c = {j: i for i, j in enumerate(CHECK)}
    Xs, res = [], []
    for idx in (CHECK, _PCHECK):
        uv2 = np.stack([uv[:, CHECK, 0], uv[:, idx, 1]], axis=2).reshape(-1, 2, 2)
        w2 = np.where(vis[..., None], np.stack([w[:, CHECK, 0], w[:, idx, 1]], axis=2), 0.0).reshape(-1, 2)
        X, r = _two_view_residual(cams, views, uv2, w2)
        Xs.append(X.reshape(F, len(CHECK), 3))
        res.append(r.reshape(F, len(CHECK)))
    m = np.isfinite(res[0]) & np.isfinite(res[1])
    p0, p1 = _median_rows(res[0], m), _median_rows(res[1], m)
    with np.errstate(invalid="ignore"):
        plausible = np.fmin(p0, p1) < PLAUSIBLE_PX
    asym = []
    for X in Xs:
        a = []
        for (la, lb), (ra, rb) in _LIMB_PAIRS:
            ll = np.linalg.norm(X[:, c[la]] - X[:, c[lb]], axis=-1)
            lr = np.linalg.norm(X[:, c[ra]] - X[:, c[rb]], axis=-1)
            a.append(np.abs(ll - lr))
        asym.append(np.stack(a, 1))                                     # (F, 4)
    ok = np.isfinite(asym[0]) & np.isfinite(asym[1])                   # limb pairs seen under both
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a0 = np.where(n > 0, np.where(ok, asym[0], 0.0).sum(1) / n, np.nan) / ASYM_M
        a1 = np.where(n > 0, np.where(ok, asym[1], 0.0).sum(1) / n, np.nan) / ASYM_M
    # both terms scored when present (the asymmetry needs limb pairs seen under both hypotheses)
    c0 = p0 / EMIT_PX + np.where(np.isfinite(a0), a0, 0.0)
    c1 = p1 / EMIT_PX + np.where(np.isfinite(a1), a1, 0.0)
    return _both(c0, c1, plausible)


def detect_swaps(bundle, pid, switch=SWITCH_COST, prior=SWAP_PRIOR) -> np.ndarray:
    """(V, F) bool: frames where a view's left/right labels are swapped."""
    views = bundle["views"]
    uv, w = pv.obs_arrays(bundle, pid, idx=list(range(133)))
    V, F = len(views), uv.shape[0]
    S = np.zeros((V, F), bool)
    if F == 0:
        return S
    if V >= 3:
        for k in range(V):
            S[k] = viterbi2(_consensus_costs(bundle["cams"], views, uv, w, k), switch, prior)
    elif V == 2:
        S[1] = viterbi2(_pair_costs(bundle["cams"], views, uv, w), switch, prior)
    return S


def apply_swaps(bundle, swaps) -> dict:
    """Un-swap the flagged frames in bundle["obs"] (in place). Returns swapped-frame counts per view."""
    counts = {v: 0 for v in bundle["views"]}
    for pid, S in swaps.items():
        for k, v in enumerate(bundle["views"]):
            m = S[k]
            if m.any():
                D = bundle["obs"][pid][v]
                D[m] = D[m][:, PERM]
                counts[v] += int(m.sum())
    return counts
