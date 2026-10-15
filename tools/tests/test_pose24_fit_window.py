"""pose24 WindowProblem24: analytic Jacobian vs finite differences, recovery from a noisy start."""
import numpy as np

import pose24_synth as syn
from analysis import pose_v2 as pv
from analysis.pose24 import fit, kpset


def _problem(T=4, noise_init=0.03, seed=2, bad_knee=True):
    rng = np.random.default_rng(seed)
    views = ("a", "b", "c", "d")
    cams = syn.rig(views)
    Xt = kpset.from133(syn.motion(F=T, z0=-3.0, walk=False)) * np.array([1.0, -1.0, 1.0])   # Y-down truth, heel planted
    uv, _ = pv.project(pv.proj_mats(cams, list(views)), Xt)
    w = np.full(uv.shape[:-1], 0.95)
    lengths = fit.estimate_lengths(Xt, np.ones(Xt.shape[:2], bool), min_n=1)
    X0 = Xt + rng.normal(0, noise_init, Xt.shape)
    if bad_knee:                                  # hyperextended knee: activates a limit term
        X0[:, 13, 2] -= 0.12
    contact = np.zeros((T, 6), bool)
    contact[:, 2] = True                          # left heel planted (kpset.CONTACT[2] = 19)
    anchor = Xt + rng.normal(0, 0.01, Xt.shape)
    anchor[0, 5] = np.nan
    prob = fit.WindowProblem24(pv.proj_mats(cams, list(views)), uv, w, X0, np.ones(T, bool),
                               fit.edge_arrays(lengths), anchor, contact, kpset.forward_sign(Xt))
    return prob, Xt, X0


def test_jacobian_matches_finite_differences():
    prob, _, X0 = _problem()
    x = X0[prob.free].ravel()
    J = prob.jacobian(x).toarray()
    eps = 1e-6
    rng = np.random.default_rng(3)
    for i in rng.choice(len(x), 40, replace=False):
        dx = np.zeros_like(x)
        dx[i] = eps
        num = (prob.residuals(x + dx) - prob.residuals(x - dx)) / (2 * eps)
        assert np.allclose(J[:, i], num, atol=1e-3 * max(1.0, np.abs(num).max()))


def test_limit_residuals_are_active_for_the_bad_knee():
    prob, _, X0 = _problem()
    r = prob.residuals(X0[prob.free].ravel())
    assert r[-prob.n_limit_res:].max() > 1.0


def test_window_recovers_skeleton_from_noisy_init():
    prob, Xt, X0 = _problem(T=8, bad_knee=False)
    Xs = prob.solve()
    assert np.linalg.norm(Xs - Xt, axis=-1).mean() < 0.01 < np.linalg.norm(X0 - Xt, axis=-1).mean()


def test_observations_behind_a_camera_give_no_residual_or_gradient():
    """A keypoint that wanders behind camera a: its projection there is meaningless (a clamped depth would flip
    its sign), so that observation contributes nothing until the point is back in front."""
    prob, _, X0 = _problem(bad_knee=False)
    K, R, t = syn.rig(("a", "b", "c", "d"))["a"]
    X = X0.copy()
    X[0, 0] = -R.T @ t - 1.0 * R[2]                     # 1 m behind camera a's centre
    x = X[prob.free].ravel()
    i = np.nonzero((prob.ot == 0) & (prob.ok == 0) & (prob.ov == 0))[0][0]
    r = prob.residuals(x)
    J = prob.jacobian(x).toarray()
    assert r[2 * i] == 0.0 and r[2 * i + 1] == 0.0
    assert not J[2 * i].any() and not J[2 * i + 1].any()
    others = np.nonzero((prob.ot == 0) & (prob.ok == 0) & (prob.ov != 0))[0]    # views it is in front of count
    assert any(r[2 * j] != 0.0 and J[2 * j].any() for j in others)
