import numpy as np
from analysis import multiview as mv
from analysis import triangulate as tri


def _three_cams():
    K = mv.intrinsics_from_fov(2000, 2000, 60.0)
    cams = {}
    for name, center in (("a", [-4, 2, -6]), ("b", [4, 2, -6]), ("c", [0, 2, 7])):
        c = np.array(center, dtype=float)
        fwd = -c / np.linalg.norm(c)          # look toward origin
        up = np.array([0, 1, 0], dtype=float)
        right = np.cross(up, fwd); right /= np.linalg.norm(right)
        trueup = np.cross(fwd, right)
        R = np.vstack([right, trueup, fwd])
        t = -R @ c
        cams[name] = mv.projection_matrix(K, R, t)
    return cams


def _project(P, X):
    uvw = P @ np.append(X, 1.0)
    return uvw[:2] / uvw[2]


def test_triangulate_recovers_point():
    cams = _three_cams()
    X = np.array([0.4, 1.2, 0.3])
    uvs = [_project(P, X) for P in cams.values()]
    Xhat = tri.triangulate_point(list(cams.values()), uvs)
    assert np.allclose(Xhat, X, atol=1e-6)


def test_triangulate_joint_confidence_and_filter():
    cams = _three_cams()
    X = np.array([-0.2, 1.0, 0.5])
    kp = {v: (*_project(P, X), 0.9) for v, P in cams.items()}
    Xhat, conf, err = tri.triangulate_joint(cams, kp, score_thr=0.3)
    assert np.allclose(Xhat, X, atol=1e-5)
    assert 0.7 < conf <= 1.0
    assert err < 1e-3
    # only one view above threshold -> not triangulable
    kp_low = {"a": (10, 10, 0.9), "b": (10, 10, 0.1), "c": (10, 10, 0.1)}
    X2, conf2, err2 = tri.triangulate_joint(cams, kp_low, score_thr=0.3)
    assert X2 is None and conf2 == 0.0 and err2 is None


def test_triangulate_joint_rejects_degenerate_rays(monkeypatch):
    # Degenerate/near-parallel camera rays can drive the DLT homogeneous
    # divide (Xh[:3] / Xh[3]) to produce a non-finite point when Xh[3] is
    # ~0. In practice, forcing this from the public API (e.g. two
    # identical cameras viewing the same pixel) still returns a finite
    # point because SVD picks a particular finite vector from the
    # degenerate null space, so we monkeypatch triangulate_point directly
    # to simulate the non-finite output and verify triangulate_joint's
    # guard drops it to None instead of leaking Inf/NaN downstream.
    cams = _three_cams()
    monkeypatch.setattr(
        tri, "triangulate_point",
        lambda projs, uvs: np.array([np.inf, 0.0, 0.0]),
    )
    kp = {v: (100.0, 100.0, 0.9) for v in cams}
    X, conf, err = tri.triangulate_joint(cams, kp, score_thr=0.3)
    assert X is None and conf == 0.0 and err is None


def test_triangulate_joint_max_reproj_gate():
    cams = _three_cams()
    X = np.array([0.4, 1.2, 0.3])
    names = list(cams)
    uv = {v: _project(cams[v], X) for v in names[:2]}
    uv[names[0]] = uv[names[0]] + np.array([30.0, 0.0])   # one view off by 30 px
    kp = {v: (u[0], u[1], 1.0) for v, u in uv.items()}
    sub = {v: cams[v] for v in names[:2]}
    Xd, _c, err = tri.triangulate_joint(sub, kp, 0.1)
    assert Xd is not None and err > 1.0
    Xn, _c2, _e2 = tri.triangulate_joint(sub, kp, 0.1, max_reproj_px=err / 2)
    assert Xn is None
