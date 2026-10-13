import numpy as np
from analysis import multiview as mv


def test_intrinsics_center_and_square():
    K = mv.intrinsics_from_fov(2160, 3840, 67.0, on="long")
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 2], 1080.0)   # cx = W/2
    assert np.isclose(K[1, 2], 1920.0)   # cy = H/2
    assert np.isclose(K[0, 0], K[1, 1])  # square pixels
    # f = (long/2) / tan(fov/2), long = 3840
    f = (3840 / 2) / np.tan(np.radians(67.0) / 2)
    assert np.isclose(K[0, 0], f, rtol=1e-6)


def test_projection_matrix_shape_and_compose():
    K = mv.intrinsics_from_fov(1000, 1000, 60.0)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5.0])
    P = mv.projection_matrix(K, R, t)
    assert P.shape == (3, 4)
    # a point 5 m in front on the axis projects to the principal point
    X = np.array([0.0, 0.0, 0.0, 1.0])
    uvw = P @ X
    uv = uvw[:2] / uvw[2]
    assert np.allclose(uv, [500.0, 500.0], atol=1e-6)


def _make_camera():
    # a camera 7 m behind the baseline, 3 m up, looking down +Z at the court
    K = mv.intrinsics_from_fov(2160, 3840, 67.0)
    # world Y-up; camera looks toward +Z and down a little
    Rz_look = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    R = Rz_look
    cam_center = np.array([0.0, 3.0, -13.0])
    t = -R @ cam_center
    return K, R, t


def test_solve_extrinsics_recovers_synthetic_camera():
    K, R, t = _make_camera()
    P = mv.projection_matrix(K, R, t)
    court_xz = np.array([[-3.05, -6.70], [3.05, -6.70], [3.05, 6.70],
                         [-3.05, 6.70], [0.0, 0.0], [-3.05, 1.98]])
    world = np.hstack([court_xz[:, :1], np.zeros((len(court_xz), 1)), court_xz[:, 1:]])
    proj = (P @ np.hstack([world, np.ones((len(world), 1))]).T).T
    image_px = proj[:, :2] / proj[:, 2:3]
    R2, t2 = mv.solve_extrinsics(court_xz, image_px, K)
    rms = mv.reprojection_rms(court_xz, image_px, K, R2, t2)
    assert rms < 1e-3
