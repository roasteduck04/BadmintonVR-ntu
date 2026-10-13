import numpy as np
import pytest

from analysis.shuttle_physics import (
    shuttle_drag_ode,
    integrate_shuttle,
    fit_shuttle_arc,
    segment_arcs,
    fit_shuttle_trajectory,
    GRAVITY,
    DRAG_K,
)


class TestShuttleDragODE:
    def test_freefall_no_drag(self):
        """With k=0, should match analytical free-fall."""
        state0 = [0, 10, 0, 5, 0, 3]
        t_eval = np.linspace(0, 1.0, 61)
        traj = integrate_shuttle(state0, (0, 1.0), t_eval, k=0, g=GRAVITY)
        assert traj is not None
        assert traj.shape == (61, 3)
        for i, t in enumerate(t_eval):
            expected_x = 5 * t
            expected_y = 10 - 0.5 * GRAVITY * t * t
            expected_z = 3 * t
            assert traj[i, 0] == pytest.approx(expected_x, abs=0.01)
            assert traj[i, 1] == pytest.approx(expected_y, abs=0.01)
            assert traj[i, 2] == pytest.approx(expected_z, abs=0.01)

    def test_terminal_velocity_convergence(self):
        """Dropping from rest with drag, vertical speed should approach v_terminal."""
        state0 = [0, 100, 0, 0, 0, 0]
        t_eval = np.linspace(0, 5.0, 301)
        traj = integrate_shuttle(state0, (0, 5.0), t_eval, k=DRAG_K, g=GRAVITY)
        assert traj is not None
        vy_final = (traj[-1, 1] - traj[-2, 1]) / (t_eval[-1] - t_eval[-2])
        assert abs(vy_final) == pytest.approx(6.7, abs=0.5)

    def test_ode_returns_correct_shape(self):
        state = [0, 5, 0, 10, 5, 3]
        deriv = shuttle_drag_ode(0, state)
        assert len(deriv) == 6

    def test_drag_decelerates_horizontal(self):
        """Horizontal-only motion should decelerate."""
        state0 = [0, 5, 0, 20, 0, 0]
        t_eval = np.array([0, 0.5, 1.0])
        traj = integrate_shuttle(state0, (0, 1.0), t_eval, k=DRAG_K, g=GRAVITY)
        vx_0 = (traj[1, 0] - traj[0, 0]) / 0.5
        vx_1 = (traj[2, 0] - traj[1, 0]) / 0.5
        assert vx_1 < vx_0


class TestSegmentArcs:
    def test_single_arc(self):
        points = [(0, np.zeros(3)), (1, np.zeros(3)), (2, np.zeros(3))]
        arcs = segment_arcs(points, gap_threshold=12)
        assert len(arcs) == 1
        assert len(arcs[0]) == 3

    def test_split_on_gap(self):
        points = [(0, np.zeros(3)), (1, np.zeros(3)),
                  (20, np.zeros(3)), (21, np.zeros(3))]
        arcs = segment_arcs(points, gap_threshold=12)
        assert len(arcs) == 2
        assert arcs[0][-1][0] == 1
        assert arcs[1][0][0] == 20

    def test_empty_input(self):
        assert segment_arcs([], gap_threshold=12) == []

    def test_single_point(self):
        arcs = segment_arcs([(5, np.zeros(3))], gap_threshold=12)
        assert len(arcs) == 1
        assert len(arcs[0]) == 1

    def test_exactly_at_threshold(self):
        points = [(0, np.zeros(3)), (12, np.zeros(3))]
        arcs = segment_arcs(points, gap_threshold=12)
        assert len(arcs) == 1

    def test_just_over_threshold(self):
        points = [(0, np.zeros(3)), (13, np.zeros(3))]
        arcs = segment_arcs(points, gap_threshold=12)
        assert len(arcs) == 2


class TestFitShuttleArc:
    def _make_synthetic_arc(self, state0, n_frames=30, fps=60, k=DRAG_K):
        """Generate a clean trajectory from known initial conditions."""
        dt = 1.0 / fps
        times = np.arange(n_frames) * dt
        traj = integrate_shuttle(state0, (0, times[-1]), times, k=k, g=GRAVITY)
        return traj, times

    def test_recovers_clean_trajectory(self):
        state0 = np.array([0, 3, 0, 8, 6, -5])
        traj, times = self._make_synthetic_arc(state0, n_frames=30)
        assert traj is not None
        fitted = fit_shuttle_arc(traj, times)
        assert fitted is not None
        assert fitted[:3] == pytest.approx(state0[:3], abs=0.1)
        assert fitted[3:] == pytest.approx(state0[3:], abs=0.5)

    def test_recovers_noisy_trajectory(self):
        rng = np.random.RandomState(42)
        state0 = np.array([1, 4, -2, 6, 8, -4])
        traj, times = self._make_synthetic_arc(state0, n_frames=40)
        noisy = traj + rng.normal(0, 0.05, traj.shape)
        fitted = fit_shuttle_arc(noisy, times)
        assert fitted is not None
        refitted = integrate_shuttle(fitted, (0, times[-1]), times)
        residuals = np.linalg.norm(refitted - traj, axis=1)
        assert np.max(residuals) < 0.5

    def test_fills_gaps(self):
        """Fit with some frames removed — the fitted trajectory should cover all."""
        state0 = np.array([0, 3, 0, 10, 5, -6])
        traj, times = self._make_synthetic_arc(state0, n_frames=30)
        keep = [0, 1, 5, 10, 15, 20, 25, 29]
        fitted = fit_shuttle_arc(traj[keep], times[keep])
        assert fitted is not None
        all_times = np.arange(30) / 60.0
        full_traj = integrate_shuttle(fitted, (0, all_times[-1]), all_times)
        residuals = np.linalg.norm(full_traj - traj, axis=1)
        assert np.mean(residuals) < 0.3

    def test_too_few_points_returns_none(self):
        pts = np.array([[0, 3, 0], [1, 3, 1], [2, 2, 2]])
        times = np.array([0, 1/60, 2/60])
        assert fit_shuttle_arc(pts, times) is None


class TestFitShuttleTrajectory:
    def test_end_to_end_synthetic(self):
        """Full pipeline with a synthetic trajectory that has a gap in the middle."""
        state0 = np.array([0, 3, 0, 8, 5, -4])
        fps = 60
        dt = 1.0 / fps
        n_frames = 60
        times = np.arange(n_frames) * dt
        traj = integrate_shuttle(state0, (0, times[-1]), times)

        raw_points = [None] * n_frames
        raw_confs = [None] * n_frames
        for f in range(n_frames):
            if 20 <= f <= 30:
                continue
            raw_points[f] = traj[f].copy()
            raw_confs[f] = 1.0

        positions, confs = fit_shuttle_trajectory(raw_points, raw_confs, fps)
        assert len(positions) == n_frames
        filled = sum(1 for p in positions if p is not None)
        assert filled >= 55

        for f in [25, 26, 27]:
            assert positions[f] is not None
            fitted_pos = np.array(positions[f])
            expected = traj[f]
            assert np.linalg.norm(fitted_pos - expected) < 0.5

    def test_short_arc_uses_sg_fallback(self):
        """Arcs with < 4 points should still produce output via SG fallback."""
        raw_points = [None] * 20
        raw_confs = [None] * 20
        for f in [5, 6, 7]:
            raw_points[f] = np.array([float(f), 2.0, 0.0])
            raw_confs[f] = 0.8

        positions, confs = fit_shuttle_trajectory(raw_points, raw_confs, fps=60)
        for f in [5, 6, 7]:
            assert positions[f] is not None

    def test_empty_input(self):
        positions, confs = fit_shuttle_trajectory([None] * 10, [None] * 10, fps=60)
        assert all(p is None for p in positions)

    def test_y_clamped_above_zero(self):
        """Fitted trajectory should not produce negative Y (below floor)."""
        state0 = np.array([0, 0.5, 0, 5, -3, 2])
        fps = 60
        dt = 1.0 / fps
        n_frames = 40
        times = np.arange(n_frames) * dt
        traj = integrate_shuttle(state0, (0, times[-1]), times)

        raw_points = [None] * n_frames
        raw_confs = [None] * n_frames
        for f in range(n_frames):
            if traj[f, 1] < -2:
                continue
            raw_points[f] = traj[f].copy()
            raw_confs[f] = 1.0

        positions, confs = fit_shuttle_trajectory(raw_points, raw_confs, fps)
        for p in positions:
            if p is not None:
                assert p[1] >= 0.0
