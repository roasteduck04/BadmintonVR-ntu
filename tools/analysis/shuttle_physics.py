"""Physics-constrained shuttlecock trajectory fitting.

Fits drag-dominated ballistic trajectories to triangulated 3D shuttle
positions. A shuttlecock has very high drag (terminal velocity ~6.7 m/s),
making its trajectory predictable between contacts. This module replaces
linear interpolation + Savitzky-Golay smoothing with physics-aware fitting
that fills gaps naturally and rejects impossible positions.

Coordinate system: X = court width, Y = up, Z = court length (meters).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

GRAVITY = 9.81
V_TERMINAL = 6.7
DRAG_K = GRAVITY / (V_TERMINAL ** 2)

_LEGACY_GAP_MAX = 8
_LEGACY_SG_WINDOW = 7
_LEGACY_SG_ORDER = 2


def shuttle_drag_ode(t, state, k=DRAG_K, g=GRAVITY):
    """Right-hand side of the shuttlecock drag ODE.

    state = [x, y, z, vx, vy, vz]
    Returns d(state)/dt.
    """
    x, y, z, vx, vy, vz = state
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    drag = k * speed
    return [vx, vy, vz,
            -drag * vx,
            -g - drag * vy,
            -drag * vz]


def integrate_shuttle(state0, t_span, t_eval, k=DRAG_K, g=GRAVITY):
    """Integrate the drag ODE and return positions at t_eval times.

    Returns ndarray of shape (len(t_eval), 3) — [x, y, z] at each time.
    """
    sol = solve_ivp(shuttle_drag_ode, t_span, state0,
                    t_eval=t_eval, args=(k, g),
                    method="RK45", rtol=1e-6, atol=1e-8)
    if not sol.success:
        return None
    return sol.y[:3].T


def _initial_guess(positions, times):
    """Estimate initial state [x0, y0, z0, vx0, vy0, vz0] from observed points."""
    p0 = positions[0]
    if len(positions) >= 2:
        dt = times[1] - times[0]
        if dt > 0:
            v0 = (positions[1] - positions[0]) / dt
        else:
            v0 = np.zeros(3)
    else:
        v0 = np.zeros(3)
    return np.concatenate([p0, v0])


def fit_shuttle_arc(observed_positions, observed_times, k=DRAG_K, g=GRAVITY):
    """Fit a drag-ballistic trajectory to observed 3D positions.

    observed_positions: ndarray (N, 3) — [x, y, z] at observed times
    observed_times: ndarray (N,) — time in seconds for each observation

    Returns fitted state0 [x0, y0, z0, vx0, vy0, vz0] or None if fit fails.
    """
    if len(observed_positions) < 4:
        return None

    t0 = observed_times[0]
    rel_times = observed_times - t0
    t_span = (0.0, rel_times[-1])

    x0 = _initial_guess(observed_positions, rel_times)

    def residual(state0):
        traj = integrate_shuttle(state0, t_span, rel_times, k=k, g=g)
        if traj is None:
            return 1e10
        diff = traj - observed_positions
        return float(np.sum(diff ** 2))

    result = minimize(residual, x0, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-10})
    if not result.success and result.fun > 1e8:
        return None
    return result.x


def segment_arcs(points, gap_threshold=12):
    """Split a list of (frame_idx, position) into contiguous arcs.

    An arc breaks wherever consecutive observed frames differ by more than
    gap_threshold. Returns list of arcs, each a list of (frame_idx, position).
    """
    if not points:
        return []

    arcs = []
    current_arc = [points[0]]

    for i in range(1, len(points)):
        if points[i][0] - points[i - 1][0] > gap_threshold:
            arcs.append(current_arc)
            current_arc = [points[i]]
        else:
            current_arc.append(points[i])

    arcs.append(current_arc)
    return arcs


def fit_shuttle_trajectory(raw_points, raw_confs, fps,
                           arc_gap=12, k=DRAG_K, g=GRAVITY,
                           sg_window=7, sg_order=2):
    """Top-level entry point: fit physics trajectories to triangulated shuttle data.

    raw_points: list of ndarray([x,y,z]) or None, length = frame_count
    raw_confs: list of float or None, length = frame_count
    fps: frames per second

    Returns (positions, confs) — same length as input, with physics-fitted
    trajectories replacing raw data where possible. Short arcs (< 4 points)
    fall back to Savitzky-Golay smoothing.
    """
    from scipy.signal import savgol_filter

    frame_count = len(raw_points)
    dt = 1.0 / fps

    observed = [(f, raw_points[f]) for f in range(frame_count)
                if raw_points[f] is not None]

    arcs = segment_arcs(observed, gap_threshold=arc_gap)

    positions = [None] * frame_count
    confs = [None] * frame_count

    n_physics = 0
    n_sg_fallback = 0

    for arc in arcs:
        frames = [f for f, _ in arc]
        pts = np.array([p for _, p in arc])
        times = np.array([f * dt for f in frames])

        f_start = frames[0]
        f_end = frames[-1]

        if len(arc) >= 4:
            state0 = fit_shuttle_arc(pts, times, k=k, g=g)

            if state0 is not None:
                t0 = times[0]
                all_times = np.array([(f - f_start) * dt for f in range(f_start, f_end + 1)])
                t_span = (0.0, all_times[-1])

                fitted = integrate_shuttle(state0, t_span, all_times, k=k, g=g)

                if fitted is not None:
                    observed_set = set(frames)
                    for i, f in enumerate(range(f_start, f_end + 1)):
                        pos = fitted[i]
                        pos[1] = max(0.0, pos[1])
                        positions[f] = [round(float(pos[0]), 4),
                                        round(float(pos[1]), 4),
                                        round(float(pos[2]), 4)]

                        if f in observed_set:
                            confs[f] = 1.0
                        else:
                            nearest_dist = min(abs(f - of) for of in frames)
                            confs[f] = round(max(0.3, 1.0 - nearest_dist * 0.1), 4)

                    n_physics += (f_end - f_start + 1)
                    continue

        if len(arc) >= sg_window:
            for ax in range(3):
                pts[:, ax] = savgol_filter(pts[:, ax], sg_window, sg_order)

        for i, f in enumerate(frames):
            positions[f] = [round(float(pts[i][0]), 4),
                            round(float(pts[i][1]), 4),
                            round(float(pts[i][2]), 4)]
            confs[f] = round(float(raw_confs[f]) if raw_confs[f] is not None else 0.5, 4)
        n_sg_fallback += len(arc)

    n_ok = sum(1 for p in positions if p is not None)
    print(f"  physics-fitted: {n_physics} frames, "
          f"SG-fallback: {n_sg_fallback} frames, "
          f"total: {n_ok}/{frame_count}")

    return positions, confs


def legacy_shuttle_trajectory(raw_points, raw_confs, fps,
                              gap_max=_LEGACY_GAP_MAX,
                              sg_window=_LEGACY_SG_WINDOW,
                              sg_order=_LEGACY_SG_ORDER):
    """Original shuttle post-processing: linear gap interpolation + Savitzky-Golay.

    Kept for A/B comparison against the physics-constrained fitter.
    """
    from scipy.signal import savgol_filter

    frame_count = len(raw_points)
    raw = [p.copy() if p is not None else None for p in raw_points]
    raw_conf = list(raw_confs)

    n_interp = 0
    for f in range(frame_count):
        if raw[f] is not None:
            continue
        left = next((k for k in range(f - 1, -1, -1) if raw[k] is not None), None)
        right = next((k for k in range(f + 1, frame_count) if raw[k] is not None), None)
        if left is None or right is None:
            continue
        gap = right - left - 1
        if gap > gap_max:
            continue
        t = (f - left) / (right - left)
        raw[f] = (1.0 - t) * raw[left] + t * raw[right]
        raw_conf[f] = (1.0 - t) * raw_conf[left] + t * raw_conf[right]
        n_interp += 1
    if n_interp:
        print(f"  legacy gaps interpolated: {n_interp} frames")

    positions = [None] * frame_count
    confs_out = [None] * frame_count
    seg_start = None
    for f in range(frame_count + 1):
        is_valid = f < frame_count and raw[f] is not None
        if is_valid and seg_start is None:
            seg_start = f
        elif not is_valid and seg_start is not None:
            seg_len = f - seg_start
            seg_pts = np.array([raw[k] for k in range(seg_start, f)])
            if seg_len >= sg_window:
                for ax in range(3):
                    seg_pts[:, ax] = savgol_filter(seg_pts[:, ax], sg_window, sg_order)
            for j, k in enumerate(range(seg_start, f)):
                positions[k] = [round(float(seg_pts[j][0]), 4),
                                round(float(seg_pts[j][1]), 4),
                                round(float(seg_pts[j][2]), 4)]
                confs_out[k] = round(float(raw_conf[k]), 4)
            seg_start = None

    n_ok = sum(1 for p in positions if p is not None)
    print(f"  legacy shuttle: {n_ok}/{frame_count} frames "
          f"({100 * n_ok / max(frame_count, 1):.1f}%)")
    return positions, confs_out
