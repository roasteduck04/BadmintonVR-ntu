"""Assemble a real analysis.json v4 from triangulated world-frame pose.

Multi-camera 2D keypoints → RANSAC DLT triangulation → SMPL-28 body-local 3D
→ court position. Racket 2D keypoints (from RacketVision) are triangulated the
same way and fed into racket.fill_player_rackets. Shuttle tracking via
TrackNetV3 triangulation. Subcommands: multi, resmooth.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis import schema
from analysis import triangulate as tri


def _write_doc(path, doc):
    """Write an analysis doc Unity reads. NaN is rejected here: JsonUtility silently drops the whole file."""
    text = json.dumps(doc, allow_nan=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _fix_3d_swaps(player_data, anchor_window=200):
    """Swap entire frames between two players when their 3D positions
    land on the wrong side of the court relative to early-frame anchors."""
    pf0 = player_data[0]["pose_frames"]
    pf1 = player_data[1]["pose_frames"]
    frame_count = min(len(pf0), len(pf1))

    def _get_z(pf, f):
        pos = pf[f].get("position") if f < len(pf) else None
        return pos[1] if pos else None

    z0 = [_get_z(pf0, f) for f in range(min(anchor_window, frame_count))]
    z1 = [_get_z(pf1, f) for f in range(min(anchor_window, frame_count))]
    z0 = [z for z in z0 if z is not None]
    z1 = [z for z in z1 if z is not None]
    if not z0 or not z1:
        return player_data

    anc0 = float(np.median(z0))
    anc1 = float(np.median(z1))
    if abs(anc0 - anc1) < 2.0:
        return player_data

    total = 0
    for f in range(frame_count):
        pos0 = pf0[f].get("position")
        pos1 = pf1[f].get("position")
        if pos0 is None or pos1 is None:
            continue
        d_normal = abs(pos0[1] - anc0) + abs(pos1[1] - anc1)
        d_swap = abs(pos0[1] - anc1) + abs(pos1[1] - anc0)
        if d_swap < d_normal - 2.0:
            pf0[f], pf1[f] = pf1[f], pf0[f]
            total += 1
    if total:
        print(f"  3D swap correction: {total} frames swapped")
    return player_data


def _fix_3d_swaps_multi(player_data, anchor_window=200):
    """Swap entire frames between N players when their 3D court positions
    land closer to a different player's anchor than their own.

    Uses Hungarian matching on (X, Z) court positions per frame.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(player_data)
    pfs = [pd["pose_frames"] for pd in player_data]
    frame_count = min(len(pf) for pf in pfs)

    def _get_xz(pf, f):
        pos = pf[f].get("position") if f < len(pf) else None
        return np.array(pos, dtype=np.float64) if pos else None

    anchors = []
    for i in range(n):
        pts = []
        for f in range(min(anchor_window, frame_count)):
            xz = _get_xz(pfs[i], f)
            if xz is not None:
                pts.append(xz)
        if pts:
            anchors.append(np.median(pts, axis=0))
        else:
            anchors.append(None)

    valid = [i for i in range(n) if anchors[i] is not None]
    if len(valid) < 2:
        return player_data

    min_sep = min(np.linalg.norm(anchors[a] - anchors[b])
                  for i, a in enumerate(valid)
                  for b in valid[i + 1:])
    if min_sep < 1.5:
        return player_data

    total = 0
    for f in range(frame_count):
        positions = {}
        for i in valid:
            xz = _get_xz(pfs[i], f)
            if xz is not None:
                positions[i] = xz
        if len(positions) < 2:
            continue

        det_idx = sorted(positions.keys())
        cost = np.zeros((len(det_idx), len(valid)))
        for r, di in enumerate(det_idx):
            for c, vi in enumerate(valid):
                cost[r, c] = np.linalg.norm(positions[di] - anchors[vi])

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            src = det_idx[r]
            dst = valid[c]
            if src != dst:
                mapping[src] = dst

        if not mapping:
            continue

        # Apply the assignment as a permutation (review 2 L7: the old pairwise write-back assumed 2-cycles and
        # corrupted 3-cycles). A player whose frame moved but who receives none (its own slot had no position)
        # takes a frame nobody claimed, which reduces to a plain swap.
        involved = sorted(set(mapping.keys()) | set(mapping.values()))
        old_frames = {i: pfs[i][f] for i in involved}
        new = {dst: old_frames[src] for src, dst in mapping.items()}
        receivers = [i for i in involved if i not in new]
        unclaimed = [i for i in involved if i not in mapping]
        for r, u in zip(receivers, unclaimed):
            new[r] = old_frames[u]
        for i, fr in new.items():
            pfs[i][f] = fr
        total += 1

    if total:
        print(f"  3D swap correction (multi): {total} frames swapped")
    return player_data


def world_to_bodylocal_and_position(joints24_world):
    j = np.asarray(joints24_world, dtype=np.float64).copy()
    pelvis = j[0].copy()
    position = [round(float(pelvis[0]), 4), round(float(pelvis[2]), 4)]
    j[:, 0] -= pelvis[0]
    j[:, 2] -= pelvis[2]
    return j, position


def _joint_names(n_joints):
    """Joint names of a 58 (SMPL-X), 28 (SMPL-28) or 24 (SMPL-24) joint layout."""
    if n_joints == schema.NUM_JOINTS:
        return list(schema.JOINT_NAMES)
    from analysis import skeleton24
    if n_joints in (24, 28):
        return list(skeleton24.SMPL28_NAMES[:n_joints])
    raise ValueError(f"no joint names for a {n_joints}-joint skeleton")


def _by_index(table, names):
    """Name-keyed table -> index-keyed, keeping the entries whose joints exist in `names` (review 2 L9: SMPL-24
    indices 22/23 are left_hand/right_hand, but left_thumb1/left_thumb2 in SMPL-X 58)."""
    pos = {nm: i for i, nm in enumerate(names)}
    out = {}
    for key, v in table.items():
        ks = (key,) if isinstance(key, str) else key
        if all(k in pos for k in ks):
            out[pos[ks[0]] if len(ks) == 1 else tuple(pos[k] for k in ks)] = v
    return out


_MAX_BONE_BY_NAME = {
    ("left_knee", "left_hip"): 0.50, ("right_knee", "right_hip"): 0.50,            # thighs
    ("left_ankle", "left_knee"): 0.50, ("right_ankle", "right_knee"): 0.50,        # shins
    ("left_elbow", "left_shoulder"): 0.40, ("right_elbow", "right_shoulder"): 0.40,  # upper arms
    ("left_wrist", "left_elbow"): 0.35, ("right_wrist", "right_elbow"): 0.35,      # forearms
    ("left_hand", "left_wrist"): 0.12, ("right_hand", "right_wrist"): 0.12,        # hands (SMPL-24/28)
    ("left_middle1", "left_wrist"): 0.12, ("right_middle1", "right_wrist"): 0.12,  # hands (SMPL-X 58)
    ("head", "neck"): 0.30,                                                          # head
    ("spine1", "pelvis"): 0.18, ("spine2", "spine1"): 0.18,                          # spine segments
    ("spine3", "spine2"): 0.18, ("neck", "spine3"): 0.18,
}
_MAX_BONE = _by_index(_MAX_BONE_BY_NAME, schema.JOINT_NAMES)      # the 58-joint table build_real uses


JOINT_Y_RANGE = (-0.3, 3.2)     # m above the floor: a jump smash reaches ~3 m (pose_v2.postprocess uses the same)
PELVIS_Y_RANGE = (-0.3, 2.0)


def _height_gate(local, conf):
    """Height plausibility of one body-local frame (local Y = height above the floor). Joints outside
    JOINT_Y_RANGE lose their confidence (pack_joints_flat then zeroes them); the frame is dropped only when the
    pelvis itself is implausible. The old gate dropped the whole frame when any joint was > 2.0 m, losing
    overhead reaches and jumps (review 2 L10). Returns (local, conf, drop)."""
    conf = np.asarray(conf, dtype=np.float64).copy()
    y = np.asarray(local, dtype=np.float64)[:, 1]
    drop = bool(conf[0] > 0.1 and not (PELVIS_Y_RANGE[0] <= y[0] <= PELVIS_Y_RANGE[1]))
    bad = (conf > 0.1) & ((y < JOINT_Y_RANGE[0]) | (y > JOINT_Y_RANGE[1]))
    conf[bad] = 0.0
    return local, conf, drop


def _enforce_bone_lengths(joints24, conf24, parents):
    """Clamp child joints along the bone direction when bones exceed max length (name-based table)."""
    j = np.asarray(joints24, dtype=np.float64).copy()
    c = np.asarray(conf24, dtype=np.float64).copy()
    table = _MAX_BONE if len(j) == schema.NUM_JOINTS else _by_index(_MAX_BONE_BY_NAME, _joint_names(len(j)))
    for (child, parent), max_len in table.items():
        if c[child] < 1e-6 or c[parent] < 1e-6:
            continue
        bone = j[child] - j[parent]
        length = np.linalg.norm(bone)
        if length > max_len and length > 1e-6:
            j[child] = j[parent] + bone * (max_len / length)
            c[child] *= max_len / length
    return j, c


def _enforce_median_bone_lengths(pose_frames, parents):
    """Measure median bone lengths from good frames, then strictly enforce
    on every frame so limb lengths stay constant across the clip.

    Traverses root-to-leaf so that moving a parent (e.g. knee) correctly
    cascades to its children (e.g. ankle, foot).
    """
    n_joints = len(parents)
    children_of = {j: [] for j in range(n_joints)}
    bone_joints = []
    for j in range(n_joints):
        p = parents[j]
        if p >= 0:
            children_of[p].append(j)
            bone_joints.append(j)

    # BFS order from root(s) guarantees parent is processed before child
    from collections import deque
    roots = [j for j in range(n_joints) if parents[j] < 0]
    bfs_order = []
    q = deque(roots)
    while q:
        node = q.popleft()
        bfs_order.append(node)
        for ch in children_of[node]:
            q.append(ch)
    # Only joints that have a parent (i.e. bones)
    bfs_bones = [j for j in bfs_order if parents[j] >= 0]

    # Collect bone lengths from confident frames
    bone_lengths = {j: [] for j in bone_joints}
    for pf in pose_frames:
        jf = pf.get("joints_flat")
        if jf is None:
            continue
        arr = np.array(jf, dtype=np.float64).reshape(-1, 4)
        for j in bone_joints:
            p = parents[j]
            if arr[j, 3] < 0.1 or arr[p, 3] < 0.1:
                continue
            bl = np.linalg.norm(arr[j, :3] - arr[p, :3])
            if 0.01 < bl < 1.0:
                bone_lengths[j].append(bl)

    med_lengths = {}
    for j, lengths in bone_lengths.items():
        if len(lengths) >= 10:
            med_lengths[j] = float(np.median(lengths))

    # Strictly enforce median length on every frame, root-to-leaf
    for pf in pose_frames:
        jf = pf.get("joints_flat")
        if jf is None:
            continue
        arr = np.array(jf, dtype=np.float64).reshape(-1, 4)
        for j in bfs_bones:
            if j not in med_lengths:
                continue
            p = parents[j]
            if arr[j, 3] < 0.1 or arr[p, 3] < 0.1:
                continue
            bone = arr[j, :3] - arr[p, :3]
            bl = np.linalg.norm(bone)
            if bl < 1e-6:
                continue
            target = med_lengths[j]
            arr[j, :3] = arr[p, :3] + bone * (target / bl)
        pf["joints_flat"] = [round(float(v), 5) for v in arr.ravel()]
    return pose_frames


def _interpolate_gaps(pose_frames, max_gap=10):
    """Fill short gaps (≤max_gap frames) by linear interpolation."""
    n = len(pose_frames)
    i = 0
    while i < n:
        if pose_frames[i].get("joints_flat") is not None:
            i += 1
            continue
        gap_start = i
        while i < n and pose_frames[i].get("joints_flat") is None:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        if gap_len > max_gap:
            continue
        if gap_start == 0 or gap_end >= n:
            continue
        prev = pose_frames[gap_start - 1]
        nxt = pose_frames[gap_end]
        if prev.get("joints_flat") is None or nxt.get("joints_flat") is None:
            continue
        jf_a = np.array(prev["joints_flat"], dtype=np.float64)
        jf_b = np.array(nxt["joints_flat"], dtype=np.float64)
        pos_a = prev.get("position")
        pos_b = nxt.get("position")
        for k in range(gap_len):
            t = (k + 1) / (gap_len + 1)
            interp = jf_a + t * (jf_b - jf_a)
            for j in range(3, len(interp), 4):
                interp[j] = min(jf_a[j], jf_b[j])
            pf = {"joints_flat": [round(float(v), 5) for v in interp]}
            if pos_a and pos_b:
                pf["position"] = [
                    round(pos_a[0] + t * (pos_b[0] - pos_a[0]), 4),
                    round(pos_a[1] + t * (pos_b[1] - pos_a[1]), 4)]
            else:
                pf["position"] = pos_a or pos_b
            pose_frames[gap_start + k] = pf
    return pose_frames


def _reject_position_jumps(pose_frames, max_jump=1.5):
    """Null out frames where position jumps impossibly far from neighbors."""
    positions = []
    for pf in pose_frames:
        pos = pf.get("position")
        positions.append(np.array(pos) if pos else None)

    n_rejected = 0
    for i in range(1, len(positions) - 1):
        if positions[i] is None:
            continue
        prev_p = next((positions[k] for k in range(i - 1, -1, -1)
                        if positions[k] is not None), None)
        next_p = next((positions[k] for k in range(i + 1, len(positions))
                        if positions[k] is not None), None)
        if prev_p is None and next_p is None:
            continue
        d_prev = np.linalg.norm(positions[i] - prev_p) if prev_p is not None else 0
        d_next = np.linalg.norm(positions[i] - next_p) if next_p is not None else 0
        if (prev_p is not None and d_prev > max_jump and
                next_p is not None and d_next > max_jump):
            pose_frames[i] = {"joints_flat": None, "position": None}
            positions[i] = None
            n_rejected += 1
        elif ((prev_p is not None and d_prev > max_jump * 2 and next_p is None) or
              (next_p is not None and d_next > max_jump * 2 and prev_p is None)):
            pose_frames[i] = {"joints_flat": None, "position": None}
            positions[i] = None
            n_rejected += 1
    if n_rejected:
        print(f"    position jump rejection: {n_rejected} frames nulled")
    return pose_frames


_KALMAN_ACCEL_NOISE = 15.0    # m/s^2 -- badminton lateral acceleration
_KALMAN_MEASURE_NOISE = 0.20  # m -- triangulation measurement noise


def _kalman_smooth_positions(pose_frames, fps=60.0):
    """Kalman filter on court (X, Z) position with constant-velocity model.

    Unlike EMA which smears fast movements, the velocity model predicts
    where the player should be next frame, so genuine fast movements are
    tracked accurately while frame-to-frame jitter is smoothed out.
    Outlier rejection is handled upstream by _reject_position_jumps.
    """
    dt = 1.0 / fps

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64)

    sa = _KALMAN_ACCEL_NOISE
    G = np.array([[dt**2 / 2, 0],
                  [0, dt**2 / 2],
                  [dt, 0],
                  [0, dt]], dtype=np.float64)
    Q = sa**2 * (G @ G.T)
    R = np.eye(2) * _KALMAN_MEASURE_NOISE**2

    state = None
    P = None

    for pf in pose_frames:
        pos = pf.get("position")
        if pos is None:
            state = None
            P = None
            continue

        meas = np.array([pos[0], pos[1]], dtype=np.float64)

        if state is None:
            state = np.array([meas[0], meas[1], 0.0, 0.0], dtype=np.float64)
            P = np.diag([_KALMAN_MEASURE_NOISE**2, _KALMAN_MEASURE_NOISE**2,
                         25.0, 25.0])
            continue

        state_pred = F @ state
        P_pred = F @ P @ F.T + Q

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        innovation = meas - H @ state_pred
        state = state_pred + K @ innovation
        P = (np.eye(4) - K @ H) @ P_pred
        pf["position"] = [round(float(state[0]), 4),
                          round(float(state[1]), 4)]

    return pose_frames


_SWAP_MIN_RUN = 10
_SWAP_ANCHOR_MARGIN = 1.5
_SWAP_ANCHOR_WINDOW = 200


def _detect_sustained_swaps(player_data):
    """Null stretches where a player's position is closer to another's anchor.

    Scans each player's 3D court position for runs of 10+ consecutive frames
    where the position is closer to a different player's anchor zone (by
    at least ANCHOR_MARGIN meters). Nulls joints_flat and position for those
    frames so _interpolate_gaps() can fill short ones.

    Must be called on the full player_data list (not per-player) since it
    compares positions across players.
    """
    n_players = len(player_data)
    if n_players < 2:
        return player_data

    pfs = [pd["pose_frames"] for pd in player_data]
    frame_count = min(len(pf) for pf in pfs)

    # Compute anchor zones from first ANCHOR_WINDOW frames
    anchors = []
    for i in range(n_players):
        pts = []
        for f in range(min(_SWAP_ANCHOR_WINDOW, frame_count)):
            pos = pfs[i][f].get("position")
            if pos is not None:
                pts.append(pos)
        if pts:
            anchors.append(np.median(pts, axis=0))
        else:
            anchors.append(None)

    # For each player, find sustained stretches in wrong anchor zone
    total_nulled = 0
    for i in range(n_players):
        if anchors[i] is None:
            continue
        own_anchor = np.array(anchors[i])

        # Build per-frame: is this frame closer to another player's anchor?
        in_wrong_zone = [False] * frame_count
        for f in range(frame_count):
            pos = pfs[i][f].get("position")
            if pos is None:
                continue
            pos_arr = np.array(pos)
            d_own = np.linalg.norm(pos_arr - own_anchor)
            for j in range(n_players):
                if j == i or anchors[j] is None:
                    continue
                d_other = np.linalg.norm(pos_arr - np.array(anchors[j]))
                if d_other < d_own - _SWAP_ANCHOR_MARGIN:
                    in_wrong_zone[f] = True
                    break

        # Find runs of 10+ consecutive wrong-zone frames
        run_start = None
        for f in range(frame_count + 1):
            if f < frame_count and in_wrong_zone[f]:
                if run_start is None:
                    run_start = f
            else:
                if run_start is not None:
                    run_len = f - run_start
                    if run_len >= _SWAP_MIN_RUN:
                        for k in range(run_start, f):
                            pfs[i][k] = {"joints_flat": None, "position": None}
                            total_nulled += 1
                    run_start = None

    if total_nulled:
        print(f"  sustained-swap detection: {total_nulled} frames nulled")
    return player_data


def _smooth_poses(pose_frames, alpha=0.25, fps=60.0):
    """Reject jumps, enforce bone lengths, interpolate gaps, Kalman-smooth
    positions, and EMA-smooth joints."""
    pose_frames = _reject_position_jumps(pose_frames)
    pose_frames = _enforce_median_bone_lengths(pose_frames, schema.PARENTS)
    pose_frames = _interpolate_gaps(pose_frames)

    # Kalman filter on court position (velocity-aware, rejects teleportation)
    pose_frames = _kalman_smooth_positions(pose_frames, fps=fps)

    # EMA on body-local joints only (position already handled by Kalman)
    return _ema_joints(pose_frames, alpha)


_CONF_VALID = 0.1      # pack_joints_flat zeroes the xyz of joints below this: they are placeholders, not data


def _ema_joints(pose_frames, alpha=0.25):
    """Causal EMA per joint. A joint is reset (not blended) when it or its previous value is a dropout
    (confidence < 0.1, xyz zeroed): blending in the zeroed placeholder put the returning joint on the floor
    (review 2 M2)."""
    prev = None
    for pf in pose_frames:
        jf = pf.get("joints_flat")
        if jf is None:
            prev = None
            continue
        cur = np.array(jf, dtype=np.float64).reshape(-1, 4)
        if prev is not None and len(prev) == len(cur):
            ok = (cur[:, 3] >= _CONF_VALID) & (prev[:, 3] >= _CONF_VALID)
            cur[ok, :3] = alpha * cur[ok, :3] + (1 - alpha) * prev[ok, :3]
            pf["joints_flat"] = [round(float(v), 5) for v in cur.ravel()]
        prev = cur
    return pose_frames


def _joint_runs(valid):
    """Contiguous True runs of a boolean array -> [(start, end)]."""
    edges = np.diff(np.r_[0, np.asarray(valid, dtype=int), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def _per_joint_runs(pose_frames):
    """(data (n, J, 4) NaN where the frame is missing, J, {joint: [(start, end)]}) with runs over frames where the
    joint is a measurement (confidence >= 0.1), so dropouts never enter a filter (review 2 M2)."""
    n = len(pose_frames)
    nj = next((len(pf["joints_flat"]) // 4 for pf in pose_frames if pf.get("joints_flat") is not None), 0)
    data = np.full((n, nj, 4), np.nan)
    for i, pf in enumerate(pose_frames):
        jf = pf.get("joints_flat")
        if jf is not None and len(jf) == nj * 4:
            data[i] = np.asarray(jf, dtype=np.float64).reshape(nj, 4)
    conf = np.nan_to_num(data[:, :, 3], nan=0.0)
    runs = {j: _joint_runs(conf[:, j] >= _CONF_VALID) for j in range(nj)}
    return data, nj, runs


def _write_back(pose_frames, data):
    for i, pf in enumerate(pose_frames):
        if pf.get("joints_flat") is not None and np.isfinite(data[i]).all():
            pf["joints_flat"] = [round(float(v), 5) for v in data[i].ravel()]
    return pose_frames


def _sg_smooth_joints(pose_frames, window=11, polyorder=3):
    """Savitzky-Golay filter on body-local joints, replacing EMA.

    SG fits a local polynomial to each joint's trajectory, preserving the
    shape of fast movements (smashes, lunges) instead of lagging behind them
    like EMA does. Each joint is filtered on its own contiguous measured runs (dropouts excluded).
    """
    from scipy.signal import savgol_filter

    data, nj, runs = _per_joint_runs(pose_frames)
    for j in range(nj):
        for s, e in runs[j]:
            seg_len = e - s
            if seg_len < window:
                w = seg_len if seg_len % 2 else seg_len - 1   # largest odd window that fits (review 2 L8)
                if w < 5:
                    continue  # segment too short for any SG window
                po = min(polyorder, w - 1)
            else:
                w = window
                po = polyorder
            data[s:e, j, :3] = savgol_filter(data[s:e, j, :3], w, po, axis=0)
    return _write_back(pose_frames, data)


def _bspline_smooth_joints(pose_frames, knot_spacing=5, spline_order=3):
    """Fit cubic B-splines to each joint's trajectory across time.

    Unlike SG (local polynomial), B-splines fit a global smooth curve with
    control points every *knot_spacing* frames.  This enforces continuity
    across the entire segment and produces very smooth output while
    preserving the overall motion shape. Each joint is fitted on its own contiguous measured runs (dropouts
    excluded).
    """
    from scipy.interpolate import make_lsq_spline

    k = spline_order
    min_points = k + 2  # need at least k+2 data points for a spline of order k
    data, nj, runs = _per_joint_runs(pose_frames)
    for j in range(nj):
        for s, e in runs[j]:
            seg_len = e - s
            if seg_len < min_points:
                continue
            t = np.arange(seg_len, dtype=np.float64)
            # Interior knots spaced every knot_spacing frames
            n_interior = max(1, (seg_len - 1) // knot_spacing - 1)
            interior_knots = np.linspace(t[1], t[-2], n_interior + 2)[1:-1]
            if len(interior_knots) == 0:
                interior_knots = np.array([(t[0] + t[-1]) / 2.0])
            # Full knot vector: k+1 copies at each end + interior
            knots = np.concatenate([np.full(k + 1, t[0]), interior_knots, np.full(k + 1, t[-1])])
            for axis in range(3):
                try:
                    spl = make_lsq_spline(t, data[s:e, j, axis], knots, k=k)
                    data[s:e, j, axis] = spl(t)
                except (ValueError, np.linalg.LinAlgError):
                    pass  # keep original if spline fit fails
    return _write_back(pose_frames, data)


# ---- Biomechanical constraints ----

# Joint angle limits: (parent, joint, child) → (min_angle_deg, max_angle_deg)
# Angle is measured at `joint` between vectors parent→joint and joint→child.
# 180° = fully extended (straight), 0° = fully folded.
_ANGLE_LIMITS_BY_NAME = {
    # Knees: can't hyperextend (>175°) or fold past ~30°
    ("left_hip", "left_knee", "left_ankle"): (30.0, 175.0),
    ("right_hip", "right_knee", "right_ankle"): (30.0, 175.0),
    # Elbows: can't hyperextend or fold past ~20°
    ("left_shoulder", "left_elbow", "left_wrist"): (20.0, 175.0),
    ("right_shoulder", "right_elbow", "right_wrist"): (20.0, 175.0),
    # Ankles: dorsiflexion/plantarflexion (toe joint: big_toe in SMPL-X 58, foot in SMPL-24/28)
    ("left_knee", "left_ankle", "left_big_toe"): (60.0, 160.0),
    ("right_knee", "right_ankle", "right_big_toe"): (60.0, 160.0),
    ("left_knee", "left_ankle", "left_foot"): (60.0, 160.0),
    ("right_knee", "right_ankle", "right_foot"): (60.0, 160.0),
    # Spine segments: limited bend per segment
    ("pelvis", "spine1", "spine2"): (120.0, 180.0),
    ("spine1", "spine2", "spine3"): (120.0, 180.0),
    ("spine2", "spine3", "neck"): (120.0, 180.0),
    # Neck
    ("spine3", "neck", "head"): (100.0, 180.0),
}
_ANGLE_LIMITS = _by_index(_ANGLE_LIMITS_BY_NAME, schema.JOINT_NAMES)

# Per-joint max velocity in m/frame at 60fps.
# Generous limits — anything beyond these is measurement noise.
_MAX_JOINT_VELOCITY_BY_NAME = {
    # Extremities can move fast relative to pelvis
    "left_wrist": 0.40, "right_wrist": 0.40,
    "left_hand": 0.45, "right_hand": 0.45,            # SMPL-24/28
    "left_middle1": 0.45, "right_middle1": 0.45,      # SMPL-X 58
    "left_elbow": 0.35, "right_elbow": 0.35,
    # Legs
    "left_ankle": 0.35, "right_ankle": 0.35,
    "left_big_toe": 0.40, "right_big_toe": 0.40, "left_foot": 0.40, "right_foot": 0.40,
    "left_knee": 0.30, "right_knee": 0.30,
    # Core (moves slower relative to pelvis)
    "head": 0.20,
    "neck": 0.15,
    "spine1": 0.10, "spine2": 0.10, "spine3": 0.10,
    "left_hip": 0.15, "right_hip": 0.15,
    "left_collar": 0.12, "right_collar": 0.12,
    "left_shoulder": 0.25, "right_shoulder": 0.25,
}
_MAX_JOINT_VELOCITY = _by_index(_MAX_JOINT_VELOCITY_BY_NAME, schema.JOINT_NAMES)


def _clamp_joint_angles(pose_frames, parents):
    """Clamp joint angles to biomechanical limits.

    For each (parent, joint, child) triplet with defined limits, compute the
    angle at `joint` and project `child` toward/away from the limit if
    violated.  Processes in BFS order so parent corrections cascade.
    """
    n_joints = len(parents)
    children_of = {j: [] for j in range(n_joints)}
    for j in range(n_joints):
        if parents[j] >= 0:
            children_of[parents[j]].append(j)

    from collections import deque
    roots = [j for j in range(n_joints) if parents[j] < 0]
    bfs_order = []
    q = deque(roots)
    while q:
        node = q.popleft()
        bfs_order.append(node)
        for ch in children_of[node]:
            q.append(ch)

    # Pre-sort angle constraints by BFS position of the joint (middle element)
    bfs_rank = {j: i for i, j in enumerate(bfs_order)}
    limits = _ANGLE_LIMITS if n_joints == schema.NUM_JOINTS else _by_index(_ANGLE_LIMITS_BY_NAME, _joint_names(n_joints))
    sorted_limits = sorted(limits.items(),
                           key=lambda x: bfs_rank.get(x[0][1], 0))

    for pf in pose_frames:
        jf = pf.get("joints_flat")
        if jf is None:
            continue
        arr = np.array(jf, dtype=np.float64).reshape(-1, 4)

        for (p_idx, j_idx, c_idx), (min_deg, max_deg) in sorted_limits:
            if (arr[p_idx, 3] < 0.1 or arr[j_idx, 3] < 0.1
                    or arr[c_idx, 3] < 0.1):
                continue

            v1 = arr[p_idx, :3] - arr[j_idx, :3]  # joint→parent
            v2 = arr[c_idx, :3] - arr[j_idx, :3]  # joint→child
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if min_deg <= angle_deg <= max_deg:
                continue

            # Clamp to nearest limit
            target_deg = min_deg if angle_deg < min_deg else max_deg
            target_rad = np.radians(target_deg)

            # Rotate child around the axis perpendicular to the plane
            axis = np.cross(v1, v2)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-8:
                continue
            axis /= axis_len

            # Angle to rotate from current to target
            delta_rad = np.radians(target_deg - angle_deg)
            cos_d = np.cos(delta_rad)
            sin_d = np.sin(delta_rad)

            # Rodrigues rotation of v2 around axis by delta_rad
            v2_rot = (v2 * cos_d
                      + np.cross(axis, v2) * sin_d
                      + axis * np.dot(axis, v2) * (1 - cos_d))
            arr[c_idx, :3] = arr[j_idx, :3] + v2_rot

        pf["joints_flat"] = [round(float(v), 5) for v in arr.ravel()]
    return pose_frames


def _clamp_joint_velocities(pose_frames, fps=60.0):
    """Cap per-joint velocity at physical limits.

    When a joint moves faster than its limit between consecutive frames,
    interpolate it toward the previous position to respect the cap.
    """
    scale = 60.0 / fps  # adjust limits if fps != 60

    prev_arr = None
    for pf in pose_frames:
        jf = pf.get("joints_flat")
        if jf is None:
            prev_arr = None
            continue
        arr = np.array(jf, dtype=np.float64).reshape(-1, 4)

        if prev_arr is not None and len(prev_arr) == len(arr):
            table = (_MAX_JOINT_VELOCITY if len(arr) == schema.NUM_JOINTS
                     else _by_index(_MAX_JOINT_VELOCITY_BY_NAME, _joint_names(len(arr))))
            for j_idx, max_vel in table.items():
                if arr[j_idx, 3] < 0.1 or prev_arr[j_idx, 3] < 0.1:
                    continue
                delta = arr[j_idx, :3] - prev_arr[j_idx, :3]
                dist = np.linalg.norm(delta)
                limit = max_vel * scale
                if dist > limit and dist > 1e-6:
                    arr[j_idx, :3] = prev_arr[j_idx, :3] + delta * (limit / dist)
            pf["joints_flat"] = [round(float(v), 5) for v in arr.ravel()]

        prev_arr = arr.copy()
    return pose_frames


def pack_joints_flat(joints24_local, conf24):
    j = np.asarray(joints24_local, dtype=np.float64).copy()
    c = np.asarray(conf24, dtype=np.float64)
    j[c < 0.1] = 0.0
    flat = []
    for i in range(schema.NUM_JOINTS):
        flat.extend([round(float(j[i, 0]), 5),
                     round(float(j[i, 1]), 5),
                     round(float(j[i, 2]), 5),
                     round(float(c[i]), 4)])
    return flat


def build_real_doc(clip_id, fps, frame_count, pose_frames, side):
    if len(pose_frames) != frame_count:
        raise ValueError(f"{len(pose_frames)} pose frames vs frame_count {frame_count}")
    frames = []
    for f, pf in enumerate(pose_frames):
        frames.append({
            "frame_id": f, "time": round(f / fps, 4),
            "joints_flat": pf.get("joints_flat"),
            "position": pf.get("position"),
            "racket": None, "grip": None, "stroke": None,
        })
    player = {"id": "P1", "side": side, "frames": frames}
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "clip_id": clip_id,
        "meta": {"fps": round(float(fps), 3), "frame_count": frame_count,
                 "source_clip": clip_id, "camera": None,
                 "producers": {"pose": "triangulated_rtmpose", "position": None,
                               "racket": None, "grip": None,
                               "stroke": None, "shuttle": None}},
        "court": schema.court_block(),
        "skeleton": {"type": schema.SKELETON_TYPE,
                     "joint_names": schema.JOINT_NAMES, "parents": schema.PARENTS},
        "players": [player],
        "shuttle": {"frames": []},
        "rallies": [],
    }


def build_real_doc_multi(clip_id, fps, frame_count, player_data,
                         cam_meta=None, mode=None):
    """Build schema-valid doc with N players.

    player_data: [{id, side, pose_frames}] where pose_frames[f] has
    'joints_flat' and 'position'.
    cam_meta: optional list of camera info dicts for meta.cameras.
    """
    players = []
    for pd in player_data:
        pf = pd["pose_frames"]
        if len(pf) != frame_count:
            raise ValueError(f"player {pd['id']}: {len(pf)} frames vs {frame_count}")
        frames = []
        for f, fr in enumerate(pf):
            frames.append({
                "frame_id": f, "time": round(f / fps, 4),
                "joints_flat": fr.get("joints_flat"),
                "position": fr.get("position"),
                "racket": None, "grip": None, "stroke": None,
            })
        players.append({"id": pd["id"], "side": pd["side"], "frames": frames})

    meta = {"fps": round(float(fps), 3), "frame_count": frame_count,
            "source_clip": clip_id, "camera": None,
            "producers": {"pose": "triangulated_rtmpose", "position": None,
                          "racket": None, "grip": None,
                          "stroke": None, "shuttle": None}}
    if cam_meta:
        meta["cameras"] = cam_meta
    if mode:
        meta["mode"] = mode

    return {
        "schema_version": schema.SCHEMA_VERSION,
        "clip_id": clip_id,
        "meta": meta,
        "court": schema.court_block(),
        "skeleton": {"type": schema.SKELETON_TYPE,
                     "joint_names": schema.JOINT_NAMES, "parents": schema.PARENTS},
        "players": players,
        "shuttle": {"frames": []},
        "rallies": [],
    }


_RACKET_KP = ["top", "bottom", "handle", "left", "right"]
_HANDLE_IDX = _RACKET_KP.index("handle")
_LWRIST, _RWRIST = 9, 10  # body keypoint indices (same for COCO-17 and Wholebody-133)


def triangulate_racket(cam_paths, racket2d_paths, frame_count, score_thr=0.15,
                       synced_rkp=None):
    """Triangulate RacketVision 5-keypoint racket from multi-view 2D detections.

    Returns {player_id: {"grip": (N,3), "head": (N,3), "side": (N,3),
    "status": [str]*N, "confidence": [float]*N}} or None if no racket data.

    synced_rkp: optional {view: [frames]} — pre-loaded and synced racket data.
    When provided, racket2d_paths is not used for loading.
    """
    from analysis import multiview as mv, triangulate as tri

    cams = {v: mv.projection_matrix(*_load_cam(p)) for v, p in cam_paths.items()}
    if synced_rkp is not None:
        rkp = synced_rkp
    else:
        rkp = {}
        for v, p in racket2d_paths.items():
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            rkp[v] = doc["frames"]

    grips = np.zeros((frame_count, 3), dtype=np.float64)
    heads = np.zeros((frame_count, 3), dtype=np.float64)
    sides = np.zeros((frame_count, 3), dtype=np.float64)
    status = ["missing"] * frame_count
    confidence = [0.0] * frame_count

    n_ok = 0
    for f in range(frame_count):
        pts_3d = {}
        confs = []
        for j_idx, j_name in enumerate(_RACKET_KP):
            per_view = {}
            for v in cams:
                frames = rkp.get(v)
                if frames is None or f >= len(frames) or frames[f] is None:
                    continue
                raw = frames[f]
                # Handle both old (single) and new (multi-candidate) formats
                if isinstance(raw[0], list) and isinstance(raw[0][0], list):
                    det = raw[0]  # best candidate
                else:
                    det = raw
                per_view[v] = det[j_idx]  # [x, y, score]
            X, conf, _err = tri.triangulate_joint(cams, per_view, score_thr)
            if X is not None:
                pts_3d[j_name] = X
                confs.append(conf)
        if "handle" in pts_3d and "top" in pts_3d:
            # Y-flip to match pose convention (PnP gives Y-down)
            for k in pts_3d:
                pts_3d[k][1] *= -1.0
            grips[f] = pts_3d["handle"]
            heads[f] = pts_3d["top"]
            # prefer left for side; fall back to right or synthesize
            if "left" in pts_3d:
                sides[f] = pts_3d["left"]
            elif "right" in pts_3d:
                sides[f] = pts_3d["right"]
            else:
                shaft = heads[f] - grips[f]
                across = np.cross(shaft, np.array([0.0, 1.0, 0.0]))
                n = np.linalg.norm(across)
                if n > 1e-9:
                    across /= n
                sides[f] = heads[f] + 0.10 * across
            status[f] = "detected"
            confidence[f] = float(np.mean(confs)) if confs else 0.5
            n_ok += 1

    print(f"  racket triangulated: {n_ok}/{frame_count} frames "
          f"({100*n_ok/max(frame_count,1):.1f}%)")
    if n_ok == 0:
        return None
    return {"P1": {"grip": grips, "head": heads, "side": sides,
                    "status": status, "confidence": confidence}}


def _to_bodylocal_racket(tracks, pose_frames):
    """Subtract each frame's pelvis XZ from racket grip/head/side to match body-local."""
    for pid, tr in tracks.items():
        for key in ("grip", "head", "side"):
            arr = tr[key].copy()
            for f in range(len(arr)):
                pf = pose_frames[f]
                pos = pf.get("position")
                if pos is not None and tr["status"][f] != "missing":
                    arr[f, 0] -= pos[0]
                    arr[f, 2] -= pos[1]
            tr[key] = arr
    return tracks


RACKET_WRIST_MAX = 150      # px — max handle-to-wrist distance for assignment
RACKET_STICKY_BONUS = 50.0  # px — temporal consistency bonus
RACKET_STICKY_PX = 40.0     # px — same racket as last frame when its handle is this close


def _sync_racket2d(racket2d_paths, offsets, frame_count):
    """Load and temporally sync racket2d frames using pose-derived offsets.

    Applies integer-rounded offsets (shift + trim to frame_count).
    Returns {view: [synced_frames]} ready for assignment/triangulation.
    """
    synced = {}
    for v, p in racket2d_paths.items():
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        frames = doc["frames"]
        off = int(round(offsets.get(v, 0.0)))
        if off > 0:
            shifted = frames[off:]
        elif off < 0:
            shifted = [None] * (-off) + frames
        else:
            shifted = list(frames)
        if len(shifted) >= frame_count:
            synced[v] = shifted[:frame_count]
        else:
            synced[v] = shifted + [None] * (frame_count - len(shifted))
    return synced


RACKET_LINK_PX = 60.0       # px — a candidate continues a racket track when its handle is this close


def _racket_candidates(raw):
    """One racket2d frame -> list of candidates (5 x [x, y, s]); accepts the old single-detection frame
    (5 x [x, y, s]) and the multi-candidate one (review 2 L6). None / empty -> []."""
    if not raw or not isinstance(raw[0], list) or not raw[0]:
        return []
    return [raw] if not isinstance(raw[0][0], list) else raw


def _link_racket_tracks(frames, link_px=RACKET_LINK_PX):
    """Per frame, a track id per candidate: candidates are re-sorted by detection score every frame, so the
    index is no identity (review 2 L5); a candidate continues the previous frame's track whose handle is
    nearest (Hungarian, <= link_px), else it starts a new track."""
    from scipy.optimize import linear_sum_assignment as lsa
    ids, prev, next_id = [], [], 0
    for cands in frames:
        cur = [None] * len(cands)
        if cands and prev:
            cost = np.array([[np.hypot(c[_HANDLE_IDX][0] - p[_HANDLE_IDX][0], c[_HANDLE_IDX][1] - p[_HANDLE_IDX][1])
                              for _tid, p in prev] for c in cands])
            for r, k in zip(*lsa(np.where(cost <= link_px, cost, 1e6))):
                if cost[r, k] <= link_px:
                    cur[r] = prev[k][0]
        for i in range(len(cands)):
            if cur[i] is None:
                cur[i] = next_id
                next_id += 1
        ids.append(cur)
        prev = list(zip(cur, cands))
    return ids


def _smooth_racket_2d(rkp_by_view, window=5, order=2):
    """SG-smooth racket keypoint (x,y) trajectories per view.

    Candidates are linked into racket tracks by handle proximity (not by candidate index, which follows the
    per-frame score order) and each track is filtered within its contiguous runs. Does not interpolate
    across gaps. Scores are preserved. Frames come back in the multi-candidate format.
    """
    from scipy.signal import savgol_filter

    result = {}
    for v, frames in rkp_by_view.items():
        n = len(frames)
        cands = [[[list(kp) for kp in c] for c in _racket_candidates(frames[f])] for f in range(n)]
        out = [c if frames[f] is not None else None for f, c in enumerate(cands)]
        ids = _link_racket_tracks(cands)
        tracks = {}
        for f in range(n):
            for ci, tid in enumerate(ids[f]):
                tracks.setdefault(tid, []).append((f, ci))
        for members in tracks.values():
            # contiguous runs of the track (frame indices consecutive)
            runs, cur = [], [members[0]]
            for m in members[1:]:
                if m[0] == cur[-1][0] + 1:
                    cur.append(m)
                else:
                    runs.append(cur)
                    cur = [m]
            runs.append(cur)
            for run in runs:
                seg_len = len(run)
                if seg_len < 3:
                    continue
                w, po = window, order
                if seg_len < w:
                    # round DOWN to the nearest odd value <= seg_len (seg_len | 1 would overshoot)
                    w = seg_len if seg_len % 2 == 1 else seg_len - 1
                    w = max(3, w)
                    po = min(po, w - 1)
                for kp_idx in range(5):
                    xs = np.array([cands[f][ci][kp_idx][0] for f, ci in run], dtype=np.float64)
                    ys = np.array([cands[f][ci][kp_idx][1] for f, ci in run], dtype=np.float64)
                    xs_s, ys_s = savgol_filter(xs, w, po), savgol_filter(ys, w, po)
                    for i, (f, ci) in enumerate(run):
                        cands[f][ci][kp_idx][0] = round(float(xs_s[i]), 2)
                        cands[f][ci][kp_idx][1] = round(float(ys_s[i]), 2)
                        # score at index 2 is preserved
        result[v] = out
    return result


def _assign_racket_to_players(matched, racket2d_paths, views, frame_count,
                               player_ids, use_hungarian=False,
                               sticky_bonus=0.0, synced_rkp=None):
    """Assign racket detections to players by 2D wrist distance.

    Supports both single-detection (old: frame = [5 kps]) and multi-candidate
    (new: frame = [[5 kps], [5 kps], ...]) formats. With multi-candidate, each
    player can receive a different candidate from the same view+frame.

    When use_hungarian=True, uses scipy.optimize.linear_sum_assignment for
    globally optimal assignment with an optional temporal sticky_bonus (px)
    that penalises switching a racket to a different player frame-to-frame.

    Returns {pid: {view: [frame_kps_or_none, ...]}} — per-player per-view racket
    keypoints ready for triangulation.
    """
    if use_hungarian:
        from scipy.optimize import linear_sum_assignment as lsa

    if synced_rkp is not None:
        rkp = synced_rkp
    else:
        rkp = {}
        for v, p in racket2d_paths.items():
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            rkp[v] = doc["frames"]

    assigned = {pid: {v: [None] * frame_count for v in views}
                for pid in player_ids}

    # Previous frame's racket handle per (view, pid) for the sticky bonus: candidates are re-sorted by score
    # every frame, so the bonus follows the racket (handle within RACKET_STICKY_PX), not a candidate index (L5)
    prev_assign = {}  # (view, pid) -> (hx, hy)

    for f in range(frame_count):
        cur_assign = {}
        for v in views:
            frames = rkp.get(v)
            if frames is None or f >= len(frames) or frames[f] is None:
                continue
            raw = frames[f]
            # Detect format: old = [[x,y,s], ...] (len-5 of 3-tuples),
            # new = [[[x,y,s], ...], [[x,y,s], ...]] (list of candidates)
            if len(raw) > 0 and isinstance(raw[0], list) and len(raw[0]) > 0:
                if isinstance(raw[0][0], list):
                    candidates = raw  # new multi-candidate format
                else:
                    candidates = [raw]  # old single-detection format
            else:
                continue

            if not use_hungarian:
                # Score each (candidate, player) pair by wrist distance
                pairs = []  # (dist, cand_idx, pid)
                for ci, cand_kps in enumerate(candidates):
                    hx, hy = cand_kps[_HANDLE_IDX][0], cand_kps[_HANDLE_IDX][1]
                    for pid in player_ids:
                        kp = matched[pid][v][f]
                        if kp is None:
                            continue
                        best_d = float("inf")
                        for widx in (_LWRIST, _RWRIST):
                            if widx < len(kp) and kp[widx] is not None:
                                wx, wy = kp[widx][0], kp[widx][1]
                                d = ((hx - wx) ** 2 + (hy - wy) ** 2) ** 0.5
                                if d < best_d:
                                    best_d = d
                        if best_d < RACKET_WRIST_MAX:
                            pairs.append((best_d, ci, pid))

                # Greedy assign: closest pair first, each candidate and player used once.
                pairs.sort()
                used_cands = set()
                used_pids = set()
                for dist, ci, pid in pairs:
                    if ci in used_cands or pid in used_pids:
                        continue
                    assigned[pid][v][f] = candidates[ci]
                    used_cands.add(ci)
                    used_pids.add(pid)
                continue

            # Hungarian assignment with wrist-max gating and sticky bonus
            n_cands = len(candidates)
            n_players = len(player_ids)
            cost = np.full((n_cands, n_players), 1e6)

            for ci, cand_kps in enumerate(candidates):
                hx, hy = cand_kps[_HANDLE_IDX][0], cand_kps[_HANDLE_IDX][1]
                for pi, pid in enumerate(player_ids):
                    kp = matched[pid][v][f]
                    if kp is None:
                        continue
                    best_d = float("inf")
                    for widx in (_LWRIST, _RWRIST):
                        if widx < len(kp) and kp[widx] is not None:
                            wx, wy = kp[widx][0], kp[widx][1]
                            d = ((hx - wx) ** 2 + (hy - wy) ** 2) ** 0.5
                            if d < best_d:
                                best_d = d
                    if best_d > RACKET_WRIST_MAX:
                        continue
                    # Apply sticky bonus if same candidate was assigned to
                    # same player in previous frame
                    bonus = 0.0
                    last = prev_assign.get((v, pid))
                    if sticky_bonus > 0 and last is not None and \
                            ((hx - last[0]) ** 2 + (hy - last[1]) ** 2) ** 0.5 <= RACKET_STICKY_PX:
                        bonus = sticky_bonus
                    cost[ci, pi] = best_d - bonus

            row_ind, col_ind = lsa(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e6:
                    continue
                pid = player_ids[c]
                assigned[pid][v][f] = candidates[r]
                cur_assign[(v, pid)] = (candidates[r][_HANDLE_IDX][0], candidates[r][_HANDLE_IDX][1])

        prev_assign = cur_assign

    return assigned


_RACKET_LENGTH = 0.66   # regulation racket length in meters
_RACKET_GAP_MAX = 10    # max gap in frames to interpolate across


def triangulate_racket_multi(cams, assigned, player_ids, frame_count,
                             score_thr=0.15):
    """Triangulate per-player racket from assigned 2D detections.

    Three passes:
    1. Full keypoint triangulation (handle+top required) → "detected"
    2. Centroid fallback for frames where some kpts triangulate but not
       both handle+top — marks "partial", orientation borrowed from neighbors
    3. Gap interpolation for gaps ≤ _RACKET_GAP_MAX frames
    """
    tracks = {}
    for pid in player_ids:
        grips = np.zeros((frame_count, 3), dtype=np.float64)
        heads = np.zeros((frame_count, 3), dtype=np.float64)
        sides = np.zeros((frame_count, 3), dtype=np.float64)
        status = ["missing"] * frame_count
        confidence = [0.0] * frame_count
        has_side = [False] * frame_count
        n_full = 0
        n_partial = 0

        # Pass 1+2: triangulate keypoints
        prev_racket_3d = {}
        for f in range(frame_count):
            pts_3d = {}
            confs = []
            for j_idx, j_name in enumerate(_RACKET_KP):
                per_view = {}
                for v in cams:
                    rk = assigned[pid].get(v)
                    if rk is None or rk[f] is None:
                        continue
                    per_view[v] = rk[f][j_idx]
                prior = prev_racket_3d.get(j_name)
                X, conf, _err = tri.triangulate_joint(
                    cams, per_view, score_thr, prior_X=prior)
                if X is not None:
                    pts_3d[j_name] = X
                    confs.append(conf)
                    prev_racket_3d[j_name] = X.copy()

            if not pts_3d:
                continue

            for k in pts_3d:
                pts_3d[k][1] *= -1.0

            if "handle" in pts_3d and "top" in pts_3d:
                grips[f] = pts_3d["handle"]
                heads[f] = pts_3d["top"]
                if "left" in pts_3d:
                    sides[f] = pts_3d["left"]
                    has_side[f] = True
                elif "right" in pts_3d:
                    sides[f] = pts_3d["right"]
                    has_side[f] = True
                else:
                    shaft = heads[f] - grips[f]
                    across = np.cross(shaft, np.array([0.0, 1.0, 0.0]))
                    n = np.linalg.norm(across)
                    if n > 1e-9:
                        across /= n
                    sides[f] = heads[f] + 0.10 * across
                status[f] = "detected"
                confidence[f] = float(np.mean(confs)) if confs else 0.5
                n_full += 1
            else:
                # Partial: at least one keypoint triangulated.
                # Use centroid as grip position; head/side filled later.
                centroid = np.mean(list(pts_3d.values()), axis=0)
                grips[f] = centroid
                status[f] = "partial"
                confidence[f] = float(np.mean(confs)) * 0.5 if confs else 0.25
                n_partial += 1

        # Pass 2b: fill orientation for "partial" frames from nearest "detected"
        _fill_partial_orientation(grips, heads, sides, status, frame_count)

        # Pass 3: gap interpolation for short gaps
        n_interp = _interpolate_racket_gaps(
            grips, heads, sides, status, confidence, frame_count)

        n_total = sum(1 for s in status if s != "missing")
        print(f"  {pid} racket: {n_total}/{frame_count} frames "
              f"({100 * n_total / max(frame_count, 1):.1f}%) "
              f"[{n_full} full, {n_partial} partial, {n_interp} interpolated]")
        tracks[pid] = {"grip": grips, "head": heads, "side": sides,
                       "status": status, "confidence": confidence,
                       "has_side": has_side}

    has_any = any(sum(1 for s in t["status"] if s != "missing") > 0
                  for t in tracks.values())
    return tracks if has_any else None


def _fill_partial_orientation(grips, heads, sides, status, frame_count):
    """For 'partial' frames, borrow shaft direction from nearest 'detected' frame
    and synthesize head/side from grip + that direction."""
    detected = [f for f in range(frame_count) if status[f] == "detected"]
    if not detected:
        return
    for f in range(frame_count):
        if status[f] != "partial":
            continue
        # Find nearest detected frame
        nearest = min(detected, key=lambda d: abs(d - f))
        shaft = heads[nearest] - grips[nearest]
        shaft_len = np.linalg.norm(shaft)
        if shaft_len < 1e-6:
            status[f] = "missing"
            continue
        shaft_dir = shaft / shaft_len
        heads[f] = grips[f] + _RACKET_LENGTH * shaft_dir
        side_offset = sides[nearest] - heads[nearest]
        sides[f] = heads[f] + side_offset


def _interpolate_racket_gaps(grips, heads, sides, status, confidence,
                             frame_count):
    """Fill gaps ≤ _RACKET_GAP_MAX frames by linear interpolation."""
    n_interp = 0
    valid = [f for f in range(frame_count) if status[f] != "missing"]
    if len(valid) < 2:
        return 0
    for i in range(len(valid) - 1):
        a, b = valid[i], valid[i + 1]
        gap = b - a - 1
        if gap < 1 or gap > _RACKET_GAP_MAX:
            continue
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            grips[f] = grips[a] * (1 - t) + grips[b] * t
            heads[f] = heads[a] * (1 - t) + heads[b] * t
            sides[f] = sides[a] * (1 - t) + sides[b] * t
            status[f] = "interpolated"
            conf_a = confidence[a] if confidence[a] > 0 else 0.3
            conf_b = confidence[b] if confidence[b] > 0 else 0.3
            confidence[f] = (conf_a * (1 - t) + conf_b * t) * 0.7
            n_interp += 1
    return n_interp


# ARCHIVED: replaced by _shaft_blend_weight
# _SHAFT_BLEND_FULL = 0.3     # blend weight for triangulated shaft on "detected" frames
# _SHAFT_BLEND_OTHER = 0.0    # blend weight for partial/interpolated (body-only)
# _SHAFT_EMA = 0.4  # ARCHIVED: replaced by SG smoothing (_SHAFT_SG_WINDOW)
_SHAFT_SG_WINDOW = 7   # frames (~117ms at 60fps)
_SHAFT_SG_ORDER = 2    # quadratic polynomial


def _reconstruct_side_dirs(shaft_dirs, tri_sides, tri_heads, has_side, valid,
                           window=_SHAFT_SG_WINDOW, order=_SHAFT_SG_ORDER):
    """Build smoothed side (face-width) direction from triangulated side data.

    On frames with has_side=True, project the triangulated side vector onto
    the plane perpendicular to the shaft. On frames without, synthesize from
    cross(shaft, world-up). Then align signs and SG-smooth the result.

    Returns (n, 3) array of unit side directions.
    """
    from racket_smoothing import align_signs

    n = len(shaft_dirs)
    side_dirs = np.zeros((n, 3))

    for f in range(n):
        if not valid[f]:
            continue
        s = shaft_dirs[f]
        if has_side[f]:
            # Project triangulated side onto plane perpendicular to shaft
            raw = tri_sides[f] - tri_heads[f]
            raw_perp = raw - np.dot(raw, s) * s
            norm = np.linalg.norm(raw_perp)
            if norm > 1e-9:
                side_dirs[f] = raw_perp / norm
                continue
        # Fallback: cross(shaft, world-up)
        across = np.cross(s, np.array([0.0, 1.0, 0.0]))
        an = np.linalg.norm(across)
        if an > 1e-9:
            side_dirs[f] = across / an
        else:
            across = np.cross(s, np.array([1.0, 0.0, 0.0]))
            an = np.linalg.norm(across)
            if an > 1e-9:
                side_dirs[f] = across / an

    # Align signs to prevent flip-through-zero during smoothing
    side_dirs = align_signs(side_dirs, valid)

    # SG smooth contiguous segments
    side_dirs = _sg_smooth_shaft_dirs(side_dirs, valid, window, order)

    # Re-orthogonalize against shaft
    for f in range(n):
        if valid[f]:
            s = shaft_dirs[f]
            side_dirs[f] = side_dirs[f] - np.dot(side_dirs[f], s) * s
            norm = np.linalg.norm(side_dirs[f])
            if norm > 1e-9:
                side_dirs[f] /= norm

    return side_dirs


def _sg_smooth_shaft_dirs(dirs, valid, window=_SHAFT_SG_WINDOW, order=_SHAFT_SG_ORDER):
    """Savitzky-Golay smooth shaft direction XYZ on contiguous valid segments.

    Returns (n, 3) array of smoothed unit-length directions. Invalid entries
    are left as-is. Follows the same contiguous-segment pattern as
    _sg_smooth_joints.
    """
    from scipy.signal import savgol_filter

    out = dirs.copy()
    n = len(dirs)
    valid = np.asarray(valid, dtype=bool)

    # Find contiguous valid segments
    segments = []
    start = None
    for i in range(n + 1):
        if i < n and valid[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i))
                start = None

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < 3:
            continue
        w = window
        po = order
        if seg_len < w:
            # Clamp to the largest odd window that still fits in the segment
            # (seg_len | 1 can round UP past seg_len for even lengths).
            w = seg_len if seg_len % 2 == 1 else seg_len - 1
            w = max(3, w)
            po = min(po, w - 1)
        seg = out[seg_start:seg_end]
        for axis in range(3):
            seg[:, axis] = savgol_filter(seg[:, axis], w, po)

    # Re-normalize to unit length
    for i in range(n):
        if valid[i]:
            norm = np.linalg.norm(out[i])
            if norm > 1e-9:
                out[i] /= norm

    return out


def _shaft_blend_weight(confidence, status):
    """Confidence-adaptive triangulated shaft blend weight.

    High-confidence detected frames trust the triangulated direction more;
    low-confidence or non-detected frames fall back to body-derived direction.
    """
    if status != "detected":
        return 0.30
    if confidence > 0.4:
        return 0.70
    if confidence > 0.2:
        return 0.50
    return 0.30


def _unit_safe(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _dominant_hand(hands):
    """Return the majority hand ('left' or 'right') from a list, or None if empty."""
    if not hands:
        return None
    left = sum(1 for h in hands if h == "left")
    right = len(hands) - left
    return "left" if left > right else "right"


def _to_bodylocal_racket_multi(tracks, player_data):
    """Convert racket world coords to body-local, snap grip to hand,
    and derive shaft direction from wrist→hand body pose vector.

    The body pose wrist→hand vector is far more stable than the noisy
    5-kpt triangulated shaft direction. We use it as the primary shaft
    direction, blending in the triangulated direction only on high-confidence
    "detected" frames. A Savitzky-Golay pass then smooths the final shaft direction.
    """
    from analysis import skeleton
    names = skeleton.JOINT_NAMES
    hand_name = "middle1" if "left_middle1" in names else "hand"
    _LH = names.index(f"left_{hand_name}")
    _RH = names.index(f"right_{hand_name}")
    _LW = names.index("left_wrist")
    _RW = names.index("right_wrist")
    _LE = names.index("left_elbow")
    _RE = names.index("right_elbow")
    _HAND_CONF_MIN = 0.1

    pd_map = {pd["id"]: pd["pose_frames"] for pd in player_data}
    for pid, tr in tracks.items():
        pf = pd_map.get(pid)
        if pf is None:
            continue
        n = min(len(tr["grip"]), len(pf))

        # Step 1: convert to body-local
        for key in ("grip", "head", "side"):
            arr = tr[key].copy()
            for f in range(n):
                pos = pf[f].get("position")
                if pos is not None and tr["status"][f] != "missing":
                    arr[f, 0] -= pos[0]
                    arr[f, 2] -= pos[1]
            tr[key] = arr

        # Step 2: snap grip to dominant hand, derive shaft from wrist→hand
        def _jxyz(jf, idx):
            return np.array(jf[idx * 4:idx * 4 + 3], dtype=np.float64)

        def _hand_xyz(jf, hand_idx, wrist_idx):
            """Middle-finger MCP position, falling back to the wrist when
            the MCP joint's confidence is below _HAND_CONF_MIN (occluded hand)."""
            base = hand_idx * 4
            conf = jf[base + 3]
            if conf < _HAND_CONF_MIN:
                return _jxyz(jf, wrist_idx)
            return _jxyz(jf, hand_idx)

        # First pass: determine dominant hand by majority vote
        hand_votes = []
        for f in range(n):
            if tr["status"][f] == "missing":
                continue
            jf = pf[f].get("joints_flat")
            if jf is None:
                continue
            lh, rh = _hand_xyz(jf, _LH, _LW), _hand_xyz(jf, _RH, _RW)
            grip = tr["grip"][f]
            is_left = np.linalg.norm(grip - lh) < np.linalg.norm(grip - rh)
            hand_votes.append("left" if is_left else "right")

        dominant = _dominant_hand(hand_votes)
        use_left = (dominant == "left")

        # Second pass: snap to dominant hand, derive shaft
        shaft_dirs = [None] * n
        for f in range(n):
            if tr["status"][f] == "missing":
                continue
            jf = pf[f].get("joints_flat")
            if jf is None:
                continue

            hand = _hand_xyz(jf, _LH, _LW) if use_left else _hand_xyz(jf, _RH, _RW)
            wrist_idx, elbow_idx = (_LW, _LE) if use_left else (_RW, _RE)
            wrist = _jxyz(jf, wrist_idx)

            # Snap grip to dominant hand; triangulated shaft = triangulated head - snapped grip
            tr["grip"][f] = hand.copy()
            tri_shaft = _unit_safe(tr["head"][f] - tr["grip"][f])

            # Body-derived shaft direction: wrist → hand; when the hand joint is unseen (hand == wrist, e.g.
            # build_real triangulates no hand keypoints) the forearm (elbow → wrist) instead of a constant
            # -Z (review 2 H2); no confident forearm either: the triangulated shaft alone
            body_shaft = _unit_safe(hand - wrist)
            if body_shaft is None and jf[elbow_idx * 4 + 3] >= _HAND_CONF_MIN \
                    and jf[wrist_idx * 4 + 3] >= _HAND_CONF_MIN:
                body_shaft = _unit_safe(wrist - _jxyz(jf, elbow_idx))
            if body_shaft is None:
                shaft_dirs[f] = tri_shaft
                continue

            # Blend with triangulated direction on full-confidence frames
            blend = _shaft_blend_weight(tr["confidence"][f], tr["status"][f])
            if blend > 0:
                if tri_shaft is not None:
                    blended = body_shaft * (1 - blend) + tri_shaft * blend
                    shaft_dirs[f] = _unit_safe(blended)
                else:
                    shaft_dirs[f] = body_shaft
            else:
                shaft_dirs[f] = body_shaft

        # Step 3: SG smooth shaft direction
        # ARCHIVED: replaced by SG smoothing below
        # prev_dir = None
        # for f in range(n):
        #     if shaft_dirs[f] is None:
        #         prev_dir = None
        #         continue
        #     if prev_dir is not None:
        #         shaft_dirs[f] = _unit_safe(
        #             _SHAFT_EMA * shaft_dirs[f] + (1 - _SHAFT_EMA) * prev_dir)
        #         if shaft_dirs[f] is None:
        #             shaft_dirs[f] = prev_dir
        #     prev_dir = shaft_dirs[f]
        shaft_arr = np.zeros((n, 3))
        shaft_valid = np.zeros(n, dtype=bool)
        for f in range(n):
            if shaft_dirs[f] is not None:
                shaft_arr[f] = shaft_dirs[f]
                shaft_valid[f] = True
        shaft_arr = _sg_smooth_shaft_dirs(shaft_arr, shaft_valid)
        for f in range(n):
            if shaft_valid[f]:
                shaft_dirs[f] = shaft_arr[f]

        # ARCHIVED: Step 4 (old) reconstructed head + side from smoothed shaft,
        # always synthesizing side from cross(shaft, world-up) and discarding
        # any real triangulated left/right keypoints. Replaced below by
        # _reconstruct_side_dirs, which preserves real face orientation where
        # available.
        # for f in range(n):
        #     if tr["status"][f] == "missing" or shaft_dirs[f] is None:
        #         continue
        #     shaft_d = shaft_dirs[f]
        #     tr["head"][f] = tr["grip"][f] + _RACKET_LENGTH * shaft_d
        #     across = np.cross(shaft_d, np.array([0.0, 1.0, 0.0]))
        #     an = np.linalg.norm(across)
        #     if an > 1e-9:
        #         across /= an
        #     else:
        #         across = np.cross(shaft_d, np.array([1.0, 0.0, 0.0]))
        #         an = np.linalg.norm(across)
        #         if an > 1e-9:
        #             across /= an
        #     tr["side"][f] = tr["head"][f] + 0.10 * across

        # Step 4: reconstruct head + side with real face orientation
        # Save pre-body-local side and head for face reconstruction
        pre_sides = tr["side"].copy()
        pre_heads = tr["head"].copy()
        has_side = tr.get("has_side", [False] * n)

        # Reconstruct head from smoothed shaft
        for f in range(n):
            if tr["status"][f] == "missing" or shaft_dirs[f] is None:
                continue
            tr["head"][f] = tr["grip"][f] + _RACKET_LENGTH * shaft_dirs[f]

        # Reconstruct side using real face orientation where available
        side_valid = np.array([shaft_dirs[f] is not None and tr["status"][f] != "missing"
                               for f in range(n)])
        side_result = _reconstruct_side_dirs(
            shaft_arr, pre_sides, pre_heads,
            np.array(has_side, dtype=bool), side_valid)
        for f in range(n):
            if side_valid[f]:
                tr["side"][f] = tr["head"][f] + 0.10 * side_result[f]

    return tracks


_SHUTTLE_MIN_VIEWS = 2       # require at least this many views
_SHUTTLE_STUCK_MIN = 6       # min run of near-identical 2D positions to flag as stuck
_SHUTTLE_STUCK_PX = 8.0      # pixel movement threshold for "stuck"
_SHUTTLE_MAX_SPEED = 50.0    # m/s — shuttle rarely exceeds this in amateur play
_SHUTTLE_GAP_MAX = 8         # max gap (frames) to interpolate within a trajectory
_SHUTTLE_SG_WINDOW = 7       # Savitzky-Golay smoothing window (must be odd)
_SHUTTLE_SG_ORDER = 2        # polynomial order for SG filter


def _filter_stuck_2d(frames, stuck_min=_SHUTTLE_STUCK_MIN, px_thr=_SHUTTLE_STUCK_PX):
    """Null out runs of near-identical 2D positions (TrackNetV3 false positives).
    A real shuttle always moves between frames at 60fps."""
    out = list(frames)
    n = len(out)
    i = 0
    total_killed = 0
    while i < n:
        if out[i] is None:
            i += 1
            continue
        run_start = i
        while i + 1 < n and out[i + 1] is not None:
            dx = abs(out[i + 1][0] - out[run_start][0])
            dy = abs(out[i + 1][1] - out[run_start][1])
            if dx > px_thr or dy > px_thr:
                break
            i += 1
        run_len = i - run_start + 1
        if run_len >= stuck_min:
            for k in range(run_start, run_start + run_len):
                out[k] = None
            total_killed += run_len
        i += 1
    return out, total_killed


def _triangulate_shuttle_raw(cams, shuttle_paths, frame_count, fps=60.0,
                             score_thr=0.1):
    """Triangulate shuttle 3D from multi-view 2D and reject outliers.

    Returns (raw_points, raw_confs) — per-frame ndarray or None, after
    stuck-filter + triangulation + velocity + median rejection. No smoothing.
    """
    from analysis import triangulate as tri

    shuttle_data = {}
    for v, p in shuttle_paths.items():
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        shuttle_data[v] = doc["frames"]

    total_stuck = 0
    for v in list(shuttle_data.keys()):
        filtered, killed = _filter_stuck_2d(shuttle_data[v])
        shuttle_data[v] = filtered
        if killed:
            print(f"    {v}: removed {killed} stuck-position frames")
            total_stuck += killed
    if total_stuck:
        print(f"  total stuck 2D frames removed: {total_stuck}")

    raw = [None] * frame_count
    raw_conf = [None] * frame_count
    prev_X = None
    for f in range(frame_count):
        per_view = {}
        for v in cams:
            frames = shuttle_data.get(v)
            if frames is None or f >= len(frames) or frames[f] is None:
                continue
            x, y = frames[f]
            per_view[v] = (x, y, 1.0)

        if len(per_view) < _SHUTTLE_MIN_VIEWS:
            continue

        X, conf, _err = tri.triangulate_joint(
            cams, per_view, score_thr, prior_X=prev_X)
        if X is not None:
            X[1] *= -1.0
            if abs(X[0]) > 8.0 or abs(X[2]) > 10.0 or X[1] < -1.0 or X[1] > 12.0:
                continue
            raw[f] = X.copy()
            raw_conf[f] = conf
            prev_X = X.copy()
            prev_X[1] *= -1.0

    n_raw = sum(1 for x in raw if x is not None)
    print(f"  raw triangulated: {n_raw}/{frame_count}")

    dt = 1.0 / fps
    n_vel_rejected = 0
    for f in range(frame_count):
        if raw[f] is None:
            continue
        prev_f = next((k for k in range(f - 1, max(f - 4, -1), -1)
                       if raw[k] is not None), None)
        next_f = next((k for k in range(f + 1, min(f + 4, frame_count))
                       if raw[k] is not None), None)
        speeds = []
        if prev_f is not None:
            d = np.linalg.norm(raw[f] - raw[prev_f])
            speeds.append(d / ((f - prev_f) * dt))
        if next_f is not None:
            d = np.linalg.norm(raw[next_f] - raw[f])
            speeds.append(d / ((next_f - f) * dt))
        if speeds and all(s > _SHUTTLE_MAX_SPEED for s in speeds):
            raw[f] = None
            raw_conf[f] = None
            n_vel_rejected += 1
    if n_vel_rejected:
        print(f"  shuttle velocity outliers rejected: {n_vel_rejected}")

    win = 7
    n_med_rejected = 0
    for f in range(frame_count):
        if raw[f] is None:
            continue
        neighbours = []
        for k in range(max(0, f - win), min(frame_count, f + win + 1)):
            if k != f and raw[k] is not None:
                neighbours.append(raw[k])
        if len(neighbours) < 3:
            continue
        med = np.median(neighbours, axis=0)
        dist = np.linalg.norm(raw[f] - med)
        if dist > 3.0:
            raw[f] = None
            raw_conf[f] = None
            n_med_rejected += 1
    if n_med_rejected:
        print(f"  shuttle median outliers rejected: {n_med_rejected}")

    return raw, raw_conf


def _shuttle_velocity(shuttle_frames, f, fps):
    """Compute shuttle velocity at frame f from triangulated positions.

    Returns velocity in m/s, or None if positions are missing.
    """
    if f <= 0 or f >= len(shuttle_frames):
        return None
    pos_cur = shuttle_frames[f].get("position")
    pos_prev = shuttle_frames[f - 1].get("position")
    if pos_cur is None or pos_prev is None:
        return None
    dx = pos_cur[0] - pos_prev[0]
    dy = pos_cur[1] - pos_prev[1]
    dz = pos_cur[2] - pos_prev[2]
    dist = (dx**2 + dy**2 + dz**2) ** 0.5
    return dist * fps


def _player_wrist_speeds(player_data, f, fps):
    """Compute per-player wrist speed at frame f from 3D pose data.

    Returns {pid: speed_m_s}. Uses the faster-moving wrist (left or right).
    joints_flat is SMPL-24 packed as x,y,z,confidence per joint (stride 4,
    96 floats total); tolerate a bare x,y,z stride-3 layout (72 floats) too.
    """
    speeds = {}
    for pd in player_data:
        pid = pd["id"]
        frames = pd["pose_frames"]
        if f <= 0 or f >= len(frames):
            speeds[pid] = 0.0
            continue
        jf_cur = frames[f].get("joints_flat")
        jf_prev = frames[f - 1].get("joints_flat")
        if jf_cur is None or jf_prev is None:
            speeds[pid] = 0.0
            continue
        n = len(jf_cur)
        if n == 72:
            stride = 3
        elif n == 96:
            stride = 4
        else:
            stride = 4
        # SMPL-24 wrist indices: 20 (left), 21 (right). Each has 3 coords.
        best_speed = 0.0
        for wrist_idx in (20, 21):
            base = wrist_idx * stride
            if base + 2 >= len(jf_cur) or base + 2 >= len(jf_prev):
                continue
            dx = jf_cur[base] - jf_prev[base]
            dy = jf_cur[base + 1] - jf_prev[base + 1]
            dz = jf_cur[base + 2] - jf_prev[base + 2]
            speed = ((dx**2 + dy**2 + dz**2) ** 0.5) * fps
            if speed > best_speed:
                best_speed = speed
        speeds[pid] = best_speed
    return speeds


def triangulate_shuttle_multi(cams, shuttle_paths, frame_count, fps=60.0,
                              score_thr=0.1):
    """Triangulate shuttle 3D with legacy SG smoothing."""
    from analysis.shuttle_physics import legacy_shuttle_trajectory

    raw, raw_conf = _triangulate_shuttle_raw(
        cams, shuttle_paths, frame_count, fps=fps, score_thr=score_thr)

    positions, confs_out = legacy_shuttle_trajectory(
        raw, raw_conf, fps)

    n_ok = sum(1 for p in positions if p is not None)
    print(f"  shuttle final: {n_ok}/{frame_count} frames "
          f"({100 * n_ok / max(frame_count, 1):.1f}%)")
    return positions, confs_out


def _load_cam(path):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return (np.array(d["K"], dtype=np.float64),
            np.array(d["R"], dtype=np.float64),
            np.array(d["t"], dtype=np.float64))


def assemble_multi(cam_paths, seeds_path, kp2d_multi_paths,
                    clip_id, mode, fps, score_thr=0.3,
                    racket2d_paths=None, shuttle_paths=None,
                    shuttle_legacy_paths=None,
                    video_dir=None):
    """Multi-player assembly: person-match → per-player triangulate → doc.

    cam_paths: {view: cam.json path}
    kp2d_multi_paths: {view: multi-person kp2d.json path}
    seeds_path: path to seeds JSON
    mode: "singles" or "doubles"
    racket2d_paths: optional {view: racket2d.json path}
    shuttle_paths: optional {view: shuttle.json path}
    video_dir: unused. Previously enabled appearance-based (HSV torso
        histogram) re-ID during person matching; PlayerTracker uses body
        proportions instead, so this is accepted for CLI compatibility
        but has no effect.
    """
    from analysis import multiview as mv, triangulate as tri
    from analysis import coco17
    from analysis.player_tracker import PlayerTracker
    from analysis.frame_sync import (find_offsets_with_errors,
                                     find_offsets_multi_player, apply_offsets)

    players_list = ["P1", "P2"] if mode == "singles" else \
                   ["P1", "P2", "P3", "P4"]

    # Auto-detect keypoint format from first kp2d file
    first_kp2d = next(iter(kp2d_multi_paths.values()))
    with open(first_kp2d, "r", encoding="utf-8") as fh:
        kp_meta = json.load(fh)
    kp_format = kp_meta.get("keypoint_format", "coco17")
    if kp_format.startswith("coco_wholebody"):
        from analysis import wholebody
        to_smpl58 = wholebody.wholebody133_to_smpl58
        n_kpts = 133
        print(f"  keypoint format: {kp_format} ({n_kpts} keypoints)")
    else:
        to_smpl58 = coco17.coco17_to_smpl58
        n_kpts = 17
        print(f"  keypoint format: {kp_format} ({n_kpts} keypoints)")

    print(f"  person tracking ({mode}, {len(cam_paths)} cameras)...")
    tracker = PlayerTracker(cam_paths, kp2d_multi_paths, seeds_path, mode)
    matched = tracker.run()

    print("  detecting frame offsets (multi-player)...")
    n_frames = min(len(matched[players_list[0]][v]) for v in cam_paths)
    offsets, sync_errors = find_offsets_multi_player(
        cam_paths, matched, players_list,
        sample_end=min(600, n_frames))
    SYNC_REPROJ_MAX = 50.0
    for v, err in sync_errors.items():
        if err > SYNC_REPROJ_MAX and offsets[v] != 0:
            print(f"  WARNING: {v} sync reproj {err:.1f}px > {SYNC_REPROJ_MAX}px"
                  f" -- resetting offset to 0")
            offsets[v] = 0.0
    if any(v != 0 for v in offsets.values()):
        for pid in matched:
            matched[pid], _ = apply_offsets(matched[pid], offsets)

    cams = {v: mv.projection_matrix(*_load_cam(p))
            for v, p in cam_paths.items()}
    views = sorted(cam_paths.keys())

    frame_count = min(len(matched[players_list[0]][v]) for v in views)

    all_errs = []
    player_data = []
    # Triangulate body+foot (0-22) plus face landmarks needed for SMPL-28
    tri_indices = list(range(min(n_kpts, 23)))
    if n_kpts >= 71:
        tri_indices += [31, 53] + list(range(59, 71))

    for pid in players_list:
        side = tracker.player_side(pid)
        kp = matched[pid]  # {view: frames_list}

        pose_frames = []
        last_pos = None
        prev_world = None  # temporal prior: previous frame's world coords
        for f in range(frame_count):
            world = np.zeros((n_kpts, 3), dtype=np.float64)
            wconf = np.zeros(n_kpts, dtype=np.float64)
            ok_any = False
            for j in tri_indices:
                per_view = {}
                for v in views:
                    row = kp[v][f]
                    if row is None:
                        continue
                    per_view[v] = row[j]
                prior = prev_world[j] if prev_world is not None else None
                X, conf, err = tri.triangulate_joint(
                    cams, per_view, score_thr, prior_X=prior)
                if X is not None:
                    world[j] = X
                    wconf[j] = conf
                    ok_any = True
                    if err is not None:
                        all_errs.append(err)
            if not ok_any:
                pose_frames.append({"joints_flat": None, "position": None})
                prev_world = None
                continue
            prev_world = world.copy()
            world[:, 1] *= -1.0
            j58, c58 = to_smpl58(world, wconf)
            j58, c58 = _enforce_bone_lengths(j58, c58, schema.PARENTS)
            local, position = world_to_bodylocal_and_position(j58)
            if abs(position[0]) > 5.0 or abs(position[1]) > 8.0:
                if last_pos is not None:
                    position = list(last_pos)
                else:
                    pose_frames.append({"joints_flat": None, "position": None})
                    continue
            local, c58, drop = _height_gate(local, c58)
            if drop:
                pose_frames.append({"joints_flat": None, "position": None})
                continue
            last_pos = position
            pose_frames.append({"joints_flat": pack_joints_flat(local, c58),
                                "position": position})

        pose_frames = _smooth_poses(pose_frames, fps=fps)
        n_ok = sum(1 for pf in pose_frames if pf["joints_flat"] is not None)
        print(f"  {pid}: {n_ok}/{frame_count} frames with pose "
              f"({100*n_ok/max(frame_count,1):.0f}%)")
        player_data.append({"id": pid, "side": side,
                            "pose_frames": pose_frames})

    # Identity maintained continuously by PlayerTracker — post-hoc swap
    # correction is no longer needed. Functions kept for fallback comparison.
    # if len(player_data) >= 2:
    #     player_data = _detect_sustained_swaps(player_data)
    #     # Re-smooth to fill gaps created by swap nulling
    #     for pd in player_data:
    #         pd["pose_frames"] = _interpolate_gaps(pd["pose_frames"])
    #
    # if len(player_data) == 2:
    #     player_data = _fix_3d_swaps(player_data)
    # elif len(player_data) > 2:
    #     player_data = _fix_3d_swaps_multi(player_data)

    cam_meta = []
    for v in views:
        K, R, t = _load_cam(cam_paths[v])
        with open(cam_paths[v], "r", encoding="utf-8") as fh:
            cd = json.load(fh)
        cam_meta.append({
            "name": v,
            "K": K.tolist(), "R": R.tolist(), "t": t.tolist(),
            "width": cd.get("width"), "height": cd.get("height"),
            "fov_deg": cd.get("fov_deg"),
        })

    doc = build_real_doc_multi(clip_id, fps, frame_count, player_data,
                               cam_meta=cam_meta, mode=mode)

    # Racket: assign detections to players by 2D wrist proximity, triangulate
    if racket2d_paths:
        from analysis import racket as racket_mod
        print("  racket triangulation...")
        synced_rkp = _sync_racket2d(racket2d_paths, offsets, frame_count)
        synced_rkp = _smooth_racket_2d(synced_rkp)  # NEW: 2D temporal filter
        rack_assigned = _assign_racket_to_players(
            matched, racket2d_paths, views, frame_count, players_list,
            use_hungarian=True, sticky_bonus=RACKET_STICKY_BONUS,
            synced_rkp=synced_rkp)
        rtracks = triangulate_racket_multi(
            cams, rack_assigned, players_list, frame_count, score_thr=0.15)
        if rtracks:
            rtracks = _to_bodylocal_racket_multi(rtracks, player_data)
            racket_mod.fill_player_rackets(
                doc, rtracks, "racketvision_triangulated")

    # Shuttle: triangulate from multi-view 2D detections
    if shuttle_paths:
        from analysis.shuttle import fill_shuttle
        print("  shuttle triangulation...")
        sh_positions, sh_confs = triangulate_shuttle_multi(
            cams, shuttle_paths, frame_count, fps=fps)
        fill_shuttle(doc, sh_positions, sh_confs,
                     "tracknetv3_triangulated")

        # Rally state: classify frames as rally/break/transition
        from analysis.rally_state import RallyStateMachine
        print("  rally segmentation...")
        shuttle_doc_frames = doc.get("shuttle", {}).get("frames", [])
        rsm = RallyStateMachine(fps)
        for f in range(frame_count):
            sh_frame = shuttle_doc_frames[f] if f < len(shuttle_doc_frames) else {}
            sh_pos = sh_frame.get("position")
            sh_vel = _shuttle_velocity(shuttle_doc_frames, f, fps)
            wrist_speeds = _player_wrist_speeds(player_data, f, fps)
            rsm.update(f, sh_pos, sh_vel, wrist_speeds)
        doc["rallies"] = rsm.rallies()
        n_rallies = len(doc["rallies"])
        total_rally_s = sum(r["duration_s"] for r in doc["rallies"])
        print(f"  found {n_rallies} rallies, {total_rally_s:.1f}s total")

    # Shuttle legacy: triangulate old data with SG-only smoothing for A/B comparison
    if shuttle_legacy_paths:
        from analysis.shuttle_physics import legacy_shuttle_trajectory
        print("  shuttle legacy triangulation (720p + SG)...")
        sh_leg_raw, sh_leg_conf = _triangulate_shuttle_raw(
            cams, shuttle_legacy_paths, frame_count, fps=fps)
        sh_leg_pos, sh_leg_confs = legacy_shuttle_trajectory(
            sh_leg_raw, sh_leg_conf, fps)
        fps_val = float(doc["meta"]["fps"])
        leg_frames = []
        for i, (pos, conf) in enumerate(zip(sh_leg_pos, sh_leg_confs)):
            leg_frames.append({"frame_id": i, "time": round(i / fps_val, 4),
                               "position": pos, "confidence": conf})
        doc["shuttle_legacy"] = {"frames": leg_frames}

    # Facing direction estimation
    from analysis.facing import fill_facing, frames_sign
    print("  facing direction...")
    for p in doc["players"]:      # handedness from the feet (cams here are 3x4 P; fallback: real analysis world)
        fill_facing(p["frames"], fps, sign=frames_sign(p["frames"]))

    errs = schema.validate_analysis(doc)
    if errs:
        raise ValueError("assembled doc invalid:\n  " + "\n  ".join(errs))
    doc["meta"]["reprojection_px_median"] = (
        round(float(np.median(all_errs)), 3) if all_errs else None)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble analysis.json from "
                                             "calibrated views + RTMPose keypoints.")
    sub = ap.add_subparsers(dest="command")

    # Phase 4 multi-player mode
    p_multi = sub.add_parser("multi", help="Phase 4 multi-player assembly")
    p_multi.add_argument("--cam", action="append", required=True,
                         metavar="view=cam.json")
    p_multi.add_argument("--kp2d-multi", action="append", required=True,
                         metavar="view=multi_kp2d.json",
                         help="Multi-person kp2d files from Colab")
    p_multi.add_argument("--seeds", required=True, help="Seeds JSON path")
    p_multi.add_argument("--mode", choices=["singles", "doubles"],
                         default="singles")
    p_multi.add_argument("--clip", default="singles_1x")
    p_multi.add_argument("--fps", type=float, default=60.0)
    p_multi.add_argument("--racket2d", action="append", default=None,
                         metavar="view=racket2d.json",
                         help="RacketVision 2D keypoint files per view")
    p_multi.add_argument("--shuttle", action="append", default=None,
                         metavar="view=shuttle.json",
                         help="TrackNetV3 shuttle detection files per view")
    p_multi.add_argument("--shuttle-legacy", action="append", default=None,
                         metavar="view=shuttle.json",
                         help="Old shuttle files for A/B comparison (720p SG)")
    p_multi.add_argument("--score-thr", type=float, default=0.3)
    p_multi.add_argument("--video-dir", default=None,
                         help="unused (kept for old command lines); the HSV appearance "
                              "re-ID it enabled was removed")
    p_multi.add_argument("--out", default=None)

    # Re-smooth existing analysis
    p_resmooth = sub.add_parser("resmooth",
                                help="Re-apply smoothing on existing analysis.json")
    p_resmooth.add_argument("input", help="Path to existing analysis.json")
    p_resmooth.add_argument("--mode",
                            choices=["bonelen", "sgsmooth", "bspline",
                                     "physics_sg", "physics_bsp"],
                            default="bonelen",
                            help="bonelen = strict bone enforcement; "
                                 "sgsmooth = Savitzky-Golay + strict bones; "
                                 "bspline = B-spline temporal fitting + strict bones; "
                                 "physics_sg = strict bones + SG + angle/velocity limits; "
                                 "physics_bsp = strict bones + B-spline + angle/velocity limits")
    p_resmooth.add_argument("--sg-window", type=int, default=11,
                            help="SG filter window length (odd, default 11)")
    p_resmooth.add_argument("--sg-poly", type=int, default=3,
                            help="SG polynomial order (default 3)")
    p_resmooth.add_argument("--knot-spacing", type=int, default=5,
                            help="B-spline knot spacing in frames (default 5)")
    p_resmooth.add_argument("--out", default=None,
                            help="Output path (default: input with mode suffix)")

    args = ap.parse_args(argv)

    def kv(items):
        out = {}
        for it in items:
            k, v = it.split("=", 1)
            out[k] = v
        return out

    if args.command == "multi":
        out_path = args.out or os.path.join(
            "data", "analysis", f"{args.clip}.analysis.json")
        print(f"Multi-player assembly: {args.mode}, clip={args.clip}")
        racket2d = kv(args.racket2d) if args.racket2d else None
        shuttle = kv(args.shuttle) if args.shuttle else None
        shuttle_legacy = kv(args.shuttle_legacy) if args.shuttle_legacy else None
        doc = assemble_multi(kv(args.cam), args.seeds,
                             kv(args.kp2d_multi),
                             args.clip, args.mode, args.fps, args.score_thr,
                             racket2d_paths=racket2d,
                             shuttle_paths=shuttle,
                             shuttle_legacy_paths=shuttle_legacy,
                             video_dir=args.video_dir)
        _write_doc(out_path, doc)
        med = doc["meta"].get("reprojection_px_median")
        print(f"wrote {out_path}: {doc['meta']['frame_count']} frames, "
              f"{len(doc['players'])} players, median reprojection "
              f"{med if med is not None else 'n/a'} px")
        return 0

    if args.command == "resmooth":
        in_path = args.input
        suffix = f"_{args.mode}"
        out_path = args.out or in_path.replace(".analysis.json",
                                               f"{suffix}.analysis.json")
        with open(in_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        # the doc's own skeleton (shipped base docs are 28-joint schema 4.0; review 2 L1)
        parents = (doc.get("skeleton") or {}).get("parents") or schema.PARENTS
        for p in doc["players"]:
            pf = p["frames"]
            _enforce_median_bone_lengths(pf, parents)
            if args.mode == "sgsmooth":
                _sg_smooth_joints(pf, window=args.sg_window,
                                  polyorder=args.sg_poly)
            elif args.mode == "bspline":
                _bspline_smooth_joints(pf, knot_spacing=args.knot_spacing)
                _enforce_median_bone_lengths(pf, parents)
            elif args.mode == "physics_sg":
                _sg_smooth_joints(pf, window=args.sg_window,
                                  polyorder=args.sg_poly)
                _clamp_joint_angles(pf, parents)
                _clamp_joint_velocities(pf)
                _enforce_median_bone_lengths(pf, parents)
            elif args.mode == "physics_bsp":
                _bspline_smooth_joints(pf, knot_spacing=args.knot_spacing)
                _clamp_joint_angles(pf, parents)
                _clamp_joint_velocities(pf)
                _enforce_median_bone_lengths(pf, parents)
        from analysis.facing import doc_joint_indices, fill_facing, frames_sign
        fps = doc["meta"].get("fps", 30)
        ix = doc_joint_indices(doc)               # the doc's own face / hip / shoulder joints (28 / 24 / 58)
        for p in doc["players"]:
            fill_facing(p["frames"], fps, sign=frames_sign(p["frames"]), **ix)
        _write_doc(out_path, doc)
        labels = {
            "sgsmooth": "SG smooth (window=%d, poly=%d) + strict bones"
                        % (args.sg_window, args.sg_poly),
            "bspline": "B-spline (knot_spacing=%d) + strict bones"
                       % args.knot_spacing,
            "physics_sg": "physics: strict bones + SG + angle/velocity limits",
            "physics_bsp": "physics: strict bones + B-spline + angle/velocity limits",
            "bonelen": "strict bone enforcement",
        }
        label = labels[args.mode]
        print(f"wrote {out_path}: {label}, "
              f"{len(doc['players'])} players, "
              f"{doc['meta']['frame_count']} frames")
        return 0

    if args.command is None:
        ap.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
