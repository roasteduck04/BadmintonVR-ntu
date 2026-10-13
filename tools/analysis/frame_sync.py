"""Auto-detect temporal offsets between multi-view cameras.

Phones started manually don't share a common clock. This module finds the
frame offset for each camera relative to a reference by minimising pairwise
triangulation reprojection error across a sweep of offsets.

Supports sub-frame (fractional) offsets: after finding the best integer
offset, refines to 0.1-frame resolution by interpolating 2D keypoints
between adjacent frames.

Must be called AFTER person matching so it uses correctly matched keypoints.
"""
from __future__ import annotations

import numpy as np

from analysis import multiview as mv, triangulate as tri


def _interp_kp(kp_a, kp_b, t):
    """Linearly interpolate two keypoint lists at fraction t in [0, 1]."""
    if kp_a is None or kp_b is None:
        return kp_a if t < 0.5 else kp_b
    out = []
    for a, b in zip(kp_a, kp_b):
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        s = min(a[2], b[2])
        out.append([x, y, s])
    return out


def _get_kp_at(frames, f_float):
    """Get keypoints at a fractional frame index via interpolation."""
    f_int = int(np.floor(f_float))
    frac = f_float - f_int
    if f_int < 0 or f_int >= len(frames):
        return None
    if frac < 1e-6:
        return frames[f_int]
    if f_int + 1 >= len(frames):
        return frames[f_int]
    return _interp_kp(frames[f_int], frames[f_int + 1], frac)


def _eval_offset(projs_ref, projs_v, matched_ref, matched_v,
                 offset, test_frames, test_joints):
    """Evaluate reprojection error for a given (possibly fractional) offset."""
    errs = []
    for f in test_frames:
        f2 = f + offset
        kp_ref = _get_kp_at(matched_ref, f)
        kp_v = _get_kp_at(matched_v, f2)
        if kp_ref is None or kp_v is None:
            continue
        for j in test_joints:
            sr = kp_ref[j][2]
            sv = kp_v[j][2]
            if sr < 0.3 or sv < 0.3:
                continue
            p = [projs_ref, projs_v]
            uv = [(kp_ref[j][0], kp_ref[j][1]),
                  (kp_v[j][0], kp_v[j][1])]
            try:
                X = tri.triangulate_point(p, uv)
                if np.isfinite(X).all():
                    errs.append(tri.reprojection_error(p, uv, X))
            except Exception:
                pass
    return float(np.median(errs)) if errs else float("inf")


def find_offsets(cam_paths, matched_player, ref_view=None,
                 max_offset=30, sample_step=50, sample_start=60,
                 sample_end=None, subframe=True):
    """Find per-camera frame offsets relative to a reference camera.

    matched_player: {view: [frame0_kps, frame1_kps, ...]} — single-player
        matched keypoints (e.g. from player_tracker or build_real).
    sample_end: if set, only use frames up to this index for offset
        estimation (useful to avoid tracker-swap contamination in tail).
    Returns {view: offset} where offset is a float (fractional frames).
    """
    views = sorted(cam_paths.keys())
    if ref_view is None:
        ref_view = views[0]

    import json
    projs = {}
    for v in views:
        with open(cam_paths[v], "r", encoding="utf-8") as fh:
            d = json.load(fh)
        K = np.array(d["K"], dtype=np.float64)
        R = np.array(d["R"], dtype=np.float64)
        t = np.array(d["t"], dtype=np.float64)
        projs[v] = mv.projection_matrix(K, R, t)

    frame_count = min(len(matched_player[v]) for v in views)
    end = min(sample_end or frame_count, frame_count) - max_offset
    test_frames = list(range(sample_start, end, sample_step))
    if not test_frames:
        test_frames = list(range(0, end, max(1, end // 20)))

    test_joints = [5, 6, 11, 12, 15, 16]

    offsets = {ref_view: 0.0}

    for v in views:
        if v == ref_view:
            continue
        # coarse: integer sweep
        best_offset, best_err = 0, float("inf")
        for offset in range(-max_offset, max_offset + 1):
            err = _eval_offset(projs[ref_view], projs[v],
                               matched_player[ref_view], matched_player[v],
                               float(offset), test_frames, test_joints)
            if err < best_err:
                best_err, best_offset = err, offset

        # fine: sub-frame refinement around best integer
        if subframe:
            coarse = best_offset
            for step in (0.5, 0.2, 0.1):
                for delta in (-step, 0.0, step):
                    candidate = best_offset + delta
                    if abs(candidate - coarse) > 1.5:
                        continue
                    err = _eval_offset(projs[ref_view], projs[v],
                                       matched_player[ref_view],
                                       matched_player[v],
                                       candidate, test_frames, test_joints)
                    if err < best_err:
                        best_err, best_offset = err, candidate

        offsets[v] = best_offset
        print(f"  {v} vs {ref_view}: offset={best_offset:+.1f} "
              f"(reproj {best_err:.1f}px)")

    return offsets


def find_offsets_with_errors(cam_paths, matched_player, ref_view=None,
                             max_offset=30, sample_step=50, sample_start=60,
                             sample_end=None, subframe=True):
    """Like find_offsets but also returns per-view reproj errors."""
    views = sorted(cam_paths.keys())
    if ref_view is None:
        ref_view = views[0]

    import json
    projs = {}
    for v in views:
        with open(cam_paths[v], "r", encoding="utf-8") as fh:
            d = json.load(fh)
        K = np.array(d["K"], dtype=np.float64)
        R = np.array(d["R"], dtype=np.float64)
        t = np.array(d["t"], dtype=np.float64)
        projs[v] = mv.projection_matrix(K, R, t)

    frame_count = min(len(matched_player[v]) for v in views)
    end = min(sample_end or frame_count, frame_count) - max_offset
    test_frames = list(range(sample_start, end, sample_step))
    if not test_frames:
        test_frames = list(range(0, end, max(1, end // 20)))
    test_joints = [5, 6, 11, 12, 15, 16]

    offsets = {ref_view: 0.0}
    errors = {ref_view: 0.0}

    for v in views:
        if v == ref_view:
            continue
        best_offset, best_err = 0, float("inf")
        for offset in range(-max_offset, max_offset + 1):
            err = _eval_offset(projs[ref_view], projs[v],
                               matched_player[ref_view], matched_player[v],
                               float(offset), test_frames, test_joints)
            if err < best_err:
                best_err, best_offset = err, offset
        if subframe:
            coarse = best_offset
            for step in (0.5, 0.2, 0.1):
                for delta in (-step, 0.0, step):
                    candidate = best_offset + delta
                    if abs(candidate - coarse) > 1.5:
                        continue
                    err = _eval_offset(projs[ref_view], projs[v],
                                       matched_player[ref_view],
                                       matched_player[v],
                                       candidate, test_frames, test_joints)
                    if err < best_err:
                        best_err, best_offset = err, candidate
        offsets[v] = best_offset
        errors[v] = best_err
        print(f"  {v} vs {ref_view}: offset={best_offset:+.1f} "
              f"(reproj {best_err:.1f}px)")

    return offsets, errors


def find_offsets_multi_player(cam_paths, matched, player_ids, ref_view=None,
                              max_offset=30, sample_step=50, sample_start=60,
                              sample_end=None, subframe=True):
    """Find per-camera offsets using the best player per view.

    Instead of using only P1's keypoints (which may be mismatched in some
    views), tries each player's data for each view and picks the
    (player, offset) pair with the lowest reprojection error.

    matched: {pid: {view: [frame_kps, ...]}} — all players' matched data.
    Returns (offsets, errors) like find_offsets_with_errors.
    """
    views = sorted(cam_paths.keys())
    if ref_view is None:
        ref_view = views[0]

    import json
    projs = {}
    for v in views:
        with open(cam_paths[v], "r", encoding="utf-8") as fh:
            d = json.load(fh)
        K = np.array(d["K"], dtype=np.float64)
        R = np.array(d["R"], dtype=np.float64)
        t = np.array(d["t"], dtype=np.float64)
        projs[v] = mv.projection_matrix(K, R, t)

    test_joints = [5, 6, 11, 12, 15, 16]

    offsets = {ref_view: 0.0}
    errors = {ref_view: 0.0}

    for v in views:
        if v == ref_view:
            continue

        best_offset = 0
        best_err = float("inf")
        best_pid = player_ids[0]

        for pid in player_ids:
            ref_data = matched[pid].get(ref_view)
            v_data = matched[pid].get(v)
            if ref_data is None or v_data is None:
                continue

            frame_count = min(len(ref_data), len(v_data))
            end = min(sample_end or frame_count, frame_count) - max_offset
            test_frames = list(range(sample_start, end, sample_step))
            if not test_frames:
                test_frames = list(range(0, end, max(1, end // 20)))
            if not test_frames:
                continue

            # Check detection rate — skip players with <30% visible frames
            n_vis = sum(1 for f in test_frames
                        if f < len(v_data) and v_data[f] is not None)
            if n_vis < 0.3 * len(test_frames):
                continue

            for offset in range(-max_offset, max_offset + 1):
                err = _eval_offset(projs[ref_view], projs[v],
                                   ref_data, v_data,
                                   float(offset), test_frames, test_joints)
                if err < best_err:
                    best_err = err
                    best_offset = offset
                    best_pid = pid

        # Sub-frame refinement using the best player
        if subframe and best_err < float("inf"):
            ref_data = matched[best_pid][ref_view]
            v_data = matched[best_pid][v]
            coarse = best_offset
            for step in (0.5, 0.2, 0.1):
                for delta in (-step, 0.0, step):
                    candidate = best_offset + delta
                    if abs(candidate - coarse) > 1.5:
                        continue
                    frame_count = min(len(ref_data), len(v_data))
                    end = min(sample_end or frame_count, frame_count) - max_offset
                    test_frames = list(range(sample_start, end, sample_step))
                    if not test_frames:
                        test_frames = list(range(0, end, max(1, end // 20)))
                    err = _eval_offset(projs[ref_view], projs[v],
                                       ref_data, v_data,
                                       candidate, test_frames, test_joints)
                    if err < best_err:
                        best_err, best_offset = err, candidate

        offsets[v] = best_offset
        errors[v] = best_err
        print(f"  {v} vs {ref_view}: offset={best_offset:+.1f} "
              f"(reproj {best_err:.1f}px, best player={best_pid})")

    return offsets, errors


def apply_offsets(kp2d_frames_by_view, offsets):
    """Shift per-view frame lists by their offsets and trim to common length.

    Supports fractional offsets via linear interpolation of 2D keypoints.
    """
    views = sorted(kp2d_frames_by_view.keys())
    shifted = {}
    lengths = []
    for v in views:
        off = offsets.get(v, 0.0)
        frames = kp2d_frames_by_view[v]
        int_off = int(np.floor(off))
        frac = off - int_off

        if abs(frac) < 1e-6:
            # integer offset — no interpolation needed
            if int_off > 0:
                shifted[v] = frames[int_off:]
            elif int_off < 0:
                shifted[v] = [None] * (-int_off) + frames
            else:
                shifted[v] = list(frames)
        else:
            # fractional offset — interpolate between adjacent frames
            new_frames = []
            for i in range(len(frames)):
                src = i + off
                new_frames.append(_get_kp_at(frames, src))
            shifted[v] = new_frames

        lengths.append(len(shifted[v]))
    frame_count = min(lengths)
    for v in views:
        shifted[v] = shifted[v][:frame_count]
    return shifted, frame_count
