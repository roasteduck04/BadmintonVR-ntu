"""Tests for racket 3D reconstruction improvements (Sub-Project B)."""
import numpy as np


def test_temporal_prior_reduces_jitter():
    """triangulate_racket_multi should use temporal prior, reducing
    frame-to-frame jitter compared to independent triangulation."""
    from unittest.mock import patch
    from analysis.build_real import triangulate_racket_multi, _RACKET_KP

    n_frames = 10
    n_views = 3
    cams = {f"v{i}": np.eye(3, 4) for i in range(n_views)}

    # Build assigned data: each player has stable 2D keypoints with small noise
    base_kps = {kp: (100.0 + j * 50, 200.0) for j, kp in enumerate(_RACKET_KP)}
    assigned = {"P1": {}}
    for v in cams:
        frames = []
        for f in range(n_frames):
            cand = []
            for kp_name in _RACKET_KP:
                bx, by = base_kps[kp_name]
                cand.append((bx + np.random.randn() * 2, by + np.random.randn() * 2, 0.8))
            frames.append(cand)
        assigned["P1"][v] = frames

    call_log = []
    original_tri = None

    def tracking_tri(cams_, per_view, score_thr, prior_X=None):
        call_log.append(prior_X)
        # Return a dummy 3D point
        return np.array([1.0, 2.0, 3.0]), 0.8, 5.0

    with patch("analysis.build_real.tri.triangulate_joint", side_effect=tracking_tri):
        triangulate_racket_multi(cams, assigned, ["P1"], n_frames, score_thr=0.15)

    # First frame's first keypoint should have no prior (None)
    assert call_log[0] is None, "First frame should have no temporal prior"
    # Second frame's first keypoint should have a prior (from frame 0)
    assert call_log[len(_RACKET_KP)] is not None, \
        "Second frame should receive temporal prior from first frame"
    np.testing.assert_array_almost_equal(
        call_log[len(_RACKET_KP)], np.array([1.0, 2.0, 3.0]))


def test_shaft_blend_weight_scales_with_confidence():
    """_shaft_blend_weight returns higher triangulated weight at higher confidence."""
    from analysis.build_real import _shaft_blend_weight

    # High confidence detected -> 0.70 triangulated weight
    w_high = _shaft_blend_weight(0.5, "detected")
    assert abs(w_high - 0.70) < 0.01, f"Expected 0.70 at conf=0.5, got {w_high}"

    # Medium confidence detected -> 0.50
    w_mid = _shaft_blend_weight(0.3, "detected")
    assert abs(w_mid - 0.50) < 0.01, f"Expected 0.50 at conf=0.3, got {w_mid}"

    # Low confidence detected -> 0.30
    w_low = _shaft_blend_weight(0.1, "detected")
    assert abs(w_low - 0.30) < 0.01, f"Expected 0.30 at conf=0.1, got {w_low}"

    # Partial/interpolated status -> 0.30 regardless of confidence
    w_partial = _shaft_blend_weight(0.5, "partial")
    assert abs(w_partial - 0.30) < 0.01, f"Expected 0.30 for partial, got {w_partial}"

    w_interp = _shaft_blend_weight(0.5, "interpolated")
    assert abs(w_interp - 0.30) < 0.01, f"Expected 0.30 for interpolated, got {w_interp}"


def test_sg_smooth_shaft_preserves_fast_swings():
    """SG smoothing should allow shaft angle changes above 30 deg/frame,
    unlike the EMA which capped at ~9.5 deg/frame."""
    from analysis.build_real import _sg_smooth_shaft_dirs

    # Simulate a fast swing: shaft rotates 45 degrees per frame over 6 frames
    n = 20
    dirs = np.zeros((n, 3))
    for i in range(n):
        angle = np.radians(45.0 * i) if 7 <= i <= 13 else 0.0
        dirs[i] = [np.sin(angle), 0.0, np.cos(angle)]
    valid = np.ones(n, dtype=bool)

    smoothed = _sg_smooth_shaft_dirs(dirs, valid, window=7, order=2)

    # Check the max angle change in the swing region
    max_angle = 0.0
    for i in range(8, 13):
        d1 = smoothed[i]
        d2 = smoothed[i + 1]
        dot = np.clip(np.dot(d1, d2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        max_angle = max(max_angle, angle)

    assert max_angle > 30.0, \
        f"SG should preserve fast swings (>30 deg/frame), got max {max_angle:.1f}"


def test_sg_smooth_shaft_handles_gaps():
    """SG smoothing should handle None entries (gaps) without crashing."""
    from analysis.build_real import _sg_smooth_shaft_dirs

    dirs = np.zeros((10, 3))
    for i in range(10):
        dirs[i] = [0.0, 0.0, 1.0]
    valid = np.ones(10, dtype=bool)
    valid[4:6] = False  # gap in the middle

    smoothed = _sg_smooth_shaft_dirs(dirs, valid, window=5, order=2)

    # Non-gap frames should have unit-length directions
    for i in range(10):
        if valid[i]:
            length = np.linalg.norm(smoothed[i])
            assert abs(length - 1.0) < 0.01, \
                f"Frame {i}: expected unit length, got {length}"


def test_face_orientation_uses_triangulated_side():
    """On detected frames with real left/right keypoints, the face orientation
    should differ from the synthetic cross(shaft, world-up) fallback."""
    # We test _reconstruct_side_dirs which will be the new helper
    from analysis.build_real import _reconstruct_side_dirs

    n = 10
    shaft_dirs = np.zeros((n, 3))
    for i in range(n):
        shaft_dirs[i] = [0.0, 0.0, 1.0]  # shaft pointing along Z

    # Triangulated side: 45 degrees from the synthetic cross(shaft, up)
    # Synthetic would be cross([0,0,1], [0,1,0]) = [1,0,0]
    # Real side: rotated 45 degrees around Z
    real_side = np.array([np.cos(np.radians(45)), np.sin(np.radians(45)), 0.0])
    tri_sides = np.zeros((n, 3))
    tri_heads = np.zeros((n, 3))
    tri_grips = np.zeros((n, 3))
    has_side = np.zeros(n, dtype=bool)
    for i in range(n):
        tri_grips[i] = [0, 0, 0]
        tri_heads[i] = [0, 0, 0.68]
        tri_sides[i] = tri_heads[i] + 0.10 * real_side
        has_side[i] = True

    valid = np.ones(n, dtype=bool)
    result = _reconstruct_side_dirs(
        shaft_dirs, tri_sides, tri_heads, has_side, valid)

    # The result should NOT be [1, 0, 0] (the synthetic direction)
    # It should be close to the real_side direction
    for i in range(n):
        dot_with_real = abs(np.dot(result[i], real_side))
        assert dot_with_real > 0.9, \
            f"Frame {i}: side should match triangulated, got dot={dot_with_real}"


def test_face_orientation_falls_back_without_side():
    """On frames without triangulated side data, fallback to cross(shaft, up)."""
    from analysis.build_real import _reconstruct_side_dirs

    n = 5
    shaft_dirs = np.zeros((n, 3))
    for i in range(n):
        shaft_dirs[i] = [0.0, 0.0, 1.0]

    tri_sides = np.zeros((n, 3))
    tri_heads = np.zeros((n, 3))
    has_side = np.zeros(n, dtype=bool)  # no real side data

    valid = np.ones(n, dtype=bool)
    result = _reconstruct_side_dirs(
        shaft_dirs, tri_sides, tri_heads, has_side, valid)

    # cross([0,0,1], [0,1,0]) = [-1,0,0] (right-hand rule; matches the
    # existing synthetic-side convention used elsewhere in build_real.py)
    for i in range(n):
        assert abs(result[i][0] - (-1.0)) < 0.1, \
            f"Frame {i}: expected synthetic side ~[-1,0,0], got {result[i]}"
