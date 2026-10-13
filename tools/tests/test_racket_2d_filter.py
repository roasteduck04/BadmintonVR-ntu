"""Tests for 2D racket temporal filtering (Sub-Project C)."""
import numpy as np
import pytest


def test_smooth_racket_2d_reduces_jitter():
    """SG filter should reduce frame-to-frame keypoint displacement."""
    from analysis.build_real import _smooth_racket_2d

    np.random.seed(42)
    n_frames = 60
    # Simulate a stable racket with noise
    base_x, base_y = 500.0, 300.0
    frames = []
    for f in range(n_frames):
        kps = []
        for kp_idx in range(5):
            x = base_x + kp_idx * 30 + np.random.randn() * 5
            y = base_y + np.random.randn() * 5
            kps.append([x, y, 0.7])
        frames.append([kps])  # 1 candidate

    rkp = {"view1": frames}
    smoothed = _smooth_racket_2d(rkp, window=5, order=2)

    # Measure jitter: mean frame-to-frame displacement of handle (kp 2)
    def jitter(view_frames):
        disps = []
        for f in range(1, len(view_frames)):
            if view_frames[f] is None or view_frames[f - 1] is None:
                continue
            dx = view_frames[f][0][2][0] - view_frames[f - 1][0][2][0]
            dy = view_frames[f][0][2][1] - view_frames[f - 1][0][2][1]
            disps.append(np.sqrt(dx**2 + dy**2))
        return np.mean(disps)

    jitter_before = jitter(frames)
    jitter_after = jitter(smoothed["view1"])

    reduction = 1.0 - jitter_after / jitter_before
    assert reduction > 0.3, \
        f"Expected >30% jitter reduction, got {reduction*100:.1f}%"


def test_smooth_racket_2d_preserves_scores():
    """Scores should not be modified by the filter."""
    from analysis.build_real import _smooth_racket_2d

    frames = []
    for f in range(20):
        kps = [[100 + f, 200, 0.42 + f * 0.01] for _ in range(5)]
        frames.append([kps])
    rkp = {"v": frames}
    smoothed = _smooth_racket_2d(rkp)

    for f in range(20):
        for kp_idx in range(5):
            assert smoothed["v"][f][0][kp_idx][2] == frames[f][0][kp_idx][2], \
                f"Score at frame {f} kp {kp_idx} was modified"


def test_smooth_racket_2d_handles_gaps():
    """None frames should pass through without error."""
    from analysis.build_real import _smooth_racket_2d

    frames = []
    for f in range(20):
        if 8 <= f <= 12:
            frames.append(None)
        else:
            kps = [[100, 200, 0.5] for _ in range(5)]
            frames.append([kps])

    rkp = {"v": frames}
    smoothed = _smooth_racket_2d(rkp)

    for f in range(8, 13):
        assert smoothed["v"][f] is None, f"Frame {f} should remain None"
    assert smoothed["v"][0] is not None
    assert smoothed["v"][15] is not None
