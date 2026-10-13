"""Tests for facing direction estimation."""
import math
import pytest
import numpy as np


def _make_joints_flat(overrides, default_conf=0.9):
    """Build a 96-element joints_flat (24 joints × 4 values)."""
    jf = [0.0] * 96
    for idx, vals in overrides.items():
        base = idx * 4
        jf[base] = vals[0]      # x
        jf[base + 1] = vals[1]  # y
        jf[base + 2] = vals[2]  # z
        jf[base + 3] = vals[3] if len(vals) > 3 else default_conf
    return jf


class TestFacing:

    def test_hip_heading_facing_positive_z(self):
        """Player facing +Z: left_hip at +X, right_hip at -X.

        hip_vec = right - left = [-2, 0, 0]
        forward = cross(hip_vec, [0, 1, 0]) = [0, 0, 2]
        heading = atan2(0, 2) = 0.0
        """
        from analysis.facing import compute_facing
        # left_hip (1) at [1, 0.9, 0], right_hip (2) at [-1, 0.9, 0]
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0),
            2: (-1.0, 0.9, 0.0),
            16: (0.5, 1.4, 0.0),  # shoulders for shoulder_heading
            17: (-0.5, 1.4, 0.0),
        })
        result = compute_facing(jf)
        assert result is not None
        assert abs(result["heading"] - 0.0) < 0.01

    def test_hip_heading_facing_positive_x(self):
        """Player facing +X: left_hip at +Z, right_hip at -Z (from player's POV).

        Actually: left_hip should be to the player's left.
        If facing +X: left_hip at (0, 0.9, -1), right_hip at (0, 0.9, 1).
        hip_vec = right - left = [0, 0, 2]
        forward = cross([0,0,2], [0,1,0]) = [2, 0, 0]
        heading = atan2(2, 0) = pi/2
        """
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (0.0, 0.9, -1.0),
            2: (0.0, 0.9, 1.0),
            16: (0.0, 1.4, -0.5),
            17: (0.0, 1.4, 0.5),
        })
        result = compute_facing(jf)
        assert result is not None
        assert abs(result["heading"] - math.pi / 2) < 0.01

    def test_shoulder_heading_diverges_from_hip(self):
        """Twisted torso: hip facing +Z but shoulders twisted toward +X."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0),   # left_hip
            2: (-1.0, 0.9, 0.0),  # right_hip → facing +Z
            16: (0.0, 1.4, -0.5),  # left_shoulder
            17: (0.0, 1.4, 0.5),   # right_shoulder → facing +X
        })
        result = compute_facing(jf)
        assert result is not None
        assert abs(result["heading"] - 0.0) < 0.01  # hip facing +Z
        assert abs(result["shoulder_heading"] - math.pi / 2) < 0.01

    def test_null_when_hips_missing(self):
        """No facing when hip confidence is below threshold."""
        from analysis.facing import compute_facing, HIP_CONF_MIN
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0, 0.05),  # below HIP_CONF_MIN
            2: (-1.0, 0.9, 0.0, 0.05),
        })
        result = compute_facing(jf)
        assert result is None

    def test_null_when_joints_flat_is_none(self):
        """No facing when joints_flat is None."""
        from analysis.facing import compute_facing
        result = compute_facing(None)
        assert result is None

    def test_confidence_is_min_hip(self):
        """Confidence should be min of left and right hip confidence."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0, 0.7),
            2: (-1.0, 0.9, 0.0, 0.5),
            16: (0.5, 1.4, 0.0, 0.8),
            17: (-0.5, 1.4, 0.0, 0.8),
        })
        result = compute_facing(jf)
        assert result is not None
        assert abs(result["confidence"] - 0.5) < 0.01


class TestFillFacing:

    def test_fill_facing_adds_to_frames(self):
        """fill_facing adds 'facing' key to each frame."""
        from analysis.facing import fill_facing
        frames = [
            {"joints_flat": _make_joints_flat({
                1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
                16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
            })},
            {"joints_flat": _make_joints_flat({
                1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
                16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
            })},
            {"joints_flat": None},
        ]
        fill_facing(frames, fps=60.0)
        assert "facing" in frames[0]
        assert frames[0]["facing"] is not None
        assert "heading" in frames[0]["facing"]
        assert frames[2]["facing"] is None

    def test_fill_facing_smooths_heading(self):
        """SG smoothing should reduce noise in heading values."""
        from analysis.facing import fill_facing
        # Create 20 frames with slight noise in hip positions
        rng = np.random.RandomState(42)
        frames = []
        for i in range(20):
            noise = rng.normal(0, 0.02)
            frames.append({"joints_flat": _make_joints_flat({
                1: (1.0 + noise, 0.9, 0.0),
                2: (-1.0 + noise, 0.9, 0.0),
                16: (0.5, 1.4, 0.0),
                17: (-0.5, 1.4, 0.0),
            })})
        fill_facing(frames, fps=60.0)
        headings = [f["facing"]["heading"] for f in frames if f["facing"] is not None]
        # After smoothing, variance should be small
        assert np.std(headings) < 0.1
