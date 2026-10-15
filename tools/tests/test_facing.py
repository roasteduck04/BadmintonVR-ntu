"""Tests for facing direction estimation."""
import math
import pytest
import numpy as np

# These bodies are right-handed (left hip at +X when facing +Z: the SMPL rest convention, forward_sign +1).
# Real analysis docs are mirror-handed (forward_sign -1): see TestMirrorHanded.
RIGHT = 1.0


def _make_joints_flat(overrides, default_conf=0.9):
    """Build a 232-element joints_flat (58 joints × 4 values)."""
    jf = [0.0] * 232
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
        result = compute_facing(jf, sign=RIGHT)
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
        result = compute_facing(jf, sign=RIGHT)
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
        result = compute_facing(jf, sign=RIGHT)
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
        result = compute_facing(jf, sign=RIGHT)
        assert result is None

    def test_null_when_joints_flat_is_none(self):
        """No facing when joints_flat is None."""
        from analysis.facing import compute_facing
        result = compute_facing(None, sign=RIGHT)
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
        result = compute_facing(jf, sign=RIGHT)
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
        fill_facing(frames, fps=60.0, sign=RIGHT)
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
        fill_facing(frames, fps=60.0, sign=RIGHT)
        headings = [f["facing"]["heading"] for f in frames if f["facing"] is not None]
        # After smoothing, variance should be small
        assert np.std(headings) < 0.1


class TestHeadHeading:

    def test_head_heading_facing_positive_z(self):
        """Nose in front of eye midpoint along +Z → head_heading ≈ 0."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
            16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
            53: (0.03, 1.58, 0.0),   # left_eye
            54: (-0.03, 1.58, 0.0),  # right_eye
            55: (0.0, 1.55, 0.1),    # nose_tip — forward in +Z
        })
        result = compute_facing(jf, sign=RIGHT)
        assert result is not None
        assert "head_heading" in result
        assert abs(result["head_heading"] - 0.0) < 0.1

    def test_head_heading_facing_positive_x(self):
        """Nose to the right of eye midpoint → head_heading ≈ pi/2."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
            16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
            53: (0.0, 1.58, 0.03),   # left_eye
            54: (0.0, 1.58, -0.03),  # right_eye
            55: (0.1, 1.55, 0.0),    # nose_tip — forward in +X
        })
        result = compute_facing(jf, sign=RIGHT)
        assert result is not None
        assert abs(result["head_heading"] - math.pi / 2) < 0.1

    def test_head_heading_absent_when_no_face_joints(self):
        """No head_heading when face joints have zero confidence."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
            16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
        })
        result = compute_facing(jf, sign=RIGHT)
        assert result is not None
        assert "head_heading" not in result

    def test_head_heading_one_eye_only(self):
        """head_heading still computed with only one eye available."""
        from analysis.facing import compute_facing
        jf = _make_joints_flat({
            1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0),
            16: (0.5, 1.4, 0.0), 17: (-0.5, 1.4, 0.0),
            53: (0.0, 1.58, 0.0, 0.9),   # left_eye
            54: (0.0, 1.58, 0.0, 0.0),   # right_eye — no confidence
            55: (0.0, 1.55, 0.1, 0.9),   # nose_tip
        })
        result = compute_facing(jf, sign=RIGHT)
        assert result is not None
        assert "head_heading" in result


class TestConfThreshold:
    """pose24 (schema 4.1) confidences are exp(-sigma/40 mm): a far player's hips sit near 0.02."""

    def _low(self, c=0.02):
        return _make_joints_flat({
            1: (1.0, 0.9, 0.0, c), 2: (-1.0, 0.9, 0.0, c),
            16: (0.5, 1.4, 0.0, c), 17: (-0.5, 1.4, 0.0, c),
            53: (0.03, 1.58, 0.0, c), 54: (-0.03, 1.58, 0.0, c), 55: (0.0, 1.55, 0.1, c),
        })

    def test_default_threshold_unchanged(self):
        from analysis.facing import compute_facing
        assert compute_facing(self._low(), sign=RIGHT) is None

    def test_low_conf_hips_with_threshold(self):
        from analysis.facing import compute_facing
        r = compute_facing(self._low(), sign=RIGHT, conf_min=1e-3)
        assert r is not None and abs(r["heading"]) < 0.01 and r["confidence"] == 0.02
        assert "head_heading" in r and abs(r["head_heading"]) < 0.1
        assert compute_facing(self._low(5e-4), sign=RIGHT, conf_min=1e-3) is None

    def test_fill_facing_passes_threshold(self):
        from analysis.facing import fill_facing
        frames = [{"joints_flat": self._low()} for _ in range(10)]
        fill_facing(frames, fps=60.0, sign=RIGHT, conf_min=1e-3)
        assert all(f["facing"] is not None for f in frames)
        fill_facing(frames, fps=60.0, sign=RIGHT)
        assert all(f["facing"] is None for f in frames)


def _smpl24_body(facing_z=1.0, mirror=True, n=24):
    """(n,4) SMPL-layout joints of a person standing at the origin facing +Z (facing_z=1) or -Z (-1), Y up.
    mirror=True: real analysis-world handedness (left hip at -X when facing +Z); False: SMPL rest convention."""
    s = -1.0 if mirror else 1.0
    J = np.zeros((n, 4))
    J[:, 3] = 0.9
    pts = {1: (0.1, 0.9, 0.0), 2: (-0.1, 0.9, 0.0), 7: (0.1, 0.08, 0.0), 8: (-0.1, 0.08, 0.0),
           10: (0.1, 0.02, 0.12), 11: (-0.1, 0.02, 0.12), 16: (0.18, 1.42, 0.0), 17: (-0.18, 1.42, 0.0)}
    for k, (x, y, z) in pts.items():
        J[k, :3] = (s * x * facing_z, y, z * facing_z)
    return J


class TestMirrorHanded:
    """Real docs live in the Y-up analysis world, which is mirror-handed (CLAUDE.md "Handedness", review 1 M1)."""

    def test_real_handed_frame_hip_heading_agrees_with_head_heading(self):
        from analysis.facing import compute_facing
        # facing +Z, left hip at -X (mirror-handed), nose ahead of the eyes
        jf = _make_joints_flat({
            1: (-1.0, 0.9, 0.0), 2: (1.0, 0.9, 0.0),
            16: (-0.5, 1.4, 0.0), 17: (0.5, 1.4, 0.0),
            53: (-0.03, 1.58, 0.0), 54: (0.03, 1.58, 0.0), 55: (0.0, 1.55, 0.1),
        })
        r = compute_facing(jf, sign=-1.0)
        assert abs(r["heading"]) < 0.01 and abs(r["shoulder_heading"]) < 0.01
        assert abs(r["head_heading"]) < 0.1

    def test_sign_is_required(self):
        from analysis.facing import compute_facing, fill_facing
        with pytest.raises(TypeError):
            compute_facing(_make_joints_flat({1: (1.0, 0.9, 0.0), 2: (-1.0, 0.9, 0.0)}))
        with pytest.raises(TypeError):
            fill_facing([], 60.0)

    def test_mirror_facing_minus_z(self):
        from analysis.facing import fill_facing
        frames = [{"joints_flat": _smpl24_body(facing_z=-1.0).ravel().tolist()} for _ in range(10)]
        fill_facing(frames, 60.0, sign=-1.0, face=None)
        assert all(abs(abs(fr["facing"]["heading"]) - math.pi) < 0.01 for fr in frames)

    def test_frames_sign_reads_handedness_from_the_feet(self):
        from analysis.facing import frames_sign
        mir = [{"joints_flat": _smpl24_body(mirror=True).ravel().tolist()} for _ in range(5)]
        rh = [{"joints_flat": _smpl24_body(facing_z=-1.0, mirror=False).ravel().tolist()} for _ in range(5)]
        assert frames_sign(mir, default=1.0) == -1.0
        assert frames_sign(rh, default=-1.0) == 1.0
        J = _smpl24_body()
        J[[7, 8, 10, 11], 3] = 0.0                      # feet never seen -> the default
        assert frames_sign([{"joints_flat": J.ravel().tolist()}, {"joints_flat": None}], default=-1.0) == -1.0
        assert frames_sign([{"joints_flat": None}], default=1.0) == 1.0

    def test_frames_sign_then_fill_facing_gives_the_true_heading(self):
        from analysis.facing import fill_facing, frames_sign
        for mirror in (True, False):
            for fz, want in ((1.0, 0.0), (-1.0, math.pi)):
                frames = [{"joints_flat": _smpl24_body(fz, mirror).ravel().tolist()} for _ in range(8)]
                fill_facing(frames, 60.0, sign=frames_sign(frames, default=-1.0), face=None)
                h = frames[0]["facing"]["heading"]
                assert abs(math.remainder(h - want, 2 * math.pi)) < 0.01, (mirror, fz, h)


class TestConventions:

    def test_heading_of_matches_sign(self):
        from analysis import conventions as cv
        L, R = np.array([-0.1, 0.9, 0.0]), np.array([0.1, 0.9, 0.0])        # mirror-handed, facing +Z
        assert abs(cv.heading_of(L, R, cv.UP_Y, -1.0)) < 1e-9
        assert abs(abs(cv.heading_of(L, R, cv.UP_Y, 1.0)) - math.pi) < 1e-9
        # the Y-down court world (Y flipped) of the same person is right-handed: sign +1, same heading
        assert abs(cv.heading_of(L * cv.Y_FLIP, R * cv.Y_FLIP, cv.UP_COURT, 1.0)) < 1e-9
        assert cv.heading_of(L, L, cv.UP_Y, 1.0) is None
        arr = cv.heading_of(np.stack([L, L]), np.stack([R, L]), cv.UP_Y, -1.0)
        assert abs(arr[0]) < 1e-9 and np.isnan(arr[1])
        with pytest.raises(ValueError):
            cv.heading_of(L, R, cv.UP_Y, float("nan"))

    def test_world_and_analysis_sign(self):
        from analysis import conventions as cv
        from analysis.pose24 import kpset
        proper = {"a": (np.eye(3), np.eye(3)), "b": (np.eye(3), np.diag([1.0, -1.0, -1.0]))}
        assert cv.world_sign(proper) == 1.0 == kpset.world_sign(proper)
        assert cv.analysis_sign(proper) == -1.0
        mirrored = {"a": (np.eye(3), np.diag([-1.0, 1.0, 1.0]))}
        assert cv.world_sign(mirrored) == -1.0 and cv.analysis_sign(mirrored) == 1.0
