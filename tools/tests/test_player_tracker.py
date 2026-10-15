import json

import numpy as np
import pytest


def _make_coco17(positions, default_score=0.9):
    """Build a 17-joint keypoint list from a dict of {joint_idx: (x, y)}.

    Joints not in ``positions`` get (0, 0, 0).
    """
    kps = [[0.0, 0.0, 0.0]] * 17
    for idx, (x, y) in positions.items():
        kps[idx] = [float(x), float(y), default_score]
    return [list(k) for k in kps]


class TestBodySignature:
    def test_from_keypoints_full_pose(self):
        """Full visible pose produces a 4-element vector of positive ratios."""
        from analysis.player_tracker import BodySignature
        kps = _make_coco17({
            0: (100, 50),    # nose
            5: (80, 100),    # left_shoulder
            6: (120, 100),   # right_shoulder
            9: (60, 100),    # left_wrist
            10: (140, 100),  # right_wrist
            11: (85, 200),   # left_hip
            12: (115, 200),  # right_hip
            15: (85, 350),   # left_ankle
            16: (115, 350),  # right_ankle
        })
        sig = BodySignature.from_keypoints(kps)
        assert sig is not None
        assert sig.vector.shape == (4,)
        assert all(v > 0 for v in sig.vector)

    def test_from_keypoints_missing_ankles_returns_none(self):
        """Missing ankles → cannot compute ratios → None."""
        from analysis.player_tracker import BodySignature
        kps = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            11: (85, 200), 12: (115, 200),
        })
        sig = BodySignature.from_keypoints(kps)
        assert sig is None

    def test_from_keypoints_low_score_returns_none(self):
        """Joints with score below 0.3 should be treated as missing."""
        from analysis.player_tracker import BodySignature
        kps = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 350), 16: (115, 350),
        }, default_score=0.2)  # all below threshold
        sig = BodySignature.from_keypoints(kps)
        assert sig is None

    def test_distance_identical(self):
        """Distance to self is 0."""
        from analysis.player_tracker import BodySignature
        kps = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 350), 16: (115, 350),
        })
        sig = BodySignature.from_keypoints(kps)
        assert sig.distance(sig) == pytest.approx(0.0)

    def test_distance_different_proportions(self):
        """Different body proportions produce nonzero distance."""
        from analysis.player_tracker import BodySignature
        kps_a = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 350), 16: (115, 350),
        })
        # Longer legs (ankles at 450 instead of 350)
        kps_b = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 450), 16: (115, 450),
        })
        sig_a = BodySignature.from_keypoints(kps_a)
        sig_b = BodySignature.from_keypoints(kps_b)
        assert sig_a.distance(sig_b) > 0.05

    def test_update_shifts_vector(self):
        """Running median update shifts the signature toward new observations."""
        from analysis.player_tracker import BodySignature
        kps_a = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 350), 16: (115, 350),
        })
        sig = BodySignature.from_keypoints(kps_a)
        old_vec = sig.vector.copy()
        # Feed several frames with different proportions
        kps_b = _make_coco17({
            0: (100, 50), 5: (80, 100), 6: (120, 100),
            9: (60, 100), 10: (140, 100),
            11: (85, 200), 12: (115, 200),
            15: (85, 450), 16: (115, 450),
        })
        for _ in range(20):
            sig.update(kps_b)
        assert not np.allclose(sig.vector, old_vec)


class TestPlayerState:
    def test_init_sets_position_and_side(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        assert ps.pid == "P1"
        assert ps.side == "near"
        assert ps.state == "ACTIVE"
        assert ps.pos[0] == pytest.approx(1.0)
        assert ps.pos[1] == pytest.approx(-3.0)

    def test_predict_advances_position(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        ps.update(np.array([1.5, -3.0]))  # give it some velocity
        pos = ps.predict()
        # After update at (1.5, -3.0), velocity is positive in x
        # Next predict should move x further right
        assert pos[0] > 1.0

    def test_predict_without_update_stays_near(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        pos = ps.predict()
        # No velocity, should stay near init
        assert abs(pos[0] - 1.0) < 0.1
        assert abs(pos[1] - (-3.0)) < 0.1

    def test_update_sets_active(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        ps.coast()
        ps.coast()
        ps.coast()
        assert ps.state == "COASTING"
        ps.update(np.array([1.0, -3.0]))
        assert ps.state == "ACTIVE"
        assert ps.coast_count == 0

    def test_coast_transitions(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        assert ps.state == "ACTIVE"
        # First 2 coasts: still ACTIVE
        ps.coast()
        ps.coast()
        assert ps.state == "ACTIVE"
        # 3rd coast: transition to COASTING
        ps.coast()
        assert ps.state == "COASTING"
        # Coast up to 30: transition to LOST
        for _ in range(27):
            ps.coast()
        assert ps.state == "LOST"

    def test_threshold_depends_on_state(self):
        from analysis.player_tracker import PlayerState
        ps = PlayerState("P1", x=1.0, z=-3.0, side="near", fps=60.0)
        assert ps.threshold() == pytest.approx(2.0)
        for _ in range(3):
            ps.coast()
        assert ps.threshold() == pytest.approx(3.0)


# --- PlayerTracker test fixtures ---
#
# NOTE: the default camera used here is deliberately NOT R=eye(3), t=[0,-5,0]
# (that fixture projects every pixel far outside the court boundary gate).
# Instead we use a camera positioned at (0, 8, -6) looking down at the court,
# verified so that pixels in the ~(800-1100, 400-700) range land in-bounds:
#   (800, 650)  -> court (1.81, -1.56)   near side, in bounds
#   (1100, 650) -> court (-1.59, -1.56)  near side, in bounds
#   (800, 400)  -> court (2.30, 2.52)    far side, in bounds
#   (1100, 400) -> court (-2.01, 2.52)   far side, in bounds
#   (400, 500)  -> court (7.27, 0.65)    bystander, outside X bounds


def _make_cam_file(tmp_path, name, K=None, R=None, t=None):
    """Write a camera calibration JSON. Returns path string."""
    if K is None:
        K = np.array([[800, 0, 960],
                      [0, 800, 540],
                      [0, 0, 1]], dtype=np.float64)
    if R is None:
        R = np.array([[-1.0, 0.0, 0.0],
                      [0.0, -0.6, -0.8],
                      [0.0, -0.8, 0.6]], dtype=np.float64)
    if t is None:
        t = np.array([0.0, 0.0, 10.0], dtype=np.float64)
    p = tmp_path / f"{name}_cam.json"
    with open(p, "w") as f:
        json.dump({"K": K.tolist(), "R": R.tolist(), "t": t.tolist()}, f)
    return str(p)


def _make_kp2d_multi_file(tmp_path, name, frames, fps=60):
    """Write a multi-person kp2d JSON. Returns path string."""
    p = tmp_path / f"{name}_kp2d.json"
    with open(p, "w") as f:
        json.dump({
            "view": name, "source": f"{name}.mov", "clip": "test",
            "width": 1920, "height": 1080, "fps": fps,
            "frame_count": len(frames),
            "keypoint_format": "coco17", "multi_person": True,
            "frames": frames,
        }, f)
    return str(p)


def _make_seeds_file(tmp_path, views_seeds, mode="singles", clip="test"):
    """Write a seeds JSON. Returns path string."""
    p = tmp_path / "seeds.json"
    with open(p, "w") as f:
        json.dump({"clip": clip, "mode": mode, "views": views_seeds}, f)
    return str(p)


def _p1_near_kps():
    """P1 near side, ankle midpoint pixel ~(800, 710) -> court ~(1.7, -2.2)."""
    return _make_coco17({
        0: (800, 610), 5: (780, 630), 6: (820, 630),
        9: (760, 630), 10: (840, 630),
        11: (785, 670), 12: (815, 670),
        15: (785, 710), 16: (815, 710),
    }, default_score=0.9)


def _p2_near_kps():
    """P2 near side, ankle midpoint pixel ~(1100, 710) -> court ~(-1.5, -2.2)."""
    return _make_coco17({
        0: (1100, 610), 5: (1080, 630), 6: (1120, 630),
        9: (1060, 630), 10: (1140, 630),
        11: (1085, 670), 12: (1115, 670),
        15: (1085, 710), 16: (1115, 710),
    }, default_score=0.9)


def _p2_far_kps():
    """P2 far side, ankle midpoint pixel (800, 400) -> court (2.30, 2.52)."""
    return _make_coco17({
        0: (800, 300), 5: (780, 320), 6: (820, 320),
        9: (760, 320), 10: (840, 320),
        11: (785, 360), 12: (815, 360),
        15: (785, 400), 16: (815, 400),
    }, default_score=0.9)


class TestPlayerTracker:
    def test_singles_basic_tracking(self, tmp_path):
        """Two players tracked across 10 frames in one camera."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = []
        for f in range(10):
            dets = [
                {"track_id": 1, "keypoints": _p1_near_kps()},
                {"track_id": 2, "keypoints": _p2_near_kps()},
            ]
            frames.append(dets)

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [1100, 700]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()

        assert "P1" in result and "P2" in result
        assert view in result["P1"]
        assert len(result["P1"][view]) == 10
        # P1 should have detections in all 10 frames
        assert all(r is not None for r in result["P1"][view])

    def test_score_gating_rejects_low_quality(self, tmp_path):
        """Detections with mean body score < 0.3 are rejected."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = []
        for f in range(5):
            dets = [
                {"track_id": 1, "keypoints": _p1_near_kps()},
                # Low-quality bystander (also outside court bounds)
                {"track_id": 99, "keypoints": _make_coco17({
                    0: (400, 450), 5: (380, 480), 6: (420, 480),
                }, default_score=0.15)},
            ]
            frames.append(dets)

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [1100, 700]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()
        # P1 should have all frames, P2 should have none (no valid detection)
        p1_ok = sum(1 for r in result["P1"][view] if r is not None)
        p2_ok = sum(1 for r in result["P2"][view] if r is not None)
        assert p1_ok == 5
        assert p2_ok == 0

    def test_player_side_derived_from_seeds(self, tmp_path):
        """Court side is derived from seed court-Z, not hardcoded."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = [[
            {"track_id": 1, "keypoints": _p1_near_kps()},
            {"track_id": 2, "keypoints": _p2_far_kps()},
        ]]

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [800, 400]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        tracker.run()
        # Sides depend on actual court-Z from pixel_to_ground,
        # just verify both sides are set and different
        s1 = tracker.player_side("P1")
        s2 = tracker.player_side("P2")
        assert s1 in ("near", "far")
        assert s2 in ("near", "far")
        assert s1 != s2

    def test_coasting_and_reconnection(self, tmp_path):
        """Player disappears for 5 frames, then reappears — reconnected."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = []
        for f in range(20):
            dets = []
            # P1 always visible
            dets.append({"track_id": 1, "keypoints": _p1_near_kps()})
            # P2 disappears frames 5-9, reappears at frame 10 with new track ID
            if f < 5 or f >= 10:
                tid = 2 if f < 5 else 20  # new track ID after gap
                dets.append({"track_id": tid, "keypoints": _p2_near_kps()})
            frames.append(dets)

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [1100, 700]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()
        # P2 should have None for frames 5-9, not-None for 0-4 and 10-19
        p2 = result["P2"][view]
        for f in range(5):
            assert p2[f] is not None, f"P2 frame {f} should not be None"
        for f in range(5, 10):
            assert p2[f] is None, f"P2 frame {f} should be None (gap)"
        for f in range(10, 20):
            assert p2[f] is not None, f"P2 frame {f} should not be None (reconnected)"

    def test_lost_player_is_reacquired_after_a_long_gap(self, tmp_path):
        """Review 2 M3: after COAST_MAX_FRAMES unmatched a player went LOST for the rest of the clip."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = []
        for f in range(60):
            dets = [{"track_id": 1, "keypoints": _p1_near_kps()}]
            if f < 5 or f >= 40:                              # P2 absent for 35 frames (> 30)
                dets.append({"track_id": 2, "keypoints": _p2_near_kps()})
            frames.append(dets)
        tracker = PlayerTracker(
            cam_paths={view: _make_cam_file(tmp_path, view)},
            kp2d_multi_paths={view: _make_kp2d_multi_file(tmp_path, view, frames)},
            seeds_path=_make_seeds_file(tmp_path, {view: {"source": f"{view}.mov", "frame_time": 0,
                                                           "seeds": {"P1": [800, 700], "P2": [1100, 700]}}}),
            mode="singles")
        result = tracker.run()
        p1, p2 = result["P1"][view], result["P2"][view]
        assert all(p2[f] is not None for f in range(40, 60))
        assert all(p1[f] is not None for f in range(60))           # P1 keeps its own detection
        assert tracker.reacquired == 1

    def test_cross_camera_fusion_for_missing_seeds(self, tmp_path):
        """Camera B has no seed for P2. P2 gets fused from camera A's track."""
        from analysis.player_tracker import PlayerTracker

        # Camera A sees P1 and P2, Camera B sees P1 only (no P2 seed)
        frames_a = []
        frames_b = []
        for f in range(10):
            frames_a.append([
                {"track_id": 1, "keypoints": _p1_near_kps()},
                {"track_id": 2, "keypoints": _p2_near_kps()},
            ])
            frames_b.append([
                {"track_id": 10, "keypoints": _p1_near_kps()},
                {"track_id": 11, "keypoints": _p2_near_kps()},
            ])

        cam_a = _make_cam_file(tmp_path, "cam_a")
        cam_b = _make_cam_file(tmp_path, "cam_b")
        kp2d_a = _make_kp2d_multi_file(tmp_path, "cam_a", frames_a)
        kp2d_b = _make_kp2d_multi_file(tmp_path, "cam_b", frames_b)
        seeds_path = _make_seeds_file(tmp_path, {
            "cam_a": {"source": "cam_a.mov", "frame_time": 0,
                      "seeds": {"P1": [800, 700], "P2": [1100, 700]}},
            "cam_b": {"source": "cam_b.mov", "frame_time": 0,
                      "seeds": {"P1": [800, 700]}},
            # Note: cam_b has NO P2 seed
        })

        tracker = PlayerTracker(
            cam_paths={"cam_a": cam_a, "cam_b": cam_b},
            kp2d_multi_paths={"cam_a": kp2d_a, "cam_b": kp2d_b},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()
        # P2 should eventually get some frames in cam_b via fusion
        p2_b_ok = sum(1 for r in result["P2"]["cam_b"] if r is not None)
        assert p2_b_ok > 0, "P2 should be fused into cam_b from cam_a's track"

    def test_output_format_matches_match_and_split(self, tmp_path):
        """Output format is {pid: {view: [kps_or_None, ...]}}."""
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = [[
            {"track_id": 1, "keypoints": _p1_near_kps()},
        ]]

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [1100, 700]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()

        # Structure check
        assert isinstance(result, dict)
        for pid in ["P1", "P2"]:
            assert pid in result
            assert view in result[pid]
            assert isinstance(result[pid][view], list)
            for frame_val in result[pid][view]:
                if frame_val is not None:
                    assert isinstance(frame_val, list)
                    assert len(frame_val) >= 17
                    assert len(frame_val[0]) == 3  # [x, y, score]


class TestBuildRealIntegration:
    def test_import_works(self):
        """build_real.py can import from player_tracker."""
        from analysis.player_tracker import PlayerTracker
        assert PlayerTracker is not None

    def test_tracker_output_compatible_with_triangulation_loop(self, tmp_path):
        """Tracker output has the same structure the triangulation loop expects.

        NOTE: pixel positions here are chosen to land in-bounds for the
        default `_make_cam_file` camera (which maps pixels around
        (800-1100, 400-700) onto the court) — see the module docstring
        above `_make_cam_file` for the verified pixel->court mapping.
        """
        from analysis.player_tracker import PlayerTracker

        view = "cam_a"
        frames = []
        for f in range(5):
            frames.append([
                {"track_id": 1, "keypoints": _p1_near_kps()},
                {"track_id": 2, "keypoints": _p2_near_kps()},
            ])

        cam_path = _make_cam_file(tmp_path, view)
        kp2d_path = _make_kp2d_multi_file(tmp_path, view, frames)
        seeds_path = _make_seeds_file(tmp_path, {
            view: {"source": f"{view}.mov", "frame_time": 0,
                   "seeds": {"P1": [800, 700], "P2": [1100, 700]}}
        })

        tracker = PlayerTracker(
            cam_paths={view: cam_path},
            kp2d_multi_paths={view: kp2d_path},
            seeds_path=seeds_path,
            mode="singles",
        )
        result = tracker.run()
        players = ["P1", "P2"]
        views = [view]
        frame_count = min(len(result[players[0]][v]) for v in views)
        assert frame_count == 5

        # Simulate the triangulation access pattern from build_real.py:1641-1655
        for pid in players:
            kp = result[pid]
            for f in range(frame_count):
                for v in views:
                    row = kp[v][f]
                    if row is not None:
                        # Access individual joints like the triangulation loop does
                        for j in range(min(17, len(row))):
                            x, y, score = row[j]
                            assert isinstance(x, (int, float))
