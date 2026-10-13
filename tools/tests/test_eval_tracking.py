"""Tests for automated tracking quality evaluation."""
import json
import math
import pytest


def _make_analysis(players, frame_count=100, fps=60.0):
    """Build a minimal valid analysis.json dict."""
    from analysis import schema
    player_list = []
    for pid, side, pose_fn in players:
        frames = []
        for f in range(frame_count):
            jf, pos = pose_fn(f)
            frames.append({
                "frame_id": f, "time": round(f / fps, 4),
                "joints_flat": jf, "position": pos,
                "racket": None, "grip": None, "stroke": None,
            })
        player_list.append({"id": pid, "side": side, "frames": frames})
    return {
        "schema_version": "3.0",
        "clip_id": "test",
        "meta": {"fps": fps, "frame_count": frame_count,
                 "source_clip": "test", "camera": None,
                 "producers": {"pose": "test"}},
        "court": schema.court_block(),
        "skeleton": {"type": "smpl-24",
                     "joint_names": schema.JOINT_NAMES,
                     "parents": schema.PARENTS},
        "players": player_list,
        "shuttle": {"frames": []},
        "rallies": [],
    }


class TestComputeMetrics:

    def test_full_coverage(self):
        """100% coverage when all frames have pose."""
        from analysis.eval_tracking import compute_metrics
        def always_present(f):
            return [0.0] * 96, [0.0, float(f) * 0.01]
        doc = _make_analysis([("P1", "near", always_present)], frame_count=50)
        m = compute_metrics(doc)
        assert m["P1"]["pose_coverage"] == 100.0
        assert m["P1"]["position_coverage"] == 100.0
        assert m["P1"]["gap_count"] == 0

    def test_gap_detection(self):
        """Detect gaps in pose coverage."""
        from analysis.eval_tracking import compute_metrics
        def with_gap(f):
            if 10 <= f < 20:
                return None, None
            return [0.0] * 96, [0.0, 0.0]
        doc = _make_analysis([("P1", "near", with_gap)], frame_count=50)
        m = compute_metrics(doc)
        assert m["P1"]["pose_coverage"] == 80.0
        assert m["P1"]["gap_count"] == 1
        assert m["P1"]["gap_max"] == 10

    def test_jump_detection(self):
        """Detect position jumps > 1m and > 2m."""
        from analysis.eval_tracking import compute_metrics
        def with_jump(f):
            if f == 25:
                return [0.0] * 96, [5.0, 0.0]  # jump of 5m in X
            return [0.0] * 96, [0.0, 0.0]
        doc = _make_analysis([("P1", "near", with_jump)], frame_count=50)
        m = compute_metrics(doc)
        assert m["P1"]["jump_count_1m"] >= 1
        assert m["P1"]["jump_count_2m"] >= 1
        assert m["P1"]["jump_max"] >= 4.9

    def test_wrong_side_detection(self):
        """Detect frames on wrong court side."""
        from analysis.eval_tracking import compute_metrics
        # "near" side = positive Z (court origin at center, Z = 13.40m total)
        def on_wrong_side(f):
            if f < 10:
                return [0.0] * 96, [0.0, -3.0]  # negative Z = far side
            return [0.0] * 96, [0.0, 3.0]
        doc = _make_analysis([("P1", "near", on_wrong_side)], frame_count=50)
        m = compute_metrics(doc)
        assert m["P1"]["wrong_side_frames"] == 10

    def test_aggregate_overlap(self):
        """Detect player overlap (two players < 0.3m apart)."""
        from analysis.eval_tracking import compute_metrics
        def p1_pos(f):
            return [0.0] * 96, [0.0, 1.0]
        def p2_pos(f):
            if f < 5:
                return [0.0] * 96, [0.0, 1.1]  # 0.1m apart → overlap
            return [0.0] * 96, [0.0, 5.0]
        doc = _make_analysis([
            ("P1", "near", p1_pos),
            ("P2", "far", p2_pos),
        ], frame_count=50)
        m = compute_metrics(doc)
        assert m["_aggregate"]["player_overlap_frames"] == 5

    def test_handles_old_analysis_without_facing(self):
        """Works on analysis.json without facing field."""
        from analysis.eval_tracking import compute_metrics
        def simple(f):
            return [0.0] * 96, [0.0, 0.0]
        doc = _make_analysis([("P1", "near", simple)], frame_count=10)
        # Don't add facing field — should still work
        m = compute_metrics(doc)
        assert "P1" in m

    def test_close_teammate_detection(self):
        """Detect same-side players closer than 0.5m."""
        from analysis.eval_tracking import compute_metrics
        def p1_pos(f):
            return [0.0] * 96, [0.0, 1.0]
        def p2_pos(f):
            if f < 8:
                return [0.0] * 96, [0.1, 1.2]  # ~0.22m apart, same side
            return [0.0] * 96, [2.0, 1.0]
        doc = _make_analysis([
            ("P1", "near", p1_pos),
            ("P2", "near", p2_pos),
        ], frame_count=50)
        m = compute_metrics(doc)
        assert m["P1"]["close_teammate_frames"] == 8


class TestCompareMetrics:

    def test_comparison_produces_deltas(self):
        """compare_metrics returns rows with delta values."""
        from analysis.eval_tracking import compute_metrics, compare_metrics
        def full(f):
            return [0.0] * 96, [0.0, 0.0]
        def gappy(f):
            if 10 <= f < 30:
                return None, None
            return [0.0] * 96, [0.0, 0.0]
        doc_new = _make_analysis([("P1", "near", full)], frame_count=50)
        doc_old = _make_analysis([("P1", "near", gappy)], frame_count=50)
        m_new = compute_metrics(doc_new)
        m_old = compute_metrics(doc_old)
        rows = compare_metrics(m_new, m_old)
        # Find the pose_coverage row for P1
        cov_row = [r for r in rows if r["player"] == "P1"
                   and r["metric"] == "pose_coverage"][0]
        assert cov_row["new"] == 100.0
        assert cov_row["old"] == 60.0
        assert cov_row["delta"] == 40.0
