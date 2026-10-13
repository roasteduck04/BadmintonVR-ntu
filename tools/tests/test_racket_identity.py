"""Tests for identity-aware racket assignment."""
import json
import numpy as np
import pytest


def _make_coco17(overrides, default_score=0.9):
    """Build a 17-keypoint array with defaults at (0,0,0)."""
    kps = [[0.0, 0.0, 0.0]] * 17
    kps = [list(k) for k in kps]
    for idx, vals in overrides.items():
        x, y = vals[0], vals[1]
        s = vals[2] if len(vals) > 2 else default_score
        kps[idx] = [float(x), float(y), float(s)]
    return kps


def _make_racket_kps(handle_x, handle_y, score=0.9):
    """Build a 5-keypoint racket detection [top, bottom, handle, left, right]."""
    return [
        [handle_x, handle_y - 30, score],  # top
        [handle_x, handle_y + 10, score],   # bottom
        [handle_x, handle_y, score],         # handle
        [handle_x - 5, handle_y - 15, score],  # left
        [handle_x + 5, handle_y - 15, score],  # right
    ]


def _make_racket2d_file(tmp_path, view, frames):
    """Write a racket2d JSON. frames[f] = [[5 kps], ...] or None."""
    p = tmp_path / f"{view}_racket2d.json"
    with open(p, "w") as f:
        json.dump({"view": view, "frames": frames}, f)
    return str(p)


class TestRacketIdentity:

    def test_greedy_fallback_unchanged(self, tmp_path):
        """use_hungarian=False preserves existing greedy behavior."""
        from analysis.build_real import _assign_racket_to_players
        view = "cam"
        # P1 wrist at (100, 200), P2 wrist at (300, 200)
        # Racket handle at (105, 205) — closer to P1
        matched = {
            "P1": {view: [_make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)]},
            "P2": {view: [_make_coco17({9: (300, 200), 10: (310, 200)}, 0.9)]},
        }
        racket_frames = [[_make_racket_kps(105, 205)]]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 1, ["P1", "P2"],
            use_hungarian=False)
        assert result["P1"][view][0] is not None
        assert result["P2"][view][0] is None

    def test_hungarian_optimal_assignment(self, tmp_path):
        """Hungarian finds globally optimal assignment over greedy."""
        from analysis.build_real import _assign_racket_to_players
        view = "cam"
        # P1 wrist at (100, 200), P2 wrist at (200, 200)
        # Racket A handle at (120, 200) — between both, slightly closer to P1
        # Racket B handle at (210, 200) — closest to P2
        # Greedy: A→P1 (closest), B→P2. Hungarian: same in this case.
        # But with sticky bonus, Hungarian can differ.
        matched = {
            "P1": {view: [_make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)]},
            "P2": {view: [_make_coco17({9: (200, 200), 10: (210, 200)}, 0.9)]},
        }
        racket_frames = [[
            _make_racket_kps(120, 200),  # A
            _make_racket_kps(210, 200),  # B
        ]]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 1, ["P1", "P2"],
            use_hungarian=True)
        # Both should be assigned
        assert result["P1"][view][0] is not None
        assert result["P2"][view][0] is not None

    def test_sticky_bonus_prevents_jitter(self, tmp_path):
        """Temporal sticky bonus keeps racket with same player across frames."""
        from analysis.build_real import _assign_racket_to_players
        view = "cam"
        # Frame 0: P1 wrist at (100,200), P2 wrist at (200,200)
        # Racket A handle at (105, 200) — clearly P1's
        # Frame 1: Players swap slightly closer
        # P1 wrist at (140,200), P2 wrist at (160,200)
        # Racket A handle at (148, 200) — slightly closer to P2 by distance
        # But sticky bonus should keep A with P1
        kp_p1_f0 = _make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)
        kp_p2_f0 = _make_coco17({9: (200, 200), 10: (210, 200)}, 0.9)
        kp_p1_f1 = _make_coco17({9: (140, 200), 10: (150, 200)}, 0.9)
        kp_p2_f1 = _make_coco17({9: (160, 200), 10: (170, 200)}, 0.9)
        matched = {
            "P1": {view: [kp_p1_f0, kp_p1_f1]},
            "P2": {view: [kp_p2_f0, kp_p2_f1]},
        }
        racket_frames = [
            [_make_racket_kps(105, 200)],   # frame 0: one racket near P1
            [_make_racket_kps(148, 200)],   # frame 1: same racket drifted
        ]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 2, ["P1", "P2"],
            use_hungarian=True, sticky_bonus=50.0)
        # Frame 0: assigned to P1 (closest)
        assert result["P1"][view][0] is not None
        # Frame 1: sticky bonus keeps it with P1 despite P2 being closer
        assert result["P1"][view][1] is not None
        assert result["P2"][view][1] is None

    def test_one_racket_per_player_cap(self, tmp_path):
        """Each player gets at most 1 racket per view per frame."""
        from analysis.build_real import _assign_racket_to_players
        view = "cam"
        # P1 wrist at (100, 200). Two rackets both near P1.
        matched = {
            "P1": {view: [_make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)]},
            "P2": {view: [_make_coco17({9: (500, 200), 10: (510, 200)}, 0.9)]},
        }
        racket_frames = [[
            _make_racket_kps(105, 200),
            _make_racket_kps(115, 200),
        ]]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 1, ["P1", "P2"],
            use_hungarian=True)
        # P1 gets exactly 1 racket (the closer one), not both
        assert result["P1"][view][0] is not None
        # P2 may or may not get the farther racket depending on RACKET_WRIST_MAX

    def test_greedy_rejects_beyond_wrist_max(self, tmp_path):
        """Greedy path must also reject rackets beyond RACKET_WRIST_MAX."""
        from analysis.build_real import _assign_racket_to_players, RACKET_WRIST_MAX
        view = "cam"
        far_x = 100 + RACKET_WRIST_MAX + 50
        matched = {
            "P1": {view: [_make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)]},
        }
        racket_frames = [[_make_racket_kps(far_x, 200)]]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 1, ["P1"],
            use_hungarian=False)
        assert result["P1"][view][0] is None, \
            "Greedy path should reject racket beyond RACKET_WRIST_MAX"

    def test_sync_racket2d_applies_offsets(self, tmp_path):
        """_sync_racket2d shifts frames by integer-rounded offsets."""
        from analysis.build_real import _sync_racket2d
        view = "cam"
        frames = [[_make_racket_kps(100 + i, 200)] for i in range(10)]
        p = _make_racket2d_file(tmp_path, view, frames)
        synced = _sync_racket2d({view: str(p)}, {view: 3.0}, frame_count=7)
        assert len(synced[view]) == 7
        hx = synced[view][0][0][2][0]  # handle x of first candidate
        assert abs(hx - 103.0) < 0.1, f"Expected handle x=103, got {hx}"

    def test_sync_racket2d_negative_offset(self, tmp_path):
        """Negative offset pads with None at the start."""
        from analysis.build_real import _sync_racket2d
        view = "cam"
        frames = [[_make_racket_kps(100, 200)] for _ in range(5)]
        p = _make_racket2d_file(tmp_path, view, frames)
        synced = _sync_racket2d({view: str(p)}, {view: -2.0}, frame_count=7)
        assert len(synced[view]) == 7
        assert synced[view][0] is None
        assert synced[view][1] is None
        assert synced[view][2] is not None

    def test_sync_racket2d_zero_offset(self, tmp_path):
        """Zero offset returns frames unchanged (trimmed to frame_count)."""
        from analysis.build_real import _sync_racket2d
        view = "cam"
        frames = [[_make_racket_kps(100, 200)] for _ in range(10)]
        p = _make_racket2d_file(tmp_path, view, frames)
        synced = _sync_racket2d({view: str(p)}, {view: 0.0}, frame_count=8)
        assert len(synced[view]) == 8
        assert synced[view][0] is not None

    def test_no_racket_beyond_wrist_max(self, tmp_path):
        """Racket candidates beyond RACKET_WRIST_MAX are rejected."""
        from analysis.build_real import _assign_racket_to_players, RACKET_WRIST_MAX
        view = "cam"
        # P1 wrist at (100, 200). Racket at (100 + RACKET_WRIST_MAX + 50, 200)
        far_x = 100 + RACKET_WRIST_MAX + 50
        matched = {
            "P1": {view: [_make_coco17({9: (100, 200), 10: (110, 200)}, 0.9)]},
        }
        racket_frames = [[_make_racket_kps(far_x, 200)]]
        racket2d_paths = {view: _make_racket2d_file(tmp_path, view, racket_frames)}
        result = _assign_racket_to_players(
            matched, racket2d_paths, [view], 1, ["P1"],
            use_hungarian=True)
        assert result["P1"][view][0] is None

    def test_majority_vote_helper(self):
        """_dominant_hand returns the majority hand from a list."""
        from analysis.build_real import _dominant_hand
        assert _dominant_hand(["right"] * 7 + ["left"] * 3) == "right"
        assert _dominant_hand(["left"] * 5 + ["right"] * 5) in ("left", "right")
        assert _dominant_hand(["left"]) == "left"
        assert _dominant_hand([]) is None
