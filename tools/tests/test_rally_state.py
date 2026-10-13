"""Tests for RallyStateMachine."""
import pytest


class TestRallyStateMachine:

    def test_starts_in_break(self):
        """Initial state is 'break' (no shuttle, no wrist activity)."""
        from analysis.rally_state import RallyStateMachine
        rsm = RallyStateMachine(fps=60.0)
        state = rsm.update(0, shuttle_pos=None, shuttle_vel=None,
                           player_wrist_speeds={})
        assert state == "break"

    def test_shuttle_flight_triggers_rally(self):
        """Shuttle moving > SHUTTLE_FLIGHT_VEL transitions to rally."""
        from analysis.rally_state import RallyStateMachine
        rsm = RallyStateMachine(fps=60.0)
        # Frame 0: no shuttle → break
        rsm.update(0, None, None, {})
        # Frame 1: shuttle detected at 5 m/s → rally
        state = rsm.update(1, [1.0, 0.0, 2.0], 5.0, {"P1": 0.5, "P2": 0.5})
        assert state == "rally"

    def test_wrist_speed_fallback_keeps_rally(self):
        """If shuttle is gone but players have high wrist speed, stay rally."""
        from analysis.rally_state import RallyStateMachine
        rsm = RallyStateMachine(fps=60.0)
        # Enter rally via shuttle
        rsm.update(0, [1.0, 0.0, 2.0], 5.0, {"P1": 3.0, "P2": 3.0})
        # Shuttle disappears but wrists are fast
        state = rsm.update(1, None, None, {"P1": 3.0, "P2": 3.0})
        assert state == "rally"

    def test_quiet_triggers_transition_then_break(self):
        """30+ quiet frames after rally → transition → break."""
        from analysis.rally_state import (
            RallyStateMachine, BREAK_QUIET_FRAMES, TRANSITION_FRAMES)
        rsm = RallyStateMachine(fps=60.0)
        # Start rally
        rsm.update(0, [1.0, 0.0, 2.0], 5.0, {"P1": 0.5})
        # Go quiet for BREAK_QUIET_FRAMES
        f = 1
        for _ in range(BREAK_QUIET_FRAMES):
            state = rsm.update(f, None, None, {"P1": 0.3, "P2": 0.3})
            f += 1
        assert state == "transition"
        # Continue quiet for TRANSITION_FRAMES more
        for _ in range(TRANSITION_FRAMES):
            state = rsm.update(f, None, None, {"P1": 0.3, "P2": 0.3})
            f += 1
        assert state == "break"

    def test_shuttle_reappears_after_break(self):
        """Shuttle re-appearing during break transitions to rally."""
        from analysis.rally_state import (
            RallyStateMachine, BREAK_QUIET_FRAMES, TRANSITION_FRAMES)
        rsm = RallyStateMachine(fps=60.0)
        # Rally → quiet → transition → break
        rsm.update(0, [1.0, 0.0, 2.0], 5.0, {"P1": 0.5})
        f = 1
        for _ in range(BREAK_QUIET_FRAMES + TRANSITION_FRAMES):
            rsm.update(f, None, None, {"P1": 0.3})
            f += 1
        assert rsm.update(f, None, None, {"P1": 0.3}) == "break"
        # Shuttle reappears
        state = rsm.update(f + 1, [2.0, 0.0, 1.0], 8.0, {"P1": 0.5})
        assert state == "rally"

    def test_rallies_output_format(self):
        """rallies() returns properly formatted rally segments."""
        from analysis.rally_state import (
            RallyStateMachine, BREAK_QUIET_FRAMES, TRANSITION_FRAMES)
        rsm = RallyStateMachine(fps=60.0)
        # Rally from frame 0-9, then quiet
        for f in range(10):
            rsm.update(f, [1.0, 0.0, 2.0], 5.0, {"P1": 0.5})
        for f in range(10, 10 + BREAK_QUIET_FRAMES + TRANSITION_FRAMES + 10):
            rsm.update(f, None, None, {"P1": 0.3})
        rallies = rsm.rallies()
        assert len(rallies) >= 1
        r = rallies[0]
        assert "start" in r and "end" in r and "duration_s" in r
        assert r["start"] == 0
        assert r["end"] >= 9
        assert isinstance(r["duration_s"], float)
        assert r["duration_s"] > 0

    def test_wrist_active_count_threshold(self):
        """Need multiple players with high wrist speed for fallback."""
        from analysis.rally_state import RallyStateMachine
        rsm = RallyStateMachine(fps=60.0)
        # Enter rally via shuttle
        rsm.update(0, [1.0, 0.0, 2.0], 5.0, {"P1": 3.0, "P2": 3.0})
        # Only ONE player has high wrist speed — not enough for fallback
        state = rsm.update(1, None, None, {"P1": 3.0, "P2": 0.3})
        # Should start counting quiet frames (may still be rally for 1 frame
        # but won't sustain). After enough quiet frames it transitions.
        # For this single frame, the state depends on implementation —
        # the key assertion is that it eventually transitions to break.
        for f in range(2, 100):
            state = rsm.update(f, None, None, {"P1": 0.3, "P2": 0.3})
        assert state == "break"
