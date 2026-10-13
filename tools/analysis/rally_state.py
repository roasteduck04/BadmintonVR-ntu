"""Rally-state segmentation for badminton matches.

Classifies each frame as rally (active play), break (between points),
or transition (buffer after rally ends). Uses shuttle velocity as the
primary signal and player wrist speeds as a fallback.
"""
from __future__ import annotations

# --- Parameters (spec Parameters) ---
SHUTTLE_FLIGHT_VEL = 1.0   # m/s - min shuttle velocity for "in flight"
WRIST_ACTIVE_VEL = 2.0     # m/s - min wrist speed for "active" player
WRIST_QUIET_VEL = 1.0      # m/s - max wrist speed for "quiet" player
BREAK_QUIET_FRAMES = 30    # consecutive quiet frames -> transition (0.5s @60fps)
TRANSITION_FRAMES = 30     # transition buffer duration (0.5s @60fps)


class RallyStateMachine:
    """Frame-by-frame rally/break classifier.

    Call update() once per frame in order. After all frames, call rallies()
    to get the list of rally segments.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self._state = "break"      # current state
        self._quiet_count = 0      # consecutive quiet frames
        self._transition_count = 0 # frames spent in transition
        self._rally_start: int | None = None
        self._rallies: list[dict] = []
        self.__last_frame = 0

    def update(self, frame: int, shuttle_pos: list | None,
               shuttle_vel: float | None,
               player_wrist_speeds: dict[str, float]) -> str:
        """Process one frame. Returns current state: rally/break/transition."""

        # Signal 1: shuttle in flight
        shuttle_active = (shuttle_pos is not None
                          and shuttle_vel is not None
                          and shuttle_vel > SHUTTLE_FLIGHT_VEL)

        # Signal 2: multiple players with high wrist speed
        n_active = sum(1 for v in player_wrist_speeds.values()
                       if v > WRIST_ACTIVE_VEL)
        wrists_active = n_active >= 2

        if self._state == "break":
            if shuttle_active:
                self._enter_rally(frame)
            # Stay in break otherwise
            self._last_frame = frame

        elif self._state == "rally":
            if shuttle_active:
                self._quiet_count = 0
            elif wrists_active:
                self._quiet_count = 0
            else:
                # Not enough activity (all quiet, or some but not
                # sufficient) - count as quiet.
                self._quiet_count += 1
                if self._quiet_count >= BREAK_QUIET_FRAMES:
                    self._enter_transition(frame)
            self._last_frame = frame

        elif self._state == "transition":
            if shuttle_active:
                self._enter_rally(frame)
            else:
                self._transition_count += 1
                if self._transition_count >= TRANSITION_FRAMES:
                    self._enter_break(frame)
            self._last_frame = frame

        return self._state

    def rallies(self) -> list[dict]:
        """Return completed rally segments. Call after all frames processed."""
        result = list(self._rallies)
        # If currently in a rally or transition (rally not yet closed),
        # close it at the last processed frame.
        if self._state in ("rally", "transition") and self._rally_start is not None:
            result.append(self._make_rally(self._rally_start, self._last_frame))
        return result

    def _enter_rally(self, frame: int) -> None:
        if self._state == "transition" and self._rally_start is not None:
            # Re-entering rally from transition - extend the current rally.
            pass
        elif self._rally_start is None:
            self._rally_start = frame
        self._state = "rally"
        self._quiet_count = 0
        self._transition_count = 0

    def _enter_transition(self, frame: int) -> None:
        self._state = "transition"
        self._transition_count = 0

    def _enter_break(self, frame: int) -> None:
        if self._rally_start is not None:
            # The rally ended when transition started, not now.
            rally_end = frame - TRANSITION_FRAMES
            self._rallies.append(self._make_rally(self._rally_start, rally_end))
            self._rally_start = None
        self._state = "break"
        self._quiet_count = 0
        self._transition_count = 0

    def _make_rally(self, start: int, end: int) -> dict:
        duration = (end - start) / self.fps
        return {"start": start, "end": end, "duration_s": round(duration, 3)}

    @property
    def _last_frame(self) -> int:
        return self.__last_frame

    @_last_frame.setter
    def _last_frame(self, v: int) -> None:
        self.__last_frame = v
