"""SP3 people: 3D-first player / non-player detection and identity ("tracker3d").

Spec: docs/superpowers/specs/2026-09-27-sp3-player-nonplayer-detection-design.md. Pipeline per clip:
candidates (cross-view 3D people per synced frame) -> tracklets -> seeds pin tracklets -> per-tracklet
player score -> per-half identity (min-cost flow) -> persons/1 sidecar.
"""
from __future__ import annotations

import json
import os

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params.json")


def load_params(path: str | None = None) -> dict:
    """Thresholds + classifier weights (params.json; hand-tuned 2026-09-29, refit once labels exist)."""
    with open(path or PARAMS_PATH, "r", encoding="utf-8") as fh:
        p = json.load(fh)
    p.pop("_comment", None)
    return p
