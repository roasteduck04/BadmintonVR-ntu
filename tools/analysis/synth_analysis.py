"""Synthetic analysis.json v3 — two players rallying, no footage, no GPU.

Pure/deterministic: identical bytes every run. Unblocks Unity + the report so the
whole base is demoable before any real producer exists. Run:
    python -m analysis.synth_analysis --out Assets/StreamingAssets/analysis/synth_rally_01.analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

from analysis import schema

STROKE_CYCLE = ["clear", "smash", "drop", "lift"]
HIT_INTERVAL = 20            # frames between shuttle contacts
_ARM_JOINTS = (17, 19, 21, 43)   # right shoulder/elbow/wrist/right_middle1 (SMPL-X 58)
_SHUTTLE_PEAK = 2.6          # meters
_BASE_Z = 3.0               # near player at -_BASE_Z, far at +_BASE_Z


def _player_position(side_sign, frame, frames):
    """Court XZ for a player. side_sign: -1 near, +1 far."""
    phase = 0.0 if side_sign < 0 else math.pi
    x = 1.6 * math.sin(2 * math.pi * frame / frames * 2 + phase)
    z = side_sign * _BASE_Z
    return [round(x, 4), round(z, 4)]


def _local_pose(frame):
    """Body-local joints_flat (232 floats) = rest pose + racket-arm swing."""
    pose = schema.REST_POSE.copy()
    s = math.sin(2 * math.pi * frame / HIT_INTERVAL)
    dy, dz = 0.30 * max(0.0, s), 0.12 * s
    for j in _ARM_JOINTS:
        pose[j, 1] += dy
        pose[j, 2] += dz
    flat = []
    for j in range(schema.NUM_JOINTS):
        flat.extend([round(float(pose[j, 0]), 5),
                     round(float(pose[j, 1]), 5),
                     round(float(pose[j, 2]), 5), 1.0])
    return flat


def _hitter_index(k):
    """Contact k is hit by player 0 (near) on even k, player 1 (far) on odd k."""
    return k % 2


def _build_player(pindex, side, frames, fps):
    side_sign = -1 if side == "near" else 1
    out = []
    for f in range(frames):
        k = f // HIT_INTERVAL
        stroke = None
        if f % HIT_INTERVAL == 0 and _hitter_index(k) == pindex:
            stroke = {"label": STROKE_CYCLE[k % len(STROKE_CYCLE)], "confidence": 0.85}
        out.append({
            "frame_id": f, "time": round(f / fps, 4),
            "joints_flat": _local_pose(f),
            "position": _player_position(side_sign, f, frames),
            "racket": None, "grip": None, "stroke": stroke,
        })
    return {"id": f"P{pindex + 1}", "side": side, "frames": out}


def _shuttle_frames(players, frames, fps):
    """Parabolic arcs between successive contacts, using hitters' court positions."""
    def court_xz(pindex, frame):
        return players[pindex]["frames"][frame]["position"]

    out = []
    for f in range(frames):
        k = f // HIT_INTERVAL
        f0 = k * HIT_INTERVAL
        f1 = min(f0 + HIT_INTERVAL, frames - 1)
        u = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
        src = court_xz(_hitter_index(k), min(f0, frames - 1))
        dst = court_xz(_hitter_index(k + 1), f1)
        x = src[0] + (dst[0] - src[0]) * u
        z = src[1] + (dst[1] - src[1]) * u
        y = _SHUTTLE_PEAK * math.sin(math.pi * u)
        out.append({"frame_id": f, "time": round(f / fps, 4),
                    "position": [round(x, 4), round(max(0.0, y), 4), round(z, 4)],
                    "confidence": 0.8})
    return out


def build_synth_doc(frames=120, fps=30.0):
    """A valid analysis.json v3 dict: two players rallying + a shuttle arc."""
    players = [_build_player(0, "near", frames, fps), _build_player(1, "far", frames, fps)]
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "clip_id": schema.CLIP_ID,
        "meta": {"fps": round(float(fps), 3), "frame_count": frames,
                 "source_clip": "synthetic", "camera": None,
                 "producers": {"pose": "synthetic", "position": "synthetic",
                               "racket": None, "grip": None,
                               "stroke": "synthetic", "shuttle": "synthetic"}},
        "court": schema.court_block(),
        "skeleton": {"type": schema.SKELETON_TYPE,
                     "joint_names": schema.JOINT_NAMES, "parents": schema.PARENTS},
        "players": players,
        "shuttle": {"frames": _shuttle_frames(players, frames, fps)},
        "rallies": [{"start": 0, "end": frames - 1, "winner": "P1"}],
    }


def main(argv=None):
    default_out = os.path.join("Assets", "StreamingAssets", "analysis",
                               "synth_rally_01.analysis.json")
    ap = argparse.ArgumentParser(description="Write a synthetic analysis.json v3.")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    doc = build_synth_doc(frames=args.frames, fps=args.fps)
    errs = schema.validate_analysis(doc)
    if errs:
        raise SystemExit("synthetic doc failed validation:\n  " + "\n  ".join(errs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    print(f"wrote {args.out}: {args.frames} frames, {len(doc['players'])} players, "
          f"{len(doc['shuttle']['frames'])} shuttle frames")


if __name__ == "__main__":
    main()
