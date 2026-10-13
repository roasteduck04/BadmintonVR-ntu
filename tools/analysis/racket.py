"""Racket + grip producer — body-local 3D racket, the second real Phase 2 producer.

The racket lives in the SAME SMPL-24 body-local frame as `joints_flat` (Y-up meters),
so shaft axis, face-normal, and which-hand are well-defined and camera-agnostic. It is
modelled as three points — grip, head, side — from which an orthonormal racket frame is
built. Grip (racket-face orientation + which-hand) is derived from the racket; finger-level
grip type is a reserved slot filled later from SMPL-X hands (close-up footage only).

Contract slice: fills each `players[i].frames[f].racket` and `.grip`. See analysis/schema.py.

Foundation-first: the real input is RacketVision 2D keypoints triangulated to 3D via
multi-view RANSAC (Phase 4 pipeline in build_real.py). The synthetic stub
`synthetic_racket_tracks` remains for tests. The old monocular 3D-lift is archived
under `_archive/old-pipeline/`.
"""
from __future__ import annotations

import numpy as np

from analysis import smpl24
from racket_smoothing import smooth_racket

_LEFT_HAND = smpl24.SMPL_JOINT_NAMES.index("left_hand")
_RIGHT_HAND = smpl24.SMPL_JOINT_NAMES.index("right_hand")


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _joint_xyz(joints_flat, idx):
    base = idx * 4  # stride 4: x, y, z, confidence
    return np.array(joints_flat[base:base + 3], dtype=np.float64)


def racket_frame(grip, head, side):
    """Orthonormal racket frame from three body-local points.

    Returns unit (axis, across, face_normal): axis = shaft dir (head-grip),
    across = face-width dir (side-head), face_normal = unit(axis x across).
    """
    grip = np.asarray(grip, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    side = np.asarray(side, dtype=np.float64)
    axis = _unit(head - grip)
    across = _unit(side - head)
    face_normal = _unit(np.cross(axis, across))
    return axis, across, face_normal


def nearest_hand(joints_flat, point):
    """'left' or 'right' — whichever SMPL hand joint is closer to `point`
    (both in the body-local frame)."""
    point = np.asarray(point, dtype=np.float64)
    lh = _joint_xyz(joints_flat, _LEFT_HAND)
    rh = _joint_xyz(joints_flat, _RIGHT_HAND)
    return "left" if np.linalg.norm(point - lh) < np.linalg.norm(point - rh) else "right"


def derive_grip(racket):
    """Grip summary from a racket dict: racket-face orientation + which hand."""
    return {"face_normal": racket["face_normal"],
            "which_hand": racket["which_hand"],
            "confidence": racket["confidence"]}


def _round3(v, nd=5):
    return [round(float(v[0]), nd), round(float(v[1]), nd), round(float(v[2]), nd)]


def fill_player_rackets(doc, tracks, producer_name, smooth=True):
    """Write each player's body-local racket + grip from grip/head/side tracks.

    tracks: {player_id: {"grip"/"head"/"side": (N,3), "status": [str]*N optional,
    "confidence": [float]*N optional}}, N == that player's frame count. A "missing"
    status frame -> racket None, grip None. Rigid-smoothed via racket_smoothing so the
    racket keeps a constant length/width. Tags meta.producers.racket and .grip.
    """
    fps = float(doc["meta"]["fps"])
    for player in doc["players"]:
        pid = player["id"]
        if pid not in tracks:
            continue
        tr = tracks[pid]
        grips = np.asarray(tr["grip"], dtype=np.float64)
        heads = np.asarray(tr["head"], dtype=np.float64)
        sides = np.asarray(tr["side"], dtype=np.float64)
        frames = player["frames"]
        n = len(frames)
        if not (len(grips) == len(heads) == len(sides) == n):
            raise ValueError(
                f"player {pid}: track length {len(grips)} != {n} frames")
        status = list(tr.get("status", ["detected"] * n))
        conf = list(tr.get("confidence", [0.8] * n))
        valid = np.array([s != "missing" for s in status])

        if smooth and valid.any():
            grips, heads, sides = smooth_racket(grips, heads, sides, fps, valid=valid)

        for i, pf in enumerate(frames):
            if not valid[i] or pf.get("joints_flat") is None:
                pf["racket"] = None
                pf["grip"] = None
                continue
            axis, _across, fn = racket_frame(grips[i], heads[i], sides[i])
            racket = {
                "grip": _round3(grips[i]), "head": _round3(heads[i]),
                "side": _round3(sides[i]), "axis": _round3(axis),
                "face_normal": _round3(fn),
                "which_hand": nearest_hand(pf["joints_flat"], grips[i]),
                "confidence": round(float(conf[i]), 4), "status": status[i],
            }
            pf["racket"] = racket
            pf["grip"] = derive_grip(racket)

    producers = doc.setdefault("meta", {}).setdefault("producers", {})
    producers["racket"] = producer_name
    producers["grip"] = producer_name
    return doc


RACKET_LENGTH_M = 0.66       # regulation shaft length
RACKET_HALF_WIDTH_M = 0.10   # regulation head half-width

_RIGHT_WRIST = smpl24.SMPL_JOINT_NAMES.index("right_wrist")


def synthetic_racket_tracks(doc):
    """Stand-in for RacketVision-2D-lifted-to-3D: a body-local racket off each player's
    right hand. Grip at the hand joint, shaft along wrist->hand, face width perpendicular.
    REPLACEMENT SEAM — swap for the real (parked) 3D-lift when footage exists.
    """
    tracks = {}
    for player in doc["players"]:
        grips, heads, sides = [], [], []
        for pf in player["frames"]:
            hand = _joint_xyz(pf["joints_flat"], _RIGHT_HAND)
            wrist = _joint_xyz(pf["joints_flat"], _RIGHT_WRIST)
            shaft = _unit(hand - wrist)
            if np.linalg.norm(shaft) < 1e-9:
                shaft = np.array([-1.0, 0.0, 0.0])
            across = _unit(np.cross(shaft, np.array([0.0, 0.0, 1.0])))
            if np.linalg.norm(across) < 1e-9:
                across = np.array([0.0, 1.0, 0.0])
            grip = hand
            head = grip + RACKET_LENGTH_M * shaft
            side = head + RACKET_HALF_WIDTH_M * across
            grips.append(grip); heads.append(head); sides.append(side)
        tracks[player["id"]] = {"grip": np.array(grips), "head": np.array(heads),
                                "side": np.array(sides)}
    return tracks


def main(argv=None):
    import argparse
    import json
    import os

    from analysis import schema, synth_analysis

    ap = argparse.ArgumentParser(
        description="Racket + grip producer: fill analysis.json with a body-local racket "
                    "(foundation demo uses a synthetic racket off the pose).")
    ap.add_argument("--out", required=True, help="output analysis.json")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    doc = synth_analysis.build_synth_doc(frames=args.frames, fps=args.fps)
    tracks = synthetic_racket_tracks(doc)
    fill_player_rackets(doc, tracks, "synthetic_racket_bodylocal")

    errs = schema.validate_analysis(doc)
    if errs:
        raise SystemExit("analysis doc failed validation:\n  " + "\n  ".join(errs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    print(f"wrote {args.out}: {args.frames} frames, body-local racket + grip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
