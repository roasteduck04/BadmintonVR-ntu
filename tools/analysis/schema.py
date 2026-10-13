"""analysis.json v5 — the single contract. Superset of skeleton.json v2.

Every producer-supplied field is nullable so synthetic, partial, and real data
all validate. Pose is the 58-joint skeleton (see analysis.skeleton) and body-local;
court placement lives in each frame's `position`. See
docs/for-claude/analysis-schema-v3.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis import skeleton as _skel

SCHEMA_VERSION = "5.0"
SKELETON_TYPE = "smplx-58"       # skeleton.type of every 58-joint doc (Unity keys on the joint count)
CLIP_ID = "synth_rally_01"

NUM_JOINTS = _skel.NUM_JOINTS          # 58
STRIDE = 4                             # x, y, z, confidence
JOINT_NAMES = list(_skel.JOINT_NAMES)
PARENTS = list(_skel.PARENTS)
REST_POSE = np.array(_skel._REST, dtype=np.float64)  # (58,3) Y-up T-pose, meters

_COURT_KEYS = ("length_z", "width_x", "singles_half_width_x",
               "baseline_z", "short_service_z", "net_z")


def court_block():
    """The `court` sub-document, read from the canonical geometry file."""
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "data", "calib", "court_geometry.json")
    with open(os.path.abspath(path), "r", encoding="utf-8") as fh:
        dims = json.load(fh)["dimensions_m"]
    return {k: round(float(dims[k]), 3) for k in _COURT_KEYS}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_vec3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3


def _check_player_frame(pf, where, errs):
    jf = pf.get("joints_flat")
    if jf is not None and len(jf) != NUM_JOINTS * STRIDE:
        errs.append(f"{where}: joints_flat must be {NUM_JOINTS * STRIDE} floats or null, got {len(jf)}")
    pos = pf.get("position")
    if pos is not None and len(pos) != 2:
        errs.append(f"{where}: position must be [x, z] (len 2) or null, got len {len(pos)}")
    st = pf.get("stroke")
    if st is not None and ("label" not in st or not _is_num(st.get("confidence"))):
        errs.append(f"{where}: stroke must be null or have string label + numeric confidence")

    rk = pf.get("racket")
    if rk is not None:
        if not isinstance(rk, dict):
            errs.append(f"{where}: racket must be null or an object")
        else:
            for key in ("grip", "head", "side", "axis", "face_normal"):
                v = rk.get(key)
                if v is not None and not _is_vec3(v):
                    errs.append(f"{where}: racket.{key} must be [x, y, z] or null")
            if rk.get("which_hand") not in ("left", "right"):
                errs.append(f"{where}: racket.which_hand must be 'left' or 'right'")

    gp = pf.get("grip")
    if gp is not None:
        if not isinstance(gp, dict):
            errs.append(f"{where}: grip must be null or an object")
        else:
            if not _is_vec3(gp.get("face_normal")):
                errs.append(f"{where}: grip.face_normal must be [x, y, z]")
            if gp.get("which_hand") not in ("left", "right"):
                errs.append(f"{where}: grip.which_hand must be 'left' or 'right'")

    fc = pf.get("facing")
    if fc is not None:
        if not isinstance(fc, dict):
            errs.append(f"{where}: facing must be null or an object")
        else:
            for key in ("heading", "shoulder_heading"):
                v = fc.get(key)
                if v is not None and not _is_num(v):
                    errs.append(f"{where}: facing.{key} must be numeric or null")
            conf = fc.get("confidence")
            if conf is not None and not _is_num(conf):
                errs.append(f"{where}: facing.confidence must be numeric or null")


def validate_analysis(doc):
    """Return a list of human-readable schema errors; empty list means valid."""
    errs = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION!r}, got {doc.get('schema_version')!r}")

    if not isinstance(doc.get("clip_id"), str) or not doc.get("clip_id"):
        errs.append("clip_id must be a non-empty string")

    meta = doc.get("meta")
    if not isinstance(meta, dict):
        errs.append("meta must be an object")
    else:
        if not _is_num(meta.get("fps")):
            errs.append("meta.fps missing or non-numeric")
        if not (isinstance(meta.get("frame_count"), int) and not isinstance(meta.get("frame_count"), bool)):
            errs.append("meta.frame_count missing or non-integer")

    skel = doc.get("skeleton") or {}
    if skel.get("joint_names") != JOINT_NAMES:
        errs.append("skeleton.joint_names must equal the SMPL-X 58 name list")
    if skel.get("parents") != PARENTS:
        errs.append("skeleton.parents must equal the SMPL-X 58 parent list")

    court = doc.get("court") or {}
    for k in _COURT_KEYS:
        if not _is_num(court.get(k)):
            errs.append(f"court.{k} missing or non-numeric")

    players = doc.get("players")
    if not isinstance(players, list) or len(players) < 1:
        errs.append("players must be a non-empty list")
        players = []
    for p in players:
        pid = p.get("id", "?")
        if p.get("side") not in ("near", "far"):
            errs.append(f"player {pid}: side must be 'near' or 'far'")
        frames = p.get("frames")
        if not isinstance(frames, list) or not frames:
            errs.append(f"player {pid}: frames must be a non-empty list")
            continue
        for i, pf in enumerate(frames):
            _check_player_frame(pf, f"player {pid} frame {i}", errs)

    sh = doc.get("shuttle") or {}
    sframes = sh.get("frames")
    if not isinstance(sframes, list):
        errs.append("shuttle.frames must be a list")
    else:
        for i, sf in enumerate(sframes):
            pos = sf.get("position")
            if pos is not None and len(pos) != 3:
                errs.append(f"shuttle frame {i}: position must be [x, y, z] (len 3) or null")
    return errs
