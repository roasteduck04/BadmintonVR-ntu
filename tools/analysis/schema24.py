"""analysis.json schema 4.1 — the SMPL-24 / SMPL-28 contract written by analysis.pose24.

4.1 = the v4 body-local layout (joints_flat J x 4, court `position`) with a labelled skeleton (`type`
smpl-24 | smpl-28, `family`, `label`, `aux_names`) and per-frame `interpolated` + `aux_flat` (14 x 4 or
null). Legacy 4.0 files (28 joints, no aux) validate read-only with allow_legacy=True. SMPL-X 58 docs
(schema 5.0) belong to analysis.schema and are rejected here.
"""
from __future__ import annotations

import math

from analysis import schema
from analysis import skeleton24 as sk

SCHEMA_VERSION = "4.1"
LEGACY_VERSIONS = ("4.0",)
STRIDE = 4
AUX_LEN = len(sk.AUX_NAMES) * STRIDE


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite_list(v):
    return all(_num(x) and math.isfinite(x) for x in v)


def _skeleton_errors(skel, legacy):
    if legacy:
        if skel.get("joint_names") != sk.SMPL28_NAMES or skel.get("parents") != sk.SMPL28_PARENTS:
            return ["legacy 4.0 skeleton must be the SMPL-28 name/parent list"], len(sk.SMPL28_NAMES)
        return [], len(sk.SMPL28_NAMES)
    kind = skel.get("type")
    if kind not in sk.TYPES:
        return [f"skeleton.type must be one of {sorted(sk.TYPES)}, got {kind!r}"], None
    names, parents, _ = sk.TYPES[kind]
    errs = []
    if skel.get("joint_names") != names:
        errs.append(f"skeleton.joint_names must equal the {kind} name list")
    if skel.get("parents") != parents:
        errs.append(f"skeleton.parents must equal the {kind} parent list")
    if skel.get("aux_names") != sk.AUX_NAMES:
        errs.append("skeleton.aux_names must equal the AUX-14 list")
    for key in ("family", "label"):
        if not isinstance(skel.get(key), str) or not skel.get(key):
            errs.append(f"skeleton.{key} must be a non-empty string")
    return errs, len(names)


def _check_frame(pf, where, nj, legacy, errs):
    if not isinstance(pf, dict):
        errs.append(f"{where}: must be an object")
        return
    if not legacy and "joints_flat" not in pf:
        errs.append(f"{where}: joints_flat missing (use null)")
    jf = pf.get("joints_flat")
    if jf is not None and not isinstance(jf, list):
        errs.append(f"{where}: joints_flat must be a list or null")
    elif jf is not None:
        if nj is not None and len(jf) != nj * STRIDE:
            errs.append(f"{where}: joints_flat must be {nj * STRIDE} floats or null, got {len(jf)}")
        elif not _finite_list(jf):
            errs.append(f"{where}: joints_flat has non-finite values")
    pos = pf.get("position")
    if pos is not None and len(pos) == 2 and not _finite_list(pos):
        errs.append(f"{where}: position has non-finite values")
    if not legacy:
        if not isinstance(pf.get("interpolated"), bool):
            errs.append(f"{where}: interpolated must be true/false")
        if "aux_flat" not in pf:
            errs.append(f"{where}: aux_flat missing (use null)")
        elif pf["aux_flat"] is not None:
            ax = pf["aux_flat"]
            if not isinstance(ax, list):
                errs.append(f"{where}: aux_flat must be a list or null")
            elif len(ax) != AUX_LEN:
                errs.append(f"{where}: aux_flat must be {AUX_LEN} floats or null, got {len(ax)}")
            elif not _finite_list(ax):
                errs.append(f"{where}: aux_flat has non-finite values")
    # position length / stroke / racket / grip / facing: the same rules as schema 5.0
    schema._check_player_frame({k: v for k, v in pf.items() if k != "joints_flat"}, where, errs)


def validate_analysis24(doc, allow_legacy=False):
    """Return a list of human-readable schema errors; empty list means valid."""
    errs = []
    ver = doc.get("schema_version")
    legacy = allow_legacy and ver in LEGACY_VERSIONS
    if ver != SCHEMA_VERSION and not legacy:
        errs.append(f"schema_version must be {SCHEMA_VERSION!r}, got {ver!r}")
    if not isinstance(doc.get("clip_id"), str) or not doc.get("clip_id"):
        errs.append("clip_id must be a non-empty string")
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        errs.append("meta must be an object")
        meta = {}
    fps, n = meta.get("fps"), meta.get("frame_count")
    if not (_num(fps) and math.isfinite(fps) and fps > 0):
        errs.append("meta.fps missing or not a positive number")
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
        errs.append("meta.frame_count missing, non-integer or negative")
        n = None
    skel = doc.get("skeleton") or {}
    if not isinstance(skel, dict):
        errs.append("skeleton must be an object")
        skel = {}
    skel_errs, nj = _skeleton_errors(skel, legacy)
    errs += skel_errs
    court = doc.get("court") or {}
    for k in schema._COURT_KEYS:
        if not _num(court.get(k)):
            errs.append(f"court.{k} missing or non-numeric")
    players = doc.get("players")
    if not isinstance(players, list) or not players:
        errs.append("players must be a non-empty list")
        players = []
    for i, p in enumerate(players):
        if not isinstance(p, dict):
            errs.append(f"player {i}: must be an object")
            continue
        pid = p.get("id", "?")
        if p.get("side") not in ("near", "far"):
            errs.append(f"player {pid}: side must be 'near' or 'far'")
        frames = p.get("frames")
        if not isinstance(frames, list) or not frames:
            errs.append(f"player {pid}: frames must be a non-empty list")
            continue
        if n is not None and len(frames) != n:
            errs.append(f"player {pid}: {len(frames)} frames but meta.frame_count is {n}")
        for i, pf in enumerate(frames):
            _check_frame(pf, f"player {pid} frame {i}", nj, legacy, errs)
    sh = doc.get("shuttle") or {}
    if not isinstance(sh.get("frames"), list):
        errs.append("shuttle.frames must be a list")
    return errs
