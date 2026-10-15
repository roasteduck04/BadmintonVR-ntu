"""SMPL-X (SMPLer-X) poses -> court-placed, per-player analysis doc `<clip>_smplx.analysis.json`.

    PYTHONPATH=tools python -m analysis.smplx_fuse raw --clip singles_05x [--raw-dir data/smplx_raw/singles_05x]
    PYTHONPATH=tools python -m analysis.smplx_fuse legacy --clip singles_05x --input <old _smplx doc>

SMPLer-X regresses root-relative joints per camera, so it cannot place a player on court or
tell players apart by itself. Both routes borrow that from pose v2 (run `analysis.pose_v2`
first): its tracker/sync cache gives each player's synced 2D keypoints per view, and its doc
gives the triangulated pelvis (court X/Z and height).

  raw     from `smplx_gpu.py` raw dumps: each SMPLer-X detection is assigned to the tracked
          player whose 2D keypoint box it overlaps (on the synced timeline), rotated into the
          court frame (R^T, then the Y flip every other producer applies), views fused with
          score x size weights and >15 cm per-view outliers dropped;
  legacy  salvage of an old in-script-fused doc (one mixed "P1", upside down, no position):
          Y flip, then each frame's pose is given to the pose v2 player it matches (body
          MPJPE < 20 cm and clearly better than the other players).

Then: placed at the pose v2 pelvis, outlier rejection + gap fill + zero-phase smoothing +
strict bone lengths (pose_v2.postprocess), racket/shuttle carried over from the base clip.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

from analysis import pose_v2 as pv
from analysis import schema
from analysis.conventions import analysis_sign
from analysis.facing import fill_facing, frames_sign

SUFFIX = "_smplx"
BODY = list(range(22))
OUTLIER_M = 0.15
MATCH_M = 0.20
ARCHIVE_DIR = os.path.join("_archive", "smplx_legacy")


def v2_pelvis(v2_player):
    """(F,3) Y-up world pelvis from a pose v2 player (NaN where no pose)."""
    F = len(v2_player["frames"])
    out = np.full((F, 3), np.nan)
    for f, fr in enumerate(v2_player["frames"]):
        jf, pos = fr.get("joints_flat"), fr.get("position")
        if jf is not None and pos is not None:
            out[f] = [pos[0], jf[1], pos[1]]
    return out


def v2_rel(v2_player):
    """(F,58,3) pelvis-relative Y-up joints and (F,58) confidences."""
    F = len(v2_player["frames"])
    J = np.full((F, schema.NUM_JOINTS, 3), np.nan)
    C = np.zeros((F, schema.NUM_JOINTS))
    for f, fr in enumerate(v2_player["frames"]):
        jf = fr.get("joints_flat")
        if jf is not None:
            a = np.asarray(jf, float).reshape(-1, 4)
            J[f] = a[:, :3] - a[0, :3]
            C[f] = a[:, 3]
    return J, C


def cam_to_court_rel(j58_cam, R):
    """Root-relative camera-space joints -> root-relative court joints, Y-up."""
    w = np.asarray(j58_cam, float) @ R
    w[..., 1] *= -1.0
    return w - w[..., :1, :]


def _kp_box(kps, w_min=0.3):
    m = np.isfinite(kps[:23, 0]) & (kps[:23, 2] >= w_min)
    if m.sum() < 5:
        return None
    lo, hi = kps[:23][m, :2].min(0), kps[:23][m, :2].max(0)
    pad = 0.1 * (hi - lo)
    return np.r_[lo - pad, hi + pad]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fuse_views(cands):
    """cands: list of (rel58 (58,3), conf58, weight) -> fused rel58, conf58."""
    P = np.stack([c[0] for c in cands])
    W = np.array([c[2] for c in cands], float)
    Cc = np.stack([c[1] for c in cands])
    keep = np.ones(len(cands), bool)
    for _ in range(2):
        mean = np.average(P[keep], axis=0, weights=W[keep])
        err = np.linalg.norm(P[:, BODY] - mean[BODY], axis=-1).mean(axis=1)
        new = err <= OUTLIER_M if keep.sum() >= 3 else keep
        if new.sum() == 0 or (new == keep).all():
            break
        keep = new
    mean = np.average(P[keep], axis=0, weights=W[keep])
    conf = np.clip(np.average(Cc[keep], axis=0, weights=W[keep]) * min(1.0, W[keep].sum()), 0, 1)
    return mean, conf


def rel_from_raw(bundle, raw_dir, pid, log=print):
    """(F,58,3) fused root-relative Y-up joints + (F,58) conf for one player from raw dumps."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from smplx_gpu import map_smplerx_to_ours
    F = bundle["frame_count"]
    per_frame = [[] for _ in range(F)]
    for v in bundle["views"]:
        path = os.path.join(raw_dir, f"{v}.npz")
        if not os.path.exists(path):
            log(f"    {v}: no raw dump")
            continue
        z = np.load(path)
        R = bundle["cams"][v][1]
        by_frame = {}
        for i, g in enumerate(z["frame"]):
            by_frame.setdefault(int(g), []).append(i)
        obs = bundle["obs"][pid][v]
        off = bundle["offsets"].get(v, 0.0)
        n_used = 0
        for f in range(F):
            box = _kp_box(obs[f])
            if box is None:
                continue
            g = int(round(f + off))
            best, bi = 0.3, None
            for i in by_frame.get(g, []):
                o = _iou(box, z["bbox"][i])
                if o > best:
                    best, bi = o, i
            if bi is None:
                continue
            j58, c58 = map_smplerx_to_ours(z["joints137"][bi].astype(float), np.full(137, float(z["score"][bi])))
            h = float(z["bbox"][bi][3] - z["bbox"][bi][1])
            per_frame[f].append((cam_to_court_rel(j58, R), c58, float(z["score"][bi]) * min(1.0, h / 800.0)))
            n_used += 1
        log(f"    {pid} {v}: {n_used} frames matched")
    J = np.full((F, schema.NUM_JOINTS, 3), np.nan)
    C = np.zeros((F, schema.NUM_JOINTS))
    for f, c in enumerate(per_frame):
        if c:
            J[f], C[f] = fuse_views(c)
    return J, C


def rel_from_legacy(doc, v2_players, log=print):
    """Split an old single-track SMPL-X doc into per-player root-relative poses by matching
    each frame against the pose v2 players."""
    F = min(len(doc["players"][0]["frames"]), len(next(iter(v2_players.values()))["frames"]))
    refs = {pid: v2_rel(p) for pid, p in v2_players.items()}
    out = {pid: (np.full((F, schema.NUM_JOINTS, 3), np.nan), np.zeros((F, schema.NUM_JOINTS)))
           for pid in v2_players}
    counts = {pid: 0 for pid in v2_players}
    for trk in doc["players"]:
        for f, fr in enumerate(trk["frames"][:F]):
            jf = fr.get("joints_flat")
            if jf is None:
                continue
            a = np.asarray(jf, float).reshape(-1, 4)
            rel = a[:, :3] - a[0, :3]
            rel[:, 1] *= -1.0
            errs = {}
            for pid, (J, C) in refs.items():
                if not np.isfinite(J[f, 0]).all():
                    continue
                m = [j for j in BODY if C[f, j] >= 0.3]
                if len(m) >= 8:
                    errs[pid] = float(np.linalg.norm(rel[m] - J[f, m], axis=1).mean())
            if not errs:
                continue
            ranked = sorted(errs.items(), key=lambda kv: kv[1])
            pid, e = ranked[0]
            if e > MATCH_M or (len(ranked) > 1 and e > 0.7 * ranked[1][1]):
                continue
            out[pid][0][f] = rel
            out[pid][1][f] = np.clip(a[:, 3], 0, 1)
            counts[pid] += 1
    log("    legacy frames matched: " + ", ".join(f"{p} {c}" for p, c in counts.items()))
    return out


def archive_previous(path, archive_dir=ARCHIVE_DIR, log=print):
    """Move an existing output to archive_dir as <name>.<mtime YYYYmmdd-HHMMSS>[-k].analysis.json, never over an
    earlier archived copy (review 2 L16: archive, don't delete). Returns the archived path, or None."""
    import time
    if not os.path.exists(path):
        return None
    os.makedirs(archive_dir, exist_ok=True)
    base = os.path.basename(path)
    stem = base[:-len(".analysis.json")] if base.endswith(".analysis.json") else os.path.splitext(base)[0]
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(path)))
    dst, k = os.path.join(archive_dir, f"{stem}.{stamp}.analysis.json"), 1
    while os.path.exists(dst):
        k += 1
        dst = os.path.join(archive_dir, f"{stem}.{stamp}-{k}.analysis.json")
    shutil.move(path, dst)
    log(f"    archived previous {base} -> {dst}")
    return dst


def finish(out_clip, bundle, v2, rel_by_pid, producer, meta_extra, analysis_dir, log=print):
    fps, n = bundle["fps"], bundle["frame_count"]
    players = []
    v2p = {p["id"]: p for p in v2["players"]}
    for pid, (Jr, Cr) in rel_by_pid.items():
        pel = v2_pelvis(v2p[pid])[:len(Jr)]
        J = Jr + pel[:, None, :]
        J[~np.isfinite(pel[:, 0])] = np.nan
        C = np.where(np.isfinite(J[..., 0]), Cr, 0.0)
        J, C, interp = pv.postprocess(J, C, fps)
        frames = pv.frames_from_world(J, C, interp)
        frames += [{"joints_flat": None, "position": None, "interpolated": False}] * (n - len(frames))
        n_ok = sum(fr["joints_flat"] is not None for fr in frames)
        n_meas = sum(fr["joints_flat"] is not None and not fr["interpolated"] for fr in frames)
        log(f"    {pid}: pose {100*n_ok/n:.1f}% (measured {100*n_meas/n:.1f}%)")
        players.append((pid, frames))
    meta = dict(meta_extra, name="smplx", placement="pose_v2 triangulated pelvis",
                views=bundle["views"], offsets={v: round(o, 2) for v, o in bundle["offsets"].items()},
                time_ref_view=bundle.get("time_ref_view"))
    doc = pv.make_doc(out_clip, fps, n, players, bundle, producer, meta)
    base_path = os.path.join(analysis_dir, f"{out_clip}.analysis.json")
    with open(base_path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    pv.carry_over(base, doc, log=log)
    for p in doc["players"]:
        fill_facing(p["frames"], fps, sign=frames_sign(p["frames"], analysis_sign(bundle["cams"])))
    errs = schema.validate_analysis(doc)
    if errs:
        raise ValueError("smplx doc invalid:\n  " + "\n  ".join(errs[:10]))
    out = os.path.join(analysis_dir, f"{out_clip}{SUFFIX}.analysis.json")
    archive_previous(out, ARCHIVE_DIR, log=log)
    text = json.dumps(doc, allow_nan=False)      # Unity's JsonUtility can't parse NaN
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"    wrote {out}")
    pv.write_racket_sidecars(analysis_dir, out_clip, f"{out_clip}{SUFFIX}", base, doc, log=log)
    return out


def _load_inputs(out_clip, analysis_dir):
    bundle = pv.load_cache(out_clip)
    with open(os.path.join(analysis_dir, f"{out_clip}_{pv.VARIANT}.analysis.json"), "r", encoding="utf-8") as fh:
        v2 = json.load(fh)
    return bundle, v2


def build_raw(out_clip, raw_dir=None, analysis_dir=pv.DEFAULT_ANALYSIS_DIR, log=print):
    raw_dir = raw_dir or os.path.join("data", "smplx_raw", out_clip)
    bundle, v2 = _load_inputs(out_clip, analysis_dir)
    log(f"[smplx_fuse raw] {out_clip}: {raw_dir}")
    rel = {}
    for p in v2["players"]:
        rel[p["id"]] = rel_from_raw(bundle, raw_dir, p["id"], log=log)
    return finish(out_clip, bundle, v2, rel, "smplerx_fused", {"source": "raw", "raw_dir": raw_dir},
                  analysis_dir, log=log)


def build_legacy(out_clip, input_path, analysis_dir=pv.DEFAULT_ANALYSIS_DIR, log=print):
    bundle, v2 = _load_inputs(out_clip, analysis_dir)
    with open(input_path, "r", encoding="utf-8") as fh:
        old = json.load(fh)
    log(f"[smplx_fuse legacy] {out_clip}: {input_path}")
    rel = rel_from_legacy(old, {p["id"]: p for p in v2["players"]}, log=log)
    return finish(out_clip, bundle, v2, rel, "smplerx_legacy_salvage",
                  {"source": "legacy", "input": os.path.basename(input_path),
                   "note": "interim: old best-view fusion (mostly the back camera), identity by pose "
                           "match; re-run smplx_gpu.py raw dump for the real multi-view result"},
                  analysis_dir, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw")
    r.add_argument("--clip", required=True)
    r.add_argument("--raw-dir", default=None)
    r.add_argument("--analysis-dir", default=pv.DEFAULT_ANALYSIS_DIR)
    lg = sub.add_parser("legacy")
    lg.add_argument("--clip", required=True)
    lg.add_argument("--input", required=True)
    lg.add_argument("--analysis-dir", default=pv.DEFAULT_ANALYSIS_DIR)
    args = ap.parse_args(argv)
    if args.cmd == "raw":
        build_raw(args.clip, args.raw_dir, args.analysis_dir)
    else:
        build_legacy(args.clip, args.input, args.analysis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
