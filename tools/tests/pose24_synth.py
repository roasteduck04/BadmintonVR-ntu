"""Synthetic multi-camera clip for the pose24 tests: a known 3D player -> 133-keypoint 2D per view.

Conventions match the pipeline: cameras and 2D live in the court world (Y-down, floor Y = 0); the
analysis world is Y-up. Template: Y-up body-local metres, facing +z, +x = the player's left.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from analysis import multiview as mv
from analysis import pose_v2 as pv
from analysis.pose24 import kpset

W, H = 2160, 3840
CENTRES = {"a": [0.0, -3.0, -11.0], "b": [-6.0, -2.5, -6.0], "c": [6.0, -2.5, -6.0], "d": [5.0, -2.5, 11.0]}
RAW_SCORE = 4.5                                   # SimCC-like raw score of a visible keypoint
UPPER = list(range(0, 11)) + list(range(23, 133))  # moves with the trunk: head, arms, face, hands
RIGHT_ARM = [8, 10] + list(range(112, 133))        # right elbow, wrist, hand (swing about the shoulder)


def look_at(center, target=(0.0, -1.0, 0.0)):
    """OpenCV camera (x right, y down, z forward) at `center` looking at `target` (court world, Y-down)."""
    c = np.asarray(center, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    R = np.vstack([right, np.cross(fwd, right), fwd])
    return R, -R @ c


def rig(views=("a", "b", "c", "d"), fov=70.0):
    K = mv.intrinsics_from_fov(W, H, fov)
    return {v: (K,) + look_at(CENTRES[v]) for v in views}


def _hand(wrist, elbow):
    """21 hand keypoints: wrist, thumb 1-4, then index/middle/ring/pinky MCP..tip (4 each)."""
    d = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    e = np.array([0.0, 0.0, 1.0]) - d * d[2]
    e /= np.linalg.norm(e)
    pts = [wrist] + [wrist + 0.02 * (i + 1) * d + 0.03 * e for i in range(4)]
    for f in range(4):
        base = wrist + 0.085 * d + (0.03 - 0.02 * f) * e
        pts += [base + 0.025 * s * d for s in range(4)]
    return np.array(pts)


def _face68():
    rng = np.random.default_rng(5)
    pts = np.c_[rng.uniform(-0.05, 0.05, 68), rng.uniform(1.56, 1.68, 68), np.full(68, 0.085)]
    for i in range(17):                                       # jaw line, chin = 8
        a = np.pi * i / 16
        pts[i] = [-0.07 * np.cos(a), 1.61 - 0.09 * np.sin(a), 0.03 + 0.05 * np.sin(a)]
    pts[30] = [0.0, 1.61, 0.115]                              # nose tip
    for base, cx in ((36, -0.035), (42, 0.035)):             # right eye ring, left eye ring
        for j in range(6):
            a = 2 * np.pi * j / 6
            pts[base + j] = [cx + 0.012 * np.cos(a), 1.66 + 0.006 * np.sin(a), 0.085]
    return pts


def template133():
    T = np.zeros((133, 3))
    T[:23] = kpset.REST35[:23]
    T[23:91] = _face68()
    T[91:112] = _hand(T[9], T[7])
    T[112:133] = _hand(T[10], T[8])
    return T


_T = template133()
LIMB_TRUE = {"upper_arm": float(np.linalg.norm(_T[7] - _T[5])), "forearm": float(np.linalg.norm(_T[9] - _T[7])),
             "thigh": float(np.linalg.norm(_T[13] - _T[11])), "shank": float(np.linalg.norm(_T[15] - _T[13]))}


def motion(F=150, fps=60.0, x0=0.0, z0=-3.0, side_bend=0.0, arm=True, walk=True, jump=None, plant=None,
           mirror=True):
    """(F,133,3) Y-up world keypoints. side_bend: trunk lateral lean amplitude (deg) about the hip centre;
    arm: right arm swings forward/up; walk: moves around (x0, z0); jump: (f0, f1) lifts the body up to
    0.5 m; plant: (f0, f1) stands still (feet fixed) while the upper body keeps moving. The player faces
    the net (+z when z0 < 0).
    mirror (default): negate the template's x so the player has real-data handedness. pose_v2 only negates
    Y between the (right-handed) court world and the analysis world, so the analysis world is mirrored vs
    the real scene: real players give kpset.forward_sign = -1 there (+1 in the Y-down court world).
    mirror=False keeps the right-handed SMPL-convention template (forward_sign = +1 in the analysis world).
    Only geometry mirrors: keypoint labels (left_*/right_*, RIGHT_ARM, UPPER) stay."""
    T = template133()
    if mirror:
        T = T * np.array([-1.0, 1.0, 1.0])
    t = np.arange(F) / fps
    hip = 0.5 * (T[11] + T[12])
    moving = np.array([walk and not (plant and plant[0] <= f < plant[1]) for f in range(F)], float)
    tt = np.cumsum(moving) / fps
    px, pz = x0 + 0.8 * np.sin(2 * np.pi * 0.5 * tt), z0 + 0.4 * np.cos(2 * np.pi * 0.35 * tt) - 0.4
    heading = Rotation.from_rotvec(np.radians(0.0 if z0 < 0 else 180.0) * np.array([0, 1.0, 0])).as_matrix()
    out = np.empty((F, 133, 3))
    for f in range(F):
        P = T.copy()
        if arm:
            a = 60.0 + 50.0 * np.sin(2 * np.pi * 1.3 * t[f])
            R = Rotation.from_rotvec(np.radians(-a) * np.array([1.0, 0, 0])).as_matrix()
            P[RIGHT_ARM] = T[6] + (P[RIGHT_ARM] - T[6]) @ R.T
        if side_bend:
            b = side_bend * np.sin(2 * np.pi * 0.4 * t[f])
            R = Rotation.from_rotvec(np.radians(b) * np.array([0, 0, 1.0])).as_matrix()
            P[UPPER] = hip + (P[UPPER] - hip) @ R.T
        lift = 0.0
        if jump and jump[0] <= f < jump[1]:
            lift = 0.5 * np.sin(np.pi * (f - jump[0]) / (jump[1] - jump[0]))
        out[f] = P @ heading.T + np.array([px[f], lift, pz[f]])
    return out


def project(Xw_up, cams, views, noise_px=0.0, rng=None, score=RAW_SCORE):
    """(F,133,3) Y-up world -> {view: (F,133,3) pixels + raw score}; points behind a camera score 0."""
    rng = rng if rng is not None else np.random.default_rng(0)
    Xd = np.asarray(Xw_up) * np.array([1.0, -1.0, 1.0])
    out = {}
    for v in views:
        uv, depth = pv.project(pv.proj_mats(cams, [v]), Xd)
        uv = uv[..., 0, :]
        if noise_px:
            uv = uv + rng.normal(0.0, noise_px, uv.shape)
        s = np.where(depth[..., 0] > 0, score, 0.0)
        out[v] = np.concatenate([uv, s[..., None]], axis=-1)
    return out


def bundle(players, cams, views, fps=60.0, noise_px=0.0, seed=0, mode="singles"):
    """A pose_v2-style observation bundle straight from known 3D (no tracking, no sync)."""
    rng = np.random.default_rng(seed)
    obs = {}
    for pid, X in players.items():
        o = project(X, cams, views, noise_px, rng)
        obs[pid] = {v: np.concatenate([o[v][..., :2], pv.norm_score(o[v][..., 2:3])], -1).astype(np.float32)
                    for v in views}
    F = len(next(iter(players.values())))
    return {"views": list(views), "cams": {v: cams[v] for v in views}, "cams_all": {v: cams[v] for v in views},
            "obs": obs, "fps": fps, "frame_count": F, "offsets": {v: 0.0 for v in views}, "excluded": {},
            "sides": {pid: ("near" if X[0, 11, 2] < 0 else "far") for pid, X in players.items()}, "mode": mode}


def write_clip(root, name, players, cams, views, fps=60.0, shifts=None, noise_px=0.5, bystander=True,
               mode="singles", seed=0):
    """Write data/calib + data/pose2d files for a synthetic clip under `root`; returns its registry entry.
    shifts[v] = k: stream v runs k frames late (raw frame f + k shows reference frame f). The bystander
    (off court, x = 5.5 m) is always detection 0 of a frame."""
    shifts = shifts or {}
    calib, pose2d = os.path.join(root, "data", "calib"), os.path.join(root, "data", "pose2d")
    os.makedirs(calib, exist_ok=True)
    os.makedirs(pose2d, exist_ok=True)
    rng = np.random.default_rng(seed)
    F = len(next(iter(players.values())))
    proj = {pid: project(X, cams, views, noise_px, rng) for pid, X in players.items()}
    by = project(motion(F, x0=5.5, z0=-1.0, walk=False, arm=False), cams, views, noise_px, rng) if bystander else None
    for v in views:
        K, R, t = cams[v]
        with open(os.path.join(calib, f"{v}_cam.json"), "w", encoding="utf-8") as fh:
            json.dump({"K": K.tolist(), "R": R.tolist(), "t": t.tolist()}, fh)
        k = int(shifts.get(v, 0))
        frames = []
        for r in range(F):
            f, dets = r - k, []
            if 0 <= f < F:
                if by is not None:
                    dets.append({"keypoints": by[v][f].tolist()})
                dets += [{"keypoints": proj[pid][v][f].tolist()} for pid in players]
            frames.append(dets)
        with open(os.path.join(pose2d, f"{v}_kp2d.json"), "w", encoding="utf-8") as fh:
            json.dump({"view": v, "clip": name, "fps": fps, "width": W, "height": H, "frame_count": F,
                       "keypoint_format": "coco_wholebody_133", "multi_person": True, "frames": frames}, fh)
    seeds = {"clip": name, "mode": mode, "views": {}}
    for v in views:
        seeds["views"][v] = {"seeds": {pid: [float(x) for x in proj[pid][v][0, [15, 16], :2].mean(axis=0)]
                                       for pid in players}}
    with open(os.path.join(calib, f"{name}_seeds.json"), "w", encoding="utf-8") as fh:
        json.dump(seeds, fh)
    return {"mode": mode, "seeds": name, "time_ref": sorted(views)[0], "views": list(views),
            "videos": {v: None for v in views}}
