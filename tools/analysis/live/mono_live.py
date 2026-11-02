"""v0.8.7 one camera (config mono.on = a single view): the live 3D without triangulation, from the offline 1-camera
fit (pose24.mono, SP5): feet planted on the court floor give each player's position, the limb lengths are fixed.

- body model (limb lengths, rest pose, handedness): one camera cannot measure the limbs, so per player, in order:
  the lengths of the live engine's last multi-camera warm-up (mono.body_models auto: <runs>/engine/warm_state.json,
  same singles / doubles mode), the player's real height x mono.height_scale (the rest template scaled to it), else
  mono.stature_m. Skeleton heights: the multi-camera builds measure this footage's players at 1.36-1.59 m (H6, a
  scale question not yet settled), so the template is 1.50 m and a real height is scaled by 0.88; a 1.70 m template
  bent the knees to fit the 2D (singles, back camera: knee 39 -> 17 deg, court position 19 -> 11 cm at 1.50 m).
  Handedness = the camera's own (kpset.world_sign of one view), as the offline mono fit.
- each step (fallback / warm-up frames): the rest template standing on the floor point of the ankles (or hips),
  turned to the hip line (mono.init_mono on one frame) - a rough stand-in until the window fit resolves.
- each window: contact from the 2D (planted heel / toe), init = mono.init_mono over the window, anchor = the planted
  heels / toes on their floor planes, then the usual WindowProblem24 with the one view (pose24 --mono's problem).
- confidence: sigma along the camera axis = mono.NO_PRIOR_DEPTH_SIGMA (0.25 m), as the offline 1-camera docs.
Camera-count study (v0.7.4): court position ~9-15 cm, joints 14-19 cm, knee 8-14 deg vs the full rig; no racket /
shuttle 3D from one camera."""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.live import stream_fit
from analysis.pose24 import fit, kpset, mono

DEPTH_SIGMA_M = mono.NO_PRIOR_DEPTH_SIGMA
DEFAULT_STATURE_M = 1.50        # skeleton height the multi-camera builds measure on this footage (configs: mono.stature_m)


def warm_state_path(cfg) -> str:
    return os.path.join(cfg["runs_dir"], "engine", "warm_state.json")


def load_saved(cfg, pids) -> tuple[dict, str | None]:
    """({pid: lengths} of the live engine's last multi-camera warm-up, its path) when mono.body_models is auto and
    that warm-up was of the same mode (singles / doubles; an older file without a mode must have the same players)."""
    if (cfg.get("mono") or {}).get("body_models", "auto") != "auto":
        return {}, None
    path = warm_state_path(cfg)
    try:
        with open(path, encoding="utf-8") as fh:
            ws = json.load(fh)
    except (OSError, ValueError):
        return {}, None
    models = ws.get("models") or {}
    same = ws.get("mode") == cfg.get("mode") if ws.get("mode") else set(models) == set(pids)
    if len(ws.get("views") or []) < 2 or not same:
        return {}, None
    return {pid: m["lengths"] for pid, m in models.items() if m and m.get("lengths") and pid in pids}, path


def body_model(cfg, cam, pid, saved=None):
    """(stream_fit.BodyModel, where its lengths came from) for one player of a 1-camera run."""
    mc = cfg.get("mono") or {}
    sgn = float(kpset.world_sign({"cam": cam}))
    if saved and saved.get(pid):
        L, src = saved[pid], "the last multi-camera warm-up"
    else:
        h = (mc.get("heights_m") or {}).get(pid)
        k = float(mc.get("height_scale", 1.0))
        stature = float(h) * k if h is not None else float(mc.get("stature_m", DEFAULT_STATURE_M))
        L = mono.lengths_for("prior", None, pid, stature_m=stature)
        src = (f"height {float(h):.2f} m x {k:g} = {stature:.2f} m" if h is not None
               else f"template {stature:.2f} m")
    D = kpset.rest_offsets_down(sgn)
    E = L["edges"]
    leg = np.linalg.norm(D[11] - D[13]) + np.linalg.norm(D[13] - D[15])
    rest = D * ((E["thigh"] + E["shank"]) / leg if E.get("thigh") and E.get("shank") else 1.0)
    return stream_fit.BodyModel(L, fit.edge_arrays(L), rest, sgn), src


def _seen_body(uv, w):
    s = mono._seen(uv, w)
    return s[:, [11, 12, 15, 16]].any(axis=1)


def step_points(uv, w, cam, model, ankle_h):
    """One step of one player: (35, 3) Y-down keypoints, NaN when neither hips nor ankles are seen. uv (1, 35, 2),
    w (1, 35) of the one view. The standing template at the floor point gives each keypoint's depth; a seen keypoint
    is then put on its pixel's ray at that depth (it reprojects exactly - the template alone sits tens of pixels off a
    near player's 2D and its frames came out empty)."""
    if not _seen_body(uv, w)[0]:
        return np.full((kpset.K, 3), np.nan)
    X = mono.init_mono(uv, w, cam, model.lengths, model.sgn, np.zeros((1, len(kpset.CONTACT)), bool), ankle_h=ankle_h)[0]
    K, R, t = (np.asarray(a, float) for a in cam)
    seen = mono._seen(uv, w)[0]
    z = (X[seen] @ R.T + t)[:, 2]                                   # camera depth of the template's keypoints
    ok = z > 0
    pix = np.c_[uv[0, seen], np.ones(int(seen.sum()))] @ np.linalg.inv(K).T     # normalised camera rays
    Xs = X[seen]
    Xs[ok] = (pix[ok] * z[ok, None] - t) @ R                     # R.T @ (p_cam - t), row-wise
    X[seen] = Xs
    return X


def window_inputs(uv, w, cam, model, ankle_h, fps):
    """A window of one player, uv (T, 35, 1, 2), w (T, 35, 1) -> (init (T, 35, 3), anchor (T, 35, 3), contact (T, 6),
    ok (T,) hips or ankles seen)."""
    u, ww = uv[:, :, 0], w[:, :, 0]
    contact = mono.contact_from_2d(u, ww, fps)
    init = mono.init_mono(u, ww, cam, model.lengths, model.sgn, contact, ankle_h=ankle_h)
    anchor = mono._foot_anchor(u, ww, cam, contact, ankle_h)
    return init, anchor, contact, _seen_body(u, ww)


def depth_axis(cam):
    a = np.asarray(cam[1], float)[2]
    return a / np.linalg.norm(a)
