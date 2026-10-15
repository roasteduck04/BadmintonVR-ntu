"""v0.7.5 monoprior: MotionBERT (DSTformer, H36M fine-tune) on CPU -> a learned 1-camera depth prior per view.

  load_model(kind)      "lite" | "full" DSTformer from models/motionbert (imported at run time, never copied)
  infer(kp, wh, model)  (F,17,3) H36M pixels + scores -> (F,17,3) root-relative 3D + (F,) frame confidence
  player_prior          one player's COCO-WholeBody-133 2D of one view -> {"kp", "rel_cam", "conf"} (or None)
  produce(clip, view)   data/smpl_mono/<clip>/<view>_motionbert-<kind>.npz (contract: analysis.pose24.mono)

Input: the pose24 debug cache's per-view 2D (already on the cache timeline, frame_count frames), mapped to H36M-17
(priors.common.coco133_to_h36m) and normalised (score as channel 3) by `norm`:
  "crop"   (default) infer_wild.py's default path, crop_scale(motion, [1, 1]): one square bbox of the score != 0
           points over the player's whole track in the view -> [-1, 1], clipped (crop_normalise)
  "image"  infer_wild.py's --pixel path: xy -> (xy - wh/2) / (min(w, h)/2) (normalise); files get a "-image" suffix
 Frames where more than half of the 17 joints are missing are
filled by linear interpolation for the network input only and get conf 0; single missing joints are interpolated in
time with score 0. Windows of CLIP_LEN (243) frames: consecutive, the last one right-aligned, overlaps averaged;
each window is flip-averaged (lib.utils.utils_data.flip_data) as in infer_wild.py.

Output axes (assumption, checked on real data in v0.7.5 Task 3): MotionBERT's H36M targets (`joint3d_image`,
lib/data/datareader_h36m.py) are image-space 3D: x right and y down in normalised image units, z the camera-frame
depth (H36M camera coordinates, z forward) relative to the root, in the same units. So the network's output is taken
as-is as OpenCV camera axes, z increasing away from the camera; the units are image-normalised (not metres), which
mono._depth_rows absorbs with its leg-length rescale. The H36M preprocessing that builds `joint3d_image` is not in
the repo, so the z sign is not provable from the code here.

Root: H36M joint 0 (hip midpoint) is subtracted, the contract's root."""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from analysis.pose24.priors import common

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
MB_DIR = os.path.join(REPO_ROOT, "models", "motionbert")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "smpl_mono")
MODELS = {"lite": ("configs/pose3d/MB_ft_h36m_global_lite.yaml",
                   "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"),
          "full": ("configs/pose3d/MB_ft_h36m.yaml",
                   "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin")}
CLIP_LEN = 243
GAP_FRAC = 0.5                  # a frame with more than this fraction of the 17 joints missing: conf 0, interpolated


NORMS = ("crop", "image")
DEFAULT_NORM = "crop"


def method_label(kind, norm=DEFAULT_NORM):
    """Prior method label (meta + filename): motionbert-<kind>, + "-image" for image normalisation."""
    if norm not in NORMS:
        raise ValueError(f"MotionBERT norm {norm!r} not one of {NORMS}")
    return f"motionbert-{kind}" + ("" if norm == DEFAULT_NORM else f"-{norm}")


def model_paths(kind):
    """(config path, checkpoint path) of a model kind under models/motionbert."""
    if kind not in MODELS:
        raise ValueError(f"MotionBERT model {kind!r} not one of {sorted(MODELS)}")
    cfg, ckpt = MODELS[kind]
    return os.path.join(MB_DIR, *cfg.split("/")), os.path.join(MB_DIR, *ckpt.split("/"))


def _mb_lib():
    """Put models/motionbert on sys.path (its `lib` package) and return the pieces used here."""
    if not os.path.isdir(os.path.join(MB_DIR, "lib")):
        raise FileNotFoundError(f"MotionBERT code not found at {MB_DIR}")
    if MB_DIR not in sys.path:
        sys.path.insert(0, MB_DIR)
    mod = sys.modules.get("lib")                      # e.g. pywin32's namespace package `win32/lib` cached first
    mb_lib = os.path.normcase(os.path.join(MB_DIR, "lib"))
    if mod is not None and mb_lib not in [os.path.normcase(p) for p in (getattr(mod, "__path__", None) or [])]:
        for name in [n for n in sys.modules if n == "lib" or n.startswith("lib.")]:
            del sys.modules[name]
    from lib.utils.learning import load_backbone
    from lib.utils.utils_data import flip_data
    return load_backbone, flip_data


def flip_fn():
    """MotionBERT's flip_data (x -> -x, left <-> right joints), for (N,T,17,C) tensors or arrays."""
    return _mb_lib()[1]


class _Cfg(dict):
    """easydict stand-in: attribute access, AttributeError when missing (load_backbone uses hasattr)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _config(path):
    """MotionBERT's get_config (needs easydict); a plain-YAML equivalent when easydict is not installed."""
    try:
        from lib.utils.tools import get_config
    except ImportError:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _Cfg(yaml.safe_load(fh))
        cfg.name = os.path.splitext(os.path.basename(path))[0]
        return cfg
    return get_config(path)


def load_model(kind="lite"):
    """The DSTformer of `kind` with its fine-tuned H36M weights (key model_pos, `module.` stripped, strict), CPU, eval."""
    import torch
    cfg_path, ckpt = model_paths(kind)
    for p in (cfg_path, ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(f"MotionBERT {kind}: missing {p}")
    load_backbone, _ = _mb_lib()
    model = load_backbone(_config(cfg_path))
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_pos"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to("cpu").eval()


def windows(F, clip_len=CLIP_LEN):
    """[(start, end)] network windows: consecutive clip_len windows, the last right-aligned to F; one (0, F) if
    F <= clip_len."""
    if F <= clip_len:
        return [(0, F)]
    W = [(s, s + clip_len) for s in range(0, F - clip_len + 1, clip_len)]
    if W[-1][1] < F:
        W.append((F - clip_len, F))
    return W


def run_windows(x, model, flip=True, clip_len=CLIP_LEN):
    """(F,17,3) normalised network input -> (F,17,3) raw network output (no_grad, CPU), flip-averaged when `flip`,
    overlapping frames averaged."""
    import torch
    x = np.asarray(x, np.float32)
    F = len(x)
    out = np.zeros((F, 17, 3))
    n = np.zeros(F)
    fl = flip_fn() if flip else None
    with torch.no_grad():
        for s, e in windows(F, clip_len):
            inp = torch.from_numpy(np.ascontiguousarray(x[s:e]))[None]
            p = model(inp)
            if flip:
                p = (p + fl(model(fl(inp)))) / 2.0
            out[s:e] += p[0].double().cpu().numpy()
            n[s:e] += 1
    return out / n[:, None, None]


def normalise(kp, image_size):
    """(F,17,3) pixels + scores -> infer_wild's image normalisation: xy -> (xy - wh/2) / (min(w,h)/2), score kept."""
    w, h = image_size
    y = np.array(kp, float)
    y[..., :2] = (y[..., :2] - np.array([w, h], float) / 2.0) / (min(w, h) / 2.0)
    return y


def crop_normalise(kp, valid=None):
    """(F,17,3) pixels + scores -> infer_wild's default normalisation (lib.utils.utils_data.crop_scale, scale_range
    [1, 1], so no randomness): the square bbox (side = the larger extent) of the `valid` points (default: score != 0)
    over all frames maps to [-1, 1]; the whole array is then clipped to [-1, 1] (scores included, as crop_scale does).
    All zeros when fewer than 4 valid points or a zero-size bbox (as crop_scale)."""
    kp = np.asarray(kp, float)
    if valid is None:
        valid = kp[..., 2] != 0
    pts = kp[..., :2][valid]
    pts = pts[np.isfinite(pts).all(-1)]
    if len(pts) < 4:
        return np.zeros(kp.shape)
    (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(kp.shape)
    y = kp.copy()
    y[..., :2] = ((kp[..., :2] - [(xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2]) / scale - 0.5) * 2
    return np.clip(y, -1, 1)


def _interp_cols(v, ok):
    """Linear interpolation (edges held) of the columns of v (F,C) over rows where ok; None when no row is ok."""
    if not ok.any():
        return None
    t = np.arange(len(v))
    return np.stack([np.interp(t, t[ok], v[ok, c]) for c in range(v.shape[1])], axis=1)


def fill_gaps(kp):
    """(F,17,3) H36M pixels + scores (missing: score 0 / NaN) -> (filled (F,17,3), conf (F,)).

    conf = mean of the 17 scores (missing = 0), 0 on frames with more than GAP_FRAC of the joints missing. Those
    frames are rebuilt by per-joint linear interpolation (xy and score) from the other frames; a missing joint on a
    kept frame gets an interpolated xy and score 0; a joint never seen: the frame's mean xy, score 0."""
    kp = np.asarray(kp, float)
    xy, s = kp[..., :2], np.nan_to_num(kp[..., 2])
    seen = np.isfinite(xy).all(-1) & (s > 0)
    bad = (~seen).sum(1) > GAP_FRAC * kp.shape[1]
    conf = np.where(seen, s, 0.0).mean(1)
    conf[bad] = 0.0
    out = np.full_like(kp, np.nan)
    out[..., 2] = 0.0
    for j in range(kp.shape[1]):
        ok = seen[:, j] & ~bad
        v = _interp_cols(np.c_[xy[:, j], s[:, j]], ok)
        if v is None:
            continue
        out[:, j, :2] = v[:, :2]
        out[:, j, 2] = np.where(ok, s[:, j], np.where(bad, v[:, 2], 0.0))
    fill = np.isnan(out[..., :2]).any(-1)
    if fill.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN frames: mean NaN -> 0 below
            mean = np.nanmean(out[..., :2], axis=1)
        out[..., :2] = np.where(fill[..., None], mean[:, None, :], out[..., :2])
        out[..., :2] = np.nan_to_num(out[..., :2])
    return out, conf


def infer(kp2d_h36m, image_size, model, flip=True, norm=DEFAULT_NORM):
    """(F,17,3) H36M pixels + scores of one player in one view (its whole track) -> (rel (F,17,3) root-relative
    network 3D in normalised units, OpenCV axes; conf (F,), see fill_gaps). norm "crop": the bbox of the observed
    (pre-fill) points over the whole track; "image": image_size."""
    method_label("x", norm)
    kp = np.asarray(kp2d_h36m, float)
    filled, conf = fill_gaps(kp)
    if norm == "crop":
        seen = np.isfinite(kp[..., :2]).all(-1) & (np.nan_to_num(kp[..., 2]) > 0)
        x = crop_normalise(filled, seen)
    else:
        x = normalise(filled, image_size)
    out = run_windows(x, model, flip=flip)
    return out - out[:, :1], conf


def player_prior(obs133, image_size, model, flip=True, norm=DEFAULT_NORM):
    """One player's (F,133,3) 2D of one view -> {"kp": (13,) kpset indices, "rel_cam": (F,13,3), "conf": (F,)},
    None when the player has no 2D in the view."""
    h36 = common.coco133_to_h36m(obs133)
    if not (h36[..., 2] > 0).any():
        return None
    rel, conf = infer(h36, image_size, model, flip=flip, norm=norm)
    hk = [h for h, _ in common.H36M_TO_KP]
    return {"kp": np.array([k for _, k in common.H36M_TO_KP], np.int64), "rel_cam": rel[:, hk], "conf": conf}


def image_size(cache, view):
    """(w, h) of `view`: the cache's image_size, else (2 cx, 2 cy) of the view's K (centred principal points)."""
    try:
        return common.view_image_size(cache, view)
    except KeyError:
        K = np.asarray(cache["cams"][view][0], float)
        return int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))


def output_path(clip, view, kind, out_dir=None, norm=DEFAULT_NORM):
    return os.path.join(out_dir or DEFAULT_OUT_DIR, clip, f"{view}_{method_label(kind, norm)}.npz")


def produce(clip, view, kind="lite", model=None, cache=None, out_dir=None, log=print, norm=DEFAULT_NORM):
    """Write the MotionBERT prior of `view` of `clip` -> its path. Every player with any 2D in the view; the prior has
    the cache's frame_count frames on the cache timeline. cache: the clip's pose24 debug cache (loaded when None);
    model: a loaded model (load_model(kind) when None)."""
    import torch
    method = method_label(kind, norm)
    torch.set_num_threads(os.cpu_count() or 1)
    if cache is None:
        from analysis.pose24 import doc as docmod, track
        cache = docmod.load_debug_cache(track.cache_path(clip))
    if view not in cache["views"]:
        raise ValueError(f"view {view!r} not in {clip}'s pose24 cache ({', '.join(cache['views'])})")
    model = model if model is not None else load_model(kind)
    F = int(cache["frame_count"])
    wh = image_size(cache, view)
    players = {}
    for pid in cache["obs"]:
        obs = cache["obs"][pid].get(view)
        if obs is None:
            continue
        if len(obs) != F:
            raise ValueError(f"{pid} {view}: {len(obs)} 2D frames, the cache {F}")
        t0 = time.perf_counter()
        p = player_prior(obs, wh, model, norm=norm)
        if p is None:
            log(f"    {pid}: no 2D in {view}: left out")
            continue
        players[pid] = p
        log(f"    {pid}: {int((p['conf'] > 0).sum())}/{F} frames with conf > 0 ({time.perf_counter() - t0:.1f} s)")
    meta = {"method": method, "clip": clip, "view": view,
            "timeline_of": cache.get("timeline_of") or clip, "fps": float(cache["fps"]), "frame_count": F,
            "build_id": cache.get("build_id"), "norm": norm}
    path = output_path(clip, view, kind, out_dir, norm=norm)
    common.write_prior(path, meta, players)
    return path
