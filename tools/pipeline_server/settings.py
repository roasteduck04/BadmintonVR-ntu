"""Streaming-build settings shown on the server page: every field maps to one key of configs/live/defaults.yaml.

The page renders FIELDS by group; a value the user changes becomes a `--set key=value` override of the live run.
Defaults shown on the page come from the merged config (defaults <- clip <- machine preset <- settings preset), so
nothing here repeats a default value. Settings presets (configs/live/settings/<name>.yaml) are flat {key: value} maps;
`best_live` is the v0.8.1 result (docs/reports/v0.8.1-speed.md).

Lengths of time are shown in FRAMES of the stream (fields with "frames": True; output_fps frames = 1 s) while the config
keeps seconds: values_from() converts seconds -> frames for the page, overrides() / save_preset() convert back. A frames
value equal to the preset's own (rounded) value passes the preset's exact seconds through, so 0.25 s at 30 fps stays
0.25 s instead of becoming 8 frames = 0.267 s.
"""
from __future__ import annotations

import os
import re

import yaml

from analysis.live import config as live_config

SETTINGS_DIR = os.path.join(live_config.CONFIG_DIR, "settings")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
BUILTIN = ("best_live", "pure_live", "laptop_realtime")     # shipped presets: Save as preset never overwrites them

# group, key, label, kind, extra. kinds: choice (options), int, float, bool, optfloat (a number or off = null),
# range (two floats). "live_only": only sent when the 2D comes from the video (backend live). "frames": the config value
# is in seconds, the page shows / sends frames of the stream (see the module docstring).
GROUPS = [
    ("source", "2D source and output", "Where the 2D keypoints come from, the output frame rate and the clock."),
    ("bystanders", "Bystander removal", "Only people standing on the court are posed (v0.8.1: 531 → 13 extra persons)."),
    ("speed2d", "2D speed", "Live 2D from the video only. Fewer detector calls and smaller frames are faster."),
    ("sync", "Camera sync", "Lines the cameras up in time. File = offsets measured once for fixed cameras."),
    ("warmup", "Player pick (warm-up)", "The first frames pick P1 (near half) and P2 (far half) without seeds."),
    ("tracker", "Tracker", "Follows each player from step to step on the court floor."),
    ("lrswap", "Left / right swap fix", "Fixes 2D models that mix up a person's left and right side."),
    ("fit", "Body fit", "Fits the SMPL body to the 3D points in short windows (the main accuracy step)."),
    ("fallback", "Smoothing when the fit is late", "Frames whose fit missed its time budget use smoothed raw 3D points."),
    ("facing", "Facing", "Body, shoulder and head heading."),
]

FIELDS = [
    # 2D source and output
    {"group": "source", "key": "backend", "label": "2D keypoints from", "kind": "choice",
     "options": [["cached", "stored 2D files"], ["live", "the video (GPU)"]],
     "help": "Stored = the offline RTMPose files replayed; video = RTMPose runs frame by frame (needs the GPU)."},
    {"group": "source", "key": "output_fps", "label": "Output frame rate", "kind": "choice", "options": [[30, "30 fps"], [60, "60 fps"]],
     "help": "60 fps is a little more accurate (v0.8.1: 3.0 vs 3.3 cm)."},
    {"group": "source", "key": "pace", "label": "Clock", "kind": "choice",
     "options": [["realtime", "real time (live broadcast)"], ["asfast", "as fast as possible"]],
     "help": "Real time = frames arrive on the camera clock; late fits fall back to smoothing."},
    {"group": "source", "key": "section_s", "label": "Part of the video", "kind": "range", "unit": "frames", "frames": True,
     "help": "Start and end frame, counted at the output frame rate from the start of the reference camera's video "
             "(6.5 s = frame 390 at 60 fps). Empty end = to the end."},
    # bystanders
    {"group": "bystanders", "key": "rtmpose.court_filter_m", "label": "Court filter margin", "kind": "optfloat", "unit": "m",
     "live_only": True, "help": "Keep a person only if their feet are on the court plus this margin. Off = pose everyone."},
    # 2D speed
    {"group": "speed2d", "key": "rtmpose.engine", "label": "Pose engine", "kind": "choice", "live_only": True,
     "options": [["fast", "fast (batched, v0.8.1)"], ["rtmlib", "rtmlib (v0.8.0)"]],
     "help": "Fast = all people of a frame in one model call. The court filter and frame size need it."},
    {"group": "speed2d", "key": "rtmpose.det_every", "label": "Person detector every", "kind": "int", "unit": "frames",
     "live_only": True, "help": "In between, boxes come from the last poses. 5 is fast; 1 is the most careful."},
    {"group": "speed2d", "key": "rtmpose.decode_short_side", "label": "Frame size", "kind": "choice", "live_only": True,
     "options": [[None, "full 4K"], [1080, "1080 px"], [720, "720 px"]],
     "help": "Shrink while decoding. 1080 costs ~1 px of 2D accuracy; 720 ~2 px for almost no gain."},
    {"group": "speed2d", "key": "rtmpose.decoder", "label": "Video reader", "kind": "choice", "live_only": True,
     "options": [["pyav", "pyav"], ["cv2", "OpenCV"], ["ffmpeg", "ffmpeg"], ["nvcodec", "GPU (PyNvVideoCodec)"]],
     "help": "pyav with 1080 px was fastest on the RTX 3060 PC."},
    {"group": "speed2d", "key": "rtmpose.model", "label": "2D model", "kind": "choice", "live_only": True,
     "options": [["wholebody", "wholebody (133 points)"], ["body_feet", "body + feet (faster, ~+1 cm)"],
                 ["body", "body only (17 points)"]],
     "help": "Lighter models are faster but less accurate in 3D (v0.8.1: body+feet +1.1 to +1.4 cm)."},
    {"group": "speed2d", "key": "rtmpose.mode", "label": "Model size", "kind": "choice", "live_only": True,
     "options": [["performance", "performance"], ["balanced", "balanced"], ["lightweight", "lightweight"]],
     "help": "Wholebody lightweight loses ~12 % of the on-court players."},
    # sync
    {"group": "sync", "key": "sync.offsets", "label": "Camera offsets", "kind": "choice",
     "options": [["file", "sync file (fixed cameras)"], ["auto", "estimate in the warm-up"], ["zero", "none"]],
     "help": "Auto from 2 s is 1-2 frames off (+0.4 cm at 60 fps). Make the file once per camera set-up."},
    {"group": "sync", "key": "sync.estimate.max_offset_s", "label": "Auto search range", "kind": "int", "unit": "frames",
     "frames": True, "help": "Auto sync looks for offsets up to this many frames."},
    {"group": "sync", "key": "sync.mode", "label": "Between source frames", "kind": "choice",
     "options": [["interp", "interpolate 2D"], ["nearest", "nearest frame"]],
     "help": "Interpolate = the 2D of the two source frames around the exact time is blended."},
    {"group": "sync", "key": "sync.partner_max_px", "label": "Interpolation partner within", "kind": "float", "unit": "px",
     "help": "The other source frame's detection must be this close, else the single detection is used."},
    # warm-up
    {"group": "warmup", "key": "warmup.seconds", "label": "Warm-up length", "kind": "int", "unit": "frames", "frames": True,
     "help": "Buffered before the body model is fixed; these frames are output too, flagged warm-up."},
    {"group": "warmup", "key": "warmup.player_height_m", "label": "Plausible standing height", "kind": "range", "unit": "m",
     "help": "Rejects children, ghosts and half-detected people."},
    {"group": "warmup", "key": "warmup.min_support_views", "label": "Seen by at least", "kind": "int", "unit": "cameras",
     "help": "A person must be seen by this many cameras to be picked as a player."},
    {"group": "warmup", "key": "warmup.min_time_in_court_frac", "label": "On court for at least", "kind": "float",
     "unit": "of warm-up", "help": "Share of the warm-up steps the candidate stands inside the court."},
    # tracker
    {"group": "tracker", "key": "tracker.gate_m", "label": "Max jump per step", "kind": "float", "unit": "m",
     "help": "A candidate farther than this from the player's last floor position is not the same player."},
    {"group": "tracker", "key": "tracker.gate_growth_mps", "label": "Gate growth while lost", "kind": "float", "unit": "m/s",
     "help": "The allowed jump grows while a player is not seen."},
    {"group": "tracker", "key": "tracker.net_band_m", "label": "Allowed past the net", "kind": "float", "unit": "m",
     "help": "A player may lean or step this far into the other half."},
    {"group": "tracker", "key": "tracker.court_margin_m", "label": "Court margin", "kind": "float", "unit": "m",
     "help": "Candidates outside the court plus this margin are ignored."},
    # left / right swap
    {"group": "lrswap", "key": "stream.history_s", "label": "Look-back", "kind": "int", "unit": "frames", "frames": True,
     "help": "How many frames of history the swap fix uses (it runs with a fixed lag)."},
    # body fit
    {"group": "fit", "key": "stream.window_s", "label": "Fit window", "kind": "int", "unit": "frames", "frames": True,
     "help": "Frames fitted together. The delay is about this long."},
    {"group": "fit", "key": "stream.solver_tol", "label": "Solver tolerance", "kind": "optfloat",
     "help": "1e-4 (v0.8.1) is 4x cheaper than the offline 1e-8 with the same accuracy. Off = offline solver."},
    {"group": "fit", "key": "stream.lsmr_maxiter", "label": "Inner iteration cap", "kind": "optfloat",
     "help": "LSMR iterations per solver step (20 in v0.8.1). Off = no cap."},
    {"group": "fit", "key": "stream.fit_every", "label": "Fit every", "kind": "int", "unit": "windows",
     "help": "2 = every second window from smoothing only (cheaper, worse: +0.4 cm, +1 deg)."},
    {"group": "fit", "key": "stream.late_policy", "label": "A late fit", "kind": "choice",
     "options": [["drop", "is dropped (delay stays short)"], ["use", "is waited for (delay grows)"]],
     "help": "Real time: drop keeps the broadcast on time; use keeps the accuracy."},
    {"group": "fit", "key": "stream.workers", "label": "Fit processes", "kind": "int",
     "help": "0 = in the main process. 2 = one per player."},
    # fallback
    {"group": "fallback", "key": "fallback.alpha", "label": "Smoothing (EMA alpha)", "kind": "float",
     "help": "Higher follows faster, lower is smoother."},
    {"group": "fallback", "key": "fallback.max_hold_s", "label": "Hold a missing joint up to", "kind": "int",
     "unit": "frames", "frames": True, "help": "A joint missing longer than this is left out."},
    # facing
    {"group": "facing", "key": "facing.window_s", "label": "Facing window", "kind": "int", "unit": "frames", "frames": True,
     "help": "Past-only smoothing of the heading over this many frames."},
]
KEYS = {f["key"] for f in FIELDS}
BYKEY = {f["key"]: f for f in FIELDS}
FRAME_KEYS = {f["key"] for f in FIELDS if f.get("frames")}


def get_key(cfg: dict, key: str):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def _fps(d: dict) -> float:
    try:
        fps = float(d.get("output_fps"))
    except (TypeError, ValueError):
        raise ValueError("output_fps is needed to convert frames to seconds")
    if fps <= 0:
        raise ValueError("output_fps must be positive")
    return fps


def _sec_to_frames(v, fps):
    if v is None or v == "":
        return None
    return int(round(float(v) * fps))


def to_frames(key: str, v, fps: float):
    """A config value (seconds) of a frames field -> frames of the stream."""
    if BYKEY[key]["kind"] == "range":
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return v
        return [_sec_to_frames(v[0], fps) or 0, _sec_to_frames(v[1], fps)]
    return _sec_to_frames(v, fps)


def to_seconds(key: str, v, fps: float):
    """A page value (frames) of a frames field -> seconds for the config."""
    if BYKEY[key]["kind"] == "range":
        a, b = v
        return [round(a / fps, 6), None if b is None else round(b / fps, 6)]
    return None if v is None else round(v / fps, 6)


def values_from(cfg: dict) -> dict:
    """{key: value} of every field in a merged config, lengths of time in frames of its output_fps."""
    out = {f["key"]: get_key(cfg, f["key"]) for f in FIELDS}
    try:
        fps = _fps(cfg)
    except ValueError:
        return out
    for k in FRAME_KEYS:
        out[k] = to_frames(k, out[k], fps)
    return out


def coerce(field: dict, v):
    """A page value -> the config value (raises ValueError on a bad one)."""
    k = field["kind"]
    if k == "choice":
        allowed = [o[0] for o in field["options"]]
        if v not in allowed:
            raise ValueError(f"{field['key']}: {v!r} is not one of {allowed}")
        return v
    if k == "bool":
        return bool(v)
    if k == "optfloat":
        return None if v is None or v == "" else float(v)
    if k == "int":
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"{field['key']}: {v!r} is not a whole number")
        return int(v)
    if k == "float":
        return float(v)
    if k == "range":
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"{field['key']}: needs [start, end]")
        a, b = v
        num = int if field.get("frames") else float
        return [num(a or 0), None if b is None or b == "" else num(b)]
    raise ValueError(f"unknown field kind {k}")


def overrides(values: dict, backend: str, base_cfg: dict | None = None) -> dict:
    """Page values -> {key: value} config overrides; unknown keys refused, 2D-speed keys dropped for stored 2D, frames
    converted to seconds at the page's output_fps. base_cfg (the merged config the page started from): a frames value
    left at its value there keeps the config's exact seconds."""
    out = {}
    for k, v in (values or {}).items():
        if k not in BYKEY:
            raise ValueError(f"unknown setting {k!r}")
        f = BYKEY[k]
        if f.get("live_only") and backend != "live":
            continue
        out[k] = coerce(f, v)
    frame_keys = [k for k in out if k in FRAME_KEYS]
    if frame_keys:
        fps = _fps({"output_fps": out.get("output_fps", (base_cfg or {}).get("output_fps"))})
        base = values_from(base_cfg) if base_cfg else {}
        same_fps = base_cfg is not None and float(base_cfg.get("output_fps") or 0) == fps
        for k in frame_keys:
            if same_fps and out[k] == base.get(k):
                out[k] = get_key(base_cfg, k)
            else:
                out[k] = to_seconds(k, out[k], fps)
    return out


def as_set_args(over: dict) -> list:
    """{key: value} -> ['key=value', ...] for live_run.py --set (values as YAML flow)."""
    out = []
    for k, v in over.items():
        out.append(f"{k}={yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}")
    return out


# ---------------------------------------------------------------------------------------------- settings presets
def preset_names() -> list:
    if not os.path.isdir(SETTINGS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SETTINGS_DIR) if f.endswith(".yaml"))


def load_preset(name: str) -> dict:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"bad preset name {name!r}")
    p = os.path.join(SETTINGS_DIR, f"{name}.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(name)
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    d.pop("_about", None)
    unknown = sorted(set(d) - KEYS)
    if unknown:
        raise ValueError(f"preset {name}: unknown keys {unknown}")
    return d


def save_preset(name: str, values: dict, about: str = "") -> str:
    if not NAME_RE.match(name or "") or name in BUILTIN:
        raise ValueError(f"bad preset name {name!r} (lowercase letters, digits, - and _; {', '.join(BUILTIN)} are built in)")
    clean = {k: coerce(BYKEY[k], v) for k, v in values.items() if k in BYKEY}
    if any(k in FRAME_KEYS for k in clean):               # the page sends frames; presets store seconds
        fps = _fps(clean)
        clean = {k: to_seconds(k, v, fps) if k in FRAME_KEYS else v for k, v in clean.items()}
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    p = os.path.join(SETTINGS_DIR, f"{name}.yaml")
    with open(p, "w", encoding="utf-8") as fh:
        if about:
            fh.write(f"_about: {yaml.safe_dump(about).strip().removesuffix('...').strip()}\n")
        yaml.safe_dump(clean, fh, sort_keys=False)
    return p
