"""v0.8.0 live-frames run config -> one resolved dict.

Layers, later wins: configs/live/defaults.yaml <- clip file <- preset <- overrides ({"a.b": v} or "a.b=v" strings).
Nothing about the recording is typed in: the views, videos and mode come from the pose24 clip registry, and the
source fps / frame count / image size from the inputs (kp2d headers for the cached backend, the videos for live).
Every length in seconds is converted to steps for the chosen output_fps, so 30 and 60 fps runs are comparable.

The resolved dict keeps the flat keys the streaming core reads (views, ref_frames, fps, warmup_steps, window, overlap,
history, offsets, sync, stride, time_ref, out_clip, timeline_frames, kp2d_paths, cam_paths, ...) next to the nested
sections (tracker, stream, fallback, ...)."""
from __future__ import annotations

import copy
import json
import os
import re

import yaml

from analysis.pose24 import config as p24config

CONFIG_DIR = os.path.join("configs", "live")
DEFAULTS = os.path.join(CONFIG_DIR, "defaults.yaml")
PRESET_DIR = os.path.join(CONFIG_DIR, "presets")
BACKENDS = ("cached", "live")
DEVICES = ("cuda", "cpu")
PACES = ("realtime", "asfast")
SYNC_MODES = ("interp", "nearest")
OFFSET_MODES = ("auto", "file", "zero")
RTM_MODES = ("performance", "balanced", "lightweight")
RTM_MODELS = ("wholebody", "body", "body_feet")
RTM_ENGINES = ("rtmlib", "fast")
RTM_DECODERS = ("cv2", "pyav", "ffmpeg", "nvdec", "nvcodec")
HEADER_BYTES = 65536


# ---------------------------------------------------------------------------- layering
def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, top: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (top or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(items) -> dict:
    """["stream.workers=0", "output_fps=60"] or {"stream.workers": 0} -> nested dict (values parsed as YAML)."""
    flat = {}
    if isinstance(items, dict):
        flat = dict(items)
    else:
        for s in items or []:
            if "=" not in s:
                raise ValueError(f"override {s!r} is not key=value")
            k, v = s.split("=", 1)
            flat[k.strip()] = yaml.safe_load(v)
    out = {}
    for k, v in flat.items():
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def preset_path(name_or_path):
    if name_or_path is None:
        return None
    if os.path.exists(name_or_path):
        return name_or_path
    p = os.path.join(PRESET_DIR, f"{name_or_path}.yaml")
    if not os.path.exists(p):
        have = sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".yaml")) if os.path.isdir(PRESET_DIR) else []
        raise ValueError(f"no preset {name_or_path!r} (have {have})")
    return p


def clip_path(name_or_path):
    if os.path.exists(name_or_path):
        return name_or_path
    p = os.path.join(CONFIG_DIR, f"{name_or_path}.yaml")
    if not os.path.exists(p):
        raise ValueError(f"no clip config {name_or_path!r} ({p} missing)")
    return p


def layered(clip, preset=None, overrides=None, defaults=DEFAULTS) -> dict:
    c = _read_yaml(defaults)
    c = deep_merge(c, _read_yaml(clip_path(clip)))
    if preset:
        c = deep_merge(c, _read_yaml(preset_path(preset)))
    c = deep_merge(c, parse_overrides(overrides))
    c["_sources"] = {"defaults": defaults, "clip": clip_path(clip), "preset": preset_path(preset),
                     "overrides": overrides if isinstance(overrides, dict) else list(overrides or [])}
    return c


# ---------------------------------------------------------------------------- source probing
def kp2d_header(path) -> dict:
    """The header fields of a kp2d JSON (everything before "frames") without parsing the frames."""
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(HEADER_BYTES)
    i = head.find('"frames"')
    if i < 0:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return {k: v for k, v in d.items() if k != "frames"}
    txt = re.sub(r",\s*$", "", head[:i].rstrip()) + "}"
    return json.loads(txt)


def video_info(path) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS)), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def probe_sources(c) -> dict:
    """{view: {fps, frame_count, width, height, source}} from the inputs the backend will read."""
    out = {}
    for v in c["views"]:
        if c["backend"] == "cached":
            p = c["kp2d_paths"][v]
            if not os.path.exists(p):
                raise ValueError(f"no 2D file for view {v}: {p}")
            h = kp2d_header(p)
            out[v] = {"fps": float(h["fps"]), "frame_count": int(h["frame_count"]), "width": int(h["width"]),
                      "height": int(h["height"]), "source": p}
        else:
            p = c["videos"].get(v)
            if not p or not os.path.exists(p):
                raise ValueError(f"no video for view {v}: {p}")
            out[v] = dict(video_info(p), source=p)
    return out


# ---------------------------------------------------------------------------- resolve
def _steps(seconds, fps, name, lo=1):
    n = int(round(float(seconds) * fps))
    if n < lo:
        raise ValueError(f"{name} = {seconds} s is {n} steps at {fps:g} fps (need >= {lo})")
    return n


def _check_choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def resolve(c, probe=probe_sources) -> dict:
    c = copy.deepcopy(c)
    if not c.get("clip"):
        raise ValueError("clip is not set (clip file or --set clip=...)")
    reg = p24config.load_registry()
    if c["clip"] not in reg:
        raise ValueError(f"clip {c['clip']!r} is not in the pose24 registry")
    entry = reg[c["clip"]]
    base = reg.get(entry.get("base"), entry) if entry.get("base") else entry
    all_views = list(entry.get("views") or base.get("views") or [])
    c["views"] = sorted(c.get("views") or all_views)
    bad = [v for v in c["views"] if v not in all_views]
    if bad:
        raise ValueError(f"views {bad} are not cameras of {c['clip']} ({all_views})")
    if len(c["views"]) < 2:
        raise ValueError(f"need at least 2 views for triangulation, got {c['views']}")
    reg_ref = entry.get("time_ref") or base.get("time_ref")
    c["time_ref"] = c.get("time_ref") or (reg_ref if reg_ref in c["views"] else c["views"][0])
    if c["time_ref"] not in c["views"]:
        raise ValueError(f"time_ref {c['time_ref']!r} is not one of the views {c['views']}")
    c["mode"] = entry.get("mode") or base.get("mode") or "singles"
    if c["mode"] != "singles":
        raise ValueError(f"live-frames handles singles only (clip mode {c['mode']!r})")
    videos = base.get("videos") or entry.get("videos") or {}
    c["videos"] = {v: videos.get(v) for v in c["views"]}
    for k, allowed in (("backend", BACKENDS), ("device", DEVICES), ("pace", PACES)):
        _check_choice(k, c[k], allowed)
    _check_choice("sync.mode", c["sync"]["mode"], SYNC_MODES)
    _check_choice("rtmpose.mode", c["rtmpose"]["mode"], RTM_MODES)
    _check_choice("rtmpose.model", c["rtmpose"]["model"], RTM_MODELS)
    rp = c["rtmpose"]
    rp.setdefault("engine", "rtmlib")
    rp.setdefault("decoder", "cv2")
    _check_choice("rtmpose.engine", rp["engine"], RTM_ENGINES)
    _check_choice("rtmpose.decoder", rp["decoder"], RTM_DECODERS)
    if rp["engine"] == "rtmlib":
        fast_only = [k for k, dflt in (("decoder", "cv2"), ("decode_short_side", None), ("batch_views", False),
                                       ("det", None), ("court_filter_m", None)) if rp.get(k, dflt) != dflt]
        if rp["model"] == "body_feet":
            fast_only.append("model=body_feet")
        if fast_only:
            raise ValueError(f"rtmpose {fast_only} need rtmpose.engine=fast")
    if rp["model"] != "wholebody":
        from analysis.live.pose2d_fast import SCORE_TO_WB
        if f"{rp['model']}/{rp['mode']}" not in SCORE_TO_WB:      # its scores could not be calibrated (all dropped)
            raise ValueError(f"rtmpose.model={rp['model']} has no score map for mode={rp['mode']} "
                             f"(have {sorted(SCORE_TO_WB)})")
    ss = rp.get("decode_short_side")
    if ss is not None and (not isinstance(ss, (int, float)) or ss < 64):
        raise ValueError(f"rtmpose.decode_short_side must be null or >= 64 px, got {ss!r}")

    paths = c["paths"]
    data = paths["data_dir"]
    c["data_dir"], c["runs_dir"] = data, paths["runs_dir"]
    c["cam_paths"] = {v: os.path.join(data, "calib", f"{v}_cam.json") for v in c["views"]}
    c["kp2d_paths"] = {v: os.path.join(data, "pose2d", f"{v}_kp2d.json") for v in c["views"]}
    paths["sync_file"] = paths["sync_file"].format(data_dir=data, clip=c["clip"])

    src = probe(c)
    c["source"] = src
    fps_set = {round(s["fps"], 3) for s in src.values()}
    if len(fps_set) != 1:
        raise ValueError(f"views have different frame rates {fps_set}")
    sfps = float(src[c["time_ref"]]["fps"])
    c["source_fps"] = sfps
    ofps = float(c["output_fps"])
    stride = sfps / ofps
    if ofps <= 0 or abs(stride - round(stride)) > 1e-6:
        raise ValueError(f"output_fps {ofps:g} must divide the source fps {sfps:g}")
    c["stride"] = int(round(stride))
    c["fps"] = sfps / c["stride"]
    n_ref = int(src[c["time_ref"]]["frame_count"])
    c["timeline_frames"] = n_ref
    a, b = c["section_s"]
    b = n_ref / sfps if b is None else b
    c["ref_start"] = int(round(float(a) * sfps))
    c["ref_end"] = min(int(round(float(b) * sfps)), n_ref)
    if not 0 <= c["ref_start"] < c["ref_end"]:
        raise ValueError(f"section_s {c['section_s']} is empty or outside the {n_ref / sfps:.2f} s video")
    c["ref_frames"] = list(range(c["ref_start"], c["ref_end"], c["stride"]))
    n = len(c["ref_frames"])

    fps = c["fps"]
    w, st = c["warmup"], c["stream"]
    c["warmup_steps"] = _steps(w["seconds"], fps, "warmup.seconds")
    if c["warmup_steps"] >= n:
        raise ValueError(f"warm-up of {c['warmup_steps']} steps does not fit the {n}-step section")
    c["window"] = _steps(st["window_s"], fps, "stream.window_s", lo=2)
    c["overlap"] = _steps(st["overlap_s"], fps, "stream.overlap_s")
    if not 0 < c["overlap"] < c["window"]:
        raise ValueError(f"need 0 < overlap < window, got overlap {c['overlap']} window {c['window']} steps")
    c["history"] = _steps(st["history_s"], fps, "stream.history_s", lo=0)
    c["fit_budget_s"] = ((c["window"] - c["overlap"]) / fps if st["fit_budget_s"] in (None, "auto")
                         else float(st["fit_budget_s"]))
    c["workers"] = int(st["workers"])
    c["fallback_max_hold"] = _steps(c["fallback"]["max_hold_s"], fps, "fallback.max_hold_s", lo=0)
    c["facing_window"] = max(1, _steps(c["facing"]["window_s"], fps, "facing.window_s", lo=0))

    offs = c["sync"]["offsets"]
    if isinstance(offs, dict):
        missing = [v for v in c["views"] if v not in offs]
        if missing:
            raise ValueError(f"no offset for views {missing}")
        c["offsets"], c["offsets_source"] = {v: float(offs[v]) for v in c["views"]}, "config"
    else:
        _check_choice("sync.offsets", offs, OFFSET_MODES)
        c["offsets"], c["offsets_source"] = {v: 0.0 for v in c["views"]}, offs
        if offs == "file":
            c["offsets"] = load_sync_file(paths["sync_file"], c["views"], c["time_ref"])
    c["offsets"][c["time_ref"]] = 0.0
    c["sync_mode"] = c["sync"]["mode"]

    c["ankle_h_m"] = None if w["ankle_h"] in (None, "auto") else float(w["ankle_h"])
    c["out_clip"] = c["out_clip"].format(clip=c["clip"], nviews=len(c["views"]), fps=f"{c['fps']:g}")
    refs = c["report"].get("reference_clips") or []
    c["reference_clips"] = [r for r in refs if r]
    return c


def load_sync_file(path, views, time_ref) -> dict:
    """data/calib/<clip>_sync.json: {"offsets": {view: frames}} (raw frame = ref frame + offset)."""
    if not os.path.exists(path):
        raise ValueError(f"sync.offsets = file but {path} does not exist (make it with live_run sync-file)")
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    offs = d.get("offsets", d)
    missing = [v for v in views if v not in offs]
    if missing:
        raise ValueError(f"{path} has no offset for {missing}")
    base = float(offs.get(time_ref, 0.0))
    return {v: float(offs[v]) - base for v in views}


def load(clip, preset=None, overrides=None, defaults=DEFAULTS, probe=probe_sources) -> dict:
    return resolve(layered(clip, preset, overrides, defaults), probe=probe)


def public(c) -> dict:
    """The resolved config without bulky / private entries, for config.resolved.yaml and the report."""
    skip = {"ref_frames"}
    out = {k: v for k, v in c.items() if k not in skip}
    out["ref_frames"] = {"first": c["ref_frames"][0], "last": c["ref_frames"][-1], "count": len(c["ref_frames"])}
    return out


def dump(c, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(public(c), default=str)), fh, sort_keys=False, allow_unicode=True)
