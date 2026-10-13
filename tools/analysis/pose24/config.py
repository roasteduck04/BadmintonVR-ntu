"""pose24 clip registry (clips.json) -> build config. Replaces pose_v2's CLIPS dict and its 5-view
fallback for this route (review L2); other tools migrate to the registry later."""
from __future__ import annotations

import json
import os

from analysis import pose_v2 as pv

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "clips.json")
DEFAULT_ANALYSIS_DIR = pv.DEFAULT_ANALYSIS_DIR


def load_registry(path: str = REGISTRY_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["clips"]


def all_clips(registry: dict) -> list:
    """Full clips first (their caches give the subset builds a timeline), then subset entries."""
    full = [c for c, e in registry.items() if "base" not in e]
    return full + [c for c, e in registry.items() if "base" in e]


def multiview_clips(registry: dict) -> list:
    """all_clips without the mono (1-camera) entries: the clips `resolve` gives a multi-view build config for
    (people / tracker CLIs, anything that needs >= 2 synced views)."""
    return [c for c in all_clips(registry) if not registry[c].get("mono")]


def resolve(clip, views=None, out_clip=None, timeline_of=None, registry=None, data_dir="data",
            analysis_dir=DEFAULT_ANALYSIS_DIR) -> dict:
    reg = registry if registry is not None else load_registry()
    if clip not in reg:
        raise KeyError(f"unknown clip {clip!r}; registry has {', '.join(sorted(reg))}")
    entry = reg[clip]
    if entry.get("mono"):
        raise ValueError(f"registry entry {clip!r} is a mono (1-camera) build: pose24 build routes it through "
                         "build --mono; it has no multi-view build config")
    base_name = entry.get("base", clip)
    if base_name not in reg:
        raise KeyError(f"registry entry {clip!r} has base {base_name!r}, which is not in the registry")
    base = reg[base_name]
    all_views = sorted(base["views"])
    entry_views = sorted(entry.get("views", base["views"]))
    views = sorted(views) if views is not None else entry_views
    unknown = sorted(set(views) - set(all_views))
    if unknown:
        raise ValueError(f"unknown views for {base_name}: {', '.join(unknown)}")
    if len(views) < 2:
        raise ValueError("pose24 needs >= 2 views (1-camera builds are a later release)")
    if out_clip is None and views != entry_views:
        # e.g. --clip singles_05x_2cam --views a,b would overwrite the registry's singles_05x_2cam build
        raise ValueError(f"views {', '.join(views)} differ from registry entry {clip!r}: give --out-clip")
    out_clip = out_clip or clip
    if views != all_views and out_clip == base_name:
        raise ValueError(f"a camera subset of {base_name} needs its own --out-clip")
    timeline_of = timeline_of if timeline_of is not None else entry.get("timeline_of")
    time_ref = base.get("time_ref") or all_views[0]
    if time_ref not in views:
        time_ref = views[0]
    calib, pose2d = os.path.join(data_dir, "calib"), os.path.join(data_dir, "pose2d")
    base_doc = os.path.join(analysis_dir, f"{out_clip}.analysis.json")
    return {"clip": base_name, "out_clip": out_clip, "mode": base["mode"], "views": views,
            "all_views": all_views, "time_ref": time_ref, "timeline_of": timeline_of,
            "cam_paths": {v: os.path.join(calib, f"{v}_cam.json") for v in views},
            "kp2d_paths": {v: os.path.join(pose2d, f"{v}_kp2d.json") for v in views},
            "seeds_path": os.path.join(calib, f"{base['seeds']}_seeds.json"),
            "videos": {v: base.get("videos", {}).get(v) for v in views},
            "base_doc_path": base_doc if os.path.exists(base_doc) else None,
            "analysis_dir": analysis_dir}
