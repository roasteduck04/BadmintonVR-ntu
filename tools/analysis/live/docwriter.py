"""Live output frames -> one schema 4.1 pose24 analysis doc on the clip's source-fps timeline, written at the end of
a run (not live): live frames at their ref frames; with a stride > 1 the frames between two consecutive valid live
frames are linearly interpolated (interpolated: true) so Unity can play / ghost it against the offline builds; all
else null. pose_variant carries the time reference + offsets so VariantGhosts' shift is 0. `header` is the same doc
with empty players: the live manifest."""
from __future__ import annotations

import time
import uuid

import numpy as np

from analysis import skeleton24 as sk
from analysis.pose24 import build, scores
from analysis.pose24 import doc as docmod


def _arrays(cfg, outs, pid, n_tl, nj):
    J, C = np.full((n_tl, nj, 3), np.nan), np.zeros((n_tl, nj))
    A, AC = np.full((n_tl, 14, 3), np.nan), np.zeros((n_tl, 14))
    interp = np.zeros(n_tl, bool)
    live = sorted(((o["ref_frame"], o["players"][pid]) for o in outs if pid in o["players"]),
                  key=lambda t: t[0])
    for f, p in live:
        if p["valid"]:
            J[f], C[f], A[f], AC[f], interp[f] = p["J"], p["C"], p["aux"], p["auxc"], p["interp"]
    for (f0, p0), (f1, p1) in zip(live, live[1:]):
        if not (p0["valid"] and p1["valid"]) or f1 - f0 > int(cfg["stride"]):
            continue
        for f in range(f0 + 1, f1):
            a = (f - f0) / (f1 - f0)
            J[f] = (1 - a) * J[f0] + a * J[f1]
            A[f] = (1 - a) * A[f0] + a * A[f1]
            C[f], AC[f], interp[f] = np.minimum(C[f0], C[f1]), np.minimum(AC[f0], AC[f1]), True
    return J, C, A, AC, interp


def _pose_variant(cfg):
    views = list(cfg["views"])
    offs = {v: float(cfg["offsets"][v]) for v in views}
    return {"name": "live", "views": views, "excluded_views": {}, "time_ref_view": cfg["time_ref"],
            "timeline_of": cfg.get("clip"), "offsets": offs,
            "offsets_list": [{"view": v, "offset": offs[v]} for v in views], "offsets_source": cfg.get("offsets_source"),
            "stride": int(cfg["stride"]), "sync": cfg.get("sync_mode"), "backend": cfg.get("backend"),
            "pace": cfg.get("pace"), "output_fps": float(cfg["fps"])}


def _bundle(cfg, cams_all, sides, image_size):
    return {"mode": cfg.get("mode", "singles"), "views": list(cfg["views"]), "excluded": {}, "cams_all": cams_all,
            "image_size": image_size or {}, "sides": dict(sides)}


def header(cfg, cams_all, sides, image_size=None, build_id=None) -> dict:
    kind = cfg["skeleton"]
    pmeta = {pid: {"forward_sign": 1.0, "coverage": 0.0} for pid in sides}
    p24 = build.pose24_meta(build_id or uuid.uuid4().hex, kind, scores.load_params(), pmeta)
    d = docmod.make_doc24(cfg["out_clip"], float(cfg["source_fps"]), int(cfg["timeline_frames"]),
                          [(pid, []) for pid in sorted(sides)], _bundle(cfg, cams_all, sides, image_size), kind,
                          _pose_variant(cfg), p24)
    d["meta"]["label"] = f"{d['meta']['label']} · simulated live {cfg['fps']:g} fps"
    return d


def to_doc(cfg, outs, cams_all, sides, image_size=None, build_id=None) -> dict:
    t_all = time.perf_counter()
    kind = cfg["skeleton"]
    sfps = float(cfg["source_fps"])
    nj = len(sk.TYPES[kind][0])
    n_tl = int(cfg["timeline_frames"])
    outs = sorted(outs, key=lambda o: o["step"])
    pids = sorted({pid for o in outs for pid in o["players"]} | set(sides))
    players, pmeta = [], {}
    for pid in pids:
        J, C, A, AC, interp = _arrays(cfg, outs, pid, n_tl, nj)
        players.append((pid, docmod.frames_from_world24(J, C, A, AC, interp, sfps)))
        signs = [o["players"][pid]["forward_sign"] for o in outs if o["players"].get(pid, {}).get("valid")]
        pmeta[pid] = {"forward_sign": -1.0 if signs and np.median(signs) < 0 else 1.0,
                      "coverage": float(np.mean([o["players"].get(pid, {}).get("valid", False) for o in outs]))
                      if outs else 0.0}
    p24 = build.pose24_meta(build_id or uuid.uuid4().hex, kind, scores.load_params(), pmeta)
    d = docmod.make_doc24(cfg["out_clip"], sfps, n_tl, players, _bundle(cfg, cams_all, sides, image_size), kind,
                          _pose_variant(cfg), p24)
    d["meta"]["label"] = f"{d['meta']['label']} · simulated live {cfg['fps']:g} fps"
    d["meta"]["live"] = {"ref_frames": [int(o["ref_frame"]) for o in outs], "fps_pipeline": float(cfg["fps"]),
                         "frame_quality": [int(o["quality"]) for o in outs],
                         "quality": {pid: [int(o["players"][pid]["quality"])
                                           if pid in o["players"] and o["players"][pid]["valid"] else 3
                                           for o in outs] for pid in pids}}
    build.finish_doc(d, sfps, kind, {}, t_all)
    return d


def write(cfg, outs, cams_all, sides, out_dir, image_size=None) -> str:
    d = to_doc(cfg, outs, cams_all, sides, image_size=image_size)
    path = build.output_path(out_dir, cfg["out_clip"])
    docmod.write_json(path, d)
    return path
