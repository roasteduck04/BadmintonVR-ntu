"""python -m analysis.people track | eval | frames | reid-extract (SP3, v0.7.2).

    PYTHONPATH=tools python -m analysis.people track --all --tracker both      # data/people_eval/<clip>.<tracker>.persons.json
                                                                               # (--reid osnet with tracker 3d -> <clip>.3d_osnet.persons.json)
    PYTHONPATH=tools python -m analysis.people eval --all                      # label-free (+ labelled when data/gt/labels exist)
    PYTHONPATH=tools python -m analysis.people frames --clip doubles_2sides    # frame list for the labelling page
    PYTHONPATH=tools python -m analysis.people reid-extract --clip doubles_2sides   # OSNet embeddings (video decode)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from analysis.people import persons
from analysis.pose24 import config, scores, track
from analysis.pose24.__main__ import _each

EVAL_DIR = os.path.join("data", "people_eval")


def resolve(clip, reg, data_dir, analysis_dir):
    """pose24's build config + its frame count (base doc when on the clip's own timeline) + timeline."""
    cfg = config.resolve(clip, registry=reg, data_dir=data_dir, analysis_dir=analysis_dir)
    timeline = track.load_timeline(cfg["timeline_of"]) if cfg["timeline_of"] else None
    if cfg["base_doc_path"] and timeline is None:
        with open(cfg["base_doc_path"], "r", encoding="utf-8") as fh:
            cfg["frame_count"] = int(json.load(fh)["meta"]["frame_count"])     # same n as pose24 build
    return cfg, timeline


def _cmd_track(a):
    reg = config.load_registry()
    clips = config.multiview_clips(reg) if a.all else (a.clip or [])   # mono entries: no tracker run
    if not clips:
        sys.exit("give --clip or --all")
    kinds = ["v2", "3d"] if a.tracker == "both" else [a.tracker]
    params = scores.load_params()

    def run(c):
        cfg, timeline = resolve(c, reg, a.data_dir, a.analysis_dir)
        feats = None
        if a.reid == "osnet":
            from analysis.people import osnet
            feats = osnet.load_cache(cfg["clip"], data_dir=a.data_dir)
        for kind in kinds:
            print(f"[people] {cfg['out_clip']}: tracker {kind}")
            b = track.track_and_sync24(cfg, params, timeline, tracker=kind, reid=a.reid, reid_feats=feats)
            tag = "3d_osnet" if (kind == "3d" and a.reid == "osnet") else kind
            out = os.path.join(a.out_dir, f"{cfg['out_clip']}.{tag}.persons.json")
            persons.write(out, b["persons"])
            st = b["persons"]["stats"]
            print(f"    -> {out}: {st['n_tracks']} tracks (player {st['n_player']}, nonplayer {st['n_nonplayer']}, "
                  f"unknown {st['n_unknown']}); player frames {dict(zip(st['player_ids'], st['player_frames']))}")
    return _each(clips, run, a.all)


def _default_sources(out_clip, a):
    src = {k: os.path.join(a.out_dir, f"{out_clip}.{k}.persons.json") for k in ("3d", "3d_osnet", "v2")}
    base = os.path.join(a.analysis_dir, f"{out_clip}.cvdebug.json")
    if os.path.exists(base):
        src["base"] = base
    return {k: p for k, p in src.items() if os.path.exists(p)}


def _cmd_eval(a):
    from analysis import appearance, cvdebug
    from analysis import pose_v2 as pv
    from analysis.people import eval_identity as ev
    reg = config.load_registry()
    clips = config.multiview_clips(reg) if a.all else (a.clip or [])   # mono entries: no tracker run
    if not clips:
        sys.exit("give --clip or --all")

    def run(c):
        cfg, _ = resolve(c, reg, a.data_dir, a.analysis_dir)
        srcs = dict(s.split("=", 1) for s in a.sources.split(",")) if a.sources else _default_sources(cfg["out_clip"], a)
        if not srcs:
            return "no persons files (run: python -m analysis.people track)"
        views = sorted(cfg["all_views"])
        cams = pv.load_cams({v: os.path.join(a.data_dir, "calib", f"{v}_cam.json") for v in views})
        raw = {v: cvdebug.load_raw_kp2d(os.path.join(a.data_dir, "pose2d", f"{v}_kp2d.json"))[0] for v in views}
        appear = appearance.load_clip(cfg["clip"], data_dir=a.data_dir)
        labels = ev.load_labels(os.path.join(a.labels_dir, f"{cfg['out_clip']}_ids.json"))
        res = {"clip": cfg["out_clip"], "n_labels": len(labels), "sources": {}}
        for name, p in srcs.items():
            d = persons.load(p)
            if d.get("format") != persons.FORMAT:             # a cvdebug sidecar
                d = persons.from_sidecar(d)
            r = {"label_free": ev.label_free(d, cams, raw, appear)}
            if labels:
                r["labelled"] = ev.labelled_metrics(labels, persons.predictions(d))
            res["sources"][name] = r
            lf = r["label_free"]["players"]
            print(f"  {cfg['out_clip']:28s} {name:5s} " + " | ".join(
                f"{pid} cov {x['coverage']:.2f} cross {x['net_crossings']} incons {x['inconsistent_frac']:.3f}"
                for pid, x in lf.items())
                + (f" || IDF1 {r['labelled']['idf1']:.3f} npFP {r['labelled']['nonplayer_fp_rate']:.3f}" if labels else ""))
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"{cfg['out_clip']}.eval.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return _each(clips, run, a.all)


def _cmd_frames(a):
    from collections import Counter
    from analysis.people.labels import LabelService
    reg = config.load_registry()
    clips = config.multiview_clips(reg) if a.all else (a.clip or [])   # mono entries: no tracker run
    if not clips:
        sys.exit("give --clip or --all")
    svc = LabelService(registry=reg, analysis_dir=a.analysis_dir, eval_dir=a.eval_dir, labels_dir=a.labels_dir,
                       data_dir=a.data_dir)

    def run(c):
        if not svc._ppath(c):
            return "no persons file (run: python -m analysis.people track)"
        fr = svc.frames(c)
        items = fr["items"]
        frames = sorted({it["f"] for it in items})
        why = Counter(r for f in frames for r in next(it["reasons"] for it in items if it["f"] == f))
        print(f"  {c:28s} {len(frames)} frames x {len(items) // max(1, len(frames))} views = {len(items)} items "
              f"(done {fr['done']}); " + ", ".join(f"{k} {n}" for k, n in sorted(why.items())))
        os.makedirs(a.labels_dir, exist_ok=True)
        out = os.path.join(a.labels_dir, f"{c}_frames.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"clip": c, "persons": os.path.basename(svc._ppath(c)), "reasons": dict(why), "items": items},
                      fh, indent=1)
        print(f"    -> {out}")
    return _each(clips, run, a.all)


def _cmd_reid_extract(a):
    from analysis.people import load_params, osnet
    reg = config.load_registry()
    clips = config.multiview_clips(reg) if a.all else (a.clip or [])   # mono entries: no tracker run
    if not clips:
        sys.exit("give --clip or --all")
    step = a.step or int(load_params()["appearance"]["osnet_step"])
    done = set()

    def run(c):
        cfg = config.resolve(c, registry=reg, data_dir=a.data_dir)
        if cfg["clip"] in done:                       # camera subsets share their base clip's cache
            return f"cache of {cfg['clip']} already extracted"
        print(f"[people] {cfg['clip']}: OSNet embeddings, every {step}th raw frame")
        out = osnet.extract_clip(cfg["clip"], cfg["all_views"], registry=reg, step=step, data_dir=a.data_dir)
        done.add(cfg["clip"])
        print(f"    -> {out}")
    return _each(clips, run, a.all)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.people", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track", help="run a tracker (sync + view health as pose24) -> persons/1")
    t.add_argument("--clip", action="append")
    t.add_argument("--all", action="store_true")
    t.add_argument("--tracker", choices=["3d", "v2", "both"], default="3d")
    t.add_argument("--reid", choices=["colour", "osnet"], default="colour")
    t.add_argument("--out-dir", default=EVAL_DIR)
    t.add_argument("--data-dir", default="data")
    t.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    e = sub.add_parser("eval", help="label-free checks (+ labelled IDF1 / HOTA-lite when labels exist)")
    e.add_argument("--clip", action="append")
    e.add_argument("--all", action="store_true")
    e.add_argument("--sources", default="", help="name=path,... (default: 3d, v2 persons files + the base cvdebug)")
    e.add_argument("--labels-dir", default=os.path.join("data", "gt", "labels"))
    e.add_argument("--out-dir", default=EVAL_DIR)
    e.add_argument("--data-dir", default="data")
    e.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    fr = sub.add_parser("frames", help="frame list for the labelling page (/labeler) -> data/gt/labels/<clip>_frames.json")
    fr.add_argument("--clip", action="append")
    fr.add_argument("--all", action="store_true")
    fr.add_argument("--labels-dir", default=os.path.join("data", "gt", "labels"))
    fr.add_argument("--eval-dir", default=EVAL_DIR)
    fr.add_argument("--data-dir", default="data")
    fr.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    r = sub.add_parser("reid-extract", help="OSNet-x0.25 embeddings of every detection (decodes the full-res videos; "
                                            "slow) -> data/pose_cache/<clip>_osnet.npz for --reid osnet")
    r.add_argument("--clip", action="append")
    r.add_argument("--all", action="store_true")
    r.add_argument("--step", type=int, default=0, help="every N-th raw frame (default params appearance.osnet_step)")
    r.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    return {"track": _cmd_track, "eval": _cmd_eval, "frames": _cmd_frames,
            "reid-extract": _cmd_reid_extract}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
