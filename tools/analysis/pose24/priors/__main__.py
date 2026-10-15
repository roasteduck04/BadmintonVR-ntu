"""python -m analysis.pose24.priors motionbert - v0.7.5 learned 1-camera depth priors (data/smpl_mono/<clip>/).

    PYTHONPATH=tools python -m analysis.pose24.priors motionbert --clip singles_05x               # kept views, lite
    PYTHONPATH=tools python -m analysis.pose24.priors motionbert --clip singles_05x --views back_051 --model full
    PYTHONPATH=tools python -m analysis.pose24.priors motionbert --clip singles_05x --norm image  # *_motionbert-lite-image.npz
    PYTHONPATH=tools python -m analysis.pose24.priors check --clip singles_05x --view back_051 --method motionbert-lite
        (prior vs the full rig's camera-frame root-relative 3D -> table + <view>_<method>.check.json)
"""
from __future__ import annotations

import argparse
import os
import sys

from analysis.pose24.priors import check as chk, motionbert as mb


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _load_cache(clip):
    from analysis.pose24 import doc as docmod, track
    path = track.cache_path(clip)
    if not os.path.exists(path):
        raise SystemExit(f"no pose24 cache of {clip} at {path}; build it first "
                         f"(`python -m analysis.pose24 build --clip {clip}`)")
    return docmod.load_debug_cache(path)


def _motionbert(ap, args):
    cache = _load_cache(args.clip)
    excluded = cache.get("excluded") or {}
    kept = [v for v in cache["views"] if v not in excluded]
    views = _csv(args.views) or kept
    bad = [v for v in views if v not in cache["views"]]
    if bad:
        ap.error(f"views {bad} not in {args.clip}'s pose24 cache ({', '.join(cache['views'])})")
    model = mb.load_model(args.model)
    for v in views:
        print(f"[priors] {args.clip} {v}: {mb.method_label(args.model, args.norm)}")
        print(f"[priors] -> {mb.produce(args.clip, v, args.model, model=model, cache=cache, norm=args.norm)}")


def _check(args):
    res = chk.run(args.clip, args.view, args.method, cache=_load_cache(args.clip))
    print(chk.format_table(res))
    print(f"[priors] -> {res['path']}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(prog="python -m analysis.pose24.priors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("motionbert", help="MotionBERT CPU prior per view -> data/smpl_mono/<clip>/<view>_motionbert-<kind>.npz")
    m.add_argument("--clip", required=True)
    m.add_argument("--views", default=None, help="comma-separated views (default: the cache's kept views)")
    m.add_argument("--model", choices=sorted(mb.MODELS), default="lite")
    m.add_argument("--norm", choices=mb.NORMS, default=mb.DEFAULT_NORM,
                   help="2D normalisation: crop = infer_wild default (track bbox), image = --pixel (image size); "
                        "image files get a -image suffix")
    c = sub.add_parser("check", help="prior vs the full rig (camera-frame root-relative 3D): per-joint r, |dz|, gate")
    c.add_argument("--clip", required=True)
    c.add_argument("--view", required=True)
    c.add_argument("--method", required=True, help="prior method label, e.g. motionbert-lite, motionbert-lite-image")
    args = ap.parse_args(argv)
    if args.cmd == "motionbert":
        _motionbert(m, args)
    elif args.cmd == "check":
        _check(args)


if __name__ == "__main__":
    main()
