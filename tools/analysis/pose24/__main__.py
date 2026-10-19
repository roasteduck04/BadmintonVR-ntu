"""python -m analysis.pose24 build | eval | calibrate-scores | list — the SMPL-24/28 pose route (v0.7.0).

    PYTHONPATH=tools python -m analysis.pose24 build --clip singles_05x
    PYTHONPATH=tools python -m analysis.pose24 build --all            # every registry clip, subsets last
    PYTHONPATH=tools python -m analysis.pose24 build --clip singles_05x --views backleft_051,frontright_051 \
            --out-clip singles_05x_2cam --timeline-of singles_05x
    PYTHONPATH=tools python -m analysis.pose24 build --clip singles_05x --mono back_051 --out-clip singles_05x_1cam_b \
            [--mono-lengths oracle|prior] [--mono-prior data/smpl_mono/singles_05x/back_051_wham.npz]
    PYTHONPATH=tools python -m analysis.pose24 list
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from analysis.pose24 import build as pb
from analysis.pose24 import config, track
from analysis.pose24.eval import OUT_DIR as EVAL_OUT_DIR


def _each(clips, run, isolate, tag="pose24"):
    """Run `run(clip)` per clip. isolate (--all): a failing clip logs its traceback and the rest still run;
    a pass/fail summary is printed and the exit code is 1 if any clip failed. run() returning a string
    means the clip was skipped (the string says why)."""
    if not isolate:
        for c in clips:
            run(c)
        return 0
    ok, skipped, failed = [], [], []
    for c in clips:
        try:
            why = run(c)
        except Exception:
            print(f"[{tag}] {c}: FAILED\n{traceback.format_exc()}", file=sys.stderr)
            failed.append(c)
            continue
        if isinstance(why, str):
            print(f"[{tag}] {c}: skipped ({why})")
            skipped.append(c)
        else:
            ok.append(c)
    print(f"[{tag}] {len(ok)}/{len(clips)} ok"
          + (f"; skipped: {', '.join(skipped)}" if skipped else "")
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def _cmd_build(a):
    reg = config.load_registry()
    clips = config.all_clips(reg) if a.all else (a.clip or [])
    if not clips:
        sys.exit("give --clip or --all")
    if (a.views or a.out_clip) and len(clips) != 1:
        sys.exit("--views / --out-clip need exactly one --clip")
    if a.all and a.timeline_of:
        sys.exit("--timeline-of applies to one --clip (--all takes each entry's timeline_of from clips.json)")
    views = a.views.split(",") if a.views else None
    mono_opts = None
    if a.mono:
        if a.all or len(clips) != 1 or not a.out_clip:
            sys.exit("--mono needs exactly one --clip and an --out-clip (e.g. <clip>_1cam_<view>)")
        if a.views or a.timeline_of:
            sys.exit("--mono takes the full build's views and timeline: no --views / --timeline-of")
        mono_opts = {"lengths": a.mono_lengths}
        if a.mono_prior:
            mono_opts["prior_path"] = a.mono_prior
    elif a.mono_prior:
        sys.exit("--mono-prior needs --mono")

    def run(c):
        kw = {"mono": a.mono, "mono_opts": mono_opts} if a.mono else {}
        fit_opts = {"passes": a.fit_passes, "use_contact": not a.no_contact, "use_limits": not a.no_limits}
        pb.build(c, views=views, out_clip=a.out_clip, timeline_of=a.timeline_of, skeleton=a.skeleton,
                 analysis_dir=a.analysis_dir, registry=reg, tracker=a.tracker, reid=a.reid,
                 fit_opts=None if a.mono else fit_opts, **kw)
    return _each(clips, run, a.all)


def _cmd_list(a):
    reg = config.load_registry()
    for c in config.all_clips(reg):
        e = reg[c]
        views = e.get("views") or reg[e["base"]]["views"]
        doc = "doc" if os.path.exists(pb.output_path(a.analysis_dir, c)) else "-"
        cache = "cache" if os.path.exists(track.cache_path(c)) else "-"
        sub = (f"  ({'mono' if e.get('mono') else 'subset'} of {e['base']}, timeline {e.get('timeline_of')})"
               if "base" in e else "")
        print(f"{c:28s} {len(views)} views  {doc:3s} {cache:5s}  {', '.join(views)}{sub}")
    return 0


def _cmd_eval(a):
    from analysis.pose24 import eval as ev
    reg = config.load_registry()
    clips = config.all_clips(reg) if a.all else (a.clip or [])

    def run(c):
        ref = a.reference or reg.get(c, {}).get("timeline_of")
        if a.all:
            need = [c] + ([ref] if ref else [])
            missing = [x for x in need if not (os.path.exists(pb.output_path(a.analysis_dir, x))
                                               and os.path.exists(track.cache_path(x)))]
            if missing:
                return f"no pose24 doc/cache for {', '.join(missing)}"
        if a.lovo and ref:
            print(f"[pose24] {c}: --lovo ignored (scored against the reference {ref} instead)")
        ev.evaluate(c, reference=ref, lovo=a.lovo and not ref, posefit=not a.no_posefit,
                    analysis_dir=a.analysis_dir, out_dir=a.out_dir)
    return _each(clips, run, a.all)


def _cmd_calibrate(a):
    from analysis.pose24 import calibrate
    clips = [c.strip() for c in a.clips.split(",") if c.strip()]
    if not clips:
        sys.exit("--clips needs at least one clip")
    calibrate.calibrate_scores(clips, out_path=a.out, data_dir=a.data_dir)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.pose24", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write <clip>_pose24.analysis.json (+ racket sidecars, debug cache)")
    b.add_argument("--clip", action="append")
    b.add_argument("--all", action="store_true")
    b.add_argument("--views", default=None, help="comma-separated camera subset")
    b.add_argument("--out-clip", default=None, help="output clip id (required for a subset)")
    b.add_argument("--timeline-of", default=None, help="reuse this clip's pose24 sync reference + offsets")
    b.add_argument("--skeleton", choices=["smpl-28", "smpl-24"], default="smpl-28")
    b.add_argument("--tracker", choices=["v2", "3d"], default="3d",
                   help="default 3d: SP3 tracker3d follows the synced stream and writes <out_clip>.persons.json; "
                        "v2 (TrackerV2 only, no persons file) is kept for comparison")
    b.add_argument("--reid", choices=["colour", "osnet"], default="colour", help="tracker3d re-id features")
    b.add_argument("--mono", default=None, metavar="VIEW",
                   help="1-camera build of this view from the clip's own pose24 cache (needs --out-clip)")
    b.add_argument("--mono-lengths", choices=["oracle", "prior"], default="oracle",
                   help="--mono limb lengths: the full build's (oracle) or the 1.70 m template (prior)")
    b.add_argument("--mono-prior", default=None, help="--mono: learned depth prior .npz (data/smpl_mono/...)")
    b.add_argument("--fit-passes", type=int, default=2, help="body-fit passes (each re-estimates limb lengths + contacts)")
    b.add_argument("--no-contact", action="store_true", help="no foot-contact term in the fit")
    b.add_argument("--no-limits", action="store_true", help="no joint-limit term in the fit")
    b.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    ls = sub.add_parser("list", help="registry clips and which pose24 outputs exist")
    ls.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    e = sub.add_parser("eval", help="held-out / LOVO / vs-reference / motion metrics -> data/pose_eval/pose24")
    e.add_argument("--clip", action="append")
    e.add_argument("--all", action="store_true")
    e.add_argument("--reference", default=None, help="full build to score a subset against (default: its timeline_of)")
    e.add_argument("--lovo", action="store_true", help="full builds: refit without each camera (slow)")
    e.add_argument("--no-posefit", action="store_true", help="skip the posefit comparison row")
    e.add_argument("--out-dir", default=EVAL_OUT_DIR)
    e.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    c = sub.add_parser("calibrate-scores", help="refit per-group score calibration from pose24 caches")
    c.add_argument("--clips", required=True, help="comma-separated full (>= 3 camera) builds")
    c.add_argument("--out", default=None)
    c.add_argument("--data-dir", default="data", help="kp2d files (the detector's raw scores) under <dir>/pose2d")
    args = ap.parse_args(argv)
    if args.cmd == "calibrate-scores" and args.out is None:
        from analysis.pose24 import scores
        args.out = scores.CALIB_PATH
    return {"build": _cmd_build, "list": _cmd_list, "eval": _cmd_eval,
            "calibrate-scores": _cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
