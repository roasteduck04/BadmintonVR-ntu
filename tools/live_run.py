"""v0.8.0 live-frames: run the simulated-live pose pipeline and report on it. Run from the repository root.

    python tools/live_run.py run --clip singles_05x --preset laptop-cached [--set output_fps=60 --set pace=asfast]
    python tools/live_run.py run --clip singles_05x --preset rtx3060 --tag test1
    python tools/live_run.py sweep --clip singles_05x --preset laptop-cached --grid output_fps=30,60 pace=realtime,asfast
    python tools/live_run.py report runs/<run folder>
    python tools/live_run.py compare runs/<a> runs/<b> ... [--out runs/compare_<name>]
    python tools/live_run.py machine
    python tools/live_run.py sync-file --clip singles_05x --from-doc Assets/StreamingAssets/analysis/singles_05x_pose24.analysis.json
    python tools/live_run.py replay runs/<run folder> [--speed 1] [--out runs/replay_<name>]   # re-write a finished
                                                     run's frame files on its clock (test the Unity live mode, no GPU)
    python tools/live_run.py bench2d --clip singles_05x --preset rtx3060 [--set rtmpose.engine=fast ...] [--seconds 10]
                                                     # v0.8.1: the live 2D step alone, as fast as possible + quality vs
                                                     # the offline 2D (runs/bench2d/<date>_<host>_<tag>/bench.md)
    python tools/live_run.py bench2d-compare runs/bench2d/<a> runs/bench2d/<b> ... [--out runs/bench2d/compare.md]

--clip is a file in configs/live/ (or a path), --preset one in configs/live/presets/ (or a path); every key of
configs/live/defaults.yaml can be changed with --set key.sub=value (repeatable). --dry-run prints the resolved config.
Each run gets its own folder runs/live_<clip>_<fps>fps_<stored2d|live2d>_<pace>_<date>_<time>[_tag]/ (config, machine, timing, samples, frames, final doc,
report)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from analysis.live import config as live_config  # noqa: E402


class Tee:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run_dir_for(cfg, tag=None):
    """runs/live_<clip>_<fps>fps_<stored|live>2d_<pace>_<date>_<time>[_tag]: clip first, so a folder of runs sorts
    by clip; the machine is in the run's machine.json and report."""
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    src = "stored2d" if cfg["backend"] == "cached" else "live2d"
    parts = ["live", cfg["clip"], f"{cfg['fps']:g}fps", src, cfg["pace"], stamp] + ([tag] if tag else [])
    return os.path.join(cfg["runs_dir"], "_".join(str(p).replace(" ", "-") for p in parts))


def cmd_run(a, overrides=None):
    from analysis.live import runner
    cfg = live_config.load(a.clip, preset=a.preset, overrides=list(a.set or []) + list(overrides or []))
    if a.dry_run:
        import yaml
        print(yaml.safe_dump(json.loads(json.dumps(live_config.public(cfg), default=str)), sort_keys=False))
        return None
    rd = a.run_dir or run_dir_for(cfg, a.tag)
    os.makedirs(rd, exist_ok=True)
    log = Tee(os.path.join(rd, "run.log"))
    log(f"live_run: {' '.join(sys.argv[1:])}")
    log(f"run folder {rd}")
    with open(os.path.join(rd, "command.json"), "w", encoding="utf-8") as fh:
        json.dump({"argv": sys.argv, "cwd": os.getcwd(), "overrides": list(a.set or []) + list(overrides or [])}, fh,
                  indent=1)
    t = time.perf_counter()
    res = runner.Runner(cfg, rd, log=log).run()
    log(f"run done in {time.perf_counter() - t:.1f} s: {res['written']}/{res['steps']} frames")
    if not a.no_report:
        from analysis.live import report
        p = report.render_run(rd, log=log)
        log(f"report {p}")
    return rd


def cmd_sweep(a):
    grid = []
    for g in a.grid:
        k, vals = g.split("=", 1)
        grid.append([f"{k}={v}" for v in vals.split(",")])
    runs = []
    for combo in itertools.product(*grid):
        print(f"=== sweep: {' '.join(combo)}", flush=True)
        a.run_dir = None
        runs.append(cmd_run(a, overrides=combo))
    if not a.dry_run and len(runs) > 1:
        from analysis.live import report
        out = os.path.join(os.path.dirname(runs[0]), f"compare_{time.strftime('%Y-%m-%d_%H%M%S')}")
        print(f"compare report {report.render_compare(runs, out)}")


def cmd_report(a):
    from analysis.live import report
    for rd in a.runs:
        print(report.render_run(rd))


def cmd_compare(a):
    from analysis.live import report
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.runs[0])), f"compare_{time.strftime('%Y-%m-%d_%H%M%S')}")
    print(report.render_compare(a.runs, out))


def cmd_machine(a):
    from analysis.live import machine
    m = machine.report()
    print(json.dumps(m, indent=1))
    print(machine.short(m))


def cmd_sync_file(a):
    cfg = live_config.load(a.clip, preset=a.preset, overrides=list(a.set or []) + ["sync.offsets=zero"])
    with open(a.from_doc, "r", encoding="utf-8") as fh:
        pvar = json.load(fh)["meta"]["pose_variant"]
    offs = {v: float(o) for v, o in pvar["offsets"].items()}
    out = a.out or cfg["paths"]["sync_file"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"clip": cfg["clip"], "time_ref": pvar.get("time_ref_view"), "offsets": offs,
                   "source": os.path.basename(a.from_doc), "made": time.strftime("%Y-%m-%d %H:%M")}, fh, indent=1)
    print(f"wrote {out}: {offs}")


def cmd_replay(a):
    """Copy a finished run into a new folder frame by frame on the original output clock (frame times), atomically,
    like the runner writes them; the final analysis doc is copied last (the viewer's "run finished" signal)."""
    import glob
    import shutil
    from analysis.live import framewriter as fw
    src = fw.frames_dir(a.run)
    if src is None:
        sys.exit(f"no {fw.FRAMES_DIR}/manifest.json (or live/manifest.json) in {a.run}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.run)), "replay_" + os.path.basename(os.path.normpath(a.run)))
    if os.path.exists(out):
        sys.exit(f"{out} exists (remove it or pass --out)")
    dst_dir = os.path.join(out, fw.FRAMES_DIR)
    os.makedirs(dst_dir)
    shutil.copyfile(os.path.join(src, "manifest.json"), os.path.join(dst_dir, "manifest.json"))
    frames = fw.frame_files(a.run)
    times = []
    for f in frames:
        with open(f, "r", encoding="utf-8") as fh:
            times.append(float(json.load(fh)["time"]))
    print(f"replaying {len(frames)} frames into {out} at {a.speed:g}x (Unity: Live run > connect to this folder)")
    t0, start = time.perf_counter(), times[0] if times else 0.0
    for f, t in zip(frames, times):
        wait = (t - start) / max(a.speed, 1e-6) - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        step = int(os.path.basename(f).replace("frame_", "")[:6])
        name = fw.FRAME_NAME.format(step=step)
        dst = os.path.join(dst_dir, name)
        shutil.copyfile(f, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        with open(os.path.join(dst_dir, "latest.json.tmp"), "w", encoding="utf-8") as fh:
            json.dump({"last_step": step, "frame": name}, fh)
        os.replace(os.path.join(dst_dir, "latest.json.tmp"), os.path.join(dst_dir, "latest.json"))
    for d in glob.glob(os.path.join(a.run, "*.analysis.json")):
        shutil.copyfile(d, os.path.join(out, os.path.basename(d)))
    print(f"done in {time.perf_counter() - t0:.1f} s")


def cmd_bench2d(a):
    from analysis.live import bench2d
    cfg = live_config.load(a.clip, preset=a.preset, overrides=list(a.set or []))
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    host = socket.gethostname().split(".")[0]
    rd = a.run_dir or os.path.join(cfg["runs_dir"], "bench2d", "_".join([stamp, host] + ([a.tag] if a.tag else [])))
    os.makedirs(rd, exist_ok=True)
    log = Tee(os.path.join(rd, "bench.log"))
    log(f"live_run: {' '.join(sys.argv[1:])}")
    with open(os.path.join(rd, "command.json"), "w", encoding="utf-8") as fh:
        json.dump({"argv": sys.argv, "cwd": os.getcwd(), "overrides": list(a.set or [])}, fh, indent=1)
    bench2d.run(cfg, rd, seconds=a.seconds, log=log, quality=not a.no_quality)
    log(f"bench folder {rd}")
    return rd


def cmd_bench2d_compare(a):
    from analysis.live import bench2d
    print(bench2d.compare(a.runs, out=a.out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python tools/live_run.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--clip", required=True, help="configs/live/<clip>.yaml (name or path)")
        p.add_argument("--preset", help="configs/live/presets/<name>.yaml (name or path)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config key (repeatable)")
        p.add_argument("--tag", help="suffix of the run folder name")
        p.add_argument("--run-dir", help="exact run folder (default runs/live_<clip>_<fps>fps_...)")
        p.add_argument("--dry-run", action="store_true", help="print the resolved config and stop")
        p.add_argument("--no-report", action="store_true", help="skip the report at the end")

    p = sub.add_parser("run", help="one run")
    common(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("sweep", help="every combination of --grid values, then a compare report")
    common(p)
    p.add_argument("--grid", nargs="+", required=True, metavar="KEY=V1,V2")
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("report", help="(re)render the report of run folders")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("compare", help="one report across run folders")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("machine", help="machine report (CPU, RAM, GPU, drivers, onnxruntime providers)")
    p.set_defaults(fn=cmd_machine)
    p = sub.add_parser("replay", help="re-write a finished run's frames on its clock (test the Unity live mode)")
    p.add_argument("run")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bench2d", help="the live 2D step alone: speed per stage, GPU / CPU, quality vs the offline 2D")
    p.add_argument("--clip", required=True)
    p.add_argument("--preset")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--seconds", type=float, help="only the first N seconds of the section")
    p.add_argument("--tag")
    p.add_argument("--run-dir")
    p.add_argument("--no-quality", action="store_true")
    p.set_defaults(fn=cmd_bench2d)
    p = sub.add_parser("bench2d-compare", help="one table across bench2d folders (first = baseline)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bench2d_compare)
    p = sub.add_parser("sync-file", help="write data/calib/<clip>_sync.json from an analysis doc's offsets")
    p.add_argument("--clip", required=True)
    p.add_argument("--preset")
    p.add_argument("--set", action="append")
    p.add_argument("--from-doc", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_sync_file)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
