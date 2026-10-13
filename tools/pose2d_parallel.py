"""Run the 2D pose step (RTMPose Wholebody, GPU) on every camera of a clip at the same time.

Each camera view is its own process of tools/cost/pose2d_gpu_timed.py (same model and the same kp2d output as
_archive/phase4_gpu/pose2d_gpu.py, plus timing), `--parallel` of them at once on the one GPU. Writes
<out>/<view>_kp2d.json per view, logs in <out>/_logs/, and <out>/<clip>_parallel_timing.json with the wall time and
per-camera numbers.

    python tools/pose2d_parallel.py --src data/15-08/singles/0.5 --clip singles_05x --out data/pose2d
    python tools/pose2d_parallel.py ... --parallel 1        # one camera after another, for comparison
    python tools/pose2d_parallel.py ... --views back_051,frontleft_051
    python tools/pose2d_parallel.py ... --max-frames 1 --start-frame 900 --out data/pose2d_frames   # 1-frame test

On the GPU PC run it inside the Docker `tools` service (docs/setup/pose24-remote-pc-quickstart.md).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TIMED = os.path.join(HERE, "cost", "pose2d_gpu_timed.py")


def clip_cameras():
    spec = importlib.util.spec_from_file_location("pose2d_gpu_timed", TIMED)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # module level imports only stdlib (the GPU stack loads in main())
    return m.CLIP_CAMERAS


def select_views(cams, views):
    """cams {view: filename}; views "a,b" or None -> the chosen subset in the clip's own order."""
    if not views:
        return dict(cams)
    want = [v.strip() for v in views.split(",") if v.strip()]
    unknown = [v for v in want if v not in cams]
    if unknown:
        sys.exit(f"ERROR: unknown view(s) {unknown}; this clip has {list(cams)}")
    return {k: v for k, v in cams.items() if k in want}


def run(src, clip, out, cams, parallel=5, worker=TIMED, python=sys.executable, start_frame=0, max_frames=None):
    """Run one worker process per view, `parallel` at a time; returns (and writes) the summary dict."""
    for view, fn in cams.items():
        if not os.path.isfile(os.path.join(src, fn)):
            sys.exit(f"ERROR: video not found for {view}: {os.path.join(src, fn)}")
    logs = os.path.join(out, "_logs")
    os.makedirs(logs, exist_ok=True)
    extra = (["--start-frame", str(start_frame)] if start_frame else []) + \
            (["--max-frames", str(max_frames)] if max_frames is not None else [])
    queue, running, views = list(cams), {}, {}
    t0 = time.time()
    while queue or running:
        while queue and len(running) < parallel:
            view = queue.pop(0)
            log = open(os.path.join(logs, f"{clip}_{view}.log"), "w")
            p = subprocess.Popen([python, worker, "--src", src, "--clip", clip, "--out", out, "--views", view] + extra,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[view] = (p, log, time.time())
            print(f"  start {view}", flush=True)
        for view, (p, log, ts) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            te = time.time()
            info = {"start_s": round(ts - t0, 2), "wall_s": round(te - ts, 2), "returncode": p.returncode,
                    "log": log.name}
            tpath = os.path.join(out, f"{clip}_{view}_gpu_timing.json")
            if p.returncode == 0 and os.path.isfile(tpath):
                with open(tpath) as fh:
                    t = json.load(fh)["views"].get(view, {})
                info.update(loop_s=t.get("seconds"), frames=t.get("frames"), fps=t.get("fps_processed"),
                            model_load_s=t.get("model_load_s"), frame_ms=t.get("frame_ms"))
            views[view] = info
            del running[view]
            state = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode}, see {log.name})"
            print(f"  done  {view}: {te - ts:.1f} s {state}", flush=True)
        time.sleep(0.2)
    wall = time.time() - t0
    frames = [v.get("frames") for v in views.values() if v.get("frames")]
    per_view = max(frames) if frames else None
    loops = [v["loop_s"] / v["frames"] for v in views.values() if v.get("loop_s") and v.get("frames")]
    summary = {"clip": clip, "parallel": parallel, "n_views": len(cams), "wall_s": round(wall, 2),
               "start_frame": start_frame, "max_frames": max_frames,
               "model_ms_per_frame_mean": round(1000 * sum(loops) / len(loops), 1) if loops else None,
               "frames_per_view": per_view,
               "ms_per_camera_frame": round(1000 * wall / (per_view * len(cams)), 2) if per_view else None,
               "ms_per_synced_frame": round(1000 * wall / per_view, 2) if per_view else None,
               "ok": all(v["returncode"] == 0 for v in views.values()), "views": views}
    with open(os.path.join(out, f"{clip}_parallel_timing.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def print_summary(s, out):
    print(f"\n{s['clip']}: {s['n_views']} cameras, {s['parallel']} at a time -> {s['wall_s']:.1f} s total")
    if s["frames_per_view"]:
        print(f"  {s['frames_per_view']} frames per camera: {s['ms_per_camera_frame']} ms per camera frame, "
              f"{s['ms_per_synced_frame']} ms per {s['n_views']}-camera frame")
        if s.get("model_ms_per_frame_mean") is not None:
            print(f"  model + tracking only (no start-up / loading): {s['model_ms_per_frame_mean']} ms per frame per camera")
    print(f"  {'view':<16} {'start s':>8} {'total s':>8} {'load s':>7} {'frames':>7} {'fps':>6}")
    for v, i in s["views"].items():
        load = i.get("model_load_s")
        print(f"  {v:<16} {i['start_s']:>8} {i['wall_s']:>8} {('-' if load is None else round(load, 1)):>7} "
              f"{str(i.get('frames', '-')):>7} {str(i.get('fps', '-')):>6}{'' if i['returncode'] == 0 else '  FAILED'}")
        if i.get("frame_ms") and len(i["frame_ms"]) <= 20:
            print(f"  {'':<16} per-frame ms: {i['frame_ms']}")
    print(f"  summary: {os.path.join(out, s['clip'] + '_parallel_timing.json')}")


def main():
    ap = argparse.ArgumentParser(description="2D pose (GPU) on every camera of a clip at once")
    ap.add_argument("--src", required=True, help="folder with the clip's .mov files")
    ap.add_argument("--clip", required=True, help="clip id, e.g. singles_05x")
    ap.add_argument("--out", default="data/pose2d", help="output folder for <view>_kp2d.json")
    ap.add_argument("--parallel", type=int, default=5, help="cameras at a time (default 5; 1 = one after another)")
    ap.add_argument("--views", default=None, help="comma list of views (default: every camera of the clip)")
    ap.add_argument("--start-frame", type=int, default=0, help="first frame (short tests; default 0)")
    ap.add_argument("--max-frames", type=int, default=None, help="frames per camera (short tests; default: whole clip)")
    a = ap.parse_args()
    cams = clip_cameras()
    if a.clip not in cams:
        sys.exit(f"ERROR: unknown clip {a.clip}; known: {list(cams)}")
    cams = select_views(cams[a.clip], a.views)
    os.makedirs(a.out, exist_ok=True)
    last = "end" if a.max_frames is None else a.start_frame + a.max_frames - 1
    span = "" if a.max_frames is None and not a.start_frame else f", frames {a.start_frame}..{last}"
    print(f"{a.clip}: {len(cams)} cameras, {a.parallel} at a time{span}")
    s = run(a.src, a.clip, a.out, cams, parallel=max(1, a.parallel), start_frame=a.start_frame, max_frames=a.max_frames)
    print_summary(s, a.out)
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
