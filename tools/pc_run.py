"""Video -> 2D pose (GPU, all cameras at once) -> pose24 3D build -> the Unity file, on the GPU PC. No Docker.

Run from the project folder with the Python that has tools/requirements-pc.txt installed (Windows Python or the
WSL conda env). Each step is printed, its output is shown live and saved to results/<step>.log, and its time is
added to results/step_times.txt. Stops at the first failed step. Guide: START-HERE.md.

    python tools/pc_run.py                         # check, gpu, pose2d, pose24, collect
    python tools/pc_run.py --step pose24           # one step: check | gpu | pose2d | pose24 | collect
    python tools/pc_run.py --step pose2d --parallel 1 --out data/pose2d_seq     # cameras one after another
    python tools/pc_run.py --dry-run               # print the commands only
    python tools/pc_run.py --max-frames 1 --start-frame 900    # 1-frame test: check, gpu, 2D on frame 900 only
                                                   # (-> data/pose2d_frames; the 3D build needs the whole clip: skipped)
A timing summary of every step (2D per camera, 3D build stages) is printed at the end and saved to
results/timing_summary.txt.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CLIP = "singles_05x"
VIDEO_DIR = os.path.join("data", "15-08", "singles", "0.5")
VIDEOS = ["15.8 back 0.5-1.mov", "15.8 back left 0.5-1.mov", "15.8 back right 0.5-1.mov",
          "15.8 front left 0.5-1.mov", "15.8 front right 0.5-1.mov"]
UNITY_FILES = [f"{CLIP}_pose24.analysis.json", f"{CLIP}.persons.json"]
STEPS = ["check", "gpu", "pose2d", "pose24", "collect"]
PACKAGES = ["cv2", "numpy", "scipy", "networkx", "matplotlib", "threadpoolctl", "onnxruntime", "rtmlib"]


class Runner:
    def __init__(self, results, dry_run=False):
        self.results, self.dry_run = results, dry_run
        os.makedirs(results, exist_ok=True)

    def run(self, name, cmd):
        """Run cmd (list) in the project folder with PYTHONPATH=tools; tee its output to results/<name>.log."""
        print(f"\n===== {name} =====\n> {subprocess.list2cmdline(cmd)}", flush=True)
        if self.dry_run:
            return 0
        env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   PYTHONUNBUFFERED="1")
        log = os.path.join(self.results, f"{name}.log")
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
                for line in p.stdout:
                    sys.stdout.write(line)
                    fh.write(line)
                code = p.wait()
        except FileNotFoundError as e:
            print(f"cannot start: {e}")
            code = 127
        dt = time.time() - t0
        with open(os.path.join(self.results, "step_times.txt"), "a", encoding="utf-8") as fh:
            fh.write(f"{name}\t{dt:.1f} s\texit {code}\t{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        print(f"----- {name}: {dt:.1f} s, exit code {code} (log: {log}) -----", flush=True)
        return code


def stop(code, what):
    if code != 0:
        print(f"\nSTOPPED: {what} failed (exit code {code}). Send the results folder.")
        sys.exit(code or 1)


def step_check(r):
    print("\n===== check: videos =====")
    missing = 0
    for v in VIDEOS:
        p = os.path.join(REPO, VIDEO_DIR, v)
        if os.path.isfile(p):
            print(f"  ok       {v}  ({os.path.getsize(p) // 2**20} MB)")
        else:
            print(f"  MISSING  {p}")
            missing += 1
    stop(missing, f"video check ({missing} missing)")
    print("\n===== check: Python packages =====")
    bad = 0
    for mod in PACKAGES:
        try:
            m = __import__(mod)
            print(f"  ok       {mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"  MISSING  {mod}  ({type(e).__name__}: {e})")
            bad += 1
    if not r.dry_run:
        info = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
                "python": sys.version.split()[0], "python_exe": sys.executable, "cpu": platform.processor(),
                "logical_cores": os.cpu_count()}
        with open(os.path.join(r.results, "pc_info.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
    stop(bad, f"package check ({bad} missing: pip install -r tools/requirements-pc.txt)")
    stop(r.run("check_nvidia_smi", ["nvidia-smi"]), "nvidia-smi (NVIDIA driver)")


def step_gpu(r):
    video = os.path.join(VIDEO_DIR, VIDEOS[0])
    stop(r.run("gpu_check", [sys.executable, os.path.join("tools", "gpu_check.py"), "--video", video]),
         "GPU check (the pose model is not on CUDA)")


def step_pose2d(r, parallel, out, start_frame=0, max_frames=None):
    name = f"pose2d_parallel{parallel}" + ("" if max_frames is None else f"_{max_frames}frames")
    extra = (["--start-frame", str(start_frame)] if start_frame else []) + \
            (["--max-frames", str(max_frames)] if max_frames is not None else [])
    code = r.run(name, [sys.executable, os.path.join("tools", "pose2d_parallel.py"), "--src", VIDEO_DIR,
                        "--clip", CLIP, "--out", out, "--parallel", str(parallel)] + extra)
    if not r.dry_run:
        dst = os.path.join(r.results, name)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(REPO, out)
        summary = os.path.join(src, f"{CLIP}_parallel_timing.json")
        if os.path.isfile(summary):
            shutil.copy2(summary, dst)
        if os.path.isdir(os.path.join(src, "_logs")):
            shutil.copytree(os.path.join(src, "_logs"), os.path.join(dst, "_logs"), dirs_exist_ok=True)
    stop(code, "2D pose")


def step_pose24(r):
    stop(r.run("pose24_build", [sys.executable, "-m", "analysis.pose24", "build", "--clip", CLIP]), "pose24 build")


def latest_step_times(results):
    """results/step_times.txt (appended by every run) -> {step: (seconds, exit code)}, the latest run of each step."""
    out = {}
    path = os.path.join(results, "step_times.txt")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[0]] = (float(parts[1].split()[0]), parts[2])
    return out


def pose24_stages(log_text):
    """'runtime track 259s, lrswap 12s, fit 452s, regress_confidence 2s, total 726s' -> [(stage, seconds), ...]."""
    m = re.findall(r"runtime (.+?), total ([\d.]+)s", log_text)
    if not m:
        return []
    stages = [(k, float(v)) for k, v in re.findall(r"(\w+) ([\d.]+)s", m[-1][0])]
    return stages + [("total (build's own clock)", float(m[-1][1]))]


def timing_summary(results):
    """Text summary of the latest run of every step; also written to results/timing_summary.txt."""
    steps = latest_step_times(results)
    if not steps:
        return ""
    lines = ["===== timing summary =====", f"{'step':<34}{'seconds':>10}  exit"]
    for name, (sec, code) in steps.items():
        lines.append(f"{name:<34}{sec:>10.1f}  {code.replace('exit ', '')}")
        if name.startswith("pose2d"):
            tj = os.path.join(results, name, f"{CLIP}_parallel_timing.json")
            if os.path.isfile(tj):
                with open(tj, encoding="utf-8") as fh:
                    t = json.load(fh)
                if t.get("ms_per_camera_frame") is not None:
                    lines.append(f"    {t['n_views']} cameras, {t['parallel']} at a time, {t['frames_per_view']} frames "
                                 f"each: {t['ms_per_camera_frame']} ms per camera frame (incl. start-up)")
                if t.get("model_ms_per_frame_mean") is not None:
                    lines.append(f"    model + tracking only: {t['model_ms_per_frame_mean']} ms per frame per camera")
                for v, i in t.get("views", {}).items():
                    load = i.get("model_load_s")
                    load_txt = "-" if load is None else f"{load:.1f} s"
                    lines.append(f"    {v:<16} total {i['wall_s']:>8.1f} s   model load {load_txt:>7}"
                                 f"   frames {i.get('frames', '-')}   fps {i.get('fps', '-')}")
                    if i.get("frame_ms") and len(i["frame_ms"]) <= 20:
                        lines.append(f"    {'':<16} per-frame ms {i['frame_ms']}")
        if name == "pose24_build":
            log = os.path.join(results, "pose24_build.log")
            if os.path.isfile(log):
                with open(log, encoding="utf-8", errors="replace") as fh:
                    for stage, sec2 in pose24_stages(fh.read()):
                        lines.append(f"    {stage:<30}{sec2:>10.1f}")
    total = sum(sec for sec, _ in steps.values())
    lines.append(f"{'all steps':<34}{total:>10.1f}  ({total / 60:.1f} min)")
    text = "\n".join(lines)
    with open(os.path.join(results, "timing_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def step_collect(r):
    if r.dry_run:
        return
    dst = os.path.join(r.results, "unity_files")
    os.makedirs(dst, exist_ok=True)
    for f in UNITY_FILES:
        p = os.path.join(REPO, "Assets", "StreamingAssets", "analysis", f)
        if os.path.isfile(p):
            shutil.copy2(p, dst)
            print(f"  copied {f}")
        else:
            print(f"  not found: {p}")
    print(f"\nResults are in: {r.results}  -> zip this folder and upload it to OneDrive.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=["all"] + STEPS, default="all")
    ap.add_argument("--parallel", type=int, default=5, help="cameras at a time in the 2D step (default 5)")
    ap.add_argument("--out", default=None,
                    help="2D output folder (default data/pose2d, which the 3D build reads; data/pose2d_frames with "
                         "--max-frames / --start-frame)")
    ap.add_argument("--start-frame", type=int, default=0, help="2D step: first frame (short tests; default 0)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="2D step: frames per camera (short tests, e.g. 1); the 3D build is skipped (needs the whole clip)")
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    short = a.max_frames is not None or a.start_frame > 0
    out = a.out or ("data/pose2d_frames" if short else "data/pose2d")
    r = Runner(a.results, a.dry_run)
    todo = STEPS if a.step == "all" else [a.step]
    if short and "pose24" in todo:
        print("note: --max-frames / --start-frame is a 2D timing test; the 3D build needs the whole clip, so it is skipped")
        todo = [x for x in todo if x != "pose24"]
    for s in todo:
        if s == "check":
            step_check(r)
        elif s == "gpu":
            step_gpu(r)
        elif s == "pose2d":
            step_pose2d(r, max(1, a.parallel), out, a.start_frame, a.max_frames)
        elif s == "pose24":
            step_pose24(r)
        elif s == "collect":
            step_collect(r)
    if not a.dry_run:
        text = timing_summary(a.results)
        if text:
            print("\n" + text + f"\n(saved to {os.path.join(a.results, 'timing_summary.txt')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
