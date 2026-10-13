"""Unity file from ONE frame or a SHORT clip (e.g. the first 30 frames), with the normal pose24 3D build.

The 3D build needs a stretch of frames: camera sync searches offsets of up to +-30 frames, short tracks stay "unknown"
(neither player nor bystander), and the body fit works in 60-frame windows. So the frames of a short 2D test
(tools/pose2d_parallel.py --max-frames N --start-frame F -> data/pose2d_frames) are padded to `--length` frames by
holding the first frame before them and the last frame after them, the unchanged `analysis.pose24 build` runs on that
in a scratch folder (your data/pose2d, pose cache and Unity files are not touched), and only the real frames are kept:
a schema 4.1 doc that Unity lists as clip <name> (default <clip>-f<F> for 1 frame, <clip>-f<F>-<F+N-1> for N), pose row
SMPL-24.

    python tools/pose24_short.py --frames-dir data/pose2d_frames --clip singles_05x
    python tools/pose24_short.py --frames-dir data/pose2d_frames --clip singles_05x \
        --analysis-dir /mnt/c/Users/<you>/BadmintonVR-unity/Assets/StreamingAssets/analysis

Limits: the held frames carry no motion, so the camera sync rests on the real frames only (the cameras' real offset
is up to ~2 frames); no smoothing beyond the clip; P1 / P2 come from the seed clicks only when the clip covers a seed
time (1 s = frame 60, 5 s = frame 300), otherwise the tracker decides from court side and size.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from analysis.pose24 import build as pb      # noqa: E402
from analysis.pose24 import config           # noqa: E402
from analysis.pose24 import doc as docmod    # noqa: E402
from analysis import schema24                # noqa: E402


def pad_counts(n, length):
    """n real frames padded to `length` (at least n): (frames held before, frames held after)."""
    extra = max(length - n, 0)
    return extra // 2, extra - extra // 2


def padded_kp2d(src_dir, views, dst_dir, length):
    """<src_dir>/<view>_kp2d.json (a short 2D test, N frames) -> <dst_dir>/<view>_kp2d.json padded to `length`
    frames (first frame held before, last frame held after). Returns (first video frame, N, frames held before)."""
    os.makedirs(dst_dir, exist_ok=True)
    first = n = None
    for v in views:
        p = os.path.join(src_dir, f"{v}_kp2d.json")
        if not os.path.isfile(p):
            sys.exit(f"ERROR: no 2D file for {v}: {p} (run tools/pose2d_parallel.py --max-frames N ... first)")
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        frames = d["frames"]
        if not frames:
            sys.exit(f"ERROR: {p} has no frames")
        start = int(d.get("start_frame", 0))
        if first is None:
            first, n = start, len(frames)
        elif (start, len(frames)) != (first, n):
            sys.exit(f"ERROR: the cameras' 2D tests differ: frames {first}..{first + n - 1} vs "
                     f"{start}..{start + len(frames) - 1} in {v} (run them with the same --start-frame / --max-frames)")
        before, after = pad_counts(n, length)
        out = {k: val for k, val in d.items() if k not in ("frames", "start_frame", "video_frame_count")}
        out["frames"] = [frames[0]] * before + frames + [frames[-1]] * after
        out["frame_count"] = len(out["frames"])
        with open(os.path.join(dst_dir, f"{v}_kp2d.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    return first, n, pad_counts(n, length)[0]


def trim_doc(doc, keep_from, n, name, first_frame, length):
    """The padded doc -> its n real frames only (renumbered 0..n-1), renamed to clip `name`."""
    fps = float(doc["meta"]["fps"])
    for p in doc["players"]:
        kept = []
        for i, f in enumerate(p["frames"][keep_from:keep_from + n]):
            f = dict(f)
            f.update(frame_id=i, time=round(i / fps, 4))
            kept.append(f)
        p["frames"] = kept
    meta = doc["meta"]
    span = f"video frame {first_frame}" if n == 1 else f"video frames {first_frame}-{first_frame + n - 1}"
    meta.update(frame_count=n, source_clip=name)
    meta["label"] = f"{meta.get('label', '')} · {span}"
    meta["short_clip"] = {"first_video_frame": first_frame, "frames": n, "padded_to": length, "kept_from": keep_from,
                          "method": "first / last frame held to pad the clip, unchanged pose24 build, real frames kept"}
    doc["clip_id"] = name
    doc["rallies"] = []
    doc["shuttle"] = {"frames": []}
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unity file from one frame or a short clip with the normal pose24 build")
    ap.add_argument("--frames-dir", default="data/pose2d_frames", help="the short 2D test's output folder")
    ap.add_argument("--clip", default="singles_05x")
    ap.add_argument("--length", type=int, default=120,
                    help="pad short clips to this many frames for the build (default 120 = 2 s at 60 fps)")
    ap.add_argument("--name", default=None, help="clip name in Unity (default <clip>-f<first>[-<last>])")
    ap.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR,
                    help="where the Unity file goes (default Assets/StreamingAssets/analysis)")
    ap.add_argument("--data-dir", default="data", help="the project's data folder (calibrations are read from it)")
    ap.add_argument("--keep-scratch", action="store_true", help="keep the scratch build folder (for debugging)")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    views = config.resolve(a.clip, data_dir=a.data_dir)["views"]
    scratch = tempfile.mkdtemp(prefix="pose24_short_", dir=os.path.abspath("."))
    try:
        sdata = os.path.join(scratch, "data")
        shutil.copytree(os.path.join(a.data_dir, "calib"), os.path.join(sdata, "calib"))
        first, n, before = padded_kp2d(a.frames_dir, views, os.path.join(sdata, "pose2d"), a.length)
        length = max(a.length, n)
        name = a.name or (f"{a.clip}-f{first}" if n == 1 else f"{a.clip}-f{first}-{first + n - 1}")
        print(f"{a.clip}: {n} frame(s) from video frame {first}, {len(views)} cameras, padded to {length} "
              f"frames for the 3D build ({before} held before, {length - n - before} after)")
        sout = os.path.join(scratch, "out")
        t_build = time.perf_counter()
        pb.build(a.clip, analysis_dir=sout, cache_dir=os.path.join(scratch, "cache"), data_dir=sdata, log=print)
        build_s = time.perf_counter() - t_build
        with open(os.path.join(sout, f"{a.clip}_pose24.analysis.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        doc = trim_doc(doc, before, n, name, first, length)
        errs = schema24.validate_analysis24(doc)
        if errs:
            sys.exit("ERROR: the doc is invalid:\n  " + "\n  ".join(errs[:10]))
        os.makedirs(a.analysis_dir, exist_ok=True)
        out = os.path.join(a.analysis_dir, f"{name}_pose24.analysis.json")
        docmod.write_json(out, doc)
    finally:
        if not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nwrote {out}")
    for p in doc["players"]:
        seen = sum(1 for f in p["frames"] if f.get("joints_flat") is not None)
        print(f"  {p['id']}: pose in {seen}/{n} frames")
    print(f"  Unity clip '{name}', pose row SMPL-24; 3D build {build_s:.1f} s, total {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
