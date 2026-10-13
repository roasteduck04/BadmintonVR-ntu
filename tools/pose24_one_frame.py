"""Unity file from ONE frame, with the normal pose24 3D build.

The 3D build needs a stretch of frames (camera sync, tracking, 60-frame body-fit windows), so the one frame of a short
2D test (tools/pose2d_parallel.py --max-frames 1 --start-frame F -> data/pose2d_frames) is repeated `--copies` times as
a frozen clip, the unchanged `analysis.pose24 build` runs on that in a scratch folder (your data/pose2d, pose cache and
Unity files are not touched), and only the middle copy is kept: a 1-frame schema 4.1 doc that Unity lists as clip
<name> (default <clip>-f<F>), pose row SMPL-24.

    python tools/pose24_one_frame.py --frames-dir data/pose2d_frames --clip singles_05x
    python tools/pose24_one_frame.py --frames-dir data/pose2d_frames --clip singles_05x \
        --analysis-dir /mnt/c/Users/<you>/BadmintonVR-unity/Assets/StreamingAssets/analysis

Limits (a single moment, not a clip): no motion for camera sync (the cameras are taken as in step, their real offset
is up to ~2 frames), no smoothing over time, and P1 / P2 come from the seed clicks only when F is close to a seed time
(1 s = frame 60, 5 s = frame 300); otherwise the tracker decides from court side and size.
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


def freeze_kp2d(src_dir, views, dst_dir, copies, index=0):
    """<src_dir>/<view>_kp2d.json (a short 2D test) -> <dst_dir>/<view>_kp2d.json with frame `index` repeated
    `copies` times. Returns the video frame number of that frame (the test's start_frame + index)."""
    os.makedirs(dst_dir, exist_ok=True)
    video_frame = None
    for v in views:
        p = os.path.join(src_dir, f"{v}_kp2d.json")
        if not os.path.isfile(p):
            sys.exit(f"ERROR: no 2D file for {v}: {p} (run tools/pose2d_parallel.py --max-frames 1 ... first)")
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if index >= len(d["frames"]):
            sys.exit(f"ERROR: {p} has {len(d['frames'])} frame(s), --index {index} is out of range")
        vf = int(d.get("start_frame", 0)) + index
        if video_frame is None:
            video_frame = vf
        elif vf != video_frame:
            sys.exit(f"ERROR: the cameras' 2D tests start at different frames ({video_frame} vs {vf} in {v})")
        frame = d["frames"][index]
        out = {k: val for k, val in d.items() if k not in ("frames", "start_frame", "video_frame_count")}
        out.update(frames=[frame] * copies, frame_count=copies)
        with open(os.path.join(dst_dir, f"{v}_kp2d.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    return video_frame


def one_frame_doc(doc, keep, name, video_frame, copies):
    """The full frozen-clip doc -> frame `keep` only, renamed to clip `name`."""
    for p in doc["players"]:
        f = dict(p["frames"][keep])
        f.update(frame_id=0, time=0.0)
        p["frames"] = [f]
    meta = doc["meta"]
    meta.update(frame_count=1, source_clip=name)
    meta["label"] = f"{meta.get('label', '')} · 1 frame (video frame {video_frame})"
    meta["one_frame"] = {"video_frame": video_frame, "copies": copies, "kept_copy": keep,
                         "method": "frame repeated as a frozen clip, unchanged pose24 build, middle copy kept"}
    doc["clip_id"] = name
    doc["rallies"] = []
    doc["shuttle"] = {"frames": []}
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unity file from one frame with the normal pose24 build")
    ap.add_argument("--frames-dir", default="data/pose2d_frames", help="the short 2D test's output folder")
    ap.add_argument("--clip", default="singles_05x")
    ap.add_argument("--index", type=int, default=0, help="which frame of the short test (default the first)")
    ap.add_argument("--copies", type=int, default=120, help="frozen-clip length for the build (default 120 = 2 s)")
    ap.add_argument("--name", default=None, help="clip name in Unity (default <clip>-f<video frame>)")
    ap.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR,
                    help="where the Unity file goes (default Assets/StreamingAssets/analysis)")
    ap.add_argument("--data-dir", default="data", help="the project's data folder (calibrations are read from it)")
    ap.add_argument("--keep-scratch", action="store_true", help="keep the scratch build folder (for debugging)")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    cfg = config.resolve(a.clip, data_dir=a.data_dir)
    views = cfg["views"]
    scratch = tempfile.mkdtemp(prefix="pose24_one_frame_", dir=os.path.abspath("."))
    try:
        sdata = os.path.join(scratch, "data")
        shutil.copytree(os.path.join(a.data_dir, "calib"), os.path.join(sdata, "calib"))
        video_frame = freeze_kp2d(a.frames_dir, views, os.path.join(sdata, "pose2d"), a.copies, a.index)
        name = a.name or f"{a.clip}-f{video_frame}"
        print(f"{a.clip}: video frame {video_frame}, {len(views)} cameras, repeated {a.copies}x for the 3D build")
        sout = os.path.join(scratch, "out")
        t_build = time.perf_counter()
        pb.build(a.clip, analysis_dir=sout, cache_dir=os.path.join(scratch, "cache"), data_dir=sdata, log=print)
        build_s = time.perf_counter() - t_build
        with open(os.path.join(sout, f"{a.clip}_pose24.analysis.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        doc = one_frame_doc(doc, a.copies // 2, name, video_frame, a.copies)
        errs = schema24.validate_analysis24(doc)
        if errs:
            sys.exit("ERROR: the 1-frame doc is invalid:\n  " + "\n  ".join(errs[:10]))
        os.makedirs(a.analysis_dir, exist_ok=True)
        out = os.path.join(a.analysis_dir, f"{name}_pose24.analysis.json")
        docmod.write_json(out, doc)
    finally:
        if not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    seen = {p["id"]: p["frames"][0].get("joints_flat") is not None for p in doc["players"]}
    print(f"\nwrote {out}")
    print(f"  Unity clip '{name}', pose row SMPL-24; players with a pose: "
          + ", ".join(f"{pid} {'yes' if ok else 'NO'}" for pid, ok in seen.items()))
    print(f"  3D build {build_s:.1f} s, total {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
