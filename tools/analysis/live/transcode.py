"""v0.8.3: smaller copies of the camera videos, to stream them as if the cameras recorded at that size (e.g. 720p).

    python tools/live_run.py transcode --clip <clip> --short-side 720 [--tag 720p] [--views a,b] [--force]

Each video of the clip's live views -> <video folder>_<tag>/<video name>.mp4 (config.variant_video), every frame kept in
order at the same frame rate (so frame N of the copy is frame N of the original and the sync offsets still hold),
shorter side = --short-side pixels, H.264 (libx264 CRF 18; else NVENC; else MPEG-4 at a high bit rate). A sidecar
<copy>.json records the source, size, codec and frame counts. A run streams the copies with --set video_variant=<tag>;
the live config rescales the 2D to the calibration's pixels (config.video_to_calib), so the 4K calibration still holds.
"""
from __future__ import annotations

import json
import os
import time
from fractions import Fraction

CODECS = ("libx264", "h264_nvenc", "mpeg4")


def pick_codec(prefer=None):
    import av
    avail = set(av.codecs_available)
    for c in ([prefer] if prefer and prefer != "auto" else []) + list(CODECS):
        if c in avail:
            return c
    raise RuntimeError(f"no H.264 / MPEG-4 encoder in this PyAV build (have {sorted(avail)[:20]}...)")


def transcode(src, dst, short_side, codec="auto", crf=18, log=print):
    """src video -> dst (mp4) with the shorter side = short_side (never enlarged; even sizes). Returns the sidecar."""
    import av
    from .pose2d_fast import target_size
    codec = pick_codec(codec)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    part = dst + ".part.mp4"
    t = time.perf_counter()
    with av.open(src) as inp:
        ist = inp.streams.video[0]
        ist.thread_type = "AUTO"
        w, h = ist.codec_context.width, ist.codec_context.height
        ow, oh, _ = target_size(w, h, short_side)
        rate = ist.average_rate or ist.guessed_rate or Fraction(60, 1)
        rate = Fraction(rate).limit_denominator(100000)
        n_src = ist.frames or 0
        with av.open(part, "w") as out:
            ost = out.add_stream(codec, rate=rate)
            ost.width, ost.height, ost.pix_fmt = ow, oh, "yuv420p"
            if codec == "libx264":
                ost.options = {"crf": str(int(crf)), "preset": "veryfast"}
            elif codec == "h264_nvenc":
                ost.options = {"rc": "vbr", "cq": str(int(crf) + 1), "preset": "p4"}
            else:
                ost.bit_rate = int(ow * oh * float(rate) * 0.25)
            tb = Fraction(rate.denominator, rate.numerator)
            n = 0
            for fr in inp.decode(ist):
                o = fr.reformat(width=ow, height=oh, format="yuv420p", interpolation="AREA")
                o.pts, o.time_base = n, tb
                for p in ost.encode(o):
                    out.mux(p)
                n += 1
                if n % 600 == 0:
                    log(f"    {os.path.basename(dst)}: {n} frames")
            for p in ost.encode(None):
                out.mux(p)
    os.replace(part, dst)
    side = {"source": src, "source_size": [w, h], "source_frames": n_src, "frames": n, "size": [ow, oh],
            "fps": float(rate), "codec": codec, "crf": crf if codec != "mpeg4" else None,
            "seconds": round(time.perf_counter() - t, 1), "made": time.strftime("%Y-%m-%d %H:%M")}
    with open(dst + ".json", "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=1)
    if n_src and abs(n - n_src) > 1:
        log(f"  WARNING {os.path.basename(dst)}: {n} frames written, the source says {n_src}")
    return side


def frame_count(path):
    import cv2
    c = cv2.VideoCapture(path)
    try:
        return int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        c.release()


def transcode_clip(cfg, short_side, tag=None, views=None, codec="auto", crf=18, force=False, log=print) -> list:
    """Copies of the clip's videos (cfg = a resolved live config on the ORIGINAL videos). Existing copies with a
    matching sidecar are kept unless force."""
    from .config import variant_video
    tag = tag or f"{int(short_side)}p"
    out = []
    for v in views or cfg["views"]:
        src = cfg["videos"][v]
        dst = variant_video(src, tag)
        side_p = dst + ".json"
        if not force and os.path.exists(dst) and os.path.exists(side_p):
            with open(side_p, "r", encoding="utf-8") as fh:
                side = json.load(fh)
            want = min(int(short_side), min(side.get("source_size") or [10 ** 6]))
            if min(side.get("size") or [0, 0]) == want:
                log(f"  {v}: {dst} exists ({side['frames']} frames, {side['size'][0]}x{side['size'][1]}, {side['codec']})")
                out.append({"view": v, "path": dst, **side, "kept": True})
                continue
        log(f"  {v}: {src} -> {dst} (short side {short_side})")
        side = transcode(src, dst, short_side, codec=codec, crf=crf, log=log)
        n = frame_count(dst)
        log(f"  {v}: {side['frames']} frames, {side['size'][0]}x{side['size'][1]}, {side['codec']}, "
            f"{side['seconds']} s; read back {n} frames")
        out.append({"view": v, "path": dst, **side, "read_back_frames": n})
    return out
