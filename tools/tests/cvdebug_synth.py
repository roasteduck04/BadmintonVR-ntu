"""Shared SP2 test helpers: videos with the frame number burned in (16 black/white blocks, robust to
compression) and a synthetic pose24 build (Task 3 adds `built_clip`)."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

BITS = 16


def _block(w):
    return max(4, w // (BITS + 2))


def burn_number(img, k):
    b = _block(img.shape[1])
    for i in range(BITS):
        val = 255 if (k >> i) & 1 else 0
        img[b:2 * b, (i + 1) * b:(i + 2) * b] = val
    return img


def read_number(img) -> int:
    b = _block(img.shape[1])
    g = img if img.ndim == 2 else img.mean(axis=2)
    k = 0
    for i in range(BITS):
        cell = g[b + b // 4:2 * b - b // 4, (i + 1) * b + b // 4:(i + 2) * b - b // 4]
        if cell.mean() > 127:
            k |= 1 << i
    return k


def write_numbered_video(path, n, w, h, codec="MJPG", fps=60.0):
    """n frames of w x h; frame k shows k in the top blocks on a mid-grey background."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if codec == "h264":                       # via ffmpeg (GOP > 1: tests sequential decode of P-frames)
        import subprocess
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p", "-crf", "10", path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in range(n):
            p.stdin.write(burn_number(np.full((h, w, 3), 128, np.uint8), k).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the test video")
        return path
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    for k in range(n):
        vw.write(burn_number(np.full((h, w, 3), 128, np.uint8), k))
    vw.release()
    return path


def built_clip(root, F=150, videos=False, scale_video=1.0):
    """pose24 build of a 4-view synthetic singles clip ("syn") + its 2-view subset ("syn_2cam", views b,c, held-out
    a,d) under root, with view "c" 2 frames late. videos=True writes numbered MJPG videos (full kp2d size x
    scale_video) and puts them in the registry. F = raw frames per view; the built docs are shorter by the largest
    integer sync shift (the tracker trims). F >= 107 or sync silently gives 0 (pose_v2.estimate_offsets needs 10
    samples between its +-30-frame margins); 150 as in test_pose24_pipeline recovers c's +2 within 0.3."""
    import pose24_synth as syn
    from analysis.pose24 import build as pb
    views = ("a", "b", "c", "d")
    players = {"P1": syn.motion(F, x0=-0.5, z0=-3.0), "P2": syn.motion(F, x0=0.5, z0=3.0)}
    entry = syn.write_clip(str(root), "syn", players, syn.rig(views), views, shifts={"c": 2})
    for v in views:                           # real calibration files carry the image size (held-out cameras use it)
        cp = os.path.join(str(root), "data", "calib", f"{v}_cam.json")
        with open(cp, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        c.update(width=syn.W, height=syn.H)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(c, fh)
    if videos:
        w, h = int(syn.W * scale_video), int(syn.H * scale_video)
        for v in views:
            p = os.path.join(str(root), "data", "video", f"{v}.avi")
            write_numbered_video(p, F, w, h)
            entry["videos"][v] = p
    reg = {"syn": entry, "syn_2cam": {"base": "syn", "views": ["b", "c"], "timeline_of": "syn"}}
    adir, cache = os.path.join(str(root), "analysis"), os.path.join(str(root), "cache")
    os.makedirs(adir, exist_ok=True)
    kw = dict(analysis_dir=adir, cache_dir=cache, data_dir=os.path.join(str(root), "data"), registry=reg,
              fit_opts={"passes": 1}, log=lambda *a: None,
              tracker="v2")                   # SP2 fixture: no persons file (tests attach their own); v2 pinned since
                                              # tracker3d became pb.build's default (2026-09-30)
    full = pb.build("syn", **kw)
    sub = pb.build("syn_2cam", **kw)
    return {"root": root, "reg": reg, "kw": kw, "full": full, "sub": sub, "adir": adir,
            "data_dir": kw["data_dir"], "cache_dir": cache, "players": players, "F": F}
