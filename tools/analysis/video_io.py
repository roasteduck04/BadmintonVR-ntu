"""Frame-accurate video input for the CV overlays (SP2 spec § 4.1).

The frame INDEX is the truth: kp2d was produced by sequential cv2.VideoCapture reads, so frame k here is the k-th
decoded frame. Never timestamps, never CAP_PROP_POS_FRAMES (imprecise on HEVC .mov): the reader decodes
sequentially, grab()s over skipped frames and reopens the file to go backwards. Downscaling happens right after
decode. View frame of analysis frame f = floor(f + offset_v + 0.5) (an exact half follows the tracker's raw frame,
see view_frame); the remainder is printed on the tile."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import OrderedDict

import cv2
import numpy as np

from analysis.pose24 import config


class VideoError(RuntimeError):
    pass


def view_frame(f: int, offset: float, det_shift: int | None = None) -> tuple:
    """(floor(f + offset + 0.5), remainder). On an exact half (|remainder| 0.5) both neighbouring raw frames are
    equally far in time: if the tracker's frame f + det_shift is one of them, that one is taken, so the video frame
    and the 2D drawn on it are the same raw frame."""
    x = float(f) + float(offset)
    idx = int(math.floor(x + 0.5))
    if det_shift is not None and abs(abs(x - idx) - 0.5) < 1e-6:
        d = int(f) + int(det_shift)
        if d in (int(round(x - 0.5)), int(round(x + 0.5))):
            idx = d
    return idx, x - idx


def resolve_videos(clip: str, registry: dict | None = None, root: str = ".") -> dict:
    """Every view of the clip's base registry entry (sorted) -> video path (joined to root) or None."""
    reg = registry if registry is not None else config.load_registry()
    if clip not in reg:
        raise KeyError(f"unknown clip {clip!r}; registry has {', '.join(sorted(reg))}")
    base = reg[reg[clip].get("base", clip)]
    vids = base.get("videos") or {}
    return {v: (os.path.join(root, vids[v]) if vids.get(v) else None) for v in sorted(base["views"])}


def _rate(s):
    a, _, b = str(s).partition("/")
    return float(a) / float(b or 1) if float(b or 1) else 0.0


def probe(path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise VideoError(f"ffprobe failed on {path}: {out.stderr.strip()[:200]}")
    s = json.loads(out.stdout)["streams"][0]
    frames = int(s.get("nb_frames") or s.get("nb_read_packets") or 0)
    r, a = _rate(s.get("r_frame_rate", "0/1")), _rate(s.get("avg_frame_rate", "0/1"))
    return {"width": int(s["width"]), "height": int(s["height"]), "frames": frames, "fps": r,
            "vfr": abs(r - a) > 1e-3 * max(r, 1.0), "codec": s.get("codec_name")}


def check_video(view, path, header: dict, log=print) -> dict:
    """The video must match its kp2d header (decoded frame size, frame count); VFR is logged, not fatal."""
    if not path or not os.path.exists(path):
        raise VideoError(f"{view}: video not found ({path})")
    info = probe(path)
    with VideoReader(path) as r:
        fr = r.get(0)
    if fr is None:
        raise VideoError(f"{view}: cannot decode {path}")
    hw = (int(header.get("height") or fr.shape[0]), int(header.get("width") or fr.shape[1]))
    if fr.shape[:2] != hw:
        raise VideoError(f"{view}: decoded frames are {fr.shape[1]}x{fr.shape[0]}, kp2d header says {hw[1]}x{hw[0]} ({path})")
    n = header.get("frame_count")
    if n is not None and info["frames"] != int(n):
        raise VideoError(f"{view}: frame count {info['frames']} but kp2d {int(n)} ({path})")
    if info["vfr"]:
        log(f"    {view}: variable frame rate - using frame index, not timestamps")
    return info


class VideoReader:
    """Sequential, frame-index-exact reader with a small LRU cache (the preview steps back and forth)."""

    def __init__(self, path, scale=1.0, n_frames=None, cache=16):
        self.path, self.scale, self.n_frames, self._cap_n = path, float(scale), n_frames, cache
        self._cap, self._next, self._lru = None, 0, OrderedDict()
        self.decoded, self._t = 0, 0.0

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise VideoError(f"cannot open {self.path}")
        self._next = 0

    def get(self, idx):
        if idx < 0 or (self.n_frames is not None and idx >= self.n_frames):
            return None
        if idx in self._lru:
            self._lru.move_to_end(idx)
            return self._lru[idx].copy()
        t0 = time.perf_counter()
        if self._cap is None or idx < self._next:
            self._open()
        while self._next < idx:
            if not self._cap.grab():
                return None
            self._next += 1
            self.decoded += 1
        ok, fr = self._cap.read()
        if not ok:
            return None
        self._next += 1
        self.decoded += 1
        if self.scale != 1.0:
            fr = cv2.resize(fr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self._lru[idx] = fr
        if len(self._lru) > self._cap_n:
            self._lru.popitem(last=False)
        self._t += time.perf_counter() - t0
        return fr.copy()

    @property
    def fps_achieved(self):
        return self.decoded / self._t if self._t > 0 else float("nan")

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
