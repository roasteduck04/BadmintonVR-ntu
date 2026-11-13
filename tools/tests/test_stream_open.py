"""v0.8.7 camera reader with damaged-picture detection (tools/analysis/live/stream_open.py) + its users: camera_check,
the calibration frame, the bridge keyframe options, the server's stream settings."""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from analysis.live import stream_open as so  # noqa: E402

av = pytest.importorskip("av")


def _clip(d, n=120, damage=(0.3, 0.6)):
    """A 30 fps H.264 MPEG-TS clip (keyframe every 30 frames) and a copy with two bursts of garbled TS packets - the
    kind of damage lost packets cause. Returns (clean path, damaged path)."""
    src = os.path.join(d, "clip.ts")
    out = av.open(src, "w")
    s = out.add_stream("libx264", rate=30)
    s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
    s.options = {"g": "30", "preset": "ultrafast", "tune": "zerolatency"}
    for i in range(n):
        img = np.zeros((240, 320, 3), np.uint8)
        img[:, :, 0] = (np.arange(320)[None, :] + i * 3) % 256
        img[:, :, 1] = (np.arange(240)[:, None] * 2 + i) % 256
        img[100:140, (i * 2) % 280:(i * 2) % 280 + 40] = 255
        for p in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()
    data = bytearray(open(src, "rb").read())
    rnd = random.Random(1)
    for start in (int(len(data) * f) for f in damage):
        for k in range(start, start + 188 * 6):
            if k % 188 > 4:                              # keep the TS packet headers, garble the payload
                data[k] = rnd.randrange(256)
    bad = os.path.join(d, "bad.ts")
    with open(bad, "wb") as fh:
        fh.write(data)
    return src, bad


def _read_all(path, damaged):
    """Every picture of a file through Stream (it stops at the end instead of opening the file again)."""
    holder = {}

    def log(msg):
        if "opening it again" in msg:
            holder["s"].running = False

    s = so.Stream(path, name="t", damaged=damaged, hold_max_s=2.0, log=log)
    holder["s"] = s
    got = [(flag, img) for img, flag, _ in s.frames()]
    return s, got


# ------------------------------------------------------------------------------------------------- settings
def test_options_defaults_and_checks():
    assert so.options() == {"reader": "pyav", "transport": "tcp", "damaged": "hold", "hold_max_s": 1.0}
    assert so.options({"damaged": "use", "hold_max_s": "0.5", "other": 1})["hold_max_s"] == 0.5
    for bad in ({"reader": "x"}, {"transport": "quic"}, {"damaged": "drop"}, {"hold_max_s": 0}, {"hold_max_s": "a"}):
        with pytest.raises(ValueError):
            so.options(bad)
    assert so.open_options("rtsp://h/x") == {"fflags": "nobuffer", "rtsp_transport": "tcp"}
    assert so.open_options("rtsp://h/x", "auto") == {"fflags": "nobuffer"}
    assert "rtsp_transport" not in so.open_options("http://h/x.mjpg", "udp")


def test_damage_classify_heals_at_a_keyframe_or_after_hold_max():
    d = so.Damage(hold_max_s=1.0)
    assert d.classify(False, True, 0.0) == "clean"
    d.errors = 3                                         # decoder error messages arrived for the next picture
    assert d.classify(False, False, 0.1) == "damaged" and d.errors == 0
    assert d.classify(False, False, 0.2) == "tainted"    # predicted from a damaged picture
    assert d.classify(False, True, 0.3) == "clean"       # keyframe: healed
    assert d.classify(True, False, 0.4) == "damaged"     # PyAV's own corrupt flag
    assert d.classify(False, False, 1.0) == "tainted"
    assert d.classify(False, False, 1.45) == "clean"     # no keyframe, but hold_max_s passed (intra refresh)


# ------------------------------------------------------------------------------------------------- the reader
def test_stream_flags_damaged_pictures_and_holds_the_last_clean_one(tmp_path):
    clean, bad = _clip(str(tmp_path))
    s, got = _read_all(clean, "use")
    assert len(got) >= 115 and {f for f, _ in got} == {"clean"} and s.health()["damaged_pct"] == 0.0

    s, got = _read_all(bad, "use")
    flags = [f for f, _ in got]
    h = s.health()
    assert h["damaged"] >= 1 and h["tainted"] >= 1 and h["errors"] >= 1 and h["damaged_pct"] > 0
    assert flags[0] == "clean" and flags[-1] == "clean"  # healed by the last keyframe

    s, got = _read_all(bad, "hold")
    flags = [f for f, _ in got]
    assert "damaged" not in flags and "tainted" not in flags and "held" in flags
    assert s.counts["held"] == flags.count("held") > 0
    last_clean = None
    for f, img in got:                                   # a held picture IS the last clean one
        if f == "clean":
            last_clean = img
        else:
            assert img is last_clean


def test_grab_clean_returns_a_clean_picture_or_the_error(tmp_path):
    clean, _ = _clip(str(tmp_path), n=40)
    img, info = so.grab_clean(clean, drain_s=0.0)
    assert img is not None and img.shape == (240, 320, 3) and info["damaged"] == 0
    img, info = so.grab_clean(os.path.join(str(tmp_path), "missing.ts"), open_s=2, read_s=2)
    assert img is None and "error" in info


def test_camera_reader_pyav_path_flags_frames(tmp_path):
    from analysis.live import pose2d_fast as pf
    clean, bad = _clip(str(tmp_path))
    logs = []
    rd = pf.CameraReader(bad, 0, 30.0, 0, cam={"reader": "pyav", "damaged": "hold"}, log=logs.append)
    try:
        img = rd.read(0)
        assert img is not None and img.shape[:2] == (240, 320)
        assert rd.flag_of(0) in ("clean", "held")
        assert set(rd.health()) >= {"frames", "damaged", "held", "damaged_pct"}
    finally:
        rd.close()


# ------------------------------------------------------------------------------------------------- feed check
def test_camera_check_damage_numbers_and_verdict():
    import camera_check as cc
    flags = ["clean"] * 80 + ["damaged"] + ["tainted"] * 9 + ["clean"] * 10
    st = cc.stats([i / 60.0 for i in range(100)], [255.0] + [3.0] * 99, 60, flags, {"errors": 7, "reconnects": 0})
    assert st["damaged_pct"] == 10.0 and st["damaged_frames"] == 1 and st["longest_damaged_run"] == 10
    assert st["decoder_errors"] == 7
    v = " | ".join(cc.verdict(st, 60))
    assert "10 % of pictures damaged" in v and "BEFORE this PC" in v
    st = cc.stats([i / 60.0 for i in range(100)], [255.0] + [3.0] * 99, 60, ["clean"] * 100)
    assert st["damaged_pct"] == 0.0 and cc.verdict(st, 60) == ["steady"]
    assert "damaged_pct" not in cc.stats([i / 60.0 for i in range(10)], [3.0] * 10, 60)      # OpenCV: not counted


def test_camera_check_counts_damage_in_a_damaged_file(tmp_path):
    import camera_check as cc
    _, bad = _clip(str(tmp_path))
    w = cc.Watch("v", bad)
    assert w.so is not None
    w.start()
    t_end = time.monotonic() + 20                        # a file ends (a camera would not): stop at its end
    while time.monotonic() < t_end and not (w.stream is not None and w.stream.counts["reconnects"]):
        time.sleep(0.05)
    w.running = False
    w.join(10)
    st = cc.stats(*[w.series()[i] for i in (0, 2)], 30, list(w.flags), w.damage())
    assert st["damaged_frames"] >= 1 and st["damaged_pct"] > 0
    assert any("damaged" in line for line in cc.report_lines("v", w, st, 30))


# ------------------------------------------------------------------------------------------------- calibration
def test_calibration_frame_through_pyav_is_clean(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import calib as calibmod
    import cv2
    _, bad = _clip(str(tmp_path))
    seen = {}

    def fake_grab(url, **kw):
        seen.update(url=url, **kw)
        return np.full((48, 64, 3), 200, np.uint8), {"damaged": 0}

    c = calibmod.Calib(str(tmp_path), str(tmp_path),
                       input_fn=lambda: {"kind": "camera", "stream": {"reader": "pyav", "transport": "udp"}})
    orig = calibmod.stream_open.grab_clean
    calibmod.stream_open.grab_clean = fake_grab
    try:
        c.grab({"view": "unit1", "video": "rtsp://pc:8554/unit1#rotate=180", "has_video": True}, 1.0, "unit1.jpg")
    finally:
        calibmod.stream_open.grab_clean = orig
    assert seen["url"] == "rtsp://pc:8554/unit1" and seen["transport"] == "udp"
    assert cv2.imread(os.path.join(c.frames_dir, "unit1.jpg")).mean() > 150
    assert calibmod.Calib(str(tmp_path), str(tmp_path))._stream_settings() is None      # no saved source: OpenCV
    off = calibmod.Calib(str(tmp_path), str(tmp_path), input_fn=lambda: {"stream": {"reader": "opencv"}})
    assert off._stream_settings() is None


# ------------------------------------------------------------------------------------------------- server page
def test_server_saves_the_camera_reader_settings_and_passes_them_on(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import server as srv
    from fastapi import HTTPException
    s = srv.Server(unity_project=tmp_path, machine="laptop-cached", repo=ROOT)
    assert s.get_input()["stream"] == so.DEFAULTS
    cams = [{"name": "cam1", "url": "rtsp://pi1:8554/cam", "use": True}]
    d = s.set_input({"kind": "camera", "cameras": cams, "stream": {"damaged": "use", "transport": "udp"}})
    assert d["stream"]["damaged"] == "use" and d["stream"]["transport"] == "udp" and d["stream"]["reader"] == "pyav"
    sets = s.input_sets()
    assert "input.damaged=use" in sets and "input.transport=udp" in sets and "input.hold_max_s=1.0" in sets
    assert s.set_input({"kind": "camera", "cameras": cams})["stream"]["damaged"] == "use"     # kept when not sent
    with pytest.raises(HTTPException):
        s.set_input({"kind": "camera", "cameras": cams, "stream": {"damaged": "drop"}})
    from analysis.live import config as live_config
    cfg = live_config.load("singles_05x", overrides=sets + ["backend=live"], probe=lambda c: {
        v: {"width": 64, "height": 48, "fps": 30.0, "frame_count": 10 ** 6} for v in c["views"]})
    assert cfg["input"]["damaged"] == "use" and cfg["input"]["transport"] == "udp"


# ------------------------------------------------------------------------------------------------- bridge
def test_bridge_keyframe_every_half_second_and_intra_refresh():
    import ros_rtsp_bridge as rb
    cmd = rb.ffmpeg_cmd("unit1", fps=60)
    assert cmd[cmd.index("-g") + 1] == "30" and "-intra-refresh" not in cmd
    cmd = rb.ffmpeg_cmd("unit1", fps=30, gop=10, intra_refresh=True)
    assert cmd[cmd.index("-g") + 1] == "10" and cmd[cmd.index("-intra-refresh") + 1] == "1"
    assert cmd.index("-intra-refresh") < cmd.index("-f", cmd.index("-intra-refresh")) and cmd[-1].endswith("/unit1")
    a = rb.parse_args(["--gop", "15", "--intra-refresh"])
    assert a.gop == 15 and a.intra_refresh
