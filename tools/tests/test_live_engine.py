# tools/tests/test_live_engine.py
"""v0.8.4 live engine: warm state in the stream core, early stop, source sessions, the engine command loop, camera
input config, the camera reader against the fake cameras, and the server's Source tab / engine endpoints."""
import io
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

import live_synth as ls
from analysis.live import config as live_config
from analysis.live import engine as eng
from analysis.live import outputs
from analysis.live.stream_pose24 import StreamPose24, model_from_dict, model_to_dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _core_run(n, warm_state=None, stop_after=None):
    cams, players = ls.scene(F=2 * n + 4)
    cfg = ls.cfg(n_steps=n)
    core = StreamPose24(cfg, cams, warm_state=warm_state)
    outs = []
    for i, r in enumerate(cfg["ref_frames"]):
        if stop_after is not None and i == stop_after:
            core.stop_at(i)
            break
        outs += core.push(ls.frameset(i, r, cams, players, offsets=cfg["offsets"]))
    outs += core.flush()
    state = core.export_warm_state()
    core.close()
    return cfg, outs, state


def test_warm_state_round_trip_and_warm_start():
    cfg, outs, state = _core_run(80)
    assert state["models"]["P1"] and state["models"]["P2"] and state["ankle_h_m"] > 0
    m = model_from_dict(json.loads(json.dumps(state["models"]["P1"])))          # survives JSON
    assert np.allclose(np.asarray(m.rest), np.asarray(state["models"]["P1"]["rest"]))
    assert model_to_dict(m)["lengths"] == state["models"]["P1"]["lengths"]
    # a warm start: no warm-up frames, every frame from the fit
    _, outs2, _ = _core_run(80, warm_state=json.loads(json.dumps(state)))
    assert len(outs2) == 80 and all(o["quality"] == outputs.FITTED for o in outs2[2:])
    assert not any(o["quality"] == outputs.WARMUP for o in outs2)


def test_stop_at_publishes_only_the_steps_received():
    _, outs, _ = _core_run(80, stop_after=50)
    assert [o["step"] for o in outs] == list(range(50))


def test_cached_source_sessions():
    from analysis.live.sources import CachedSource
    src = CachedSource.__new__(CachedSource)
    src.cfg, src.frames, src.t0, src.fps, src.realtime, src.origins = {}, {"a": [[]] * 5}, None, 60.0, True, {}
    src.open()
    src.session({"a": [1, 2]}, {"a": 1.0}, realtime=False)
    src.begin(time.monotonic_ns())
    dets, tim = src.get("a", 2)
    assert dets == [] and src.realtime is False
    src.end_session()
    src.session({"a": [3]}, {"a": 3.0}, realtime=True)
    assert src.origins == {"a": 3.0} and src.t0 is None


def test_warm_keys_separate_model_and_player_settings():
    base = live_config.load("singles_05x", preset="rtx5090", probe=_probe())
    same = live_config.load("singles_05x", preset="rtx5090", overrides=["stream.window_s=0.5", "pace=asfast"],
                            probe=_probe())
    assert eng.warm_key(base) == eng.warm_key(same) and eng.player_key(base) == eng.player_key(same)
    det = live_config.load("singles_05x", preset="rtx5090", overrides=["rtmpose.det_every=3"], probe=_probe())
    assert eng.warm_key(det) != eng.warm_key(base)
    wu = live_config.load("singles_05x", preset="rtx5090", overrides=["warmup.seconds=0.5"], probe=_probe())
    assert eng.warm_key(wu) == eng.warm_key(base) and eng.player_key(wu) != eng.player_key(base)


def _probe():
    def probe(c):
        out = {}
        for v in c["views"]:
            w, h = live_config.calib_size(c["cam_paths"][v]) or (2160, 3840)
            out[v] = {"fps": 60.0, "frame_count": 1877, "width": w, "height": h, "source": "x"}
        return out
    return probe


def test_camera_input_config():
    cams = {v: f"http://pi{i}:8080/stream.mjpg" for i, v in enumerate(("back_051", "backright_051", "frontleft_051"))}
    c = live_config.load("singles_05x", preset="rtx5090",
                         overrides=["input.kind=camera", "input.cameras=" + json.dumps(cams), "input.max_minutes=1"],
                         probe=_probe())
    assert c["videos"] == cams and c["sync"]["mode"] == "nearest" and c["section_s"][0] == 0.0
    with pytest.raises(ValueError, match="no stream address"):
        live_config.load("singles_05x", preset="rtx5090", overrides=["input.kind=camera"], probe=_probe())
    with pytest.raises(ValueError, match="input.kind"):
        live_config.load("singles_05x", preset="rtx5090", overrides=["input.kind=usb"], probe=_probe())
    from analysis.live import pose2d_fast as pf
    assert pf.is_stream_url("rtsp://pi:8554/cam") and pf.is_stream_url("udp://@:5000")
    assert not pf.is_stream_url("data/15-08/singles/0.5/15.8 back 0.5-1.mov")


def test_follow_reads_lines_written_before_and_after(tmp_path):
    p = tmp_path / "cmds.jsonl"
    p.write_text('{"cmd": "a"}\n', encoding="utf-8")
    stop = threading.Event()
    got = []

    def run():
        for line in eng.follow(str(p), stop, poll_s=0.02):
            got.append(json.loads(line)["cmd"])
            if len(got) == 2:
                stop.set()
    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.1)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"cmd": "b"}\n')
    t.join(3)
    assert got == ["a", "b"]


def test_engine_loop_reports_errors_and_keeps_running(monkeypatch, capsys):
    calls = []

    class Fake(eng.LiveEngine):
        def warmup(self, sets=None):
            calls.append(("warmup", sets))
            self.warm_state = {"made": "now"}
            self._set_state("ready", warm={"players": {"P1": True}})

        def stream(self, sets=None, run_dir=None, title=None):
            raise RuntimeError("boom")
    monkeypatch.setattr(eng, "LiveEngine", Fake)
    lines = io.StringIO('{"cmd": "warmup", "set": ["a=1"]}\n{"cmd": "stream"}\n{"cmd": "nope"}\n{"cmd": "quit"}\n')
    eng.serve("singles_05x", stdin=lines)
    evs = [json.loads(ln[7:]) for ln in capsys.readouterr().out.splitlines() if ln.startswith("ENGINE ")]
    assert calls == [("warmup", ["a=1"])]
    assert any(e["ev"] == "error" and "boom" in e["message"] for e in evs)
    assert any(e["ev"] == "error" and "unknown command" in e["message"] for e in evs)
    assert evs[-1] == {"ev": "state", "state": "off"}


def _tiny_video(path, n=40, w=64, h=36):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (6 * i) % 255, np.uint8))
    vw.release()


def test_fake_camera_serves_a_stream_the_camera_reader_reads(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera
    from analysis.live import pose2d_fast as pf
    vid = tmp_path / "v.mp4"
    _tiny_video(vid)
    cam = fake_camera.Camera("v", str(vid), time.monotonic(), 0.0, None, 80)
    cam.start()
    srv = fake_camera.ThreadingServer(("127.0.0.1", 0), fake_camera.make_handler({"v": cam}))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = pf.open_reader("camera", f"http://127.0.0.1:{port}/v.mjpg", 0, 20.0, None)
        assert (r.src_w, r.src_h) == (64, 36) and r.kind == "camera"
        frames = [r.read(f) for f in range(5)]
        r.close()
        assert all(f is not None and f.shape == (36, 64, 3) for f in frames)
        _, s = _server(tmp_path)                                         # the Source tab's Test cameras + preview
        t = s.test_cameras({"v": f"http://127.0.0.1:{port}/v.mjpg", "bad": "not-a-url"})
        assert t["v"]["ok"] and t["v"]["size"] == [64, 36] and t["v"]["thumb"].startswith("data:image/jpeg;base64,")
        assert not t["bad"]["ok"]
    finally:
        cam.running = False
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------------------------------- server
def _server(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import server as srv
    s = srv.Server(unity_project=tmp_path, machine="laptop-cached", repo=ROOT)
    return srv, s


def test_server_input_save_validate_and_sets(tmp_path):
    srv, s = _server(tmp_path)
    assert s.get_input()["kind"] == "video" and s.input_sets() == ["input.kind=video"]
    # v0.8.6: the clip + the chosen angles
    d = s.set_input({"kind": "video", "clip": "singles_05x", "views": ["back_051", "backleft_051", "frontright_051"]})
    assert d["clip"] == "singles_05x" and (tmp_path / "runs" / "input.json").exists()
    assert s.input_sets() == ["input.kind=video", "views=[back_051, backleft_051, frontright_051]"]
    # live cameras named by the user (the v0.8.4 {view: url} form is still read)
    d = s.set_input({"kind": "camera", "clip": "singles_05x", "fps": 30, "max_minutes": 2,
                     "cameras": [{"name": "cam1", "url": "rtsp://pi1:8554/cam", "use": True},
                                 {"name": "cam2", "url": "rtsp://pi2:8554/cam", "use": True},
                                 {"name": "cam3", "url": "", "use": True}]})
    assert d["kind"] == "camera" and [c["use"] for c in d["cameras"]] == [True, True, False]
    sets = s.input_sets()
    assert "input.kind=camera" in sets and "input.fps=30.0" in sets and "views=[cam1, cam2]" in sets
    assert "sync.offsets=auto" in sets
    assert json.loads(next(x for x in sets if x.startswith("input.cameras="))[len("input.cameras="):]) ==         {"cam1": "rtsp://pi1:8554/cam", "cam2": "rtsp://pi2:8554/cam"}
    assert s.set_input({"kind": "camera", "cameras": {"back_051": "rtsp://a/x", "backleft_051": "rtsp://b/x"}})["cameras"][0]         == {"name": "back_051", "url": "rtsp://a/x", "use": True}
    from fastapi import HTTPException
    for bad in ({"kind": "camera", "cameras": {"back_051": "just-a-name", "b": "rtsp://b/x"}},     # not an address
                {"kind": "camera", "cameras": [{"name": "cam1", "url": ""}]},                        # no camera address
                {"kind": "camera", "cameras": [{"name": "../x", "url": "rtsp://a/x"}, {"name": "c2", "url": "rtsp://b/x"}]},
                {"kind": "video", "clip": "singles_05x", "views": []},                                # no angle (1 is fine, v0.8.7)
                {"kind": "video", "clip": "singles_05x", "views": ["back_11", "back_051"]},           # another clip's angle
                {"kind": "video", "clip": "nope", "views": []},
                {"kind": "webcam"}):
        with pytest.raises(HTTPException):
            s.set_input(bad)


def test_server_stream_needs_a_matching_warm_up(tmp_path, monkeypatch):
    srv, s = _server(tmp_path)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        s.stream_engine("singles_05x", "laptop_realtime", {})
    assert e.value.detail["code"] == "not_warm"
    sent = []

    class FakeEngine(srv.jobmod.EngineJob):
        def start(self):
            self.state = "running"
            self.engine["state"] = "cold"
            return self

        def send(self, obj):
            sent.append(obj)

        @property
        def running(self):
            return True
    monkeypatch.setattr(srv.jobmod, "EngineJob", FakeEngine)
    s.warmup("singles_05x", "laptop_realtime", {})
    assert sent[-1]["cmd"] == "warmup" and "input.kind=video" in sent[-1]["set"]
    j = s.engine_job()
    j.engine.update(state="ready", warm={"players": {"P1": True, "P2": True}})
    monkeypatch.setattr(s, "unity_status", lambda: {"in_play": True})
    s.stream_engine("singles_05x", "laptop_realtime", {})
    assert sent[-1]["cmd"] == "stream" and sent[-1]["run_dir"].endswith("realtime_" + sent[-1]["run_dir"][-17:])
    with pytest.raises(HTTPException) as e:
        s.stream_engine("singles_05x", "laptop_realtime", {"warmup.seconds": 30})
    assert e.value.detail["code"] == "rewarm" and "warmup.seconds" in e.value.detail["message"]


def test_engine_job_phases_from_status_lines():
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import jobs as jm
    j = jm.EngineJob(["x"], ROOT, "c", "p")
    for ev in ({"ev": "state", "state": "loading"}, {"ev": "models", "seconds": 4.2}, {"ev": "state", "state": "warming"},
               {"ev": "state", "state": "ready", "warm": {"players": {"P1": True}, "seconds": 3.0}}):
        j._engine_line(json.dumps(ev))
    assert j.engine["state"] == "ready" and j.engine["warm"]["players"] == {"P1": True}
    assert [p["state"] for p in j.phases] == ["done", "done", "done"] and "4.2" in j.phases[0]["detail"]
    j._engine_line(json.dumps({"ev": "stream_start", "run_dir": "/x/live_c_run", "steps": 90}))
    assert j.run_dir == "/x/live_c_run" and j.progress["steps"] == 90 and j.phases[0]["id"] == "stream"
    j._engine_line(json.dumps({"ev": "stream_done", "run_dir": "/x/live_c_run", "written": 90, "steps": 90}))
    assert all(p["state"] == "done" for p in j.phases) and j.engine["last_stream"]["written"] == 90
    j._engine_line(json.dumps({"ev": "error", "message": "bad", "cmd": "stream"}))
    assert j.engine["error"]["message"] == "bad"


# ------------------------------------------------------------------ v0.8.7 one-click test cameras
class _FakeCamJob:
    """Stands in for jobs.Job: no process (the real fake_camera.py needs the clip's videos)."""
    started = []

    def __init__(self, kind, argv, cwd, title="", meta=None, **kw):
        self.kind, self.argv, self.meta, self.lines = kind, list(argv), dict(meta or {}), []
        self.running, self._stop_requested, self.error, self.returncode = False, False, None, None
        self.proc = type("P", (), {"pid": 4242})()

    def start(self):
        self.running = True
        _FakeCamJob.started.append(self.argv)
        return self

    def stop(self, timeout_s=10.0):
        self.running, self._stop_requested = False, True


def test_server_test_cameras_one_click(tmp_path, monkeypatch):
    srv, s = _server(tmp_path)
    _FakeCamJob.started = []
    monkeypatch.setattr(srv.jobmod, "Job", _FakeCamJob)
    answers = iter([False] + [True] * 20)              # the port is free, then the cameras listen
    monkeypatch.setattr(srv, "port_open", lambda port, *a, **k: next(answers))
    monkeypatch.setattr(srv, "video_fps", lambda repo, clip, view: 60.0)
    s.set_input({"kind": "video", "clip": "singles_05x", "views": ["back_051", "frontleft_051"]})
    r = s.testcams_start()
    a = _FakeCamJob.started[-1]
    assert a[1].endswith("fake_camera.py") and a[a.index("--clip") + 1] == "singles_05x"
    assert a[a.index("--views") + 1] == "back_051,frontleft_051" and a[a.index("--start-s") + 1] == "6.5"
    inp = r["input"]
    assert inp["kind"] == "camera" and [c["name"] for c in inp["cameras"]] == ["back_051", "frontleft_051"]
    assert inp["cameras"][0] == {"name": "back_051", "url": "http://localhost:8090/back_051.mjpg", "use": True}
    assert inp["fps"] == 60.0 and inp["views"] == ["back_051", "frontleft_051"]      # the video angles are kept
    assert "views=[back_051, frontleft_051]" in s.input_sets()
    assert r["status"]["running"] and (tmp_path / "runs" / "testcams.pid").read_text() == "4242"
    st = s.testcams_stop()
    assert not st["running"] and st["error"] is None and not (tmp_path / "runs" / "testcams.pid").exists()


def test_server_test_cameras_refuse_a_busy_port_and_bad_angles(tmp_path, monkeypatch):
    srv, s = _server(tmp_path)
    from fastapi import HTTPException
    monkeypatch.setattr(srv.jobmod, "Job", _FakeCamJob)
    monkeypatch.setattr(srv, "port_open", lambda port, *a, **k: True)
    with pytest.raises(HTTPException) as e:
        s.testcams_start("singles_05x", ["back_051"])
    assert e.value.status_code == 409 and "8090" in e.value.detail
    with pytest.raises(HTTPException) as e:
        s.testcams_start("singles_05x", ["back_11"])
    assert e.value.status_code == 400


# ------------------------------------------------------------------ v0.8.7 USB cameras on this PC
def test_fake_camera_quit_token_and_webcam_names():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import urllib.error
    import urllib.request
    import fake_camera
    quits = []
    srv = fake_camera.ThreadingServer(("127.0.0.1", 0), fake_camera.make_handler({}, "abc", lambda: quits.append(1)))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/quit?token=nope", timeout=5)
        assert e.value.code == 403 and not quits
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/quit?token=abc", timeout=5) as r:
            assert r.status == 200 and r.read() == b"bye"
        assert quits == [1]
    finally:
        srv.shutdown()
        srv.server_close()
    with pytest.raises(SystemExit):                     # one different name per camera, checked before any opens
        fake_camera.main(["--webcam", "0,1", "--webcam-names", "back"])


def test_fake_camera_probe_lists_the_cameras_that_send_a_frame(monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera

    class Cap:
        def read(self):
            return True, np.zeros((36, 64, 3), np.uint8)

        def get(self, prop):
            return 30.0

        def release(self):
            pass

    asked = []

    def fake_open(cv2, index, size=None, fps=30.0, fourcc="MJPG"):
        asked.append(index)
        return Cap() if index == 1 else None
    monkeypatch.setattr(fake_camera, "open_webcam", fake_open)
    monkeypatch.setattr(fake_camera, "dshow_names", lambda: ["Front", "Capture card"])
    out = io.StringIO()
    assert fake_camera.probe_webcams(4, skip={2}, out=out, measure_s=0.05) == 1
    rows = [json.loads(ln) for ln in out.getvalue().splitlines()]
    assert asked == [0, 1, 3]                           # 2 is in use: left alone
    assert rows[0]["index"] == 1 and rows[0]["name"] == "Capture card" and rows[0]["size"] == [64, 36]
    assert rows[0]["fps"] > 0 and rows[0]["thumb"]                       # measured, not the asked rate
    assert rows[0]["thumb"].startswith("data:image/jpeg;base64,")
    assert rows[1] == {"index": 2, "name": None, "skipped": True} and rows[-1] == {"done": True, "found": 1}


class _FakeUsbJob:
    """Stands in for jobs.Job: the relay 'opens' its cameras at once."""
    started = []

    def __init__(self, kind, argv, cwd, title="", meta=None, **kw):
        self.kind, self.argv, self.meta = kind, list(argv), dict(meta or {})
        self.lines, self.running, self._stop_requested = [], False, False
        self.error, self.returncode = None, None
        self.proc = type("P", (), {"pid": 777 + len(_FakeUsbJob.started)})()

    def start(self):
        self.running = True
        self.lines.append("fake cameras on port 8091:")
        _FakeUsbJob.started.append(self.argv)
        return self

    def wait(self, timeout_s=None):
        return "running" if self.running else "stopped"

    def stop(self, timeout_s=10.0):
        self.running, self._stop_requested = False, True


def _usb_server(tmp_path, monkeypatch, wsl=False, reach=("127.0.0.1",)):
    srv, s = _server(tmp_path)
    _FakeUsbJob.started = []
    monkeypatch.setattr(srv.jobmod, "Job", _FakeUsbJob)
    monkeypatch.setattr(srv, "in_wsl", lambda: wsl)
    quits = []
    monkeypatch.setattr(srv, "_quit_relay", lambda info, hosts: quits.append((dict(info or {}), list(hosts))) or True)
    # nothing listens before a relay starts; afterwards it answers on the reachable addresses
    monkeypatch.setattr(srv, "port_open", lambda port, host="127.0.0.1", **k: bool(
        s.usbcams is not None and s.usbcams.running and host in reach))
    return srv, s, quits


def test_server_usb_rows_start_the_relay_and_fill_their_addresses(tmp_path, monkeypatch):
    srv, s, quits = _usb_server(tmp_path, monkeypatch)
    rows = [{"name": "back", "type": "usb", "device": 2, "use": True}, {"name": "net", "url": "rtsp://pi1:8554/cam"}]
    inp = s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": rows, "fps": 60})
    assert inp["cameras"] == [{"name": "back", "url": "http://127.0.0.1:8091/back.mjpg", "use": True, "type": "usb",
                               "device": 2},
                              {"name": "net", "url": "rtsp://pi1:8554/cam", "use": True}]
    a = _FakeUsbJob.started[-1]
    assert a[0] == sys.executable and a[1] == "-u" and a[2].endswith("fake_camera.py")
    assert a[a.index("--webcam") + 1] == "2" and a[a.index("--webcam-names") + 1] == "back"
    assert a[a.index("--host") + 1] == "127.0.0.1" and a[a.index("--fps") + 1] == "60"
    info = json.loads((tmp_path / "runs" / "usbcams.json").read_text())
    assert info["host"] == "127.0.0.1" and info["token"] == a[a.index("--quit-token") + 1]
    st = s.usb_status()
    assert st["running"] and st["devices"] == {"back": 2} and st["where"] == "this PC"
    s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": rows, "fps": 60})
    assert len(_FakeUsbJob.started) == 1                # same cameras: the relay keeps running
    assert "input.cameras=" + json.dumps({"back": "http://127.0.0.1:8091/back.mjpg", "net": "rtsp://pi1:8554/cam"}) \
        in s.input_sets()
    s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": rows, "fps": 30})
    assert len(_FakeUsbJob.started) == 2 and quits    # another frame rate: restarted (the old one asked to quit)
    s.set_input({"kind": "video", "clip": "singles_05x", "views": ["back_051", "frontleft_051"]})
    assert not s.usb_status()["running"] and not (tmp_path / "runs" / "usbcams.json").exists()


def test_server_usb_rows_are_checked(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch)
    from fastapi import HTTPException
    for rows, part in (([{"name": "a", "type": "usb", "device": 0}, {"name": "b", "type": "usb", "device": 0}],
                        "already a"),
                       ([{"name": "a", "type": "usb", "device": 12}], "0-9"),
                       ([{"name": "a", "type": "usb", "device": "x"}], "0-9")):
        with pytest.raises(HTTPException) as e:
            s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": rows})
        assert e.value.status_code == 400 and part in e.value.detail
    s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": [            # an unticked duplicate is fine
        {"name": "a", "type": "usb", "device": 0}, {"name": "b", "type": "usb", "device": 0, "use": False}]})
    a = _FakeUsbJob.started[-1]
    assert a[a.index("--webcam") + 1] == "0" and a[a.index("--webcam-names") + 1] == "a"


def test_server_usb_in_wsl_runs_the_relay_on_windows(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch, wsl=True, reach=("172.20.0.1",))
    from fastapi import HTTPException
    monkeypatch.setattr(srv, "find_windows_python", lambda: [])
    with pytest.raises(HTTPException) as e:
        s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": [{"name": "back", "type": "usb", "device": 0}]})
    assert e.value.status_code == 400 and "opencv-python" in e.value.detail and not _FakeUsbJob.started
    monkeypatch.setattr(srv, "find_windows_python", lambda: ["/mnt/c/Windows/py.exe", "-3"])
    monkeypatch.setattr(srv, "win_path", lambda p: r"\\wsl.localhost\Ubuntu\repo\tools\fake_camera.py")
    monkeypatch.setattr(srv, "wsl_windows_hosts", lambda: ["127.0.0.1", "172.20.0.1"])
    inp = s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": [{"name": "back", "type": "usb", "device": 0}]})
    a = _FakeUsbJob.started[-1]
    assert a[:4] == ["/mnt/c/Windows/py.exe", "-3", "-u", r"\\wsl.localhost\Ubuntu\repo\tools\fake_camera.py"]
    assert a[a.index("--host") + 1] == "0.0.0.0"       # WSL reaches Windows over its virtual network
    assert inp["cameras"][0]["url"] == "http://172.20.0.1:8091/back.mjpg"
    assert s.usb_status()["where"] == "Windows"


def test_server_usb_unreachable_relay_is_stopped_with_a_firewall_hint(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch, wsl=True, reach=())
    from fastapi import HTTPException
    monkeypatch.setattr(srv, "find_windows_python", lambda: ["py.exe"])
    monkeypatch.setattr(srv, "win_path", lambda p: p)
    monkeypatch.setattr(srv, "wsl_windows_hosts", lambda: ["127.0.0.1"])
    monkeypatch.setattr(srv.time, "sleep", lambda s_: None)
    with pytest.raises(HTTPException) as e:
        s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": [{"name": "back", "type": "usb", "device": 0}]})
    assert e.value.status_code == 502 and "Firewall" in e.value.detail and not s.usb_status()["running"]


def test_server_usb_ensure_restarts_the_relay_and_moves_addresses(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch, wsl=True, reach=("172.29.0.1",))
    monkeypatch.setattr(srv, "find_windows_python", lambda: ["py.exe"])
    monkeypatch.setattr(srv, "win_path", lambda p: p)
    monkeypatch.setattr(srv, "wsl_windows_hosts", lambda: ["127.0.0.1", "172.29.0.1"])
    old = {"kind": "camera", "clip": "singles_05x", "views": [], "fps": 60, "max_minutes": 5, "video_variant": None,
           "cameras": [{"name": "back", "url": "http://172.20.0.1:8091/back.mjpg", "use": True, "type": "usb",
                        "device": 0}]}                  # saved before a reboot moved WSL's address of Windows
    (tmp_path / "runs" / "input.json").write_text(json.dumps(old))
    sets = s.input_sets()
    assert "input.cameras=" + json.dumps({"back": "http://172.29.0.1:8091/back.mjpg"}) in sets
    assert s.get_input()["cameras"][0]["url"] == "http://172.29.0.1:8091/back.mjpg"


def test_server_usb_test_rows_and_find(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch)
    tested = {}
    monkeypatch.setattr(s, "test_cameras", lambda cams, timeout_s=5.0: tested.update(cams) or
                        {k: {"ok": True, "size": [1920, 1080]} for k in cams})
    r = s.test_rows([{"name": "back", "type": "usb", "device": 1}, {"name": "net", "url": "rtsp://pi/1"},
                     {"name": "empty", "url": ""}], fps=60)
    assert tested == {"back": "http://127.0.0.1:8091/back.mjpg", "net": "rtsp://pi/1"}
    assert r["back"]["url"] == "http://127.0.0.1:8091/back.mjpg" and "url" not in r["net"]

    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        out = "\n".join([json.dumps({"index": 0, "name": "Front", "size": [1280, 720], "fps": 30.0, "thumb": None}),
                         json.dumps({"index": 1, "name": "Capture card", "skipped": True}),
                         json.dumps({"done": True, "found": 1})])
        return type("R", (), {"stdout": out, "stderr": "[ WARN ] DSHOW", "returncode": 0})()
    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    f = s.usb_find(fps=60)                              # the relay holds camera 1: skipped, listed as in use
    a = calls[-1]
    assert a[a.index("--probe") + 1] == "6" and a[a.index("--skip") + 1] == "1"
    assert f["where"] == "this PC" and [c["index"] for c in f["cameras"]] == [0, 1]
    assert f["cameras"][1] == {"index": 1, "name": "Capture card", "in_use": "back"}


def test_find_windows_python_skips_the_store_stand_in(tmp_path, monkeypatch):
    srv, _ = _server(tmp_path)
    monkeypatch.setattr(srv, "_WIN_PY", [])
    monkeypatch.delenv("BVR_WINDOWS_PYTHON", raising=False)
    monkeypatch.setattr(srv.shutil, "which", lambda exe: {"python.exe": "/mnt/c/Users/u/AppData/Local/Microsoft/"
                                                          "WindowsApps/python.exe"}.get(exe))
    monkeypatch.setattr(srv.subprocess, "run", lambda argv, **kw: type("R", (), {"returncode": 9009})())
    assert srv.find_windows_python() == []              # the Store stand-in: no Python
    monkeypatch.setattr(srv.shutil, "which", lambda exe: {"py.exe": "/mnt/c/Windows/py.exe",
                                                          "python.exe": "/mnt/c/x/python.exe"}.get(exe))
    seen = []
    monkeypatch.setattr(srv.subprocess, "run",
                        lambda argv, **kw: seen.append(argv) or type("R", (), {"returncode": 0})())
    assert srv.find_windows_python() == ["/mnt/c/Windows/py.exe", "-3"]
    assert seen[0][:3] == ["/mnt/c/Windows/py.exe", "-3", "-c"]
    assert srv.find_windows_python() == ["/mnt/c/Windows/py.exe", "-3"] and len(seen) == 1     # found once, kept


def test_server_usb_quits_a_relay_left_by_an_earlier_server(tmp_path, monkeypatch):
    srv, s, quits = _usb_server(tmp_path, monkeypatch)
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / "usbcams.json").write_text(json.dumps({"pid": 1, "token": "old", "port": 8091,
                                                                "hosts": ["127.0.0.1"], "where": "Windows",
                                                                "host": "127.0.0.1"}))
    inp = s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": [{"name": "back", "type": "usb", "device": 0}]})
    assert quits and quits[0][0]["token"] == "old" and quits[0][1] == ["127.0.0.1"]     # asked to quit first
    assert inp["cameras"][0]["url"] == "http://127.0.0.1:8091/back.mjpg" and len(_FakeUsbJob.started) == 1
    assert json.loads((tmp_path / "runs" / "usbcams.json").read_text())["token"] != "old"
    (tmp_path / "runs" / "usbcams.json").write_text(json.dumps({"pid": 2, "token": "older", "port": 8091,
                                                                "hosts": ["127.0.0.1"], "where": "Windows"}))
    s.usbcams = None                                    # a fresh server, the source has no USB rows: freed too
    s.set_input({"kind": "video", "clip": "singles_05x", "views": ["back_051", "frontleft_051"]})
    assert quits[-1][0]["token"] == "older" and not (tmp_path / "runs" / "usbcams.json").exists()


# ---------------------------------------------------------------------------------- review fixes (2026-11-10)
def test_job_output_is_kept_in_a_log_file(tmp_path, monkeypatch):
    srv, s = _server(tmp_path)
    assert srv.jobmod.LOG_DIR == str(tmp_path / "runs" / "server_logs")
    j = srv.jobmod.Job("probe", [sys.executable, "-c", "print('hello'); raise SystemExit(3)"], cwd=str(tmp_path)).start()
    assert j.wait(30) == "failed"
    j._thread.join(5)
    text = open(j.log_path, encoding="utf-8").read()
    assert "hello" in text and "failed, exit code 3" in text and "raise SystemExit(3)" in text.splitlines()[1]


def test_camera_test_says_whether_the_camera_was_reached(tmp_path, monkeypatch):
    srv, s = _server(tmp_path)
    assert srv.stream_host_port("rtsp://user:pw@10.0.0.9/stream1") == ("10.0.0.9", 554)
    assert srv.stream_host_port("http://pi1:8090/back.mjpg") == ("pi1", 8090)
    assert srv.stream_host_port("nonsense") == (None, None)
    assert srv.stream_reachable("udp://239.0.0.1:5000") is None
    monkeypatch.setattr(srv, "port_open", lambda port, host="127.0.0.1", timeout_s=0.3: False)
    monkeypatch.setattr(srv, "in_wsl", lambda: True)
    r = srv.Server._test_camera("rtsp://user:pw@10.0.0.9/stream1", 1.0)
    assert not r["ok"] and "cannot connect to 10.0.0.9:554 from WSL" in r["error"] and "pw" not in r["error"]
    assert "ping" in r["error"]


def test_usb_status_reports_a_camera_that_stopped_sending(tmp_path, monkeypatch):
    srv, s, _ = _usb_server(tmp_path, monkeypatch)
    s.usb_start({"back": 2}, 60)
    assert s.usb_status()["lost"] == [] and s.usb_status()["error"] is None
    s.usbcams.lines.append("camera back: no frames from webcam 2")
    st = s.usb_status()
    assert st["running"] and st["lost"] == ["back"] and "unplugged" in st["error"]
    s.usbcams.lines.append("camera back: frames again from webcam 2")
    assert s.usb_status()["lost"] == [] and s.usb_status()["error"] is None


def test_usb_relay_is_not_restarted_under_the_engine(tmp_path, monkeypatch):
    from fastapi import HTTPException
    srv, s, _ = _usb_server(tmp_path, monkeypatch)
    s.usb_start({"back": 2}, 60)
    monkeypatch.setattr(s, "cameras_in_use", lambda: True)
    assert s.usb_start({"back": 2}, 60)                  # the same cameras: nothing restarts
    with pytest.raises(HTTPException) as e:
        s.usb_start({"back": 2}, 30)
    assert e.value.status_code == 409 and "engine" in e.value.detail
    assert len(_FakeUsbJob.started) == 1 and s.usb_status()["running"]


def test_other_angles_need_a_new_warm_up(tmp_path):
    srv, s = _server(tmp_path)
    assert srv.Server._warm_part(["views=[back_051, frontleft_051]", "output_fps=30"]) == {
        "views": "[back_051, frontleft_051]"}


def test_engine_left_by_an_earlier_server_is_stopped(tmp_path):
    pytest.importorskip("psutil")
    import subprocess
    srv, s = _server(tmp_path)
    pf = tmp_path / "runs" / "engine" / "engine.pid"
    pf.parent.mkdir(parents=True, exist_ok=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "live_run.py", "engine"])
    try:
        pf.write_text(str(other.pid))
        assert s.stop_stale_engine() is False and other.poll() is None and not pf.exists()   # not an engine: left alone
        pf.write_text(str(mine.pid))
        assert s.stop_stale_engine() is True and not pf.exists()
        assert mine.wait(10) is not None
        assert s.stop_stale_engine() is False            # no file: nothing to do
    finally:
        for p in (other, mine):
            if p.poll() is None:
                p.kill()


def test_webcam_reports_a_lost_camera_and_opens_it_again(monkeypatch, capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera as fc

    class Cap:
        def __init__(self, n):
            self.n, self.released = n, False

        def read(self):
            if self.n <= 0:
                return False, None
            self.n -= 1
            return True, np.zeros((48, 64, 3), np.uint8)

        def get(self, prop):
            return 30.0

        def release(self):
            self.released = True

    caps = [Cap(2), Cap(10 ** 6)]
    opened = []
    monkeypatch.setattr(fc, "open_webcam", lambda cv2, index, *a: (opened.append(index), caps[len(opened) - 1])[1])
    monkeypatch.setattr(fc, "LOST_S", 0.05)
    monkeypatch.setattr(fc, "REOPEN_S", 0.01)
    cam = fc.Webcam("back", 3, None, 80)
    cam.start()
    try:
        t_end = time.monotonic() + 10
        while len(opened) < 2 and time.monotonic() < t_end:
            time.sleep(0.02)
        seq = cam.seq
        jpeg, new = cam.wait_next(seq, timeout=5.0)
        assert new > seq and jpeg                        # frames flow again from the reopened camera
    finally:
        cam.running = False
        cam.join(5)
    out = capsys.readouterr().out
    assert opened == [3, 3] and caps[0].released
    assert "camera back: no frames from webcam 3" in out and "camera back: frames again" in out


def test_mjpeg_stream_never_resends_an_old_frame():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera as fc
    import urllib.request
    from http.server import ThreadingHTTPServer

    class Still:                                         # a camera that sent one frame and then stopped
        size, fps = (64, 48), 30.0
        cond = threading.Condition()
        jpeg, seq, running = b"\xff\xd8one\xff\xd9", 1, True
        wait_next = fc.Camera.wait_next

    cam = Still()
    cam.wait_next = lambda seq, timeout=0.2: fc.Camera.wait_next(cam, seq, timeout=0.2)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fc.make_handler({"back": cam}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/back.mjpg", timeout=1.0)
        got = b""
        try:
            while True:
                got += r.read(1)
        except Exception:  # noqa: BLE001 - the read times out once the one frame is through
            pass
        assert got.count(b"\xff\xd8one\xff\xd9") == 1
    finally:
        cam.running = False
        srv.shutdown()


def test_live_camera_calibration_frame_is_taken_again_after_a_new_source(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import calib as calibmod
    saved = [0.0]
    c = calibmod.Calib(str(tmp_path), str(tmp_path), input_mtime_fn=lambda: saved[0])
    os.makedirs(c.frames_dir)
    grabs = []
    c.grab = lambda cam, t, name, frame=None: grabs.append((cam["view"], name))
    cams = {"cam1": {"view": "cam1", "video": "http://127.0.0.1:8091/cam1.mjpg", "has_video": True},
            "back_051": {"view": "back_051", "video": "data/x/back_051.mp4", "has_video": True}}
    c.camera = lambda view: cams[view]
    for v in cams:
        open(os.path.join(c.frames_dir, v + ".jpg"), "wb").write(b"jpg")
    c.frame_path("cam1.jpg")
    assert grabs == []                                   # the frame is newer than the saved source
    saved[0] = time.time() + 100                         # the source was saved after the frame was taken
    c.frame_path("cam1.jpg")
    c.frame_path("back_051.jpg")                         # a recorded video never changes: its frame stays
    assert grabs == [("cam1", "cam1.jpg")]


# ---------------------------------------------------------------------------------- cameras mounted upside down
def test_rotate_tag_travels_with_the_camera_address(tmp_path, monkeypatch):
    assert live_config.with_rotate("rtsp://pc:8554/unit1", 180) == "rtsp://pc:8554/unit1#rotate=180"
    assert live_config.with_rotate("rtsp://pc:8554/unit1#rotate=180", 180) == "rtsp://pc:8554/unit1#rotate=180"
    assert live_config.with_rotate("rtsp://pc:8554/unit1", None) == "rtsp://pc:8554/unit1"
    assert live_config.split_rotate("rtsp://pc:8554/unit1#rotate=180") == ("rtsp://pc:8554/unit1", 180)
    assert live_config.split_rotate("rtsp://pc:8554/unit1") == ("rtsp://pc:8554/unit1", 0)
    img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert np.array_equal(live_config.upright(img, 180), img[::-1, ::-1]) and live_config.upright(img, 0) is img
    srv, s, _ = _usb_server(tmp_path, monkeypatch)
    rows = [{"name": "unit1", "url": "rtsp://pc:8554/unit1", "rotate": 180},
            {"name": "unit2", "url": "rtsp://pc:8554/unit2"},
            {"name": "desk", "type": "usb", "device": 0, "rotate": True}]
    d = s.set_input({"kind": "camera", "clip": "singles_05x", "cameras": rows, "fps": 60})
    assert [c.get("rotate") for c in d["cameras"]] == [180, None, 180] and "rotate" not in d["cameras"][1]
    cams = json.loads(next(x for x in s.input_sets() if x.startswith("input.cameras="))[len("input.cameras="):])
    assert cams == {"unit1": "rtsp://pc:8554/unit1#rotate=180", "unit2": "rtsp://pc:8554/unit2",
                    "desk": "http://127.0.0.1:8091/desk.mjpg#rotate=180"}
    tested = {}
    monkeypatch.setattr(s, "test_cameras", lambda cams, timeout_s=5.0: tested.update(cams) or {k: {"ok": True} for k in cams})
    s.test_rows(rows)
    assert tested["unit1"].endswith("#rotate=180") and tested["desk"].endswith("desk.mjpg#rotate=180")


def test_camera_reader_and_camera_test_turn_an_upside_down_camera(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera as fc
    from http.server import ThreadingHTTPServer
    import cv2
    from analysis.live import pose2d_fast as pf

    img = np.zeros((48, 64, 3), np.uint8)
    img[:24] = 255                                       # white on top, black below
    ok, buf = cv2.imencode(".jpg", img)

    class Cam:
        size, fps, running = (64, 48), 30.0, True
        cond = threading.Condition()
        jpeg, seq = buf.tobytes(), 0

        def wait_next(self, seq, timeout=2.0):
            time.sleep(0.03)
            Cam.seq += 1
            return Cam.jpeg, Cam.seq

    http = ThreadingHTTPServer(("127.0.0.1", 0), fc.make_handler({"v": Cam()}))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{http.server_address[1]}/v.mjpg"
    try:
        for tag, top_white in (("", True), ("#rotate=180", False)):
            rd = pf.open_reader("camera", url + tag, 0, 30.0)
            try:
                frame = rd.read(0)
                assert (frame[:12].mean() > 200) == top_white and (frame[-12:].mean() > 200) != top_white
            finally:
                rd.close()
        srv, s = _server(tmp_path)
        t = s.test_cameras({"up": url, "down": url + "#rotate=180"})
        assert t["up"]["ok"] and t["down"]["ok"] and t["up"]["thumb"] != t["down"]["thumb"]
        assert live_config.probe_camera(url + "#rotate=180", 30, 1)["width"] == 64
    finally:
        Cam.running = False
        http.shutdown()


def test_calibration_frame_of_an_upside_down_camera_is_saved_upright(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import calib as calibmod
    import cv2
    img = np.zeros((48, 64, 3), np.uint8)
    img[:24] = 255

    class Cap:
        def __init__(self, src, *a):
            Cap.src = src

        def read(self):
            return True, img.copy()

        def release(self):
            pass

    monkeypatch.setattr(calibmod.cv2, "VideoCapture", Cap)
    c = calibmod.Calib(str(tmp_path), str(tmp_path))
    c.grab({"view": "unit1", "video": "rtsp://pc:8554/unit1#rotate=180", "has_video": True}, 1.0, "unit1.jpg")
    assert Cap.src == "rtsp://pc:8554/unit1"             # the tag never reaches FFmpeg
    saved = cv2.imread(os.path.join(c.frames_dir, "unit1.jpg"))
    assert saved[:12].mean() < 50 and saved[-12:].mean() > 200


def test_ros_rtsp_bridge_command_and_restart():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import ros_rtsp_bridge as rb
    cmd = rb.ffmpeg_cmd("unit2", "rtsp://127.0.0.1:8554/", fps=60, flip=True)
    assert cmd[-1] == "rtsp://127.0.0.1:8554/unit2" and cmd[cmd.index("-vf") + 1] == "hflip,vflip"
    assert cmd.index("-use_wallclock_as_timestamps") < cmd.index("-i") < cmd.index("-fps_mode")
    assert "-vf" not in rb.ffmpeg_cmd("unit1") and rb.ffmpeg_cmd("unit1")[-1].endswith(":8554/unit1")

    class Proc:
        made = []

        def __init__(self, cmd, stdin=None):
            self.dead, self.got = False, []
            self.stdin = self
            Proc.made.append(self)

        def write(self, data):
            self.got.append(data)

        def poll(self):
            return 1 if self.dead else None

    now, logs = [0.0], []
    p = rb.Pipe("unit1", ["ffmpeg"], log=logs.append, popen=Proc, clock=lambda: now[0])
    assert p.write(b"a") and Proc.made[0].got == [b"a"]
    Proc.made[0].dead = True                             # ffmpeg exited (the RTSP server went away)
    assert not p.write(b"b") and len(logs) == 1 and len(Proc.made) == 1
    now[0] = rb.RESTART_S + 0.1
    assert p.write(b"c") and len(Proc.made) == 2 and Proc.made[1].got == [b"c"] and p.restarts == 1
    assert rb.parse_args(["--flip-units", "unit3"]).flip_units == "unit3"


# ---------------------------------------------------------------------------------- camera check (what the feed does)
def test_camera_check_numbers_and_plain_words():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import camera_check as cc
    even = [i / 60.0 for i in range(600)]
    st = cc.stats(even, [255.0] + [3.0] * 599, 60)
    assert st["fps"] == 60.0 and st["freezes"] == 0 and st["late_gaps"] == 0 and st["still_pct"] == 0
    assert abs(st["gap_ms_p50"] - 16.7) < 0.1 and cc.verdict(st, 60) == ["steady"]
    # half the frames lost and repeated by the bridge, one freeze, frames in bursts of four
    t, bad = 0.0, []
    for i in range(400):
        t += 0.001 if i % 4 else 0.064
        if i == 200:
            t += 1.2
        bad.append(t)
    st = cc.stats(bad, [255.0] + [0.0 if i % 2 else 3.0 for i in range(399)], 60)
    v = " | ".join(cc.verdict(st, 60))
    assert st["freezes"] == 1 and st["longest_freeze_s"] >= 1.2 and st["burst_pct"] > 60 and 45 < st["still_pct"] < 55
    assert "froze 1 time" in v and "bursts" in v and "no change" in v
    assert cc.verdict(cc.stats([0.0], [255.0], 60), 60) == ["no frames arrived"]
    assert cc.hide_login("rtsp://user:pw@10.0.0.9/s") == "rtsp://<login>@10.0.0.9/s"


def test_camera_check_flash_delay_and_saved_source(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import camera_check as cc
    toggles = [(5.0, 1), (9.0, -1), (13.0, 1)]
    times = [i / 30.0 for i in range(30 * 17)]
    delay = 0.8

    def level(t):                                        # the camera shows each flash 0.8 s late
        lit = False
        for t0, sign in toggles:
            if t >= t0 + delay:
                lit = sign > 0
        return 180.0 if lit else 40.0

    d = cc.flash_delays(times, [level(t) for t in times], toggles)
    assert all(x is not None and abs(x - delay) < 0.05 for x in d)
    assert cc.flash_delays(times, [90.0] * len(times), toggles) == [None, None, None]     # never saw the screen
    p = tmp_path / "input.json"
    p.write_text(json.dumps({"kind": "camera", "fps": 60, "cameras": [
        {"name": "unit1", "url": "rtsp://pc:8554/unit1", "use": True, "rotate": 180},
        {"name": "unit2", "url": "rtsp://pc:8554/unit2", "use": False},
        {"name": "unit3", "url": "", "use": True}]}))
    assert cc.cameras_from_input(str(p)) == ({"unit1": "rtsp://pc:8554/unit1#rotate=180"}, 60.0)
    out = io.StringIO()
    tg = cc.run_latency({}, n=3, gap_s=(0.0, 0.0), out=out, sleep=lambda s: None)
    assert [s for _, s in tg] == [1, -1, 1] and out.getvalue().endswith("\x1b[0m\x1b[2J\x1b[H")


def test_camera_check_reads_a_stream_end_to_end(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import camera_check as cc
    import fake_camera as fc
    import cv2
    from http.server import ThreadingHTTPServer

    class Cam:
        size, fps, running = (64, 48), 30.0, True
        cond = threading.Condition()
        seq = 0

        def wait_next(self, seq, timeout=2.0):
            time.sleep(1 / 30.0)
            Cam.seq += 1
            img = np.full((48, 64, 3), (Cam.seq * 7) % 255, np.uint8)       # every frame differs
            return cv2.imencode(".jpg", img)[1].tobytes(), Cam.seq

    http = ThreadingHTTPServer(("127.0.0.1", 0), fc.make_handler({"v": Cam()}))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "check"
        rc = cc.main(["--url", f"v=http://127.0.0.1:{http.server_address[1]}/v.mjpg", "--seconds", "2", "--fps", "30",
                      "--out", str(out)])
    finally:
        Cam.running = False
        http.shutdown()
    doc = json.loads((out / "report.json").read_text())
    st = doc["cameras"]["v"]["stats"]
    assert rc == 0 and doc["cameras"]["v"]["size"] == [64, 48] and 20 < st["fps"] < 40 and st["freezes"] == 0
    assert (out / "v.jpg").exists() and (out / "report.txt").read_text().count("fps") >= 2
    assert (out / "v_frames.csv").read_text().splitlines()[0] == "t_s,gap_ms,brightness,change"


def test_live_calibration_frame_is_the_newest_one(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import calib as calibmod
    import cv2

    class Cap:                                           # a stream that opens on 5 buffered (old) frames
        def __init__(self, src, *a):
            self.n = 0

        def read(self):
            self.n += 1
            if self.n > 40:
                return False, None
            return True, np.full((48, 64, 3), 10 if self.n <= 5 else 200, np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(calibmod.cv2, "VideoCapture", Cap)
    c = calibmod.Calib(str(tmp_path), str(tmp_path))
    c.grab({"view": "unit1", "video": "rtsp://pc:8554/unit1", "has_video": True}, 1.0, "unit1.jpg")
    assert cv2.imread(os.path.join(c.frames_dir, "unit1.jpg")).mean() > 150       # not the first, old frame


# ---------------------------------------------------------------------------------- setup helpers on the page
def test_calibration_suggests_the_points_not_clicked_yet(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "pipeline_server"))
    import calib as calibmod
    import cv2
    c = calibmod.Calib(str(tmp_path), str(tmp_path))
    cp = calibmod._court_points()
    names = list(cp)
    H = np.array([[60.0, 8.0, 900.0], [-4.0, -35.0, 600.0], [0.002, -0.03, 1.0]])      # a court seen in perspective
    px = cv2.perspectiveTransform(np.array([cp[n] for n in names], np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    spread = [names[i] for i in (0, len(names) // 5, 2 * len(names) // 5, 3 * len(names) // 5, len(names) - 1)]
    body = {"points": [{"name": n, "px": float(px[names.index(n)][0]), "py": float(px[names.index(n)][1])}
                       for n in spread]}
    r = c.suggest(body)
    assert set(r["points"]) == set(names) - set(spread) and r["fit_px"] < 1.0
    for n, (x, y) in r["points"].items():
        assert abs(x - px[names.index(n)][0]) < 1.0 and abs(y - px[names.index(n)][1]) < 1.0
    assert c.suggest({"points": body["points"][:3]}) == {"points": {}, "fit_px": None}         # 4 clicks needed
    body["points"][0]["px"] += 900                       # one click far off its line crossing: no guess is shown
    body["points"][1]["py"] -= 700
    assert c.suggest(body)["points"] == {}


def test_live_preview_feed_check_and_delay_test(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import fake_camera as fc
    import cv2
    from fastapi import HTTPException
    from http.server import ThreadingHTTPServer

    class Cam:
        size, fps, running = (64, 48), 30.0, True
        cond = threading.Condition()
        seq, level = 0, 40

        def wait_next(self, seq, timeout=2.0):
            time.sleep(1 / 30.0)
            Cam.seq += 1
            img = np.full((48, 64, 3), Cam.level + Cam.seq % 3, np.uint8)
            return cv2.imencode(".jpg", img)[1].tobytes(), Cam.seq

    http = ThreadingHTTPServer(("127.0.0.1", 0), fc.make_handler({"v": Cam()}))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{http.server_address[1]}/v.mjpg"
    srv, s = _server(tmp_path)
    rows = [{"name": "unit1", "url": url, "rotate": 180}, {"name": "empty", "url": ""}]
    try:
        p = s.preview_set(rows)
        assert p["cameras"] == ["unit1"] and s.preview_cams["unit1"].endswith("#rotate=180")
        gen = s.preview_frames("unit1")
        part = next(gen)
        assert part.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in part
        s.preview_stop()                                 # the page left step 1: the stream ends
        assert list(gen) == [] and s.preview_cams == {}
        r = s.feed_check(rows, fps=30, seconds=2)
        st = r["cameras"]["unit1"]["stats"]
        assert 20 < st["fps"] < 40 and list(r["cameras"]["unit1"]["size"]) == [64, 48] and r["flashes"] == 0
        assert (tmp_path / "runs").glob("camera_check_*/report.txt") and "unit1" in r["cameras"]
        with pytest.raises(HTTPException):
            s.feed_check([{"name": "empty", "url": ""}])
        with pytest.raises(HTTPException):
            s.delay_flash(True)                          # no test running
        monkeypatch.setattr(srv.time, "sleep", time.sleep)
        assert s.delay_start(rows, fps=30)["cameras"] == ["unit1"]
        for white in (True, False, True):
            time.sleep(0.7)
            Cam.level = 200 if white else 40             # the "screen" the camera looks at
            s.delay_flash(white)
        time.sleep(0.7)
        d = s.delay_finish()["cameras"]["unit1"]
        assert d["flashes_seen"] == 3 and 0 <= d["delay_s"] < 0.6 and len(d["delays_s"]) == 3
    finally:
        Cam.running = False
        http.shutdown()


def test_camera_warm_up_is_as_short_as_a_video_warm_up(tmp_path):
    # the config turns camera input into "from now, up to input.max_minutes": a warm-up must not run that long
    assert eng.warmup_sets({"input": {"kind": "video"}}, ["a=1"], 6.5, 7.5) == ["a=1", "section_s=[6.5, 7.5]"]
    sets = eng.warmup_sets({"input": {"kind": "camera"}}, None, 0.0, 32 / 30.0)
    assert sets[0].startswith("section_s=[0.0, ") and sets[1].startswith("input.max_minutes=")
    minutes = float(sets[1].split("=")[1])
    assert abs(minutes * 60 - 32 / 30.0) < 1e-9
    srv, s = _server(tmp_path)
    assert s.cpu_sets("live2d") == ["device=cpu"] and s.cpu_sets("stored2d") == []      # a PC without a GPU
    s.machine = "rtx5090"
    assert s.cpu_sets("live2d") == []
