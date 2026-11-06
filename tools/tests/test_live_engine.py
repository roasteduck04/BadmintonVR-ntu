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
