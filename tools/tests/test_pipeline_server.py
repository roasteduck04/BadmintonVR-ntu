# tools/tests/test_pipeline_server.py
"""v0.8.2 pipeline server: settings <-> live config mapping, presets, the job runner, now_playing.json, security."""
import json
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline_server"))

import jobs as jobmod  # noqa: E402
import settings as st  # noqa: E402
from analysis.live import config as live_config  # noqa: E402

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402


def test_every_field_maps_to_a_real_config_key():
    cfg = live_config.layered("singles_05x", "rtx3060", {})
    vals = st.values_from(cfg)
    missing = [k for k, v in vals.items() if st.get_key(cfg, k) is None and v is None
               and not any(k == f["key"] and f["kind"] == "optfloat" for f in st.FIELDS)]
    assert not missing, missing
    assert {f["group"] for f in st.FIELDS} <= {g for g, _, _ in st.GROUPS}


def test_best_live_preset_is_the_v081_result():
    p = st.load_preset("best_live")
    assert (p["output_fps"], p["sync.offsets"], p["rtmpose.det_every"], p["rtmpose.court_filter_m"]) == (60, "file", 5, 1.0)
    cfg = live_config.layered("singles_05x", "rtx3060", p)
    assert cfg["stream"]["solver_tol"] == pytest.approx(1e-4) and cfg["rtmpose"]["engine"] == "fast"


def test_overrides_coerce_refuse_and_drop_2d_keys_for_stored_2d():
    o = st.overrides({"output_fps": 30, "rtmpose.det_every": "3", "warmup.player_height_m": ["1", "2.2"],
                      "rtmpose.court_filter_m": None}, backend="live")
    assert o == {"output_fps": 30, "rtmpose.det_every": 3, "warmup.player_height_m": [1.0, 2.2],
                 "rtmpose.court_filter_m": None}
    assert "rtmpose.det_every" not in st.overrides({"rtmpose.det_every": 3}, backend="cached")
    with pytest.raises(ValueError, match="unknown setting"):
        st.overrides({"paths.data_dir": "/etc"}, backend="live")
    with pytest.raises(ValueError, match="not one of"):
        st.overrides({"pace": "warp"}, backend="live")
    args = st.as_set_args({"rtmpose.court_filter_m": None, "warmup.player_height_m": [1.0, 2.3]})
    assert args == ["rtmpose.court_filter_m=null", "warmup.player_height_m=[1.0, 2.3]"]
    assert live_config.parse_overrides(args) == {"rtmpose": {"court_filter_m": None}, "warmup": {"player_height_m": [1.0, 2.3]}}


def test_user_preset_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SETTINGS_DIR", str(tmp_path))
    st.save_preset("my_test", {"output_fps": 30, "stream.fit_every": 2, "not.a.key": 1}, about="x")
    assert st.load_preset("my_test") == {"output_fps": 30, "stream.fit_every": 2}
    for bad in ("best_live", "../x", "Has Space", ""):
        with pytest.raises(ValueError):
            st.save_preset(bad, {})


def test_job_runs_parses_progress_and_stops():
    code = ("import time,sys\n"
            "for i in range(0, 31, 10):\n    print(f'    step {i}/30  written {i}  (0.1 s)', flush=True); time.sleep(0.05)\n"
            "print('run done in 0.2 s: 30/30 frames', flush=True)\n")
    j = jobmod.Job("t", [sys.executable, "-c", code], cwd=HERE).start()
    assert j.wait(20) == "done" and j.progress["written"] == 30 and j.progress["steps"] == 30
    slow = jobmod.Job("t", [sys.executable, "-c", "import time\nprint('x', flush=True)\ntime.sleep(60)"], cwd=HERE).start()
    time.sleep(0.5)
    t = time.time()
    slow.stop()
    assert slow.state == "stopped" and time.time() - t < 15
    bad = jobmod.Job("t", [sys.executable, "-c", "raise ValueError('boom')"], cwd=HERE).start()
    assert bad.wait(20) == "failed" and "boom" in (bad.error or "")


def test_one_job_at_a_time():
    m = jobmod.JobManager()
    m.start(jobmod.Job("a", [sys.executable, "-c", "import time; time.sleep(30)"], cwd=HERE))
    with pytest.raises(RuntimeError, match="running"):
        m.start(jobmod.Job("b", [sys.executable, "-c", "pass"], cwd=HERE))
    m.stop()


def _app(tmp_path):
    unity = tmp_path / "unity"
    (unity / "Assets" / "StreamingAssets" / "analysis").mkdir(parents=True)
    (unity / "Assets" / "StreamingAssets" / "analysis" / "singles_05x_pose24.analysis.json").write_text("{}", encoding="utf-8")
    run = unity / "runs" / "live_singles_05x_30fps_stored2d_realtime_2026-10-19_120000"
    (run / "unity_live_frames").mkdir(parents=True)
    (run / "unity_live_frames" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "unity_live_frames" / "frame_000000.json").write_text("{}", encoding="utf-8")
    s = srv.Server(unity_project=unity, machine="laptop-cached")
    return s, TestClient(srv.create_app(s))


def test_api_lists_play_pointer_and_settings(tmp_path):
    s, c = _app(tmp_path)
    info = c.get("/api/info").json()
    assert info["machine"] == "laptop-cached" and info["default_settings"] == "laptop_realtime"
    assert [x["name"] for x in c.get("/api/clips").json()] == ["singles_05x"]
    v = c.get("/api/settings", params={"clip": "singles_05x", "preset": "laptop_realtime"}).json()["values"]
    assert v["backend"] == "cached" and v["output_fps"] == 30 and v["stream.solver_tol"] == pytest.approx(1e-4)
    runs = c.get("/api/runs").json()
    assert runs[0]["frames"] == 1 and runs[0]["layout"] == "unity_live_frames"
    np1 = c.post("/api/play", json={"kind": "run", "name": runs[0]["name"]}).json()
    np2 = c.post("/api/play", json={"kind": "clip", "name": "singles_05x_pose24.analysis.json"}).json()
    assert np2["seq"] == np1["seq"] + 1 and np2["file"] == "singles_05x_pose24.analysis.json"
    on_disk = json.loads((s.runs / "now_playing.json").read_text(encoding="utf-8"))
    assert on_disk == np2 and "run" not in on_disk
    assert c.post("/api/play", json={"kind": "run", "name": "../../etc"}).status_code == 400
    assert c.post("/api/play", json={"kind": "clip", "name": "missing.analysis.json"}).status_code == 404
    assert c.get("/api/runs/..%2F..%2Fx/report").status_code in (400, 404)


def test_stream_start_builds_the_live_run_command(tmp_path, monkeypatch):
    s, c = _app(tmp_path)
    started = {}

    def fake_start(job):
        started["job"] = job
        job.state = "running"
        return job
    monkeypatch.setattr(s.jobs, "start", fake_start)
    r = c.post("/api/stream/start", json={"oneshot": True, "clip": "singles_05x", "preset": "laptop_realtime", "force": True,
                                          "values": {"output_fps": 60, "rtmpose.det_every": 3}})
    assert r.status_code == 200, r.text
    argv = started["job"].argv
    assert argv[argv.index("--preset") + 1] == "laptop-cached" and "output_fps=60" in argv
    assert not any(a.startswith("rtmpose.") for a in argv)                  # stored 2D: 2D-speed keys dropped
    assert os.path.basename(argv[argv.index("--run-dir") + 1]).startswith("live_singles_05x_60fps_stored2d_realtime_")
    started["job"].state = "done"
    assert c.post("/api/stream/start", json={"oneshot": True, "clip": "nope"}).status_code == 400
    assert c.post("/api/stream/start", json={"oneshot": True, "clip": "singles_05x", "force": True, "values": {"pace": "warp"}}).status_code == 400


def test_foreign_host_and_cross_site_post_are_refused(tmp_path):
    s, _ = _app(tmp_path)
    c = TestClient(srv.create_app(s), base_url="http://evil.example")
    assert c.get("/api/info").status_code == 400
    c2 = TestClient(srv.create_app(s))
    assert c2.post("/api/play", json={"kind": "stop"}, headers={"origin": "http://evil.example"}).status_code == 403
    assert c2.post("/api/play", json={"kind": "stop"}, headers={"origin": "http://127.0.0.1:8770"}).status_code == 200


# ---------------------------------------------------------------- compare + build-step progress
def test_job_parses_build_steps():
    code = "print('STEP 1/3 archive', flush=True)\nprint('STEP 2/3 3D build (pose24)', flush=True)\nprint('build done in 5 s: x', flush=True)\n"
    j = jobmod.Job("build", [sys.executable, "-c", code], cwd=HERE).start()
    assert j.wait(20) == "done" and j.progress["steps"] == 3 and j.progress["label"] == "done"


def test_compare_lists_only_differing_settings(tmp_path):
    s, c = _app(tmp_path)
    for name, fps in (("run_a", 30), ("run_b", 60)):
        d = s.runs / name
        (d / "unity_live_frames").mkdir(parents=True)
        (d / "unity_live_frames" / "manifest.json").write_text("{}", encoding="utf-8")
        (d / "config.resolved.yaml").write_text(f"output_fps: {fps}\npace: realtime\nbackend: cached\n", encoding="utf-8")
    r = c.get("/api/compare", params={"runs": "run_a,run_b"}).json()
    assert [x["key"] for x in r["differs"]] == ["output_fps"] and r["rows"][1]["settings"]["output_fps"] == 60
    assert c.get("/api/compare", params={"runs": "run_a,../x"}).status_code == 400


def test_compare_reads_numeric_text_as_numbers(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "config.resolved.yaml").write_text("stream:\n  solver_tol: 1e-4\n", encoding="utf-8")
    assert srv.flat_settings(str(d))["stream.solver_tol"] == pytest.approx(1e-4)



def test_stream_start_checks_that_unity_is_in_play_mode(tmp_path, monkeypatch):
    s, c = _app(tmp_path)
    monkeypatch.setattr(s.jobs, "start", lambda job: job)
    assert c.get("/api/unity").json() == {"in_play": False, "seen": False}
    r = c.post("/api/stream/start", json={"oneshot": True, "clip": "singles_05x", "preset": "laptop_realtime"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "unity_not_playing"
    hb = s.runs / "unity_status.json"
    hb.write_text(json.dumps({"v": 1, "playing": True, "t": time.time() - 10, "mode": "idle"}), encoding="utf-8")
    assert c.get("/api/unity").json()["in_play"] is False                    # stale heartbeat: Unity stopped
    hb.write_text(json.dumps({"v": 1, "playing": True, "t": time.time(), "mode": "broadcast", "received": 5}),
                  encoding="utf-8")
    u = c.get("/api/unity").json()
    assert u["in_play"] is True and u["mode"] == "broadcast"
    assert c.post("/api/stream/start", json={"oneshot": True, "clip": "singles_05x", "preset": "laptop_realtime"}).status_code == 200
    hb.write_text(json.dumps({"v": 1, "playing": False, "t": time.time()}), encoding="utf-8")   # Play mode ended
    assert c.get("/api/unity").json()["in_play"] is False


# ---------------------------------------------------------------- tab 1 Calibrate
def test_calibrator_page_is_served_with_its_paths_under_calib(tmp_path):
    s, c = _app(tmp_path)
    html = c.get("/calibrate").text
    assert "'/calib/api/cameras'" in html and "'/api/" not in html and "window.EMBED_LIVE_SERVER" in html
    pts = c.get("/calib/api/court-points").json()
    assert len(pts) >= 4 and "corner_nl" in pts
    cams = {x["view"]: x for x in c.get("/calib/api/cameras").json()}
    assert cams["back_051"]["clip"] == "singles_05x" and cams["back_051"]["video"].endswith(".mov")
    assert "singles_05x" in c.get("/calib/api/clips").json()
    assert c.get("/calib/frames/..%2F..%2Fsecret.jpg").status_code in (400, 404)
    assert c.post("/calib/api/calibrate", json={"view": "nope", "points": []}).status_code == 400


def test_calibrate_solves_archives_and_writes(tmp_path, monkeypatch):
    import calib as cm
    repo = tmp_path / "repo"
    (repo / "data" / "calib" / "frames").mkdir(parents=True)
    import cv2
    import numpy as np
    cv2.imwrite(str(repo / "data" / "calib" / "frames" / "v1.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    k = cm.Calib(repo, repo)
    monkeypatch.setattr(k, "cameras", lambda: [{"view": "v1", "video": "x.mov", "has_video": False, "clip": "c"}])
    # synthetic camera: project court points with a known K, R, t and click exactly there
    from analysis import multiview as mv
    pts = cm._court_points()
    names = ["corner_nl", "corner_nr", "corner_fl", "corner_fr", "ssl_nl", "ssl_nr", "net_l", "net_r"]
    K = mv.intrinsics_from_fov(1920, 1080, 70.0, on="long")
    court = np.array([pts[n] for n in names], float)
    R0, t0 = mv.solve_extrinsics(court, np.array([[400, 900], [1500, 900], [700, 300], [1200, 300],
                                                  [450, 650], [1450, 650], [520, 500], [1400, 500]], float), K)
    X = np.c_[court[:, 0], np.zeros(len(court)), court[:, 1]]
    P = (K @ (R0 @ X.T + np.asarray(t0).reshape(3, 1))).T
    px = P[:, :2] / P[:, 2:3]
    body = {"view": "v1", "fov_deg": 70.0, "points": [{"name": n, "px": float(a), "py": float(b)} for n, (a, b) in zip(names, px)]}
    (repo / "data" / "calib" / "v1_cam.json").write_text("{\"old\": true}", encoding="utf-8")
    r = k.calibrate(body)
    assert r["rms_px"] < 1.0 and not r["warning"] and r["archived"] and (repo / r["archived"]).exists()
    saved = json.loads((repo / "data" / "calib" / "v1_cam.json").read_text(encoding="utf-8"))
    assert saved["view"] == "v1" and saved["fov_deg"] == 70.0 and set(saved["court_points"]) == set(names)
    best = k.auto_fov(body)
    assert abs(best["fov_deg"] - 70.0) < 3.0


def test_sync_file_from_the_offline_build(tmp_path, monkeypatch):
    import calib as cm
    repo = tmp_path / "repo"
    (repo / "data" / "calib").mkdir(parents=True)
    a = repo / "Assets" / "StreamingAssets" / "analysis"
    a.mkdir(parents=True)
    (a / "clipx_pose24.analysis.json").write_text(json.dumps({"meta": {"pose_variant": {
        "offsets": {"v1": 0.0, "v2": -0.7}, "time_ref_view": "v1"}}}), encoding="utf-8")
    k = cm.Calib(repo, repo)
    monkeypatch.setattr(k, "registry", lambda: {"clipx": {"views": ["v1", "v2"]}, "clipy": {"views": ["v3"]}})
    st_ = {x["clip"]: x for x in k.sync_status()}
    assert st_["clipx"]["has_build"] and st_["clipx"]["sync"] is None and not st_["clipy"]["has_build"]
    out = k.make_sync("clipx")
    assert out["offsets"] == {"v1": 0.0, "v2": -0.7} and out["archived"] is None
    assert k.make_sync("clipx")["archived"]                                   # the second one archives the first
    with pytest.raises(Exception):
        k.make_sync("clipy")                                                  # no offline build to take offsets from


# ---------------------------------------------------------------------------------------------- v0.8.2 follow-up
def test_time_settings_are_frames_on_the_page_and_seconds_in_the_config():
    cfg = {"output_fps": 60, "section_s": [6.5, 16.5], "warmup": {"seconds": 2.0},           # no data files needed
           "stream": {"window_s": 1.0, "history_s": 1.0}, "facing": {"window_s": 0.25},
           "fallback": {"max_hold_s": 0.1667}, "sync": {"estimate": {"max_offset_s": 0.25}}}
    v = st.values_from(cfg)
    assert v["warmup.seconds"] == 120 and v["stream.window_s"] == 60 and v["facing.window_s"] == 15
    assert v["section_s"] == [390, 990]                                       # 6.5-16.5 s at 60 fps
    assert all(f["unit"] == "frames" for f in st.FIELDS if f.get("frames"))
    # changed frames -> seconds at the page's fps; unchanged ones keep the config's exact seconds
    cfg30 = dict(cfg, output_fps=30)
    v30 = st.values_from(cfg30)
    assert v30["facing.window_s"] == 8                                         # 0.25 s * 30 = 7.5 -> 8 frames
    page = {k: v30[k] for k in st.FRAME_KEYS | {"output_fps"}}
    o = st.overrides({**page, "warmup.seconds": 90}, backend="cached", base_cfg=cfg30)
    assert o["warmup.seconds"] == 3.0 and o["facing.window_s"] == 0.25 and o["section_s"] == [6.5, 16.5]
    o = st.overrides({"output_fps": 60, "warmup.seconds": 30, "section_s": [60, None]}, backend="cached")
    assert o["warmup.seconds"] == 0.5 and o["section_s"] == [1.0, None]
    with pytest.raises(ValueError, match="whole number"):
        st.overrides({"output_fps": 60, "warmup.seconds": 1.5}, backend="cached")
    with pytest.raises(ValueError, match="output_fps"):
        st.overrides({"warmup.seconds": 30}, backend="cached")


def test_saved_preset_stores_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SETTINGS_DIR", str(tmp_path))
    st.save_preset("frames_test", {"output_fps": 60, "warmup.seconds": 180, "section_s": [390, None]})
    assert st.load_preset("frames_test") == {"output_fps": 60, "warmup.seconds": 3.0, "section_s": [6.5, None]}
    for name in st.BUILTIN:
        with pytest.raises(ValueError):
            st.save_preset(name, {})


def test_pure_live_preset_takes_nothing_from_an_offline_build():
    p = st.load_preset("pure_live")
    assert p["backend"] == "live" and p["sync.offsets"] == "auto"
    own = ("sync.offsets", "warmup.seconds")          # pure_live keeps the 120-frame warm-up for its sync estimate
    assert "warmup.seconds" not in p
    assert {k: v for k, v in p.items() if k not in own} == \
        {k: v for k, v in st.load_preset("best_live").items() if k not in own}


def test_stream_start_passes_untouched_seconds_through(tmp_path, monkeypatch):
    s, c = _app(tmp_path)
    started = {}
    monkeypatch.setattr(s.jobs, "start", lambda job: started.setdefault("job", job))
    r = c.post("/api/stream/start", json={"oneshot": True, "clip": "singles_05x", "preset": "laptop_realtime", "force": True,
                                          "values": {"output_fps": 60, "warmup.seconds": 60}})
    assert r.status_code == 200, r.text
    job = started["job"]
    sets = [a for a in job.argv if "=" in a]
    assert "warmup.seconds=1.0" in sets and "facing.window_s=0.25" in sets and "section_s=[6.5, 16.5]" in sets
    assert job.meta["warmup_steps"] == 60 and job.meta["views"] == 3 and job.meta["fps"] == 60.0
    assert job.meta["frames_dir"].endswith("unity_live_frames")


def test_stream_job_phases_and_frame_count(tmp_path):
    run = tmp_path / "run"
    (run / "unity_live_frames").mkdir(parents=True)
    code = ("import time,json,sys\n"
            "p=sys.argv[1]\n"
            "print('12:00:00 run folder x', flush=True); time.sleep(0.3)\n"
            "for v in ('a','b'): print(f'12:00:01   2D worker {v}: model ready in 1.0 s, providers', flush=True)\n"
            "time.sleep(0.3)\n"
            "print('12:00:02   10 steps at 30 fps (realtime, backend live), setup 2.0 s (source ready 1.0 s)', flush=True)\n"
            "time.sleep(0.3)\n"
            "json.dump({'n_written': 4}, open(p+'/unity_live_frames/latest.json','w')); time.sleep(0.6)\n"
            "json.dump({'n_written': 10}, open(p+'/unity_live_frames/latest.json','w')); time.sleep(0.6)\n"
            "print('12:00:05 run done in 3.0 s: 10/10 frames', flush=True); time.sleep(0.3)\n"
            "print('12:00:06 report x/report.md', flush=True)\n")
    j = jobmod.Job("stream", [sys.executable, "-c", code, str(run)], cwd=HERE, run_dir=str(run),
                   meta={"views": 2, "warmup_steps": 3, "backend": "live"}).start()
    seen = set()
    while j.running:
        s_ = j.status()
        act = [p["id"] for p in s_["phases"] if p["state"] == "active"]
        seen.update(act)
        if "models" in act:
            assert "cameras" in next(p for p in s_["phases"] if p["id"] == "models")["detail"]
        time.sleep(0.05)
    final = j.status()
    assert final["state"] == "done" and all(p["state"] == "done" for p in final["phases"])
    assert {"models", "stream", "save", "report"} <= seen
    assert final["progress"]["written"] == 10 and final["phases"][1]["detail"].startswith("2 / 2 cameras · setup 2.0 s")


def test_error_line_is_picked_up():
    assert jobmod.ERR_RE.match("ValueError: boom") and jobmod.ERR_RE.match("RuntimeError: 2D worker failed")
    assert not jobmod.ERR_RE.match("12:00:00 step 1/10")


def test_run_timing_reads_the_report(tmp_path):
    run = tmp_path / "live_x"
    run.mkdir()
    (run / "run.log").write_text("10:00:00 live_run: run\n10:00:05 run done in 4.0 s: 10/10 frames\n"
                                 "10:00:09 report x/report.md\n", encoding="utf-8")
    (run / "report.json").write_text(json.dumps({"timing": {
        "steps": 10, "written": 10, "setup_s": 1.5, "wall_s": 3.0, "realtime_factor": 0.5, "achieved_output_fps": 15.0,
        "target_fps": 30.0, "stages": {"e2e": {"p50": 900.0, "p90": 1200.0, "max": 1500.0}, "infer": {"p50": 20.0}}},
        "late_fraction_after_warmup": 0.0, "config": {"pace": "realtime"}}), encoding="utf-8")
    t = srv.run_timing(str(run))
    assert t["has_report"] and t["job_s"] == 9 and t["report_s"] == 4 and t["setup_s"] == 1.5
    assert [s["key"] for s in t["stages"]] == ["infer", "e2e"] and t["e2e"]["p50"] == 900.0
    assert srv.run_summary(str(run))["setup_s"] == 1.5 and srv.run_summary(str(run))["delay_p50_ms"] == 900.0


def _synthetic_clicks(fov, names):
    import calib as cm
    import numpy as np
    from analysis import multiview as mv
    pts = cm._court_points()
    K = mv.intrinsics_from_fov(1920, 1080, fov, on="long")
    court = np.array([pts[n] for n in names], float)
    R0, t0 = mv.solve_extrinsics(court[:4], np.array([[400, 900], [1500, 900], [700, 300], [1200, 300]], float), K)
    X = np.c_[court[:, 0], np.zeros(len(court)), court[:, 1]]
    P = (K @ (R0 @ X.T + np.asarray(t0).reshape(3, 1))).T
    return court, P[:, :2] / P[:, 2:3]


def test_calibrate_finds_the_field_of_view_itself(tmp_path, monkeypatch):
    import calib as cm
    import cv2
    import numpy as np
    repo = tmp_path / "repo"
    (repo / "data" / "calib" / "frames").mkdir(parents=True)
    cv2.imwrite(str(repo / "data" / "calib" / "frames" / "v1.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    k = cm.Calib(repo, repo)
    monkeypatch.setattr(k, "cameras", lambda: [{"view": "v1", "video": "x.mov", "has_video": False, "clip": "c"}])
    names = ["corner_nl", "corner_nr", "corner_fl", "corner_fr", "ssl_nl", "ssl_nr", "net_l", "net_r"]
    _, px = _synthetic_clicks(73.3, names)
    px[4] += [6.0, 0.0]                                                       # one sloppy click
    r = k.calibrate({"view": "v1", "points": [{"name": n, "px": float(a), "py": float(b)} for n, (a, b) in zip(names, px)]})
    assert abs(r["fov_deg"] - 73.3) < 1.0 and r["worst"]["name"] == "ssl_nl" and len(r["points"]) == len(names)
    saved = json.loads((repo / "data" / "calib" / "v1_cam.json").read_text(encoding="utf-8"))
    assert saved["fov_auto"] is True and saved["court_points"]["ssl_nl"]["err_px"] > 1.0
    rp = cm.Calib.reprojected(saved)
    assert set(rp) == set(names) and rp["ssl_nl"][2] == max(v[2] for v in rp.values())


def test_solve_camera_warns_about_points_on_a_line():
    import numpy as np
    from analysis import multiview as mv
    names = ["corner_nl", "corner_nr", "corner_fl", "corner_fr"]
    _, px4 = _synthetic_clicks(70.0, names)
    court = np.array([[-3.05, -6.7], [-1.0, -6.7], [1.0, -6.7], [3.05, -6.7], [0.0, -6.6]], float)
    K = mv.intrinsics_from_fov(1920, 1080, 70.0)
    import calib as cm
    pts = cm._court_points()
    R, t = mv.solve_extrinsics(np.array([pts[n] for n in names], float), px4, K)
    X = np.c_[court[:, 0], np.zeros(len(court)), court[:, 1]]
    P = (K @ (R @ X.T + t.reshape(3, 1))).T
    s = mv.solve_camera(1920, 1080, court, P[:, :2] / P[:, 2:3], fov_deg=70.0)
    assert any("one line" in n for n in s["notes"]) and s["rms_px"] < 1.0


def test_find_best_fov_refines_below_the_coarse_step():
    from analysis import multiview as mv
    names = ["corner_nl", "corner_nr", "corner_fl", "corner_fr", "net_l", "net_r", "ssl_ctr_n"]
    court, px = _synthetic_clicks(81.37, names)
    fov, rms = mv.find_best_fov(1920, 1080, court, px)
    assert abs(fov - 81.37) < 0.1 and rms < 0.5
