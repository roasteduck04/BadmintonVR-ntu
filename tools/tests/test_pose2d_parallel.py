"""pose2d_parallel launcher (N camera views at once on one GPU) + pose2d_gpu_timed --views."""
import importlib.util
import json
import os
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(ROOT, "tools", "pose2d_parallel.py")
TIMED = os.path.join(ROOT, "tools", "cost", "pose2d_gpu_timed.py")

FAKE_WORKER = textwrap.dedent('''
    """Stands in for pose2d_gpu_timed.py: same CLI, writes the same files, sleeps instead of running a model."""
    import argparse, json, os, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--clip"); ap.add_argument("--out"); ap.add_argument("--views", nargs="*")
    ap.add_argument("--start-frame", type=int, default=0); ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    (view,) = a.views
    if view.startswith("bad"):
        print("boom"); sys.exit(3)
    time.sleep(0.6)
    json.dump({"view": view, "frames": [[], []]}, open(os.path.join(a.out, f"{view}_kp2d.json"), "w"))
    n = 2 if a.max_frames is None else a.max_frames
    v = {"seconds": 0.5, "frames": n, "fps_processed": 4.0, "model_load_s": 3.0}
    if a.max_frames is not None:
        v.update(start_frame=a.start_frame, frame_ms=[100.0] * n)
    json.dump({"clip": a.clip, "device": "fake", "views": {view: v}},
              open(os.path.join(a.out, f"{a.clip}_{view}_gpu_timing.json"), "w"))
    print("Done!")
''')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    src = tmp_path / "videos"
    src.mkdir()
    cams = {f"v{i}": f"cam {i}.mov" for i in range(5)}
    for fn in cams.values():
        (src / fn).write_bytes(b"x")
    return _load(LAUNCHER, "pose2d_parallel"), worker, src, cams, tmp_path / "out"


def test_runs_views_at_once_and_writes_summary(setup):
    m, worker, src, cams, out = setup
    summary = m.run(str(src), "clipx", str(out), cams, parallel=5, worker=str(worker))
    assert summary["ok"] and summary["parallel"] == 5 and summary["n_views"] == 5
    assert summary["wall_s"] < 5 * 0.6                     # overlapped, not one after another
    for v in cams:
        assert (out / f"{v}_kp2d.json").is_file()
        assert summary["views"][v]["returncode"] == 0 and summary["views"][v]["loop_s"] == 0.5
        assert summary["views"][v]["frames"] == 2
    assert summary["frames_per_view"] == 2 and summary["ms_per_camera_frame"] > 0
    saved = json.loads((out / "clipx_parallel_timing.json").read_text())
    assert saved["wall_s"] == summary["wall_s"]
    assert (out / "_logs" / "clipx_v0.log").read_text().strip().endswith("Done!")


def test_parallel_1_runs_one_after_another(setup):
    m, worker, src, cams, out = setup
    two = dict(list(cams.items())[:2])
    s = m.run(str(src), "clipx", str(out), two, parallel=1, worker=str(worker))
    starts = sorted(v["start_s"] for v in s["views"].values())
    assert starts[1] >= 0.5                                # the second view waited for the first


def test_failed_view_is_reported_not_hidden(setup):
    m, worker, src, cams, out = setup
    (src / "bad.mov").write_bytes(b"x")
    s = m.run(str(src), "clipx", str(out), {"v0": "cam 0.mov", "bad0": "bad.mov"}, parallel=2, worker=str(worker))
    assert not s["ok"] and s["views"]["bad0"]["returncode"] == 3 and s["views"]["v0"]["returncode"] == 0


def test_missing_video_stops_before_starting(setup):
    m, worker, src, cams, out = setup
    with pytest.raises(SystemExit):
        m.run(str(src), "clipx", str(out), {"v0": "nope.mov"}, parallel=1, worker=str(worker))


def test_select_views(setup):
    m = setup[0]
    cams = {"a": "1", "b": "2", "c": "3"}
    assert m.select_views(cams, None) == cams
    assert list(m.select_views(cams, "c,a")) == ["a", "c"]
    with pytest.raises(SystemExit):
        m.select_views(cams, "zz")


def test_timed_script_timing_name_per_view_subset(tmp_path):
    t = _load(TIMED, "pose2d_gpu_timed")
    full = t.write_timing(str(tmp_path), "clipx", "dev", {"a": (1.0, 10)})
    sub = t.write_timing(str(tmp_path), "clipx", "dev", {"a": (1.0, 10)}, name="clipx_a")
    assert os.path.basename(full) == "clipx_gpu_timing.json"          # unchanged default (camsubsets cost reads it)
    assert os.path.basename(sub) == "clipx_a_gpu_timing.json"
    assert json.loads(open(sub).read())["clip"] == "clipx"


def test_timed_script_has_views_option():
    src = open(TIMED, encoding="utf-8").read()
    assert '"--views"' in src


PC_RUN = os.path.join(ROOT, "tools", "pc_run.py")
GPU_CHECK = os.path.join(ROOT, "tools", "gpu_check.py")


def test_pc_run_stops_when_videos_are_missing(tmp_path, monkeypatch, capsys):
    m = _load(PC_RUN, "pc_run")
    monkeypatch.setattr(m, "REPO", str(tmp_path))              # no data/15-08 here
    with pytest.raises(SystemExit) as e:
        m.main(["--step", "check", "--results", str(tmp_path / "results")])
    assert e.value.code == 5 and "MISSING" in capsys.readouterr().out


def test_pc_run_dry_run_prints_every_command(tmp_path, capsys):
    m = _load(PC_RUN, "pc_run")
    for step in ("gpu", "pose2d", "pose24"):
        assert m.main(["--step", step, "--dry-run", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "gpu_check.py" in out and "pose2d_parallel.py" in out and "--parallel 5" in out
    assert "analysis.pose24 build --clip singles_05x" in out
    assert not (tmp_path / "step_times.txt").exists()          # dry run writes nothing


def test_pc_run_logs_and_times_a_step(tmp_path):
    m = _load(PC_RUN, "pc_run")
    r = m.Runner(str(tmp_path))
    assert r.run("hello", [sys.executable, "-c", "print('hi there')"]) == 0
    assert "hi there" in (tmp_path / "hello.log").read_text()
    assert (tmp_path / "step_times.txt").read_text().startswith("hello\t")
    assert r.run("bad", [sys.executable, "-c", "import sys; sys.exit(4)"]) == 4


def test_gpu_check_and_timed_script_import_no_gpu_stack(monkeypatch):
    before = set(sys.modules)
    _load(GPU_CHECK, "gpu_check")
    t = _load(TIMED, "pose2d_gpu_timed")
    assert not {"cv2", "rtmlib", "onnxruntime"} & (set(sys.modules) - before)
    assert callable(t.preload_cuda_dlls)


def test_frame_options_reach_every_camera(setup):
    m, worker, src, cams, out = setup
    s = m.run(str(src), "clipx", str(out), cams, parallel=5, worker=str(worker), start_frame=900, max_frames=1)
    assert s["ok"] and s["start_frame"] == 900 and s["max_frames"] == 1 and s["frames_per_view"] == 1
    for v in cams:
        assert s["views"][v]["frame_ms"] == [100.0] and s["views"][v]["model_load_s"] == 3.0
    assert s["model_ms_per_frame_mean"] == 500.0                 # loop_s 0.5 / 1 frame


def test_pc_run_short_test_skips_3d_and_uses_own_folder(tmp_path, capsys):
    m = _load(PC_RUN, "pc_run")
    m.main(["--max-frames", "1", "--start-frame", "900", "--step", "pose2d", "--dry-run", "--results", str(tmp_path)])
    m.main(["--max-frames", "1", "--step", "pose24", "--dry-run", "--results", str(tmp_path)])
    out = capsys.readouterr().out
    assert "--max-frames 1" in out and "--start-frame 900" in out and "data/pose2d_frames" in out
    assert "analysis.pose24 build" not in out and "3D build needs the whole clip" in out


def test_timing_summary_reads_steps_cameras_and_pose24_stages(tmp_path):
    m = _load(PC_RUN, "pc_run")
    rows = [("gpu_check", "7.3 s"), ("pose2d_parallel5", "1876.9 s"), ("pose24_build", "683.2 s")]
    (tmp_path / "step_times.txt").write_text("".join(f"{n}\t{t}\texit 0\tx\n" for n, t in rows))
    d = tmp_path / "pose2d_parallel5"; d.mkdir()
    (d / "singles_05x_parallel_timing.json").write_text(json.dumps({
        "n_views": 5, "parallel": 5, "frames_per_view": 1877, "ms_per_camera_frame": 199.96,
        "model_ms_per_frame_mean": 190.0,
        "views": {"back_051": {"wall_s": 1284.9, "model_load_s": 8.1, "frames": 1877, "fps": 1.4}}}))
    (tmp_path / "pose24_build.log").write_text("    wrote x; runtime track 259s, lrswap 12s, fit 452s, "
                                               "regress_confidence 2s, total 726s\n")
    text = m.timing_summary(str(tmp_path))
    for want in ("gpu_check", "1876.9", "199.96 ms per camera frame", "190.0 ms", "back_051", "model load", "8.1 s",
                 "track", "259.0", "fit", "452.0", "all steps", "2567.4"):
        assert want in text, want
    assert (tmp_path / "timing_summary.txt").read_text().startswith("===== timing summary")
