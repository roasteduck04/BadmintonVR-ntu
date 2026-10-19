# tools/tests/test_live_runner.py
import json
import os
import time

import numpy as np
import pytest

import live_synth as ls
from analysis import schema24
from analysis.live import framewriter, runner, sync


class FakeSource:
    """Serves the synthetic scene's raw detections; raw frame f of view v shows scene time f - offsets[v]."""
    name = "fake"

    def __init__(self, cams, players, offsets):
        self.cams, self.players, self.offsets = cams, players, offsets
        self.info = {"fake": True}

    def start(self, plan, origins):
        self.plan, self.origins = plan, origins

    def begin(self, t0):
        self.t0 = t0

    def try_get(self, v, f):
        t = time.monotonic_ns()
        dets = ls.raw_dets(self.cams, self.players, f, offset=self.offsets.get(v, 0.0))[v]
        return dets, {"t_avail_ns": t, "t_done_ns": time.monotonic_ns(), "decode_ns": 0, "infer_ns": 1000,
                      "wait_ns": 0, "dropped": False, "first_call": False}

    def close(self):
        pass


def _cfg(n=80, warm=30, pace="asfast", offsets_source="config", true_offsets=None):
    c = ls.cfg(n_steps=n, warmup=warm)
    c["ref_frames"] = [r + 20 for r in c["ref_frames"]]                   # room for negative offsets
    c.update({"pace": pace, "backend": "fake", "out_clip": "synth_3cam_live30", "timeline_frames": 2 * n + 60,
              "offsets_source": offsets_source,
              "source": {v: {"fps": 60.0, "frame_count": 2 * n + 60, "width": 1920, "height": 1080} for v in ls.VIEWS}})
    c["output"] = dict(c["output"], frames=True, final_doc=True, copy_to_analysis_dir=False)
    c["sync"] = dict(c["sync"], estimate={"max_offset_s": 0.25, "step_frames": 1, "min_samples": 10})
    if offsets_source == "config" and true_offsets:
        c["offsets"] = dict(true_offsets)
    return c


def _run(tmp_path, c, true_offsets=None):
    cams, players = ls.scene(F=c["timeline_frames"] + 10)
    src = FakeSource(cams, players, true_offsets or {v: 0.0 for v in ls.VIEWS})
    res = runner.Runner(c, str(tmp_path / "run"), log=lambda *a: None, source=src, cams=cams).run()
    return res, tmp_path / "run"


def test_asfast_run_writes_every_frame_the_manifest_and_a_valid_doc(tmp_path):
    c = _cfg()
    res, run = _run(tmp_path, c)
    assert res["steps"] == 80 and res["written"] == 80
    files = sorted(f for f in os.listdir(run / "unity_live_frames") if f.startswith("frame_"))
    assert len(files) == 80 and files[0] == "frame_000000.json" and not any(f.endswith(".tmp") for f in files)
    man = json.loads((run / "unity_live_frames" / "manifest.json").read_text(encoding="utf-8"))
    assert man["live"]["output_fps"] == 30.0 and [p["id"] for p in man["players"]] == ["P1", "P2"]
    assert man["skeleton"]["type"] == "smpl-28"
    latest = json.loads((run / "unity_live_frames" / "latest.json").read_text(encoding="utf-8"))
    assert latest["last_step"] == 79 and latest["n_written"] == 80
    fr = json.loads((run / "unity_live_frames" / "frame_000050.json").read_text(encoding="utf-8"))
    p1 = fr["players"][0]
    assert p1["id"] == "P1" and p1["quality_name"] == "fitted" and len(p1["frame"]["joints_flat"]) == 28 * 4
    assert p1["frame"]["facing"] is not None and fr["frame_id"] == c["ref_frames"][50]
    assert json.loads((run / "unity_live_frames" / "frame_000005.json").read_text(encoding="utf-8"))["players"][0]["warmup"]
    doc = json.loads((run / res["final_doc"]).read_text(encoding="utf-8"))
    assert schema24.validate_analysis24(doc) == []
    for f in ("config.resolved.yaml", "machine.json", "timing.jsonl", "run_summary.json"):
        assert (run / f).exists(), f
    evs = [json.loads(line) for line in (run / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(e["ev"] == "frames" for e in evs) == 80 and sum(e["ev"] == "out" for e in evs) == 80
    assert any(e["ev"] == "window" for e in evs)


def test_auto_sync_recovers_the_offsets_after_warmup(tmp_path):
    true = {"a": 0.0, "b": 0.6, "c": -1.4}
    c = _cfg(n=70, warm=40, offsets_source="auto")
    res, run = _run(tmp_path, c, true_offsets=true)
    est = res["sync"]["offsets"]
    assert est["a"] == 0.0
    for v in ("b", "c"):
        assert abs(est[v] - true[v]) <= 0.45, (v, est[v])
    assert res["offsets"] == est
    sched = sync.schedule(dict(c, offsets=est))
    fr = json.loads((run / "unity_live_frames" / "frame_000060.json").read_text(encoding="utf-8"))
    assert fr["ref_frame"] == sched[60]["ref_frame"]


def test_realtime_pace_takes_about_the_section_length(tmp_path):
    c = _cfg(n=45, warm=30, pace="realtime")
    t = time.perf_counter()
    res, run = _run(tmp_path, c)
    took = time.perf_counter() - t
    assert res["written"] == 45 and took >= 45 / 30.0 - 0.1
    lat = [json.loads(open(f, encoding="utf-8").read())["latency_ms"] for f in framewriter.frame_files(str(run))]
    assert all(x is not None and x >= 0 for x in lat)


def test_run_folder_inside_assets_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Assets"):
        runner.Runner(_cfg(), str(tmp_path / "Assets" / "run"))


def test_frame_files_reads_both_run_layouts(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    (new / "unity_live_frames").mkdir(parents=True)
    (old / "live" / "frames").mkdir(parents=True)
    for d in (new / "unity_live_frames", old / "live"):
        (d / "manifest.json").write_text("{}", encoding="utf-8")
    for s in (1, 0):
        (new / "unity_live_frames" / f"frame_{s:06d}.json").write_text("{}", encoding="utf-8")
        (old / "live" / "frames" / f"{s:06d}.json").write_text("{}", encoding="utf-8")
    (new / "unity_live_frames" / "latest.json").write_text("{}", encoding="utf-8")
    assert [os.path.basename(f) for f in framewriter.frame_files(str(new))] == ["frame_000000.json", "frame_000001.json"]
    assert [os.path.basename(f) for f in framewriter.frame_files(str(old))] == ["000000.json", "000001.json"]
    assert framewriter.frames_dir(str(tmp_path)) is None and framewriter.frame_files(str(tmp_path)) == []


def test_run_folder_name_starts_with_the_clip():
    import importlib.util
    spec = importlib.util.spec_from_file_location("live_run", os.path.join(os.path.dirname(__file__), "..", "live_run.py"))
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)
    d = lr.run_dir_for({"runs_dir": "runs", "clip": "singles_05x", "fps": 60.0, "backend": "live", "pace": "asfast"}, "x")
    name = os.path.basename(d)
    assert name.startswith("live_singles_05x_60fps_live2d_asfast_20") and name.endswith("_x")
