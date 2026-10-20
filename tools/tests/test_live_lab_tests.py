# tools/tests/test_live_lab_tests.py
"""v0.8.3 lab test kit: smaller video copies (transcode + calibration scaling), TensorRT helpers, bench detections,
the campaign runner (plan, resume, STOP, Unity group, pack) and its laptop summary."""
import json
import os
import sys
import time
import zipfile

import numpy as np
import pytest

from analysis.live import bench2d, campaign, config as live_config, pose2d_fast as pf


def _probe(div=1, swap=False):
    """Probe stub: each view's video = its calibration size / div (swap = the wrong orientation)."""
    def probe(c):
        out = {}
        for v in c["views"]:
            w, h = live_config.calib_size(c["cam_paths"][v])
            w, h = (h, w) if swap else (w, h)
            out[v] = {"fps": 60.0, "frame_count": 1877, "width": w // div, "height": h // div}
        return out
    return probe


def test_variant_video_path():
    p = live_config.variant_video(os.path.join("data", "15-08", "singles", "0.5", "15.8 back 0.5-1.mov"), "720p")
    assert p == os.path.join("data", "15-08", "singles", "0.5_720p", "15.8 back 0.5-1.mp4")
    assert live_config.variant_video("a/b.mov", None) == "a/b.mov"


def test_smaller_video_is_rescaled_to_the_calibration():
    full = live_config.load("singles_05x", preset="rtx5090", probe=_probe())
    assert set(full["video_to_calib"].values()) == {1.0}
    small = live_config.load("singles_05x", preset="rtx5090", overrides=["video_variant=720p"], probe=_probe(3))
    assert small["video_variant"] == "720p" and set(small["video_to_calib"].values()) == {3.0}
    assert all("0.5_720p" in p and p.endswith(".mp4") for p in small["videos"].values())
    with pytest.raises(ValueError, match="aspect"):
        live_config.load("singles_05x", preset="rtx5090", overrides=["video_variant=720p"], probe=_probe(3, swap=True))
    with pytest.raises(ValueError, match="engine=fast"):
        live_config.load("singles_05x", preset="rtx5090", overrides=["video_variant=720p", "rtmpose.engine=rtmlib",
                                                                     "rtmpose.decoder=cv2", "rtmpose.decode_short_side=null",
                                                                     "rtmpose.court_filter_m=null"], probe=_probe(3))
    with pytest.raises(ValueError, match="backend live"):
        live_config.load("singles_05x", preset="laptop-cached", overrides=["video_variant=720p"], probe=_probe(3))


def test_tensorrt_settings_are_checked():
    ok = live_config.load("singles_05x", preset="rtx5090", overrides=["rtmpose.ep=trt"], probe=_probe())
    assert ok["rtmpose"]["ep"] == "trt"
    with pytest.raises(ValueError, match="rtmpose.ep"):
        live_config.load("singles_05x", preset="rtx5090", overrides=["rtmpose.ep=openvino"], probe=_probe())
    assert pf._fixed(("N", 3, "h", 288), (8, 3, 384, 288)) == (8, 3, 384, 288)
    assert pf._shape_str("input", (1, 3, 384, 288)) == "input:1x3x384x288"


@pytest.mark.skipif(campaign.has_module("tensorrt_libs") or campaign.has_module("tensorrt"), reason="TensorRT installed")
def test_preload_tensorrt_says_what_to_install():
    with pytest.raises(RuntimeError, match="not installed"):
        pf.preload_tensorrt()


def test_bench_detections_round_trip_at_quarter_pixels(tmp_path):
    rng = np.random.default_rng(0)
    res = {("a", 10): [rng.uniform(0, 3800, (133, 3)).astype(np.float32) for _ in range(2)], ("b", 11): []}
    for d in res[("a", 10)]:
        d[:, 2] = rng.uniform(0, 9, 133)
    p = str(tmp_path / "dets.npz")
    bench2d.save_dets(p, res)
    back = bench2d.load_dets(p)
    assert set(back) == set(res) and back[("b", 11)] == []
    for x, y in zip(res[("a", 10)], back[("a", 10)]):
        assert np.abs(x[:, :2] - y[:, :2]).max() <= 0.125 + 1e-4 and np.abs(x[:, 2] - y[:, 2]).max() <= 5e-4


def test_transcode_keeps_every_frame_in_order(tmp_path):
    av = pytest.importorskip("av")
    from analysis.live import transcode
    src = str(tmp_path / "cam" / "v.mov")
    os.makedirs(os.path.dirname(src))
    with av.open(src, "w") as out:
        st = out.add_stream(transcode.pick_codec(), rate=30)
        st.width, st.height, st.pix_fmt = 128, 72, "yuv420p"
        for i in range(24):
            fr = av.VideoFrame.from_ndarray(np.full((72, 128, 3), 10 * i, np.uint8), format="rgb24")
            for pk in st.encode(fr):
                out.mux(pk)
        for pk in st.encode(None):
            out.mux(pk)
    dst = live_config.variant_video(src, "36p")
    side = transcode.transcode(src, dst, 36, log=lambda *a: None)
    assert side["frames"] == 24 and side["size"] == [64, 36] and os.path.exists(dst + ".json")
    import cv2
    cap = cv2.VideoCapture(dst)
    means = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        means.append(float(img.mean()))
    cap.release()
    assert len(means) == 24 and all(abs(m - 10 * i) < 6 for i, m in enumerate(means))


# ---------------------------------------------------------------------------------------------------- campaign
def test_lab_plan_expands_repeats_groups_and_stages():
    p = campaign.load_plan(campaign.plan_path("lab_5090"))
    ids = [t["id"] for t in p["expanded"]]
    assert len(ids) == len(set(ids)) and "B01_best_live_r1" in ids and "B01_best_live_r2" in ids
    by = {t["id"]: t for t in p["expanded"]}
    assert by["A06_trt16_1080"]["kind"] == "bench2d" and by["A06_trt16_1080"]["requires"] == ["tensorrt"]
    assert by["U01_best_live_unity"]["group"] == "unity" and by["U01_best_live_unity"]["stage"] == "U"
    assert {t["stage"] for t in p["expanded"]} == {"A", "B", "U"}
    assert "sync.offsets=file" in campaign.settings_sets("best_live")
    assert "sync.offsets=auto" in campaign.settings_sets("pure_live")


FAKE = r"""
import json, os, sys
d = sys.argv[1]
os.makedirs(os.path.join(d, "unity_live_frames"), exist_ok=True)
json.dump({"frames": []}, open(os.path.join(d, "unity_live_frames", "manifest.json"), "w"))
json.dump({"step": 0}, open(os.path.join(d, "unity_live_frames", "frame_000000.json"), "w"))
json.dump({"timing": {"achieved_output_fps": 60.0}}, open(os.path.join(d, "report.json"), "w"))
import time; time.sleep(0.4)
if os.path.basename(d).startswith("T2"):
    print("ValueError: boom"); sys.exit(3)
"""


def _plan(tmp_path):
    p = tmp_path / "mini.yaml"
    p.write_text("""name: mini
clip: singles_05x
machine: rtx5090
defaults: {settings: best_live}
tests:
  - {id: T1_ok, note: works}
  - {id: T2_bad, note: fails}
  - {id: T3_skip, requires: ["video:999p"]}
  - {id: T4_unity, group: unity, stage: U}
""", encoding="utf-8")
    return str(p)


def test_campaign_runs_resumes_skips_and_packs(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "argv_for", lambda t, plan, d, python=None: [sys.executable, "-c", FAKE, d])
    from analysis.live import machine
    monkeypatch.setattr(machine, "report", lambda: {"host": "test"})
    unity = tmp_path / "unity"
    (unity / "runs").mkdir(parents=True)
    out = campaign.run_campaign(_plan(tmp_path), group="main", unity_dir=str(unity), log=lambda *a: None)
    s = json.loads(open(os.path.join(out, "campaign.json"), encoding="utf-8").read())["tests"]
    assert s["T1_ok"]["status"] == "done" and s["T2_bad"]["status"] == "failed" and "boom" in s["T2_bad"]["error"]
    assert s["T3_skip"]["status"] == "skipped" and "999p" in s["T3_skip"]["reason"] and "T4_unity" not in s
    assert out == os.path.join(str(unity), "runs", "campaign", "mini")
    mtime = os.path.getmtime(os.path.join(out, "T1_ok", "test.json"))
    campaign.run_campaign(_plan(tmp_path), group="main", unity_dir=str(unity), log=lambda *a: None)
    assert os.path.getmtime(os.path.join(out, "T1_ok", "test.json")) == mtime          # done tests are kept
    with zipfile.ZipFile(out + ".zip") as zf:
        names = zf.namelist()
    assert "mini/T1_ok/report.json" in names and not any("unity_live_frames" in n for n in names)
    # unity group: skipped without a fresh heartbeat; with one, Unity is pointed at the run
    campaign.run_campaign(_plan(tmp_path), group="unity", unity_dir=str(unity), log=lambda *a: None)
    s = json.loads(open(os.path.join(out, "campaign.json"), encoding="utf-8").read())["tests"]
    assert s["T4_unity"]["status"] == "skipped" and "Play mode" in s["T4_unity"]["reason"]
    (unity / "runs" / "unity_status.json").write_text(json.dumps({"playing": True, "t": time.time() + 60}), encoding="utf-8")
    campaign.run_campaign(_plan(tmp_path), group="unity", unity_dir=str(unity), log=lambda *a: None)
    np_ = json.loads((unity / "runs" / "now_playing.json").read_text(encoding="utf-8"))
    assert np_["kind"] == "run" and np_["run"] == "campaign/mini/T4_unity" and np_["seq"] == 1
    # STOP stops before the next test
    open(os.path.join(out, "STOP"), "w").close()
    lines = []
    campaign.run_campaign(_plan(tmp_path), group="main", unity_dir=str(unity), retry_failed=True, log=lines.append)
    assert any("STOP" in ln for ln in lines)


def test_campaign_report_summarises_runs_and_benches(tmp_path, monkeypatch):
    f = tmp_path / "lab"
    (f / "B1").mkdir(parents=True)
    (f / "A1").mkdir()
    (f / "B1" / "report.json").write_text(json.dumps({
        "timing": {"achieved_output_fps": 59.5, "target_fps": 60.0, "realtime_factor": 0.99, "setup_s": 4.0,
                   "stages": {"e2e": {"p50": 1500.0, "p90": 2100.0}}},
        "late_fraction_after_warmup": 0.0, "config": {"views": ["a", "b", "c"]},
        "quality": {"clip": {}, "clip_3cam": {"groups": {"after warm-up": {"w_mpjpe_cm": {"median": 2.4},
                                                                       "pa_mpjpe_cm": {"median": 2.0}}},
                                         "identity": {"P1": {"swap_transitions": 0}}}}}), encoding="utf-8")
    (f / "A1" / "bench.json").write_text(json.dumps({
        "label": "fast", "timing": {"steps_per_s": 120.0, "realtime_x": 2.0, "infer_ms": {"p50": 8.0}},
        "quality": {"parts": {"body": {"median_px": 1.5}}, "matched_frac": 0.98}}), encoding="utf-8")
    (f / "campaign.json").write_text(json.dumps({"plan": "lab", "tests": {
        "A1": {"id": "A1", "kind": "bench2d", "status": "done", "note": "trt"},
        "B1": {"id": "B1", "kind": "run", "status": "done", "note": "best"},
        "B2": {"id": "B2", "kind": "run", "status": "skipped", "reason": "no TensorRT"}}}), encoding="utf-8")
    from analysis.live import report as live_report
    monkeypatch.setattr(live_report, "render_run", lambda d, log=None: None)
    monkeypatch.setattr(bench2d, "rescore", lambda d, log=None: None)
    md = open(campaign.report(str(f), log=lambda *a: None), encoding="utf-8").read()
    assert "| A1 | trt | 120.0 | 2.00 |" in md and "1.50" in md
    assert "| B1 | best |" in md and "59.5 / 60" in md and "1.50 / 2.10" in md and "2.40" in md
    assert "B2" in md and "no TensorRT" in md


def test_tensorrt_session_never_falls_back_to_the_cpu_quietly(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    class FakeSession:
        def __init__(self, path, providers):
            self.providers = providers

        def get_providers(self):
            return ["CPUExecutionProvider"]                 # what onnxruntime does when TensorRT cannot start
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    with pytest.raises(pf.TensorRTUnavailable, match="fell back"):
        pf.trt_session("model.onnx", {"trt_cache": str(tmp_path)}, {"input": ((1, 3, 4, 4),) * 3})
