# tools/tests/test_live_config.py
import json
import os
import re

import pytest
import yaml

from analysis.live import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLIP = os.path.join(ROOT, "configs", "live", "singles_05x.yaml")
DEFAULTS = os.path.join(ROOT, "configs", "live", "defaults.yaml")


def fake_probe(fps=60.0, n=1877):
    """Every view at its calibration's size (portrait or landscape, v0.8.3 checks it against the calibration)."""
    def probe(c):
        out = {}
        for v in c["views"]:
            w, h = config.calib_size(c["cam_paths"][v]) or (2160, 3840)
            out[v] = {"fps": fps, "frame_count": n, "width": w, "height": h, "source": "x"}
        return out
    return probe


def load(overrides=None, preset=None, probe=None):
    return config.load(CLIP, preset=preset, overrides=overrides, defaults=DEFAULTS, probe=probe or fake_probe())


def test_clip_file_resolves_to_300_steps_at_30_fps():
    c = load()
    assert c["views"] == ["back_051", "backright_051", "frontleft_051"]
    assert c["time_ref"] == "back_051" and c["mode"] == "singles"
    assert c["fps"] == 30.0 and c["stride"] == 2 and c["source_fps"] == 60.0
    assert c["ref_frames"][0] == 390 and c["ref_frames"][-1] == 988 and len(c["ref_frames"]) == 300
    assert c["warmup_steps"] == 60 and c["window"] == 30 and c["overlap"] == 5 and c["history"] == 30
    assert c["fit_budget_s"] == pytest.approx(25 / 30)
    assert c["timeline_frames"] == 1877
    assert c["out_clip"] == "singles_05x_3cam_live30"
    assert c["reference_clips"] == ["singles_05x", "singles_05x_3cam"]
    assert c["videos"]["back_051"].endswith("15.8 back 0.5-1.mov")
    assert c["cam_paths"]["back_051"] == os.path.join("data", "calib", "back_051_cam.json")
    assert "seeds_path" not in c and "seeds" not in c


def test_60_fps_doubles_every_step_count():
    c30, c60 = load(), load(["output_fps=60"])
    assert c60["stride"] == 1 and c60["fps"] == 60.0 and len(c60["ref_frames"]) == 600
    for k in ("warmup_steps", "window", "overlap", "history"):
        assert c60[k] == 2 * c30[k], k
    assert c60["fit_budget_s"] == pytest.approx(c30["fit_budget_s"], abs=0.02)
    assert c60["out_clip"].endswith("live60")


def test_layers_merge_in_order(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"stream": {"workers": 7}, "pace": "asfast"}), encoding="utf-8")
    c = load(["stream.workers=3"], preset=str(p))
    assert c["workers"] == 3 and c["pace"] == "asfast"          # CLI beats preset beats defaults
    assert c["stream"]["window_s"] == 1.0                         # untouched nested keys survive the merge


def test_presets_exist_and_load():
    for name in ("laptop-cached", "rtx3060", "rtx5090"):
        c = load(preset=name)
        assert c["backend"] in config.BACKENDS
    gpu = load(preset="rtx3060")["rtmpose"]                       # v0.8.1 fast 2D stack on the GPU presets
    assert (gpu["engine"], gpu["court_filter_m"], gpu["det_every"]) == ("fast", 1.0, 5)
    assert load(preset="laptop-cached")["rtmpose"]["engine"] == "rtmlib"


def test_section_end_null_runs_to_the_end_of_the_video():
    c = load(["section_s=[25.0, null]"])
    assert c["ref_end"] == 1877 and c["ref_frames"][-1] == 1876


def test_fps_comes_from_the_inputs():
    c = load(probe=fake_probe(fps=50.0, n=1000), overrides=["output_fps=25"])
    assert c["stride"] == 2 and c["timeline_frames"] == 1000 and c["ref_frames"][0] == 325


@pytest.mark.parametrize("ov, msg", [
    (["sync.mode=cubic"], "sync.mode"), (["backend=tpu"], "backend"), (["pace=slow"], "pace"),
    (["stream.overlap_s=1.0"], "overlap"), (["sync.offsets={back_051: 0.0}"], "offset"), (["time_ref=back_9"], "time_ref"),
    (["warmup.seconds=20.0"], "warm-up"), (["output_fps=7"], "divide"), (["views=[back_051]"], "at least 2"),
    (["views=[back_051, nope_9]"], "not cameras"), (["section_s=[40, 50]"], "empty or outside"),
    (["rtmpose.model=hands"], "rtmpose.model"),
    (["sync.offsets=file", "paths.sync_file=no/such/sync.json"], "does not exist"),
])
def test_bad_config_is_rejected(ov, msg):
    with pytest.raises(ValueError, match=msg):
        load(ov)


def test_mixed_frame_rates_are_rejected():
    probe = lambda c: {v: {"fps": 60.0 if v != "frontleft_051" else 30.0, "frame_count": 900, "width": 1,
                           "height": 1, "source": "x"} for v in c["views"]}
    with pytest.raises(ValueError, match="different frame rates"):
        load(probe=probe)


def test_sync_file_offsets_are_relative_to_the_reference(tmp_path):
    f = tmp_path / "sync.json"
    f.write_text(json.dumps({"offsets": {"back_051": 1.0, "backright_051": 0.3, "frontleft_051": 0.6}}), encoding="utf-8")
    c = load(["sync.offsets=file", f"paths.sync_file={f.as_posix()}"])
    assert c["offsets"] == pytest.approx({"back_051": 0.0, "backright_051": -0.7, "frontleft_051": -0.4})
    assert c["offsets_source"] == "file"


def test_explicit_offsets_and_auto_default():
    c = load()
    assert c["offsets_source"] == "auto" and set(c["offsets"].values()) == {0.0}
    c = load(["sync.offsets={back_051: 0, backright_051: -0.7, frontleft_051: -0.4}"])
    assert c["offsets"]["backright_051"] == pytest.approx(-0.7) and c["offsets_source"] == "config"


def test_kp2d_header_reads_only_the_header(tmp_path):
    p = tmp_path / "v_kp2d.json"
    p.write_text(json.dumps({"view": "v", "width": 4, "height": 3, "fps": 60.0, "frame_count": 2,
                             "frames": [[], []]}), encoding="utf-8")
    assert config.kp2d_header(str(p)) == {"view": "v", "width": 4, "height": 3, "fps": 60.0, "frame_count": 2}


def test_dump_writes_a_readable_resolved_config(tmp_path):
    c = load()
    config.dump(c, str(tmp_path / "config.resolved.yaml"))
    d = yaml.safe_load((tmp_path / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert d["ref_frames"] == {"first": 390, "last": 988, "count": 300} and d["tracker"]["gate_m"] == 1.5


@pytest.mark.needs_data
def test_real_inputs_probe():
    if not os.path.exists(os.path.join(ROOT, "data", "pose2d", "back_051_kp2d.json")):
        pytest.skip("needs data/pose2d")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        c = config.load(CLIP, defaults=DEFAULTS)
    finally:
        os.chdir(cwd)
    assert c["timeline_frames"] == 1877 and c["source"]["back_051"]["width"] == 2160


# Literals that must never appear in live code (they belong in configs/ or come from the inputs).
FORBIDDEN = [r"singles_05x", r"back_051", r"backright_051", r"frontleft_051", r"\b1877\b", r"SOURCE_FPS\s*=", r"\b60\.0\b"]


def test_live_code_has_no_clip_camera_or_fps_literals():
    live = os.path.join(ROOT, "tools", "analysis", "live")
    hits = []
    for f in sorted(os.listdir(live)):
        if not f.endswith(".py"):
            continue
        txt = open(os.path.join(live, f), encoding="utf-8").read()
        for pat in FORBIDDEN:
            for m in re.finditer(pat, txt):
                line = txt.count("\n", 0, m.start()) + 1
                hits.append(f"{f}:{line}: {pat}")
    assert not hits, "hardcoded values in live code:\n" + "\n".join(hits)
