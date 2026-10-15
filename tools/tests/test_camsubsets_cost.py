"""camsubsets cost table (capture MB, CPU-s, GPU-s) + the timed copy of the archived 2D GPU script."""
import ast
import csv
import importlib.util
import json
import os
import sys

import pytest

import camsubsets_synth as cs_syn
from analysis.pose24.camsubsets import __main__ as cli
from analysis.pose24.camsubsets import cost, oracle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMED = os.path.join(ROOT, "tools", "cost", "pose2d_gpu_timed.py")
ARCHIVED = os.path.join(ROOT, "_archive", "phase4_gpu", "pose2d_gpu.py")


def _videos(tmp_path, sizes):
    vids = {}
    for v, mb in sizes.items():
        p = tmp_path / "vid" / f"{v}.mov"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"\0" * int(mb * 1e6))
        vids[v] = os.path.join("vid", f"{v}.mov")                        # relative to data_dir (the repo root)
    return vids


def _timing(views, clip="c"):
    return {"clip": clip, "device": "fake", "views": {v: {"seconds": s, "frames": 100, "fps_processed": 100 / s}
                                                   for v, s in views.items()}}


def test_cost_rows_sum_views(tmp_path):
    reg = {"c": {"views": ["a", "b"], "videos": _videos(tmp_path, {"a": 1, "b": 2})}}
    assert cost.video_mb(reg, "c", data_dir=str(tmp_path)) == {"a": 1.0, "b": 2.0}
    res = {"oracle": {"fit": {"a+b": {"runtime_s": 12.5, "cpu_s": 11.0, "workers": 6}}}}
    rows = cost.cost_rows("c", reg, None, res, gpu_timing=_timing({"a": 10.0, "b": 20.0}), data_dir=str(tmp_path))
    (r,) = rows
    assert r["subset"] == "a+b" and r["n_cams"] == 2 and r["mode"] == "oracle" and r["method"] == "fit"
    assert r["capture_mb"] == 3.0 and r["gpu_2d_s"] == 30.0 and r["cpu_fit_s"] == 11.0 and r["cpu_total_s"] == 11.0
    assert r["wall_fit_s"] == 12.5 and r["workers"] == 6
    assert r["cpu_track_s"] is None and r["setup_note"] == "not measured"
    (r,) = cost.cost_rows("c", reg, None, res, data_dir=str(tmp_path))
    assert r["gpu_2d_s"] is None and r["capture_mb"] == 3.0


def test_cost_rows_missing_video_or_timed_view_and_e2e_and_full(tmp_path):
    reg = {"c": {"views": ["a", "b", "c"], "videos": dict(_videos(tmp_path, {"a": 1, "b": 2}), c="vid/none.mov")}}
    res = {"oracle": {"fit": {"a+b": {"runtime_s": 5.0}, "a+c": {"runtime_s": 6.0}},
                      "mono-len-prior": {"b": {"runtime_s": 1.5, "cpu_s": 1.25, "workers": 6}}},
           "e2e": {"fit": {"a+b": {"runtime_s": 99.0, "cpu_s": 70.0, "workers": 3,
                                   "e2e_runtime_s": {"track": 40.0, "fit": 30.0, "total": 90.0}},
                           "b+c": {"runtime_s": 80.0}}}}
    rows = cost.cost_rows("c", reg, {"track": 70.0, "fit": 50.0, "total": 150.0}, res,
                          gpu_timing=_timing({"a": 1.0, "b": 2.0}), data_dir=str(tmp_path), setup_min_per_cam=5)
    by = {(r["mode"], r["method"], r["subset"]): r for r in rows}
    assert by[("oracle", "fit", "a+c")]["capture_mb"] is None and by[("oracle", "fit", "a+c")]["gpu_2d_s"] is None
    m = by[("oracle", "mono-len-prior", "b")]
    assert m["n_cams"] == 1 and m["capture_mb"] == 2.0 and m["gpu_2d_s"] == 2.0 and m["cpu_fit_s"] == 1.25
    o = by[("oracle", "fit", "a+b")]                                       # stored before cpu_s (R18): wall only
    assert o["cpu_fit_s"] is None and o["cpu_total_s"] is None and o["wall_fit_s"] == 5.0 and o["workers"] is None
    e = by[("e2e", "fit", "a+b")]                                          # the doc's steps are wall under workers
    assert e["cpu_track_s"] is None and e["cpu_fit_s"] is None and e["cpu_total_s"] == 70.0
    assert e["wall_track_s"] == 40.0 and e["wall_fit_s"] == 30.0 and e["wall_total_s"] == 99.0 and e["workers"] == 3
    b = by[("e2e", "fit", "b+c")]                                          # built before e2e_runtime_s / cpu_s
    assert b["wall_track_s"] is None and b["cpu_total_s"] is None and b["wall_total_s"] == 80.0
    f = by[("full", "fit", "a+b+c")]
    assert f["n_cams"] == 3 and f["cpu_track_s"] == 70.0 and f["cpu_fit_s"] == 50.0 and f["cpu_total_s"] == 150.0
    assert f["wall_fit_s"] == 50.0 and f["workers"] == 1
    assert m["setup_note"] == "5 min (1 x 5 min)" and f["setup_note"] == "15 min (3 x 5 min)"
    assert rows[0]["mode"] == "full"


def test_load_timing_file_or_dir(tmp_path):
    t = _timing({"a": 1.0})
    p = tmp_path / "c_gpu_timing.json"
    p.write_text(json.dumps(t), encoding="utf-8")
    assert cost.load_timing(str(p), "c") == t and cost.load_timing(str(tmp_path), "c") == t
    assert cost.load_timing(str(p), "other") is None and cost.load_timing(str(tmp_path), "other") is None
    assert cost.load_timing(None, "c") is None


def test_write_csv_blanks_none_and_nan(tmp_path):
    p = cost.write_csv([{"clip": "c", "capture_mb": None, "gpu_2d_s": float("nan"), "cpu_fit_s": 1.0}],
                       str(tmp_path / "x" / "cost.csv"))
    with open(p, newline="", encoding="utf-8") as fh:
        (r,) = list(csv.DictReader(fh))
    assert r["capture_mb"] == "" and r["gpu_2d_s"] == "" and r["cpu_fit_s"] == "1.0"
    assert list(r) == list(cost.COLUMNS)


def test_cli_cost_reads_the_result_store(tmp_path, capsys):
    cache_path, cache, _ = cs_syn.full_cache(tmp_path, F=10)
    vids = _videos(tmp_path, {"a": 1, "b": 1})
    regp = tmp_path / "clips.json"
    regp.write_text(json.dumps({"clips": {"synfull": {"mode": "singles", "views": cs_syn.VIEWS, "videos": vids}}}),
                    encoding="utf-8")
    out_dir = str(tmp_path / "subsets")
    meta = oracle.expected_meta(cache, "fit", None, None)
    oracle.save_result(oracle.result_path(out_dir, "synfull", "oracle", "fit", ("a", "b")),
                       {"players": {}, "views": ["a", "b"], "method": "fit", "runtime_s": 4.0}, dict(meta, cpu_s=4.0))
    oracle.save_result(oracle.result_path(out_dir, "synfull", "oracle", "fit", ("a", "c")),
                       {"players": {}, "views": ["a", "c"], "method": "fit", "runtime_s": 4.0},
                       dict(meta, build_id="stale"))
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "synfull_gpu_timing.json").write_text(json.dumps(_timing({"a": 3.0, "b": 4.0}, "synfull")),
                                                            encoding="utf-8")
    csvp = str(tmp_path / "cost.csv")
    rc = cli.main(["cost", "--clip", "synfull", "--gpu-timing", str(tmp_path / "t"), "--out", csvp,
                   "--registry", str(regp), "--cache-dir", str(tmp_path), "--out-dir", out_dir,
                   "--analysis-dir", str(tmp_path / "an"), "--data-dir", str(tmp_path)])
    assert rc == 0
    with open(csvp, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["subset"] for r in rows] == ["a+b"]                           # the stale a+c result skipped
    assert rows[0]["capture_mb"] == "2.0" and rows[0]["gpu_2d_s"] == "7.0" and rows[0]["cpu_fit_s"] == "4.0"
    out = capsys.readouterr().out
    assert "a+b" in out and "stale" in out and csvp in out


def _load_timed():
    spec = importlib.util.spec_from_file_location("pose2d_gpu_timed", TIMED)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_timed_script_imports_no_gpu_stack_at_module_level():
    tree = ast.parse(open(TIMED, encoding="utf-8").read())
    top = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            top |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            top.add((n.module or "").split(".")[0])
    assert not top & {"cv2", "torch", "rtmlib", "byte_tracker", "onnxruntime"}
    before = set(sys.modules)
    m = _load_timed()
    assert not {"cv2", "rtmlib", "torch"} & (set(sys.modules) - before)
    assert m.CLIP_CAMERAS and callable(m.run_view_multi)


# test_timed_script_keeps_the_archived_2d_code removed in the NTU copy: the 2D script lives at tools/pose2d_gpu.py here, not _archive/phase4_gpu/.
def test_timed_script_writes_the_timing_json(tmp_path, monkeypatch):
    m = _load_timed()
    frames = [[], [{"track_id": 1, "keypoints": []}]] * 50
    p = m.write_timing(str(tmp_path), "singles_05x", "Fake GPU", {"back_051": (2.0, len(frames)),
                                                                  "back_052": (1.0, len(frames))})
    assert p == os.path.join(str(tmp_path), "singles_05x_gpu_timing.json")
    got = json.load(open(p, encoding="utf-8"))
    assert got == {"clip": "singles_05x", "device": "Fake GPU",               # first view: CUDA / model warm-up
                   "views": {"back_051": {"seconds": 2.0, "frames": 100, "fps_processed": 50.0,
                                          "warmup_included": True},
                             "back_052": {"seconds": 1.0, "frames": 100, "fps_processed": 100.0}}}
    monkeypatch.setitem(sys.modules, "torch", None)                        # no torch -> "cuda"
    assert m.device_name() == "cuda"
