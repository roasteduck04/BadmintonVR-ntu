import json
import os
import pathlib
import subprocess
import sys

from analysis import schema, synth_analysis as sa


def test_build_synth_doc_validates():
    doc = sa.build_synth_doc(frames=120, fps=30.0)
    assert schema.validate_analysis(doc) == []


def test_has_two_players_with_sides():
    doc = sa.build_synth_doc()
    sides = {p["side"] for p in doc["players"]}
    assert len(doc["players"]) == 2
    assert sides == {"near", "far"}


def test_frame_counts_are_consistent():
    doc = sa.build_synth_doc(frames=60)
    assert doc["meta"]["frame_count"] == 60
    for p in doc["players"]:
        assert len(p["frames"]) == 60
    assert len(doc["shuttle"]["frames"]) == 60


def test_players_actually_move():
    doc = sa.build_synth_doc(frames=120)
    xs = [pf["position"][0] for pf in doc["players"][0]["frames"]]
    assert max(xs) - min(xs) > 0.5  # nonzero footwork


def test_positions_stay_in_court_bounds():
    doc = sa.build_synth_doc()
    c = doc["court"]
    for p in doc["players"]:
        for pf in p["frames"]:
            x, z = pf["position"]
            assert abs(x) <= c["width_x"] / 2 + 1e-6
            assert abs(z) <= c["baseline_z"] + 1e-6


def test_shuttle_arc_has_height_and_stays_up():
    doc = sa.build_synth_doc()
    ys = [sf["position"][1] for sf in doc["shuttle"]["frames"]]
    assert max(ys) > 1.5           # arcs rise
    assert min(ys) >= -1e-6        # never below floor


def test_strokes_present_on_contact_frames_only():
    doc = sa.build_synth_doc(frames=120)
    labelled = [(p["id"], pf["frame_id"], pf["stroke"]["label"])
                for p in doc["players"] for pf in p["frames"]
                if pf["stroke"] is not None]
    assert len(labelled) >= 4
    for _pid, fid, label in labelled:
        assert fid % sa.HIT_INTERVAL == 0
        assert label in sa.STROKE_CYCLE


def test_time_respects_fps():
    doc = sa.build_synth_doc(frames=10, fps=60.0)
    assert doc["meta"]["fps"] == 60.0
    # frame 1 at 60 fps is 1/60 s, not 1/30
    assert doc["players"][0]["frames"][1]["time"] == round(1 / 60.0, 4)
    assert doc["shuttle"]["frames"][1]["time"] == round(1 / 60.0, 4)


def test_output_is_deterministic():
    a = json.dumps(sa.build_synth_doc(frames=48), sort_keys=True)
    b = json.dumps(sa.build_synth_doc(frames=48), sort_keys=True)
    assert a == b


def test_cli_writes_valid_file(tmp_path):
    out = tmp_path / "synth.analysis.json"
    repo = pathlib.Path(__file__).resolve().parents[2]
    tools_dir = repo / "tools"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tools_dir) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "analysis.synth_analysis",
                        "--out", str(out), "--frames", "40"],
                       cwd=str(repo), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_analysis(doc) == []
    assert doc["clip_id"] == "synth_rally_01"
