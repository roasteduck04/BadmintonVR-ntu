import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from analysis import shuttle as sh
from analysis import schema, synth_analysis as sa


def test_parabola_point_apex_and_ends():
    assert sh.parabola_point([0.0, 3.0], [2.0, -3.0], 0.0, 2.6)[1] == 0.0
    assert sh.parabola_point([0.0, 3.0], [2.0, -3.0], 1.0, 2.6)[1] == 0.0
    mid = sh.parabola_point([0.0, 3.0], [2.0, -3.0], 0.5, 2.6)
    assert mid == [1.0, 2.6, 0.0]      # XZ midpoint, apex height


def test_build_track_length_and_gaps():
    contacts = [(10, [0.0, 3.0]), (30, [1.0, -3.0])]
    positions, confs = sh.build_shuttle_track(contacts, frame_count=40, peak_h=2.6)
    assert len(positions) == 40 and len(confs) == 40
    assert positions[0] is None and positions[9] is None       # before first contact
    assert positions[31] is None and positions[39] is None     # after last contact
    assert positions[10][1] == 0.0                             # at a contact, on the floor
    assert positions[20][1] > 2.0                              # apex mid-arc


def test_build_track_endpoints_match_contacts():
    contacts = [(5, [-2.0, 3.0]), (25, [2.0, -3.0])]
    positions, _ = sh.build_shuttle_track(contacts, frame_count=30)
    assert positions[5][0] == -2.0 and positions[5][2] == 3.0
    assert positions[25][0] == 2.0 and positions[25][2] == -3.0


def test_build_track_too_few_contacts_all_none():
    positions, confs = sh.build_shuttle_track([(3, [0.0, 0.0])], frame_count=10)
    assert all(p is None for p in positions)
    assert all(c is None for c in confs)


def test_contacts_from_doc_reads_stroke_frames():
    doc = sa.build_synth_doc(frames=60)   # synth sets cosmetic strokes on contact frames
    contacts = sh.contacts_from_doc(doc)
    assert len(contacts) >= 2
    frames = [f for f, _ in contacts]
    assert frames == sorted(set(frames))               # ascending, unique
    for _f, xz in contacts:
        assert len(xz) == 2


def test_fill_shuttle_keeps_doc_valid():
    doc = sa.build_synth_doc(frames=60)
    contacts = sh.contacts_from_doc(doc)
    positions, confs = sh.build_shuttle_track(contacts, frame_count=60)
    sh.fill_shuttle(doc, positions, confs, "parabola_from_contacts")
    assert schema.validate_analysis(doc) == []
    assert len(doc["shuttle"]["frames"]) == 60


def test_fill_shuttle_provenance_and_nulls():
    doc = sa.build_synth_doc(frames=60)
    positions = [None] * 60
    positions[10] = [0.0, 2.0, 3.0]
    confs = [None] * 60
    confs[10] = 0.8
    sh.fill_shuttle(doc, positions, confs, "parabola_from_contacts")
    assert doc["meta"]["producers"]["shuttle"] == "parabola_from_contacts"
    assert doc["shuttle"]["frames"][0]["position"] is None
    assert doc["shuttle"]["frames"][10]["position"] == [0.0, 2.0, 3.0]
    assert doc["shuttle"]["frames"][10]["time"] == round(10 / doc["meta"]["fps"], 4)


from analysis import racket as _rk, strokes as _stk


def test_synthetic_shuttle_track_has_airborne_points():
    doc = sa.build_synth_doc(frames=120)
    _rk.fill_player_rackets(doc, _rk.synthetic_racket_tracks(doc), "synthetic_racket_bodylocal")
    _stk.fill_player_strokes(doc, min_peak_speed=2.0)
    positions, confs = sh.synthetic_shuttle_track(doc)
    assert len(positions) == 120
    ys = [p[1] for p in positions if p is not None]
    assert ys and max(ys) > 1.5             # arcs rise off the floor


def test_cli_writes_valid_analysis_with_shuttle(tmp_path):
    out = tmp_path / "shuttle.analysis.json"
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "analysis.shuttle",
         "--out", str(out), "--frames", "120", "--min-peak-speed", "2.0"],
        cwd=str(repo), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_analysis(doc) == []
    assert doc["meta"]["producers"]["shuttle"] == "parabola_from_contacts"
    assert len(doc["shuttle"]["frames"]) == 120
    ys = [f["position"][1] for f in doc["shuttle"]["frames"] if f["position"] is not None]
    assert ys and max(ys) > 1.5
