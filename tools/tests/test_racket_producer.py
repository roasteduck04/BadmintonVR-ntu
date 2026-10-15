import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from analysis import racket as rk
from analysis import schema, skeleton, synth_analysis as sa


def test_racket_frame_is_orthonormal():
    grip = [0.0, 1.4, 0.0]
    head = [-0.66, 1.4, 0.0]     # shaft along -x
    side = [-0.66, 1.5, 0.0]     # face width along +y
    axis, across, fn = rk.racket_frame(grip, head, side)
    assert np.allclose(np.linalg.norm(axis), 1.0)
    assert np.allclose(np.linalg.norm(fn), 1.0)
    assert abs(float(axis @ fn)) < 1e-9       # normal perpendicular to shaft
    assert np.allclose(fn, [0.0, 0.0, -1.0])  # x cross y = -z here


def _rest_flat():
    flat = []
    for j in range(skeleton.NUM_JOINTS):
        flat.extend([float(skeleton._REST[j, 0]), float(skeleton._REST[j, 1]),
                     float(skeleton._REST[j, 2]), 1.0])
    return flat


def test_nearest_hand_picks_right():
    flat = _rest_flat()
    rh = skeleton.JOINT_NAMES.index("right_middle1")
    point = skeleton._REST[rh]                    # exactly on the right hand
    assert rk.nearest_hand(flat, point) == "right"


def test_nearest_hand_picks_left():
    flat = _rest_flat()
    lh = skeleton.JOINT_NAMES.index("left_middle1")
    point = skeleton._REST[lh]
    assert rk.nearest_hand(flat, point) == "left"


def test_derive_grip_extracts_face_and_hand():
    racket = {"face_normal": [0.0, 0.0, -1.0], "which_hand": "right",
              "confidence": 0.8, "axis": [1.0, 0.0, 0.0]}
    g = rk.derive_grip(racket)
    assert g == {"face_normal": [0.0, 0.0, -1.0], "which_hand": "right",
                 "confidence": 0.8}


def _doc_with_racket(racket, grip):
    doc = sa.build_synth_doc(frames=5)
    for p in doc["players"]:
        for pf in p["frames"]:
            pf["racket"] = racket
            pf["grip"] = grip
    return doc


def test_null_racket_and_grip_still_valid():
    doc = _doc_with_racket(None, None)
    assert schema.validate_analysis(doc) == []


def test_well_formed_racket_validates():
    racket = {"grip": [0, 1.4, 0], "head": [-0.66, 1.4, 0], "side": [-0.66, 1.5, 0],
              "axis": [-1, 0, 0], "face_normal": [0, 0, -1],
              "which_hand": "right", "confidence": 0.8, "status": "detected"}
    grip = {"face_normal": [0, 0, -1], "which_hand": "right", "confidence": 0.8}
    doc = _doc_with_racket(racket, grip)
    assert schema.validate_analysis(doc) == []


def test_bad_which_hand_rejected():
    racket = {"face_normal": [0, 0, -1], "which_hand": "middle", "confidence": 0.8}
    doc = _doc_with_racket(racket, None)
    errs = schema.validate_analysis(doc)
    assert any("which_hand" in e for e in errs)


def test_bad_face_normal_length_rejected():
    grip = {"face_normal": [0, 0], "which_hand": "right", "confidence": 0.8}
    doc = _doc_with_racket(None, grip)
    errs = schema.validate_analysis(doc)
    assert any("face_normal" in e for e in errs)


def _bodylocal_tracks(doc, length=0.66, half_width=0.10):
    """Grip at each player's right hand, shaft along wrist->hand, face width along +y."""
    rh = skeleton.JOINT_NAMES.index("right_middle1")
    rw = skeleton.JOINT_NAMES.index("right_wrist")
    tracks = {}
    for p in doc["players"]:
        grips, heads, sides = [], [], []
        for pf in p["frames"]:
            hand = rk._joint_xyz(pf["joints_flat"], rh)
            wrist = rk._joint_xyz(pf["joints_flat"], rw)
            shaft = rk._unit(hand - wrist)
            if np.linalg.norm(shaft) < 1e-9:
                shaft = np.array([-1.0, 0.0, 0.0])
            across = rk._unit(np.cross(shaft, np.array([0.0, 0.0, 1.0])))
            if np.linalg.norm(across) < 1e-9:
                across = np.array([0.0, 1.0, 0.0])
            grip = hand
            head = grip + length * shaft
            side = head + half_width * across
            grips.append(grip); heads.append(head); sides.append(side)
        tracks[p["id"]] = {"grip": np.array(grips), "head": np.array(heads),
                           "side": np.array(sides)}
    return tracks


def test_fill_rackets_keeps_doc_valid():
    doc = sa.build_synth_doc(frames=30)
    rk.fill_player_rackets(doc, _bodylocal_tracks(doc), "synthetic_racket_bodylocal")
    assert schema.validate_analysis(doc) == []


def test_fill_rackets_face_normal_is_unit_and_hand_right():
    doc = sa.build_synth_doc(frames=30)
    rk.fill_player_rackets(doc, _bodylocal_tracks(doc), "synthetic_racket_bodylocal")
    for p in doc["players"]:
        for pf in p["frames"]:
            fn = np.array(pf["racket"]["face_normal"])
            assert np.allclose(np.linalg.norm(fn), 1.0, atol=1e-4)
            assert pf["racket"]["which_hand"] == "right"
            assert pf["grip"]["which_hand"] == "right"


def test_fill_rackets_stays_rigid():
    doc = sa.build_synth_doc(frames=30)
    rk.fill_player_rackets(doc, _bodylocal_tracks(doc), "synthetic_racket_bodylocal")
    lengths = []
    for pf in doc["players"][0]["frames"]:
        g = np.array(pf["racket"]["grip"]); h = np.array(pf["racket"]["head"])
        lengths.append(np.linalg.norm(h - g))
    assert max(lengths) - min(lengths) < 1e-3   # shaft length constant


def test_missing_status_yields_null_racket_and_grip():
    doc = sa.build_synth_doc(frames=30)
    tracks = _bodylocal_tracks(doc)
    first = doc["players"][0]["id"]
    tracks[first]["status"] = ["detected"] * 30
    tracks[first]["status"][7] = "missing"
    rk.fill_player_rackets(doc, tracks, "synthetic_racket_bodylocal")
    assert doc["players"][0]["frames"][7]["racket"] is None
    assert doc["players"][0]["frames"][7]["grip"] is None
    assert doc["players"][0]["frames"][6]["racket"] is not None


def test_fill_rackets_provenance_and_length_guard():
    doc = sa.build_synth_doc(frames=30)
    rk.fill_player_rackets(doc, _bodylocal_tracks(doc), "synthetic_racket_bodylocal")
    assert doc["meta"]["producers"]["racket"] == "synthetic_racket_bodylocal"
    assert doc["meta"]["producers"]["grip"] == "synthetic_racket_bodylocal"
    bad = {doc["players"][0]["id"]: {"grip": np.zeros((5, 3)),
                                     "head": np.zeros((5, 3)), "side": np.zeros((5, 3))}}
    with pytest.raises(ValueError):
        rk.fill_player_rackets(doc, bad, "x")


def test_synthetic_racket_tracks_shape():
    doc = sa.build_synth_doc(frames=20)
    tracks = rk.synthetic_racket_tracks(doc)
    assert set(tracks) == {p["id"] for p in doc["players"]}
    for p in doc["players"]:
        for key in ("grip", "head", "side"):
            assert np.asarray(tracks[p["id"]][key]).shape == (20, 3)


def test_cli_writes_valid_analysis_with_rackets(tmp_path):
    out = tmp_path / "racket.analysis.json"
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "analysis.racket",
         "--out", str(out), "--frames", "40"],
        cwd=str(repo), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_analysis(doc) == []
    assert doc["meta"]["producers"]["racket"] == "synthetic_racket_bodylocal"
    for p in doc["players"]:
        assert all(pf["racket"] is not None for pf in p["frames"])
        assert all(pf["grip"] is not None for pf in p["frames"])
