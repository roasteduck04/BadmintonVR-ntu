import numpy as np
import pytest

from analysis import schema


def _minimal_valid_doc():
    """Smallest document that must validate: 2 players, 1 frame each, shuttle 1 frame."""
    court = schema.court_block()
    pf = {
        "frame_id": 0, "time": 0.0,
        "joints_flat": [0.0] * (schema.NUM_JOINTS * schema.STRIDE),
        "position": [0.0, -3.0], "racket": None, "grip": None,
        "stroke": {"label": "clear", "confidence": 0.9},
    }
    return {
        "schema_version": "5.0", "clip_id": "synth_rally_01",
        "meta": {"fps": 30.0, "frame_count": 1, "source_clip": "synthetic",
                 "camera": None,
                 "producers": {"pose": "synthetic", "position": "synthetic",
                               "racket": None, "grip": None,
                               "stroke": "synthetic", "shuttle": "synthetic"}},
        "court": court,
        "skeleton": {"type": "smpl-x58",
                     "joint_names": schema.JOINT_NAMES, "parents": schema.PARENTS},
        "players": [
            {"id": "P1", "side": "near", "frames": [dict(pf)]},
            {"id": "P2", "side": "far",
             "frames": [dict(pf, position=[0.0, 3.0])]},
        ],
        "shuttle": {"frames": [{"frame_id": 0, "time": 0.0,
                                "position": [0.0, 2.0, 0.0], "confidence": 0.9}]},
        "rallies": [{"start": 0, "end": 0, "winner": "P1"}],
    }


def test_topology_is_58_joints():
    assert schema.NUM_JOINTS == 58 and schema.STRIDE == 4
    assert len(schema.JOINT_NAMES) == 58 and len(schema.PARENTS) == 58
    assert schema.PARENTS[0] == -1
    assert schema.REST_POSE.shape == (58, 3)


def test_court_block_matches_regulation():
    c = schema.court_block()
    assert c["length_z"] == pytest.approx(13.40)
    assert c["width_x"] == pytest.approx(6.10)
    assert c["singles_half_width_x"] == pytest.approx(2.59)
    assert c["baseline_z"] == pytest.approx(6.70)
    assert c["short_service_z"] == pytest.approx(1.98)
    assert c["net_z"] == pytest.approx(0.0)


def test_minimal_doc_validates():
    assert schema.validate_analysis(_minimal_valid_doc()) == []


def test_nullable_fields_are_accepted():
    doc = _minimal_valid_doc()
    doc["players"][0]["frames"][0]["joints_flat"] = None
    doc["players"][0]["frames"][0]["position"] = None
    doc["players"][0]["frames"][0]["stroke"] = None
    doc["shuttle"]["frames"][0]["position"] = None
    doc["meta"]["producers"]["pose"] = None
    assert schema.validate_analysis(doc) == []


def test_wrong_schema_version_is_rejected():
    doc = _minimal_valid_doc()
    doc["schema_version"] = "2.0"
    errs = schema.validate_analysis(doc)
    assert any("schema_version" in e for e in errs)


def test_wrong_joint_count_is_rejected():
    doc = _minimal_valid_doc()
    doc["players"][0]["frames"][0]["joints_flat"] = [0.0] * 40  # not 96
    errs = schema.validate_analysis(doc)
    assert any("joints_flat" in e for e in errs)


def test_missing_players_is_rejected():
    doc = _minimal_valid_doc()
    doc["players"] = []
    errs = schema.validate_analysis(doc)
    assert any("players" in e for e in errs)


def test_bad_position_length_is_rejected():
    doc = _minimal_valid_doc()
    doc["players"][0]["frames"][0]["position"] = [1.0, 2.0, 3.0]  # must be 2
    errs = schema.validate_analysis(doc)
    assert any("position" in e for e in errs)


def test_missing_clip_id_is_rejected():
    doc = _minimal_valid_doc()
    del doc["clip_id"]
    errs = schema.validate_analysis(doc)
    assert any("clip_id" in e for e in errs)


def test_missing_meta_is_rejected():
    doc = _minimal_valid_doc()
    doc["meta"] = None
    errs = schema.validate_analysis(doc)
    assert any("meta" in e for e in errs)


def test_non_integer_frame_count_is_rejected():
    doc = _minimal_valid_doc()
    doc["meta"]["frame_count"] = 1.5
    errs = schema.validate_analysis(doc)
    assert any("frame_count" in e for e in errs)
