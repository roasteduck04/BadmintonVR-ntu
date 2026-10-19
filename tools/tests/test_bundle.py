# tools/tests/test_bundle.py
import json
import os

from analysis import bundle


def _doc(d, name="clipA_pose24", bid="b1", frames=3):
    p = os.path.join(d, f"{name}.analysis.json")
    json.dump({"clip_id": name, "meta": {"frame_count": frames, "pose24": {"build_id": bid}}, "players": []},
              open(p, "w", encoding="utf-8"))
    return p


def _cv(d, name="clipA_pose24", bid="b1", frames=3):
    p = os.path.join(d, f"{name}.cvdebug.json")
    json.dump({"format": "cvdebug/1", "clip_id": name, "frame_count": frames, "source_build_id": bid, "cameras": [1]},
              open(p, "w", encoding="utf-8"))
    open(p + ".meta", "w").close()
    return p


def test_bundle_embeds_cvdebug_and_moves_sidecars_out(tmp_path):
    a, side = str(tmp_path / "analysis"), str(tmp_path / "side")
    os.makedirs(a)
    doc = _doc(a)
    _cv(a)
    open(os.path.join(a, "clipA.persons.json"), "w").write("{}")
    open(os.path.join(a, "clipA_pose24.racket_rigid.json"), "w").write("{}")
    res = bundle.bundle(doc, side, log=lambda *x: None)
    assert res["embedded"] == ["cvdebug"] and sorted(res["moved"]) == ["clipA.persons.json", "clipA_pose24.cvdebug.json"]
    assert sorted(os.listdir(a)) == ["clipA_pose24.analysis.json", "clipA_pose24.racket_rigid.json"]
    d = json.load(open(doc, encoding="utf-8"))
    assert d["cvdebug"]["format"] == "cvdebug/1" and d["meta"]["bundle"]["embedded"] == ["cvdebug"]
    assert os.path.exists(os.path.join(side, "clipA_pose24", "clipA_pose24.cvdebug.json.meta"))
    bundle.unbundle(doc, side, log=lambda *x: None)
    assert "clipA_pose24.cvdebug.json" in os.listdir(a) and "cvdebug" not in json.load(open(doc, encoding="utf-8"))


def test_a_cvdebug_from_another_build_is_not_embedded_or_moved(tmp_path):
    a = str(tmp_path)
    doc = _doc(a, bid="new")
    _cv(a, bid="old")
    res = bundle.bundle(doc, str(tmp_path / "side"), log=lambda *x: None)
    assert res["embedded"] == [] and "build" in res["skipped"]["cvdebug"]
    assert os.path.exists(os.path.join(a, "clipA_pose24.cvdebug.json"))


def test_dry_run_changes_nothing(tmp_path):
    a = str(tmp_path)
    doc = _doc(a)
    _cv(a)
    before = open(doc, encoding="utf-8").read()
    bundle.bundle(doc, str(tmp_path / "side"), dry_run=True, log=lambda *x: None)
    assert open(doc, encoding="utf-8").read() == before and os.path.exists(os.path.join(a, "clipA_pose24.cvdebug.json"))
