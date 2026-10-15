"""SP4 additions to the cvdebug/1 sidecar: proxy_clip, hotspots, short non-player track pruning (SP3 F19)."""
import copy

import numpy as np
import pytest

from analysis import cvdebug as cd
import cvdebug_synth as cs

V, F = 2, 100


@pytest.fixture(scope="module")
def syn_built(tmp_path_factory):
    return cs.built_clip(tmp_path_factory.mktemp("cvd4"))


def _build(b, clip="syn", **kw):
    return cd.build(clip, variant="pose24", analysis_dir=b["adir"], cache_dir=b["cache_dir"],
                    data_dir=b["data_dir"], registry=b["reg"], log=lambda *a: None, write=False, **kw)


def _sc():
    return {"frame_count": F, "cameras": [{"name": "a", "det_shift": 0}, {"name": "b", "det_shift": 0}],
            "persons": [], "hotspots": [], "persons_dropped": 0}


def _track(tid, label, first, last, pid=""):
    n = last - first + 1
    return {"track_id": tid, "label": label, "player_id": pid, "score": 0.5, "first_frame": first,
            "last_frame": last, "ambiguous": 0, "pinned": 0, "floor_xz": [0.0] * (2 * n), "support": [1] * n,
            "det_index": np.full((V, n), -1).ravel().tolist()}


def _pdoc():
    return {"format": "persons/1", "views": ["a", "b"], "views_used": ["a", "b"], "det_shift": [0, 0],
            "frame_count": F, "source_build_id": "abc",
            "tracks": [_track(0, "player", 0, 5, "P1"),        # short but a player: always kept
                       _track(1, "nonplayer", 0, 9),           # 10 frames: dropped at min_frames 30
                       _track(2, "unknown", 10, 60),           # 51 frames: kept
                       _track(3, "unknown", 70, 80)],          # 11 frames: dropped
            "stats": {"hotspots": [{"x": -3.5, "z": 7.5, "frames": 483}]}}


def test_sidecar_names_its_proxy_clip(syn_built):
    assert _build(syn_built, "syn")[1]["proxy_clip"] == "syn"
    assert _build(syn_built, "syn_2cam")[1]["proxy_clip"] == "syn"      # subset clips share the base's proxies


def test_sidecar_without_persons_has_empty_hotspots(syn_built):
    _, sc = _build(syn_built)
    assert sc["hotspots"] == [] and sc["persons_dropped"] == 0 and cd.validate_sidecar(sc) == []


def test_attach_persons_default_keeps_every_track():
    sc = _sc()
    assert cd.attach_persons(sc, _pdoc(), "abc") == ""
    assert [t["track_id"] for t in sc["persons"]] == [0, 1, 2, 3] and sc["persons_dropped"] == 0


def test_attach_persons_drops_short_non_player_tracks():
    sc = _sc()
    assert cd.attach_persons(sc, _pdoc(), "abc", min_frames=30) == ""
    assert [t["track_id"] for t in sc["persons"]] == [0, 2] and sc["persons_dropped"] == 2


def test_attach_persons_copies_hotspots_and_stays_json_safe():
    sc = _sc()
    cd.attach_persons(sc, _pdoc(), "abc")
    assert sc["hotspots"] == [{"x": -3.5, "z": 7.5, "frames": 483}]
    assert cd.validate_sidecar(sc) == []


def test_attach_persons_that_does_not_pair_leaves_hotspots_empty():
    d = copy.deepcopy(_pdoc())
    d["views"] = ["b", "a"]
    sc = _sc()
    assert "camera order" in cd.attach_persons(sc, d, "abc")
    assert sc["hotspots"] == [] and sc["persons"] == []


def test_cli_persons_min_frames_reaches_build(monkeypatch):
    monkeypatch.setattr(cd.config, "load_registry", lambda *a, **k: {"x": {"views": ["a"]}})
    seen = []
    monkeypatch.setattr(cd, "build", lambda clip, variant, **kw: seen.append(kw["persons_min_frames"]) or ("p", {}))
    cd.main(["build", "--clip", "x"])
    cd.main(["build", "--clip", "x", "--persons-min-frames", "0"])
    assert seen == [30, 0]


# --- shared Unity fixture (Assets/Editor/Tests/CvDebugTests.cs reads these files) ------------------------------------
import json  # noqa: E402
import os  # noqa: E402

import cv_fixture  # noqa: E402
from analysis import video_io  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _shape(x):
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in sorted(x.items()) if k != "source_build_id"}
    if isinstance(x, list):
        return ["list", len(x), _shape(x[0]) if x and isinstance(x[0], dict) else type(x[0]).__name__ if x else None]
    return type(x).__name__


def test_committed_unity_fixture_is_what_cvdebug_produces(tmp_path):
    fresh, _ = cv_fixture.make(tmp_path)
    committed = json.load(open(os.path.join(FIX, "cvdebug_small.json"), encoding="utf-8"))
    assert _shape(committed) == _shape(fresh)      # same fields, types and array lengths: regenerate with cv_fixture.py
    assert cd.validate_sidecar(committed) == []
    cams = committed["cameras"]
    assert any(c["held_out"] for c in cams) and any(c["excluded"] for c in cams)
    assert {p["label"] for p in committed["persons"]} == {"player", "nonplayer", "unknown"}
    assert committed["hotspots"] and committed["proxy_clip"] == "syn"


def test_frame_cases_match_video_io():
    cases = json.load(open(os.path.join(FIX, "cv_cases.json"), encoding="utf-8"))
    assert len(cases["frames"]) >= 12
    for c in cases["frames"]:
        assert video_io.view_frame(c["f"], c["offset"], c["det_shift"])[0] == c["expected"], c
    assert any(abs(c["offset"] - round(c["offset"])) == 0.5 for c in cases["frames"])   # exact halves covered
    assert cases["projections"] and len(cases["shifts"]) == 3
