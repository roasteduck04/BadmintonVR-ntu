"""pose24 with --tracker 3d: sync unchanged, persons doc in the bundle, excluded views empty, build writes persons.json."""
import json
import os

import numpy as np
import pytest

import people_synth as ps
from analysis.people import persons
from analysis.pose24 import build as pb
from analysis.pose24 import config, scores, track

QUIET = dict(log=lambda *a: None)


def _cfg(tmp_path, reg, clip, **kw):
    return config.resolve(clip, registry=reg, data_dir=str(tmp_path / "data"), analysis_dir=str(tmp_path), **kw)


def test_track_and_sync24_tracker3d(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), shifts={"c": 3})
    cfg, params = _cfg(tmp_path, reg, "syns"), scores.load_params(path=None)
    b = track.track_and_sync24(cfg, params, tracker="3d", **QUIET)
    # sync unchanged: the same TrackerV2 sync as --tracker v2. (This scene's sync reads c at +3.5 frames for a true
    # +3 - slow far player - so shifts["c"] rounds to 4 for both trackers; the brief's `== 3` assumed an exact sync.)
    b2 = track.track_and_sync24(cfg, params, tracker="v2", **QUIET)
    assert b["offsets"] == b2["offsets"] and b["shifts"] == b2["shifts"] and b["excluded"] == b2["excluded"]
    assert b["offsets"]["c"] == pytest.approx(3.0, abs=0.6) and b["shifts"]["c"] == int(np.round(b["offsets"]["c"]))
    assert b["tracker"] == "3d" and set(b["obs"]) == {"P1", "P2"}
    doc = b["persons"]
    assert doc["format"] == "persons/1" and persons.validate(doc) == [] and doc["det_shift"][2] == b["shifts"]["c"]
    pred = persons.predictions(doc)
    p1 = [k for k, n in truth.items() if n == "p1"]
    assert np.mean([pred.get(k) == "P1" for k in p1]) > 0.95
    # det_index of the bundle and the persons doc agree for P1 in view a
    k = b["views"].index("a")
    D = b["det_index"]["P1"][k]
    f = int(np.nonzero(D >= 0)[0][10])
    assert pred[("a", f + b["shifts"]["a"], int(D[f]))] == "P1"
    assert b["sync_disagreement"]["P1"] is not None and b["sync_disagreement"]["P1"] < 1.0


def test_dropped_view_has_no_person_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path))
    bad = tmp_path / "data" / "calib" / "d_cam.json"                     # break d's calibration (as frontright_21)
    c = json.load(open(bad))
    c["t"] = (np.array(c["t"]) + [1.5, 0.0, 0.0]).tolist()
    json.dump(c, open(bad, "w"))
    b = track.track_and_sync24(_cfg(tmp_path, reg, "syns"), scores.load_params(path=None), tracker="3d", **QUIET)
    assert "d" in b["excluded"]
    doc = b["persons"]
    ci, V = doc["views"].index("d"), len(doc["views"])
    assert "d" not in doc["views_used"]
    for t in doc["tracks"]:
        D = np.asarray(t["det_index"]).reshape(V, -1)
        assert (D[ci] == -1).all()


def test_build_writes_persons_and_v2_removes_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path), F=90)
    out = pb.build("syns", registry=reg, analysis_dir=str(tmp_path), data_dir=str(tmp_path / "data"),
                   cache_dir=str(tmp_path / "cache"), tracker="3d", fit_opts={"passes": 1}, **QUIET)
    d = json.load(open(out))
    p = persons.path(str(tmp_path), "syns")
    assert os.path.exists(p) and d["meta"]["pose_variant"]["tracker"] == "3d"
    assert persons.load(p)["source_build_id"] == d["meta"]["pose24"]["build_id"]
    # review 1 L2: tracker3d's ankle keypoint height reaches persons stats and the debug cache (mono reads it)
    from analysis.pose24 import doc as docmod
    from analysis.pose24 import mono
    ah = persons.load(p)["stats"]["ankle_h_m"]
    assert 0.0 <= ah <= 0.3
    cache = docmod.load_debug_cache(track.cache_path("syns", str(tmp_path / "cache")))
    assert cache["ankle_h_m"] == ah and mono._ankle_h(cache) == ah
    pb.build("syns", registry=reg, analysis_dir=str(tmp_path), data_dir=str(tmp_path / "data"),
             cache_dir=str(tmp_path / "cache"), tracker="v2", fit_opts={"passes": 1}, **QUIET)
    assert not os.path.exists(p)


def test_build_default_tracker_is_3d(tmp_path, monkeypatch):
    """Since the SP3 real-clip acceptance (2026-09-30) pb.build without `tracker` runs tracker3d."""
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path), F=90)
    out = pb.build("syns", registry=reg, analysis_dir=str(tmp_path), data_dir=str(tmp_path / "data"),
                   cache_dir=str(tmp_path / "cache"), fit_opts={"passes": 1}, **QUIET)
    d = json.load(open(out))
    assert d["meta"]["pose_variant"]["tracker"] == "3d"
    assert os.path.exists(persons.path(str(tmp_path), "syns"))


def test_people_track_cli(tmp_path, monkeypatch):
    from analysis.people.__main__ import main
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path), F=90)
    monkeypatch.setattr(config, "load_registry", lambda path=None: reg)
    assert main(["track", "--clip", "syns", "--tracker", "both", "--out-dir", str(tmp_path / "pe"),
                 "--data-dir", str(tmp_path / "data"), "--analysis-dir", str(tmp_path)]) == 0
    assert (tmp_path / "pe" / "syns.3d.persons.json").exists() and (tmp_path / "pe" / "syns.v2.persons.json").exists()


def test_v2_persons_on_dropped_view(tmp_path, monkeypatch):
    """persons.from_v2: valid doc, excluded view's rows empty, P1 detections = the bundle's det_index."""
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path))
    bad = tmp_path / "data" / "calib" / "d_cam.json"
    c = json.load(open(bad))
    c["t"] = (np.array(c["t"]) + [1.5, 0.0, 0.0]).tolist()
    json.dump(c, open(bad, "w"))
    b = track.track_and_sync24(_cfg(tmp_path, reg, "syns"), scores.load_params(path=None), tracker="v2", **QUIET)
    assert "d" in b["excluded"] and b["tracker"] == "v2" and b["sync_disagreement"] == {}
    doc = b["persons"]
    assert doc["tracker"] == "v2" and persons.validate(doc) == [] and json.dumps(doc, allow_nan=False)
    V, ci = len(doc["views"]), doc["views"].index("d")
    assert "d" not in doc["views_used"]
    for t in doc["tracks"]:
        assert (np.asarray(t["det_index"]).reshape(V, -1)[ci] == -1).all()
    pred = persons.predictions(doc)
    n_checked = 0
    for k, v in enumerate(b["views"]):                  # det_index rows are in tr.views (= bundle views) order
        D = b["det_index"]["P1"][k]
        for f in np.nonzero(D >= 0)[0]:
            assert pred[(v, int(f) + b["shifts"][v], int(D[f]))] == "P1"
            n_checked += 1
    assert n_checked > 100


def test_sync_disagreement_follows_the_same_person_under_a_label_permutation():
    # T15 doubles_2sides: TrackerV2 is seeded by stale 1 s seeds, tracker3d by the 5 s seeds, so v2's P3 is
    # tracker3d's P4 all clip long (2.0 m "disagreement"). The sync only needs v2 to have followed one person per
    # track: the check compares each tracker3d player with the nearest v2 track and names the permutation.
    F = 100
    x = np.stack([np.linspace(-1, 1, F), np.full(F, 3.0)], axis=1)
    y = np.stack([np.linspace(1, -1, F), np.full(F, 5.0)], axis=1)
    d = track.sync_disagreement({"P3": x, "P4": y}, {"P3": y + [0.05, 0.0], "P4": x})
    assert d["P3"] == pytest.approx(0.05, abs=1e-6) and d["P4"] == pytest.approx(0.0, abs=1e-6)
    assert track.sync_matches({"P3": x, "P4": y}, {"P3": y + [0.05, 0.0], "P4": x}) == {"P3": "P4", "P4": "P3"}
    assert track.sync_disagreement({"P3": x}, {"P3": x + [3.0, 0.0]})["P3"] == pytest.approx(3.0)


def test_sync_check_is_one_to_one_and_warns_on_a_v2_track_following_nobody():
    # nearest-only matching let both tracker3d players match v2 P3 (P4 would follow nobody unnoticed); the Hungarian
    # assignment gives v2 P4 to 3d P4, 3 m away -> warning
    F = 100
    x = np.stack([np.linspace(-1, 1, F), np.full(F, 3.0)], axis=1)
    far = x + [0.0, 3.0]
    notes, warns = [], []
    d, m = track.sync_check({"P3": x, "P4": far}, {"P3": x, "P4": x + [0.3, 0.0]}, 1.0, log=lambda s: (
        warns if "WARNING" in s else notes).append(s))
    assert m == {"P3": "P3", "P4": "P4"} and d["P4"] == pytest.approx(3.0, abs=0.05)
    assert any("P4" in w for w in warns)


def test_sync_check_label_permutation_is_a_note_not_a_warning():
    F = 100
    x = np.stack([np.linspace(-1, 1, F), np.full(F, 3.0)], axis=1)
    y = np.stack([np.linspace(1, -1, F), np.full(F, 5.0)], axis=1)
    logs = []
    d, m = track.sync_check({"P3": x, "P4": y}, {"P3": y, "P4": x}, 1.0, log=logs.append)
    assert m == {"P3": "P4", "P4": "P3"} and max(d.values()) < 1e-9
    assert not any("WARNING" in s for s in logs) and sum("note" in s for s in logs) == 2


def test_sync_check_warns_on_an_unmatched_v2_track():
    F = 100
    x = np.stack([np.linspace(-1, 1, F), np.full(F, 3.0)], axis=1)
    logs = []
    d, m = track.sync_check({"P3": x, "P4": np.full((F, 2), np.nan)}, {"P3": x, "P4": x + [0.0, 2.0]}, 1.0,
                            log=logs.append)
    assert d["P4"] is None and m["P4"] is None and any("WARNING" in s and "P4" in s for s in logs)


def test_sync_disagree_limit_comes_from_params(tmp_path, monkeypatch):
    """params.json sync_disagree_m is the threshold handed to sync_check (not the module constant)."""
    from analysis import people
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path), F=90)
    real = people.load_params()
    monkeypatch.setattr(people, "load_params", lambda path=None: dict(real, sync_disagree_m=0.123))
    seen = []
    orig = track.sync_check
    monkeypatch.setattr(track, "sync_check", lambda a, b, threshold, **kw: seen.append(threshold) or orig(a, b, threshold, **kw))
    track.track_and_sync24(_cfg(tmp_path, reg, "syns"), scores.load_params(path=None), tracker="3d", **QUIET)
    assert seen == [0.123]


def test_build_v2_ignores_osnet(tmp_path, monkeypatch):
    """--tracker v2 --reid osnet: no OSNet cache load (none exists) and pose_variant.reid is None."""
    from analysis.people import osnet
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(osnet, "load_cache", lambda clip: (_ for _ in ()).throw(AssertionError("cache loaded")))
    reg, _, _ = ps.singles_scene(str(tmp_path), F=90)
    out = pb.build("syns", registry=reg, analysis_dir=str(tmp_path), data_dir=str(tmp_path / "data"),
                   cache_dir=str(tmp_path / "cache"), fit_opts={"passes": 1}, tracker="v2", reid="osnet", **QUIET)
    pv = json.load(open(out))["meta"]["pose_variant"]
    assert pv["tracker"] == "v2" and pv["reid"] is None


@pytest.mark.parametrize("cmd", ["track", "eval", "reid-extract", "frames"])
def test_people_all_against_the_real_registry_skips_mono(cmd, monkeypatch, tmp_path):
    """review 1 M2: `people <cmd> --all` reached the mono entry and config.resolve raised (exit 1 every run)."""
    from analysis.people import __main__ as cli
    from analysis.people import labels
    reg = config.load_registry()
    seen = []
    monkeypatch.setattr(track, "load_timeline", lambda *a, **k: None)
    monkeypatch.setattr(track, "track_and_sync24", lambda cfg, *a, **k: seen.append(cfg["out_clip"]) or
                        {"persons": {"stats": {"n_tracks": 0, "n_player": 0, "n_nonplayer": 0, "n_unknown": 0,
                                               "player_ids": [], "player_frames": []}}})
    monkeypatch.setattr(persons, "write", lambda *a, **k: None)
    monkeypatch.setattr(labels.LabelService, "_ppath", lambda self, c: seen.append(c) and None)
    if cmd == "reid-extract":
        osnet = pytest.importorskip("analysis.people.osnet")
        monkeypatch.setattr(osnet, "extract_clip", lambda clip, *a, **k: seen.append(clip) or "x")
    args = [cmd, "--all"] + (["--out-dir", str(tmp_path)] if cmd in ("track", "eval") else [])
    assert cli.main(args) == 0
    mono = [c for c, e in reg.items() if e.get("mono")]
    assert mono and not set(mono) & set(seen)


def test_sync_check_compares_tracker3d_with_the_synced_trackerv2_run(tmp_path, monkeypatch):
    """review 1 L10: the unsynced TrackerV2 floor track blends views up to +/-30 frames apart, which makes the
    "wrong person" warning noisy for fast movers. The check uses the synced TrackerV2 run (same seeds, the final
    shifts and views) that view health already made - no extra tracking."""
    from analysis import pose_v2 as pv
    monkeypatch.chdir(tmp_path)
    reg, _, _ = ps.singles_scene(str(tmp_path), shifts={"c": 3})
    made = []

    class Rec(pv.TrackerV2):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    monkeypatch.setattr(pv, "TrackerV2", Rec)
    seen = []
    orig = track.sync_check
    monkeypatch.setattr(track, "sync_check", lambda a, b, threshold, **kw: seen.append(a) or orig(a, b, threshold, **kw))
    b = track.track_and_sync24(_cfg(tmp_path, reg, "syns"), scores.load_params(path=None), tracker="3d", **QUIET)
    assert len(seen) == 1 and b["shifts"]["c"] != 0
    src = next(t for t in made if t.track is seen[0])
    assert src.shifts == b["shifts"] and list(src.views) == list(b["views"])
