"""tracker3d end to end on synthetic scenes: labels, identities, half-court, occlusion, persons/1 validity."""
import json

import numpy as np

import people_synth as ps
from analysis import appearance
from analysis import pose_v2 as pv
from analysis.people import persons
from analysis.people.tracker3d import Tracker3D
from analysis.pose24 import scores

QUIET = dict(log=lambda *a: None)


def _run(tmp_path, reg, clip, shifts):
    e = reg[clip]
    params = scores.load_params(path=None)
    views = sorted(e["views"])
    cams = pv.load_cams({v: str(tmp_path / "data" / "calib" / f"{v}_cam.json") for v in views})
    kp2d, kp_idx, tids = {}, {}, {}
    for v in views:
        kp2d[v], fps, kp_idx[v], header = scores.load_kp2d24(str(tmp_path / "data" / "pose2d" / f"{v}_kp2d.json"), params)
        tids[v] = header["_track_ids"]
    seeds_views = json.load(open(tmp_path / "data" / "calib" / f"{e['seeds']}_seeds.json"))["views"]
    tr = Tracker3D(cams, kp2d, seeds_views, e["mode"], fps, shifts=shifts, kp_idx=kp_idx,
                   appear=appearance.load_clip(clip), track_ids=tids, **QUIET)
    matched = tr.run()
    doc = persons.from_tracker3d(tr, clip, views, {v: float(shifts.get(v, 0)) for v in views}, tr.frame_count)
    return tr, matched, doc


def _frac(truth, pred, person, want):
    keys = [k for k, n in truth.items() if n == person]
    return np.mean([pred.get(k, "N") in want for k in keys])


def test_singles_labels_and_identities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path))
    tr, matched, doc = _run(tmp_path, reg, "syns", {"c": 2})
    pred = persons.predictions(doc)
    assert _frac(truth, pred, "p1", {"P1"}) > 0.95 and _frac(truth, pred, "p2", {"P2"}) > 0.95
    assert _frac(truth, pred, "by", {"N"}) > 0.9 and _frac(truth, pred, "ump", {"N"}) > 0.9
    assert _frac(truth, pred, "walk", {"P1", "P2"}) < 0.05
    # P1 steps outside the doubles sideline and stays a player
    out = [k for k, n in truth.items() if n == "p1" and k[0] == "a"]
    assert np.mean([pred.get(k) == "P1" for k in out]) > 0.95
    assert persons.validate(doc) == [] and json.dumps(doc, allow_nan=False)
    assert doc["views"] == ["a", "b", "c", "d"] and doc["det_shift"] == [0, 0, 2, 0]
    assert any(h["frames"] >= 120 for h in doc["stats"]["hotspots"])
    # matched holds the loader's own arrays (pose24.track.detection_indices relies on identity)
    f = 50
    k = matched["P1"]["a"][f]
    assert k is not None and any(k is c for c in tr.kp2d["a"][f])


def test_doubles_crossing_and_occluded_player(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.doubles_scene(str(tmp_path))
    tr, matched, doc = _run(tmp_path, reg, "synd", {})
    pred = persons.predictions(doc)
    for n in ("p1", "p2", "p3", "p4"):
        assert _frac(truth, pred, n, {ps.PIDS[n]}) > 0.9, n
    # P4 seen by camera d only in frames 60-120: still covered
    assert sum(matched["P4"]["d"][f] is not None for f in range(60, 120)) >= 55


def test_half_court_rule_and_additional_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.net_scene(str(tmp_path))
    _, _, doc = _run(tmp_path, reg, "synn", {})
    pred = persons.predictions(doc)
    assert _frac(truth, pred, "p1", {"P2"}) == 0.0 and _frac(truth, pred, "p2", {"P1"}) == 0.0
    reg, truth, _ = ps.reseed_scene(str(tmp_path / "r"))
    monkeypatch.chdir(tmp_path / "r")
    _, _, doc = _run(tmp_path / "r", reg, "synr", {})
    pred = persons.predictions(doc)
    late = {k: n for k, n in truth.items() if k[1] >= 170}
    assert _frac(late, pred, "p1", {"P1"}) > 0.9 and _frac(late, pred, "p2", {"P2"}) > 0.9


def test_player_never_seen_leaves_valid_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), F=90)
    for v in ("a", "b", "c", "d"):                       # remove P2 (far) everywhere
        path = tmp_path / "data" / "pose2d" / f"{v}_kp2d.json"
        d = json.load(open(path))
        d["frames"] = [[det for i, det in enumerate(fr) if truth.get((v, r, i)) != "p2"]
                       for r, fr in enumerate(d["frames"])]
        json.dump(d, open(path, "w"))
    tr, matched, doc = _run(tmp_path, reg, "syns", {"c": 2})
    assert all(k is None for v in matched["P2"] for k in matched["P2"][v])
    assert not any(t["player_id"] == "P2" for t in doc["tracks"]) and persons.validate(doc) == []


def test_tracker_estimates_the_clip_ankle_height(tmp_path, monkeypatch):
    # T15: real ankle keypoints sit ~0.15 m up; the tracker measures it on its own triangulated candidates and puts
    # 1-view floor points on that plane
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), ankle_dh=0.07)
    tr, matched, doc = _run(tmp_path, reg, "syns", {"c": 2})
    assert abs(tr.result["ankle_h_m"] - 0.15) < 0.02
    pred = persons.predictions(doc)
    assert _frac(truth, pred, "p1", {"P1"}) > 0.95 and _frac(truth, pred, "p2", {"P2"}) > 0.95
    assert _frac(truth, pred, "by", {"N"}) > 0.9


def test_released_pins_get_their_classifier_score_back():
    from analysis.people import classify, tracker3d
    from analysis.people.candidates import Candidate
    from analysis.people.tracklets import Tracklet
    from analysis import people
    p = people.load_params()
    t = Tracklet(tid=0)
    for f in range(10):
        t.add(Candidate(frame=f, dets={"a": 0}, floor=np.array([0.0, 3.0]), support=1))
    feats = {0: {k: 0.0 for k in p["classify"]["weights"]}}
    labels, sc = {0: "player"}, {0: 1.0}
    pins = tracker3d.release_pins({0}, {0: "P3", 5: "P4"}, labels, sc, feats, [t], p)
    assert pins == {5: "P4"} and sc[0] == classify.score(feats[0], 10, p) and sc[0] < 1.0 and labels[0] != "player"
