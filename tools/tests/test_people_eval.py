"""Identity metrics on a hand-made label set (numbers worked out by hand) + label-free checks."""
import numpy as np
import pytest

import people_synth as ps
from analysis.people import eval_identity as ev
from analysis.people import persons


def _L(f, d, lab):
    return {"view": "a", "raw_frame": f, "det": d, "label": lab, "source": "manual", "proposal": ""}


def test_labelled_metrics_hand_computed():
    labels = [_L(0, 0, "P1"), _L(0, 1, "P2"), _L(0, 2, "N"),
              _L(1, 0, "P1"), _L(1, 1, "P2"), _L(1, 2, "N"),
              _L(2, 0, "P1"), _L(2, 1, "P2"),
              _L(3, 0, "P1"), _L(3, 1, "S")]
    pred = {("a", 0, 0): "P1", ("a", 0, 1): "P2", ("a", 0, 2): "N",
            ("a", 1, 0): "P2", ("a", 1, 1): "P1", ("a", 1, 2): "P1",
            ("a", 2, 0): "P1", ("a", 2, 1): "U",
            ("a", 3, 0): "P1"}
    m = ev.labelled_metrics(labels, pred)
    assert m["n_player"] == 7 and m["n_nonplayer"] == 2
    assert m["idf1"] == pytest.approx(8 / 14) and m["deta"] == pytest.approx(0.75)
    assert m["assa"] == pytest.approx(4 / 6) and m["hota_lite"] == pytest.approx(np.sqrt(0.75 * 4 / 6))
    assert m["nonplayer_fp_rate"] == pytest.approx(0.5) and m["player_fn_rate"] == pytest.approx(1 / 7)
    assert m["switches"] == 3
    assert ev.labelled_metrics([], pred)["n_player"] == 0
    pv_ = m["per_view"]["a"]
    assert {k: v for k, v in pv_.items() if k != "per_view"} == {k: v for k, v in m.items() if k != "per_view"}


def test_labelled_metrics_ignore_u_rows():
    labels = [_L(0, 0, "P1"), _L(0, 1, "N")]
    pred = {("a", 0, 0): "P1", ("a", 0, 5): "P2"}
    got = ev.labelled_metrics(labels + [_L(0, 5, "U"), {**_L(0, 6, "U"), "view": "b"}], pred)
    assert {k: v for k, v in got.items()} == ev.labelled_metrics(labels, pred) and "b" not in got["per_view"]


def test_net_crossings_with_hysteresis():
    assert ev.net_crossings([-3, -1, -0.3, 0.2, -0.2, -2]) == 0         # touches the net, never 0.5 m into the other half
    assert ev.net_crossings([-3, -1, 0.6, 2, 1, -1, -3]) == 2
    assert ev.net_crossings([]) == 0


def test_label_free_on_the_synthetic_tracker_output(tmp_path, monkeypatch):
    from test_people_tracker3d import _run
    from analysis import appearance, cvdebug
    from analysis import pose_v2 as pv
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), F=120)
    tr, _, doc = _run(tmp_path, reg, "syns", {"c": 2})
    views = doc["views"]
    cams = pv.load_cams({v: str(tmp_path / "data" / "calib" / f"{v}_cam.json") for v in views})
    raw = {v: cvdebug.load_raw_kp2d(str(tmp_path / "data" / "pose2d" / f"{v}_kp2d.json"))[0] for v in views}
    lf = ev.label_free(doc, cams, raw, appearance.load_clip("syns"))
    p1 = lf["players"]["P1"]
    assert p1["net_crossings"] == 0 and p1["coverage"] > 0.95 and p1["reproj_px_median"] < 5.0
    assert p1["inconsistent_frac"] < 0.02 and lf["shared_detections"] == 0 and p1["appearance_jumps"] == 0


def test_from_sidecar_players():
    sc = {"clip_id": "x", "fps": 60.0, "frame_count": 4, "cameras": [{"name": "a", "det_shift": 1}],
          "players": [{"id": "P1", "det_index": [-1, 0, 0, -1]}], "persons": []}
    d = persons.from_sidecar(sc)
    assert persons.validate(d) == [] and persons.predictions(d) == {("a", 2, 0): "P1", ("a", 3, 0): "P1"}


def test_validate_rejects_player_id_on_non_player_track():
    sc = {"clip_id": "x", "fps": 60.0, "frame_count": 4, "cameras": [{"name": "a", "det_shift": 1}],
          "players": [{"id": "P1", "det_index": [-1, 0, 0, -1]}], "persons": []}
    d = persons.from_sidecar(sc)
    for lab in ("nonplayer", "unknown"):
        d["tracks"][0]["label"] = lab
        assert any("player_id" in e for e in persons.validate(d))
        d["tracks"][0]["player_id"] = ""
        assert persons.validate(d) == []
        d["tracks"][0]["player_id"] = "P1"


def test_load_labels_rejects_wrong_format(tmp_path):
    p = tmp_path / "x_ids.json"
    p.write_text('{"format": "nope/9", "labels": []}')
    with pytest.raises(ValueError, match="x_ids.json"):
        ev.load_labels(str(p))


def test_heldout_view_agreement_flags_wrong_person(tmp_path, monkeypatch):
    import copy
    from test_people_tracker3d import _run
    from analysis import cvdebug
    from analysis import pose_v2 as pv
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), F=120)
    tr, _, doc = _run(tmp_path, reg, "syns", {"c": 2})
    views = doc["views"]
    V = len(views)
    cams = pv.load_cams({v: str(tmp_path / "data" / "calib" / f"{v}_cam.json") for v in views})
    raw = {v: cvdebug.load_raw_kp2d(str(tmp_path / "data" / "pose2d" / f"{v}_kp2d.json"))[0] for v in views}
    clean = ev.label_free(doc, cams, raw)["players"]["P1"]
    assert 0 <= clean["heldout_px_median"] < 8 and clean["heldout_inconsistent_frac"] < 0.05

    def dets(pid):
        out = {}
        for t in doc["tracks"]:
            if t["player_id"] == pid:
                L = t["last_frame"] - t["first_frame"] + 1
                D = t["det_index"]
                for k in range(L):
                    out[t["first_frame"] + k] = [D[ci * L + k] for ci in range(V)]
        return out
    d1, d2 = dets("P1"), dets("P2")
    bad = copy.deepcopy(doc)
    ci = 0
    n = 0
    for t in bad["tracks"]:
        if t["player_id"] != "P1":
            continue
        L = t["last_frame"] - t["first_frame"] + 1
        for k in range(L):
            f = t["first_frame"] + k
            if 20 <= f < 80 and f in d2 and d2[f][ci] >= 0 and t["det_index"][ci * L + k] >= 0:
                t["det_index"][ci * L + k] = d2[f][ci]
                n += 1
    assert n >= 30
    dirty = ev.label_free(bad, cams, raw)["players"]["P1"]
    assert dirty["heldout_inconsistent_frac"] > 0.1 or dirty["heldout_inconsistent_frac"] > 3 * clean["heldout_inconsistent_frac"]
