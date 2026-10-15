"""Labelling backend: frame list (events + regular), proposals, confirmed/corrected sources, HTTP routes."""
import json
import threading
import urllib.request

import people_synth as ps
from analysis.people import labels as lb
from analysis.people import persons


def _doc(tmp_path, monkeypatch):
    from test_people_tracker3d import _run
    monkeypatch.chdir(tmp_path)
    reg, truth, _ = ps.singles_scene(str(tmp_path), F=120)
    _, _, doc = _run(tmp_path, reg, "syns", {"c": 2})
    (tmp_path / "data" / "people_eval").mkdir(parents=True, exist_ok=True)
    persons.write(str(tmp_path / "data" / "people_eval" / "syns.3d.persons.json"), doc)
    return reg, truth, doc


def test_frame_list_and_upsert(tmp_path, monkeypatch):
    _, _, doc = _doc(tmp_path, monkeypatch)
    fl = lb.frame_list(doc, every=30, n_max=40)
    fs = [x["f"] for x in fl]
    assert {0, 30, 60, 90} <= set(fs) and len(fs) == len(set(fs)) <= 40
    other = json.loads(json.dumps(doc))
    other["tracks"][0]["player_id"] = "P2" if other["tracks"][0]["player_id"] != "P2" else "P1"
    other["tracks"][0]["label"] = "player"
    fl2 = lb.frame_list(doc, other=other, every=30, n_max=40)
    assert "disagree" in fl2[0]["reasons"]                 # disagreements come first
    d = {"format": "idlabels/1", "clip": "syns", "labels": []}
    n = lb.upsert(d, "syns", "a", 5, [{"det": 0, "label": "P1"}, {"det": 1, "label": "N"}], {0: "P1", 1: "P2"})
    assert n == 2 and [l["source"] for l in d["labels"]] == ["confirmed", "corrected"]
    lb.upsert(d, "syns", "a", 5, [{"det": 1, "label": "P2"}], {0: "P1", 1: "P2"})
    assert len(d["labels"]) == 2 and d["labels"][1]["label"] == "P2" and d["labels"][1]["source"] == "confirmed"


def test_service_and_routes(tmp_path, monkeypatch):
    reg, truth, doc = _doc(tmp_path, monkeypatch)
    svc = lb.LabelService(root=str(tmp_path), registry=reg, analysis_dir=str(tmp_path),
                          eval_dir=str(tmp_path / "data" / "people_eval"), labels_dir=str(tmp_path / "gt"),
                          data_dir=str(tmp_path / "data"))
    assert [c["clip"] for c in svc.clips()] == ["syns"]
    items = svc.frames("syns")["items"]
    it = items[0]
    fr = svc.frame("syns", it["view"], it["raw"])
    assert fr["dets"] and all(len(d["kps"]) == 23 for d in fr["dets"])
    assert any(d["proposal"] in ("P1", "P2") for d in fr["dets"])
    assert svc.image("syns", it["view"], it["raw"]) is None            # synthetic clip has no video
    import serve_calibrator as sc
    from http.server import HTTPServer
    monkeypatch.setattr(sc, "LABELS", svc)
    srv = HTTPServer(("127.0.0.1", 0), sc.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        got = json.load(urllib.request.urlopen(f"{base}/api/label/frame?clip=syns&view={it['view']}&raw={it['raw']}"))
        assert got["dets"] == fr["dets"]
        body = json.dumps({"clip": "syns", "view": it["view"], "raw": it["raw"],
                           "labels": [{"det": d["det"], "label": d["proposal"] or "N"} for d in got["dets"]]}).encode()
        req = urllib.request.Request(f"{base}/api/label/save", data=body, headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["saved"] == len(got["dets"])
        saved = json.load(open(tmp_path / "gt" / "syns_ids.json"))
        assert saved["format"] == "idlabels/1" and all(l["source"] == "confirmed" for l in saved["labels"])
        assert urllib.request.urlopen(f"{base}/labeler").status == 200
    finally:
        srv.shutdown()


def test_frames_cli(tmp_path, monkeypatch, capsys):
    reg, _, _ = _doc(tmp_path, monkeypatch)
    from analysis.people import __main__ as cli
    monkeypatch.setattr(cli.config, "load_registry", lambda *a, **k: reg)
    assert cli.main(["frames", "--clip", "syns", "--labels-dir", str(tmp_path / "gt"), "--analysis-dir", str(tmp_path),
                     "--eval-dir", str(tmp_path / "data" / "people_eval"), "--data-dir", str(tmp_path / "data")]) == 0
    out = json.load(open(tmp_path / "gt" / "syns_frames.json"))
    assert out["items"] and out["reasons"]["regular"] >= 4 and "regular" in capsys.readouterr().out


def _long_doc(F, pid):
    return {"frame_count": F, "views": ["a"], "views_used": ["a"], "det_shift": [0],
            "tracks": [{"track_id": 0, "label": "player", "player_id": pid, "first_frame": 0, "last_frame": F - 1,
                        "floor_xz": [0.0, -3.0] * F, "det_index": [0] * F}]}


def test_frame_list_spreads_dense_disagreements():
    fl = lb.frame_list(_long_doc(1400, "P1"), other=_long_doc(1400, "P2"), every=30, n_max=40)
    fs = [x["f"] for x in fl]
    assert 0 < len(fs) <= 40 and len(set(fs)) == len(fs) and max(fs) - min(fs) > 1000
    assert all(abs(a - b) >= 5 for i, a in enumerate(fs) for b in fs[i + 1:])
    assert all("disagree" in x["reasons"] for x in fl)
    # no disagreements: regular frames spread over the clip, capped
    fl = lb.frame_list(_long_doc(1400, "P1"), every=30, n_max=40)
    assert len(fl) == 40 and max(x["f"] for x in fl) - min(x["f"] for x in fl) > 1000


def test_item_done_only_when_every_detection_labelled(tmp_path, monkeypatch):
    reg, _, _ = _doc(tmp_path, monkeypatch)
    svc = lb.LabelService(root=str(tmp_path), registry=reg, analysis_dir=str(tmp_path),
                          eval_dir=str(tmp_path / "data" / "people_eval"), labels_dir=str(tmp_path / "gt"),
                          data_dir=str(tmp_path / "data"))
    it = next(i for i in svc.frames("syns")["items"]
              if sum(d["visible"] for d in svc.frame("syns", i["view"], i["raw"])["dets"]) >= 2)
    dets = [d for d in svc.frame("syns", it["view"], it["raw"])["dets"] if d["visible"]]
    done = lambda: next(i for i in svc.frames("syns")["items"] if (i["view"], i["raw"]) == (it["view"], it["raw"]))["done"]
    assert not done()
    svc.save("syns", it["view"], it["raw"], [{"det": dets[0]["det"], "label": "N"}])
    assert not done()
    svc.save("syns", it["view"], it["raw"], [{"det": d["det"], "label": "S"} for d in dets[1:]])
    assert done()
    import pytest
    with pytest.raises(KeyError):
        svc.save("syns", "nope", it["raw"], [{"det": 0, "label": "N"}])


def test_service_rejects_unregistered_clip(tmp_path, monkeypatch):
    """A clip name outside the registry (e.g. a crafted ../ path) is a KeyError (the routes' 400), nothing is read or written."""
    import pytest
    import urllib.error
    reg, _, doc = _doc(tmp_path, monkeypatch)
    (tmp_path / "evil.persons.json").write_text(json.dumps(doc))            # would be read via clip=../evil
    svc = lb.LabelService(root=str(tmp_path), registry=reg, analysis_dir=str(tmp_path / "sub"),
                          eval_dir=str(tmp_path / "data" / "people_eval"), labels_dir=str(tmp_path / "gt"),
                          data_dir=str(tmp_path / "data"))
    bad = "../evil"
    for call in (lambda: svc.persons(bad), lambda: svc.frames(bad), lambda: svc.frame(bad, "a", 0),
                 lambda: svc.image(bad, "a", 0), lambda: svc.seeds(bad, "a"),
                 lambda: svc.save(bad, "a", 0, [{"det": 0, "label": "P1"}])):
        with pytest.raises(KeyError):
            call()
    assert not (tmp_path / "evil_ids.json").exists() and not (tmp_path / "gt").exists()
    import serve_calibrator as sc
    from http.server import HTTPServer
    monkeypatch.setattr(sc, "LABELS", svc)
    srv = HTTPServer(("127.0.0.1", 0), sc.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/api/label/frames?clip=..%2Fevil")
        assert e.value.code == 400
        body = json.dumps({"clip": bad, "view": "a", "raw": 0, "labels": [{"det": 0, "label": "P1"}]}).encode()
        req = urllib.request.Request(f"{base}/api/label/save", data=body, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400
    finally:
        srv.shutdown()
