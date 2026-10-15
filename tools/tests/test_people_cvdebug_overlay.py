"""persons[] in the cvdebug sidecar (only when it pairs) and grey / dashed non-players in the dets layer."""
import copy

import numpy as np

from analysis import cvdebug, draw
from analysis import overlay_video as ov

V, F = 2, 10


def _sc():
    return {"frame_count": F, "cameras": [{"name": "a", "det_shift": 0}, {"name": "b", "det_shift": 2}], "persons": []}


def _pdoc():
    L = 5
    det = np.full((V, L), -1)
    det[1, :] = 1
    return {"format": "persons/1", "views": ["a", "b"], "views_used": ["a", "b"], "det_shift": [0, 2], "frame_count": F,
            "source_build_id": "abc",
            "tracks": [{"track_id": 7, "label": "nonplayer", "player_id": "", "score": 0.1, "first_frame": 3,
                        "last_frame": 7, "ambiguous": 0, "pinned": 0, "floor_xz": [0.0] * 10, "support": [1] * 5,
                        "det_index": det.ravel().tolist()}]}


def test_attach_persons_only_when_it_pairs():
    sc = _sc()
    assert cvdebug.attach_persons(sc, _pdoc(), "abc") == "" and sc["persons"][0]["track_id"] == 7
    assert cvdebug.validate_sidecar(sc) == []
    for mutate, why in ((lambda d: d.update(source_build_id="zzz"), "build"),
                        (lambda d: d["det_shift"].__setitem__(1, 3), "det_shift"),
                        (lambda d: d.update(frame_count=F + 1), "frame count"),
                        (lambda d: d.update(views=["b", "a"]), "camera order")):
        d = copy.deepcopy(_pdoc())
        mutate(d)
        sc = _sc()
        assert why in cvdebug.attach_persons(sc, d, "abc") and sc["persons"] == []
    d = _pdoc()
    d["source_build_id"] = ""                                   # people track output: pairs by det_shift alone
    assert cvdebug.attach_persons(_sc(), d, "abc") == ""


def test_person_map_and_draw():
    sc = _sc()
    cvdebug.attach_persons(sc, _pdoc(), "abc")
    pm = ov.person_map(sc)
    assert pm[(1, 3)] == {1: ("nonplayer", 7, "")} and (0, 3) not in pm
    img = np.zeros((200, 200, 3), np.uint8)
    uv = np.full((133, 2), np.nan)
    uv[5], uv[7] = [20, 20], [180, 180]
    draw.draw_edges_dashed(img, uv, [(5, 7)], draw.GREY)
    solid = np.zeros_like(img)
    draw.draw_edges(solid, uv, [(5, 7)], draw.GREY)
    assert 0 < (img > 0).sum() < (solid > 0).sum()


def _det(x0, y0):
    d = np.zeros((133, 3))
    for k, j in enumerate((5, 6, 7, 8, 11, 12)):
        d[j] = [x0 + 30 * (k % 2), y0 + 25 * (k // 2), 0.9]
    return d


class _Ctx:
    scale = 1.0


def _near(region, bgr, tol=40):
    return (np.abs(region.astype(int) - np.array(bgr)).max(axis=2) <= tol) & (region.sum(axis=2) > 0)


def test_draw_dets_colours_tags_and_dashes(monkeypatch):
    import cv2
    texts = []
    real = cv2.putText
    monkeypatch.setattr(ov.cv2, "putText", lambda img, t, *a, **k: (texts.append(t), real(img, t, *a, **k))[1])
    dets = [_det(10, 20), _det(110, 20), _det(210, 20)]
    img = np.zeros((150, 300, 3), np.uint8)
    ov._draw_dets(_Ctx(), img, dets, {0: 0}, {1: ("nonplayer", 7, ""), 2: ("unknown", 8, "")})
    assert _near(img[:, :100], draw.player_bgr(0)).any() and not _near(img[:, 100:], draw.player_bgr(0)).any()
    assert _near(img[:, 100:200], draw.GREY).any() and _near(img[:, 200:], draw.GREY).any()
    assert texts == ["N7", "?8"]

    def ink(persons):
        z = np.zeros((150, 300, 3), np.uint8)
        ov._draw_dets(_Ctx(), z, [_det(210, 20)], {}, persons)
        return int((z > 0).sum())
    plain_n = ink({0: ("nonplayer", 8, "")})
    assert plain_n > ink({})                                     # same edges plus the tag text
    assert 0 < ink({0: ("unknown", 8, "")}) < plain_n


def test_attach_persons_length_mismatch():
    d = _pdoc()
    d["det_shift"] = [0]
    assert "det_shift length" in cvdebug.attach_persons(_sc(), d, "abc")


def test_build_attaches_persons_when_paired(tmp_path):
    import json
    import os

    import cvdebug_synth as cs
    from analysis import cvdebug as cd
    from analysis.people import persons as pm
    b = cs.built_clip(tmp_path)

    def run(**kw):
        logs = []
        _, sc = cd.build("syn", variant="pose24", analysis_dir=b["adir"], cache_dir=b["cache_dir"],
                         data_dir=b["data_dir"], registry=b["reg"], log=logs.append, write=False, **kw)
        return sc, logs

    sc, _ = run()
    assert sc["persons"] == []                                   # no persons file: SP2 behaviour
    F, cams = sc["frame_count"], sc["cameras"]
    V, L = len(cams), 3
    pdoc = {"format": "persons/1", "views": [c["name"] for c in cams], "views_used": [c["name"] for c in cams],
            "det_shift": [c["det_shift"] for c in cams], "frame_count": F, "source_build_id": sc["source_build_id"],
            "tracks": [{"track_id": 1, "label": "nonplayer", "player_id": "", "score": 0.1, "first_frame": 2,
                        "last_frame": 4, "ambiguous": 0, "pinned": 0, "floor_xz": [0.0] * (2 * L),
                        "support": [1] * L, "det_index": [0] * (V * L)}]}
    pp = pm.path(b["adir"], "syn")
    pm.write(pp, pdoc)
    sc, logs = run(persons_min_frames=0)
    assert len(sc["persons"]) == 1 and any("1 tracks" in m for m in logs)
    sc, logs = run()                                             # default drops short non-player tracks (SP4, SP3 F19)
    assert sc["persons"] == [] and sc["persons_dropped"] == 1
    pdoc["det_shift"][0] += 1
    pm.write(pp, pdoc)
    sc, logs = run()
    assert sc["persons"] == [] and any("not attached" in m and "det_shift" in m for m in logs)
    try:
        run(persons_path=os.path.join(b["adir"], "missing.persons.json"))
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
