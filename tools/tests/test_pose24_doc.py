"""pose24 outputs + the two optional hooks outside the package (pose_v2 sidecar id map, facing face)."""
import json

import numpy as np
import pytest

import pose24_synth as syn
from analysis import pose_v2 as pv
from analysis import schema24
from analysis.facing import fill_facing
from analysis.pose24 import doc as docmod


def _sc(ids):
    return {"format": "racket_variant/1", "clip_id": "c", "frame_count": 2,
            "players": [{"id": i, "which_hand": "right",
                         "frames": [{"grip": [0.1, 1.0, 0.2], "head": None, "side": None}, {"grip": [0.1, 1.0, 0.2]}]}
                        for i in ids]}


def test_write_racket_sidecars_default_is_id_for_id(tmp_path):
    base = {"players": [{"id": "P1", "frames": [{"position": [1.0, 2.0]}, {"position": [1.0, 2.0]}]}]}
    doc = {"meta": {"frame_count": 2}, "players": [{"id": "P1", "frames": [{"position": [0.5, 2.0]}, {"position": None}]}]}
    (tmp_path / "c.racket_rigid.json").write_text(json.dumps(_sc(["P1"])))
    pv.write_racket_sidecars(str(tmp_path), "c", "c_x", base, doc, log=lambda *a: None)
    out = json.loads((tmp_path / "c_x.racket_rigid.json").read_text())
    assert out["clip_id"] == "c_x" and out["players"][0]["id"] == "P1"
    assert out["players"][0]["frames"][0]["grip"] == [0.6, 1.0, 0.2] and out["players"][0]["frames"][1] is None


def test_write_racket_sidecars_matches_players_by_position(tmp_path):
    base = {"players": [{"id": "P1", "frames": [{"position": [1.0, 0.0]}] * 2},
                        {"id": "P2", "frames": [{"position": [-1.0, 0.0]}] * 2}]}
    doc = {"meta": {"frame_count": 2}, "players": [{"id": "P1", "frames": [{"position": [-1.0, 0.0]}] * 2},
                                                   {"id": "P2", "frames": [{"position": [1.0, 0.0]}] * 2},
                                                   {"id": "P3", "frames": [{"position": [0.0, 5.0]}] * 2}]}
    (tmp_path / "c.racket_rigid.json").write_text(json.dumps(_sc(["P1", "P2"])))
    pv.write_racket_sidecars(str(tmp_path), "c", "c_x", base, doc, log=lambda *a: None)
    out = json.loads((tmp_path / "c_x.racket_rigid.json").read_text())
    assert [p["id"] for p in out["players"]] == ["P1", "P2", "P3"]        # one per doc player, doc order
    assert out["players"][1]["frames"][0]["grip"] == [0.1, 1.0, 0.2]      # base P1 -> doc P2, same place
    assert out["players"][0]["frames"][0]["grip"] == [0.1, 1.0, 0.2]
    assert out["players"][2]["frames"] == [None, None] and "_base_id" not in out["players"][0]


def test_fill_facing_with_smpl28_face_indices():
    J = np.zeros((28, 4))
    J[:, 3] = 0.9
    J[1, :3], J[2, :3] = [0.1, 0.9, 0.0], [-0.1, 0.9, 0.0]
    J[25, :3], J[26, :3], J[27, :3] = [0.03, 1.6, 0.0], [-0.03, 1.6, 0.0], [0.0, 1.58, 0.1]
    frames = [{"joints_flat": J.ravel().tolist()} for _ in range(3)]
    fill_facing(frames, 60.0, sign=1.0, face=(25, 26, 27))
    assert "head_heading" in frames[0]["facing"]
    frames = [{"joints_flat": J[:24].ravel().tolist()} for _ in range(3)]
    fill_facing(frames, 60.0, sign=1.0, face=None)
    assert "head_heading" not in frames[0]["facing"] and "heading" in frames[0]["facing"]


def test_frames_from_world24_body_local_and_aux():
    J = np.zeros((2, 24, 3))
    J[:, :, 1] = 1.0
    J[:, 0] = [2.0, 0.95, -3.0]
    J[1] = np.nan
    aux = np.full((2, 14, 3), np.nan)
    aux[0, 0] = [2.1, 0.02, -3.0]
    fr = docmod.frames_from_world24(J, np.full((2, 24), 0.8), aux, np.full((2, 14), 0.7), [True, False], 60.0)
    assert fr[0]["position"] == [2.0, -3.0] and fr[0]["interpolated"] is True and fr[0]["time"] == 0.0
    assert fr[0]["joints_flat"][:4] == [0.0, 0.95, 0.0, 0.8] and len(fr[0]["aux_flat"]) == 56
    assert fr[0]["aux_flat"][:4] == [pytest.approx(0.1), 0.02, 0.0, 0.7] and fr[0]["aux_flat"][7] == 0.0
    assert fr[1]["joints_flat"] is None and fr[1]["aux_flat"] is None and fr[1]["interpolated"] is False


def test_make_doc24_validates_for_both_kinds():
    views = ("a", "b")
    b = syn.bundle({"P1": syn.motion(3)}, syn.rig(views), views)
    b["image_size"] = {"a": [2160, 3840], "b": [2160, 3840]}
    for kind, nj in (("smpl-24", 24), ("smpl-28", 28)):
        J = np.zeros((3, nj, 3))
        fr = docmod.frames_from_world24(J, np.ones((3, nj)), np.zeros((3, 14, 3)), np.ones((3, 14)), [False] * 3, 60.0)
        d = docmod.make_doc24("c", 60.0, 3, [("P1", fr)], b, kind, {"name": "pose24", "offsets": {"a": np.float64(0.0)}},
                              {"x": float("nan")})
        assert schema24.validate_analysis24(d) == [], kind
        assert d["meta"]["pose24"]["x"] is None and d["meta"]["cameras"][0]["width"] == 2160
    assert d["meta"]["label"] == "SMPL-28 · triangulated · 2 cams (a, b)"


def test_match_players_by_position_and_rejects_far_pairs():
    def p(pid, x):
        return {"id": pid, "frames": [{"position": [x, 0.0]} for _ in range(20)]}
    base = {"players": [p("P1", 3.0), p("P2", -3.0)]}
    doc = {"players": [p("P1", -3.1), p("P2", 3.05), p("P3", 0.0)]}
    assert docmod.match_players(base, doc) == {"P1": "P2", "P2": "P1"}


def test_match_players_leaves_a_pair_beyond_max_m_unmatched():
    def p(pid, x):
        return {"id": pid, "frames": [{"position": [x, 0.0]} for _ in range(20)]}
    base, doc = {"players": [p("P1", 0.0)]}, {"players": [p("P1", 0.8)]}
    assert docmod.match_players(base, doc) == {}
    assert docmod.match_players(base, doc, max_m=1.0) == {"P1": "P1"}


def test_cyclic_player_ids_carry_rackets_to_the_right_player(tmp_path):
    """Three players whose ids are a 3-cycle (not self-inverse, unlike a 2-player swap): the doc -> base match,
    its inversion and the sidecar remap must each go the right way."""
    xs = {"P1": 2.0, "P2": 0.0, "P3": -2.0}
    cyc = {"P1": "P2", "P2": "P3", "P3": "P1"}                 # doc id -> the base id at the same place

    def p(pid, x):
        return {"id": pid, "frames": [{"position": [x, 0.0]} for _ in range(20)]}
    base = {"players": [p(b, xs[b]) for b in xs]}
    doc = {"meta": {"frame_count": 20}, "players": [p(d, xs[cyc[d]]) for d in xs]}
    doc_to_base = docmod.match_players(base, doc)
    assert doc_to_base == cyc
    base_to_doc = docmod.base_to_doc(doc_to_base)
    assert base_to_doc == {"P2": "P1", "P3": "P2", "P1": "P3"}
    sc = {"format": "racket_variant/1", "clip_id": "c", "frame_count": 20,
          "players": [{"id": b, "which_hand": "right", "frames": [{"grip": [0.0, 1.0, float(i)]}] * 20}
                      for i, b in enumerate(xs)]}                              # grip z = base index
    (tmp_path / "c.racket_rigid.json").write_text(json.dumps(sc))
    pv.write_racket_sidecars(str(tmp_path), "c", "c_x", base, doc, log=lambda *a: None)
    out = {q["id"]: q for q in json.loads((tmp_path / "c_x.racket_rigid.json").read_text())["players"]}
    for d, b in cyc.items():
        assert out[d]["frames"][0]["grip"][2] == list(xs).index(b), d


def test_base_to_doc_refuses_a_map_that_is_not_one_to_one():
    with pytest.raises(ValueError):
        docmod.base_to_doc({"P1": "P1", "P2": "P1"})


def test_carry_over_sanitises_nan(tmp_path):
    base = {"meta": {"producers": {"racket": "rigid", "shuttle": "tracknet"}},
            "shuttle": {"frames": [{"position": [0.0, float("nan"), 0.0]}]}, "rallies": [],
            "players": [{"id": "B1", "frames": [{"position": [1.0, 0.0], "racket": {"grip": [0.0, 1.0, 0.0],
                                                  "which_hand": "right", "confidence": float("nan")}, "stroke": None}]}]}
    doc = {"meta": {"producers": {}}, "players": [{"id": "P1", "frames": [{"position": [0.7, 0.0]}]}]}
    docmod.carry_over24(base, doc, log=lambda *a: None)
    assert doc["shuttle"]["frames"][0]["position"][1] is None
    r = doc["players"][0]["frames"][0]["racket"]
    assert r["grip"] == [pytest.approx(0.3), 1.0, 0.0] and r["confidence"] is None
    docmod.write_json(str(tmp_path / "x.json"), doc)


def test_debug_cache_round_trip(tmp_path):
    views = ("a", "b")
    b = syn.bundle({"P1": syn.motion(5)}, syn.rig(views), views)
    extra = {"sigma_mm": np.ones((5, 28), np.float32), "det_index": np.zeros((2, 5), np.int32)}
    path = str(tmp_path / "c" / "x_pose24.npz")
    docmod.save_debug_cache(path, b, {"P1": extra}, {"time_ref_view": "a", "offsets_all": {"a": 0.0}})
    c = docmod.load_debug_cache(path)
    assert c["format"] == docmod.CACHE_FORMAT and c["time_ref_view"] == "a" and c["views"] == ["a", "b"]
    assert np.allclose(c["obs"]["P1"]["a"], b["obs"]["P1"]["a"], equal_nan=True)
    assert c["extra"]["P1"]["det_index"].dtype == np.int32 and c["extra"]["P1"]["sigma_mm"].shape == (5, 28)


def test_head_from_nose_fallback_keeps_a_halved_nose_and_trunk_confidence():
    """review 1 L1: ears never measured -> their sigma sits at CAP, so max(nose, ears) gave the fallback head
    conf ~1e-11 (written 0.0). The fallback head's sigma comes from the sources it really uses (nose + trunk)."""
    from analysis import skeleton24 as sk
    from analysis.pose24 import build as pb
    from analysis.pose24 import confidence, regress
    F = 3
    sig35 = np.full((F, 35), 25.0)
    sig35[:, 0] = 20.0                                   # nose
    sig35[:, [3, 4]] = confidence.CAP_MM                 # ears never measured
    sig35[:, [5, 6, 11, 12]] = 30.0                      # trunk
    info = {"head_fallback": np.array([True, True, False]), "hand_fallback": np.zeros((F, 2), bool)}
    sigJ, C = pb.joint_sigma_conf(sig35, info)
    h = sk.SMPL28_NAMES.index("head")
    assert sigJ[0, h] == pytest.approx(30.0) and C[0, h] == pytest.approx(0.5 * np.exp(-30.0 / 40.0))
    assert C[2, h] < 1e-6                                 # no fallback: the ears' (capped) sigma rules
    others = [j for j in range(len(sk.SMPL28_NAMES)) if j != h]
    want = confidence.joint_sigma(sig35, regress.SOURCE_LIST)
    assert np.allclose(sigJ[:, others], want[:, others])
