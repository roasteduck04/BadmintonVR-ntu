"""Identity-safe carry-over of the base racket / grip / stroke onto pose variants (review 2 H1, review 1 L13)."""
import json

import pytest

from analysis import pose_v2 as pv
from analysis.pose24 import doc as docmod

F = 40
SWAP = 20                        # base P3 / P4 swap identities from this frame on
LOCAL = [0.3, 1.0, 0.1]          # every base racket's body-local grip
PLACES = {"P1": (-2.0, -4.0), "P2": (2.0, -4.0), "P3": (-2.0, 4.0), "P4": (2.0, 4.0)}


def _base_place(pid, f):
    if f >= SWAP and pid in ("P3", "P4"):
        return PLACES["P4" if pid == "P3" else "P3"]
    return PLACES[pid]


def _racket(tag):
    return {"grip": list(LOCAL), "head": [0.3, 1.6, 0.1], "side": [0.4, 1.6, 0.1], "which_hand": tag[1],
            "confidence": 0.9, "tag": tag[0]}


def _pair():
    """Base doc whose P3 / P4 identities swap mid-clip relative to the variant doc (as the doubles_2sides base
    tracker does); each base frame's racket / stroke is tagged with its base player id."""
    hands = {"P1": "right", "P2": "left", "P3": "right", "P4": "left"}
    base = {"meta": {"producers": {"racket": "rigid"}, "frame_count": F}, "shuttle": {"frames": []}, "rallies": [],
            "players": [{"id": pid, "frames": [
                {"position": [_base_place(pid, f)[0] + 0.02, _base_place(pid, f)[1]],
                 "racket": _racket((pid, hands[pid])),
                 "grip": {"face_normal": [1.0, 0.0, 0.0], "which_hand": hands[pid], "confidence": 0.5, "tag": pid},
                 "stroke": {"label": f"stroke_{pid}", "confidence": 0.8}} for f in range(F)]}
                for pid in PLACES]}
    doc = {"meta": {"producers": {}, "frame_count": F},
           "players": [{"id": pid, "frames": [{"position": list(PLACES[pid])} for _ in range(F)]} for pid in PLACES]}
    doc["players"][0]["frames"][5]["position"] = [-2.0, -3.2]          # 0.8 m from every base player: over the gate
    return base, doc


def _base_at(doc_pid, f):
    """The base id standing at doc player `doc_pid`'s place in frame f."""
    return next(b for b in PLACES if _base_place(b, f) == PLACES[doc_pid])


def test_match_frames_follows_a_mid_clip_identity_swap_and_gates_far_frames():
    base, doc = _pair()
    m = pv.match_frames(base, doc)
    assert m["P3"][:SWAP] == ["P3"] * SWAP and m["P3"][SWAP:] == ["P4"] * (F - SWAP)
    assert m["P4"][SWAP:] == ["P3"] * (F - SWAP)
    assert m["P1"][5] is None and m["P1"][4] == "P1"
    assert pv.match_frames(base, doc, gate_m=1.0)["P1"][5] == "P1"


def test_carry_over_takes_racket_grip_stroke_from_the_base_player_at_the_same_place():
    base, doc = _pair()
    pv.carry_over(base, doc, log=lambda *a: None)
    for p in doc["players"]:
        for f, fr in enumerate(p["frames"]):
            if p["id"] == "P1" and f == 5:
                assert fr["racket"] is None and fr["grip"] is None and fr["stroke"] is None
                continue
            want = _base_at(p["id"], f)
            assert fr["racket"]["tag"] == want and fr["grip"]["tag"] == want, (p["id"], f)
            assert fr["stroke"]["label"] == f"stroke_{want}"
            assert fr["racket"]["grip"] == pytest.approx([LOCAL[0] + 0.02, LOCAL[1], LOCAL[2]])   # stays at the hand


def test_carry_over24_is_identity_safe_and_sanitises():
    base, doc = _pair()
    base["players"][0]["frames"][0]["racket"]["confidence"] = float("nan")
    docmod.carry_over24(base, doc, log=lambda *a: None)
    fr = doc["players"][2]["frames"][SWAP + 3]
    assert fr["racket"]["tag"] == "P4" and fr["stroke"]["label"] == "stroke_P4"
    assert doc["players"][0]["frames"][0]["racket"]["confidence"] is None
    assert doc["players"][0]["frames"][5]["racket"] is None


def _write_sidecar(tmp_path, base):
    sc = {"format": "racket_variant/1", "clip_id": "c", "frame_count": F, "time_base": "video",
          "players": [{"id": p["id"], "which_hand": p["frames"][0]["racket"]["which_hand"],
                       "frames": [dict(fr["racket"]) for fr in p["frames"]]} for p in base["players"]]}
    (tmp_path / "c.racket_rigid.json").write_text(json.dumps(sc))


def test_racket_sidecars_follow_the_per_frame_match(tmp_path):
    base, doc = _pair()
    _write_sidecar(tmp_path, base)
    pv.write_racket_sidecars(str(tmp_path), "c", "c_x", base, doc, log=lambda *a: None)
    out = json.loads((tmp_path / "c_x.racket_rigid.json").read_text())
    assert [p["id"] for p in out["players"]] == ["P1", "P2", "P3", "P4"]        # one per doc player, doc order
    by = {p["id"]: p for p in out["players"]}
    for pid in PLACES:
        for f, r in enumerate(by[pid]["frames"]):
            if pid == "P1" and f == 5:
                assert r is None
                continue
            assert r["tag"] == _base_at(pid, f), (pid, f)
            assert r["grip"] == pytest.approx([LOCAL[0] + 0.02, LOCAL[1], LOCAL[2]])
    # which_hand = the most-matched base player's (a 20 / 20 tie goes to the first one seen)
    assert by["P3"]["which_hand"] == "right" and by["P4"]["which_hand"] == "left"
    assert out["carry_over"]["players"]["P3"] == {"P3": SWAP, "P4": F - SWAP}
    assert out["carry_over"]["gate_m"] == pv.CARRY_GATE_M


def test_doc_time_ref_reads_variant_meta_or_the_sorted_first_base_camera():
    assert pv.doc_time_ref({"meta": {"cameras": [{"name": "b"}, {"name": "a"}]}}) == "a"
    assert pv.doc_time_ref({"meta": {"pose_variant": {"time_ref_view": "c"}, "cameras": [{"name": "a"}]}}) == "c"
    assert pv.doc_time_ref({"meta": {}}) is None


def test_carry_over_and_sidecars_skip_a_base_on_another_timeline(tmp_path):
    """Review 2 L4 / review 1 L13: the variant was synced to "b" (its reference "a" was dropped and not pinned):
    nothing is copied by frame index."""
    base, doc = _pair()
    base["meta"]["cameras"] = [{"name": "a"}, {"name": "b"}]
    base["shuttle"] = {"frames": [{"position": [0.0, 1.0, 0.0]}]}
    doc["meta"]["pose_variant"] = {"time_ref_view": "b"}
    doc["shuttle"] = {"frames": []}
    lines = []
    pv.carry_over(base, doc, log=lines.append)
    assert doc["shuttle"] == {"frames": []} and all(fr["racket"] is None for p in doc["players"] for fr in p["frames"])
    _write_sidecar(tmp_path, base)
    pv.write_racket_sidecars(str(tmp_path), "c", "c_x", base, doc, log=lines.append)
    assert not (tmp_path / "c_x.racket_rigid.json").exists()
    assert sum("not carried" in ln for ln in lines) == 2


def test_same_timeline_refuses_a_changed_reference_with_a_log_line():
    base = {"meta": {"cameras": [{"name": "back"}, {"name": "front"}]}}
    lines = []
    assert pv.same_timeline(base, "back", log=lines.append) and not lines
    assert pv.same_timeline({"meta": {}}, "front", log=lines.append)           # unknown reference: trusted
    assert not pv.same_timeline(base, "front", log=lines.append)
    assert "synced to back" in lines[0] and "not carried" in lines[0]
