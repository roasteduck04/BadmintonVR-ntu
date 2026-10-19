# tools/tests/test_live_tracker.py
import numpy as np

import live_synth as ls
from analysis.live import sync
from analysis.live.tracker_singles import SinglesTracker
from analysis.pose24 import scores


def _cal(raw):
    p = scores.load_params()
    return {v: sync.calibrate_dets(raw[v], p)[0] for v in raw}


def test_tracker_follows_both_players_and_ignores_the_bystander():
    cams, players = ls.scene(F=120)
    tr = SinglesTracker(cams, ls.VIEWS, 30.0, ls.cfg()["tracker"])
    assert tr.sides == {"P1": "near", "P2": "far"}
    for f in range(0, 120, 2):
        out = tr.step(f, _cal(ls.raw_dets(cams, players, f)))
        assert out["P1"] == {v: 1 for v in ls.VIEWS}          # det 0 is the bystander (x = 5.5 m, off court)
        assert out["P2"] == {v: 2 for v in ls.VIEWS}


def test_tracker_reports_a_missing_player_and_reacquires():
    cams, players = ls.scene(F=60)
    tr = SinglesTracker(cams, ls.VIEWS, 30.0, ls.cfg()["tracker"])
    tr.step(0, _cal(ls.raw_dets(cams, players, 0)))
    out = tr.step(2, _cal(ls.raw_dets(cams, players, 2, drop=("P2",))))
    assert out["P2"] == {} and out["P1"] == {v: 1 for v in ls.VIEWS}
    out = tr.step(4, _cal(ls.raw_dets(cams, players, 4)))
    assert out["P2"] == {v: 2 for v in ls.VIEWS}


def test_empty_views_do_not_crash():
    cams, players = ls.scene(F=10)
    tr = SinglesTracker(cams, ls.VIEWS, 30.0, ls.cfg()["tracker"])
    assert tr.step(0, {v: [] for v in ls.VIEWS}) == {"P1": {}, "P2": {}}


def _cand(z, support=3, height=1.75, x=0.0, dets=None):
    from analysis.people.candidates import Candidate
    return Candidate(frame=0, dets=dets or {"a": 0}, floor=np.array([x, z]), support=support, height=height)


def _tracker_with(monkeypatch, frames, warm=30):
    from analysis.live import tracker_singles as ts
    it = iter(frames)
    monkeypatch.setattr(ts.candidates, "frame_candidates", lambda *a, **k: next(it))
    c = ls.cfg()
    return ts.SinglesTracker({}, [], 30.0, c["tracker"], wcfg=c["warmup"], warm_steps=warm)


def test_implausible_candidates_are_never_acquired(monkeypatch):
    kid, ghost, player = _cand(-3, height=0.8, dets={"a": 0}), _cand(-2, support=1, dets={"a": 1}), \
        _cand(-4, dets={"a": 2})
    tr = _tracker_with(monkeypatch, [[kid, ghost], [kid, ghost, player]])
    assert tr.step(0, {})["P1"] == {}
    assert tr.step(1, {})["P1"] == {"a": 2}
    assert tr.events[0]["kind"] == "acquire" and tr.events[0]["step"] == 1


def test_a_player_whose_candidates_turn_implausible_in_warmup_is_dropped(monkeypatch):
    seq = [[_cand(-3)]] + [[_cand(-3, height=0.8)]] * 12 + [[_cand(-3)]]
    tr = _tracker_with(monkeypatch, seq, warm=30)
    for i in range(len(seq)):
        tr.step(i, {})
    kinds = [e["kind"] for e in tr.events if e["pid"] == "P1"]
    assert kinds[:2] == ["acquire", "drop"] and kinds[-1] == "acquire"


def test_a_player_briefly_seen_by_one_camera_in_warmup_is_kept(monkeypatch):
    seq = [[_cand(3)]] + [[_cand(3, support=1, height=float("nan"))]] * 8 + [[_cand(3)]] * 3
    tr = _tracker_with(monkeypatch, seq, warm=30)
    for i in range(len(seq)):
        assert tr.step(i, {})["P2"] == {"a": 0}
    assert [e["kind"] for e in tr.events] == ["acquire"]


def test_no_drop_after_warmup(monkeypatch):
    seq = [[_cand(-3)]] + [[_cand(-3, height=0.8)]] * 12
    tr = _tracker_with(monkeypatch, seq, warm=1)
    for i in range(len(seq)):
        tr.step(i, {})
    assert [e["kind"] for e in tr.events] == ["acquire"]


def test_ankle_height_is_estimated_from_warmup_candidates(monkeypatch):
    from analysis.people.candidates import Candidate
    mk = lambda h: Candidate(frame=0, dets={"a": 0}, floor=np.array([0.0, -3.0]), support=3, height=1.7, tri=True,
                             ankle_h=h)
    seq = [[mk(0.14 + 0.001 * (i % 3))] for i in range(40)]
    tr = _tracker_with(monkeypatch, seq, warm=40)
    assert tr.ankle_auto
    for i in range(40):
        tr.step(i, {})
    res = tr.estimate_ankle()
    assert abs(res["ankle_h_m"] - 0.141) < 0.002 and not res["used_default"] and tr.ankle_h == res["ankle_h_m"]
