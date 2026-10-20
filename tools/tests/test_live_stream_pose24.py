import numpy as np
import pytest

import live_synth as ls
from analysis import skeleton24 as sk
from analysis.live import outputs
from analysis.live.stream_pose24 import StreamPose24

PELVIS = sk.SMPL28_NAMES.index("pelvis")
LW = sk.SMPL28_NAMES.index("left_wrist")


def _run(n=80, budget_s=None, workers=0, offsets=None, drop_steps=(), skip_steps=(), stream=None, capture_lag_s=None):
    cams, players = ls.scene(F=2 * n + 4)
    cfg = ls.cfg(n_steps=n)
    cfg["stream"].update(stream or {})
    if offsets:
        cfg["offsets"] = offsets
    events = []
    s = StreamPose24(cfg, cams, budget_s=budget_s, workers=workers,
                     events=events.append)
    outs = []
    for i, r in enumerate(cfg["ref_frames"]):
        if i in skip_steps:
            continue
        fs = ls.frameset(i, r, cams, players, offsets=cfg["offsets"], drop=("P2",) if i in drop_steps else ())
        if capture_lag_s is not None:              # captured this long before it reaches the core (stream behind)
            fs["t_capture_ns"] = s.clock() - int(capture_lag_s * 1e9)
        outs += s.push(fs)
    outs += s.flush()
    s.close()
    return cfg, players, outs, events


def test_asfast_all_steps_in_order_warmup_then_fitted_and_accurate():
    cfg, players, outs, events = _run()
    assert [o["step"] for o in outs] == list(range(80))
    assert all(o["quality"] == outputs.WARMUP for o in outs[:30])
    assert all(o["quality"] == outputs.FITTED for o in outs[30:])
    for o in outs[30:]:
        J = o["players"]["P1"]["J"]
        assert np.linalg.norm(J[LW] - ls.truth(players, "P1", o["ref_frame"])[9]) < 0.03
    assert any(e["ev"] == "window" for e in events) and any(e["ev"] == "step" for e in events)


def test_zero_budget_degrades_every_window_but_keeps_order():
    _, _, outs, events = _run(budget_s=0.0)
    assert [o["step"] for o in outs] == list(range(80))                          # Review Focus 5
    assert all(o["quality"] == outputs.LATE for o in outs[30:])
    assert all(set(e["status"].values()) == {"late"} for e in events if e["ev"] == "window")


def test_interpolated_offsets_still_accurate():
    _, players, outs, _ = _run(offsets={"a": 0.0, "b": -0.7, "c": -0.4})
    for o in outs[40:]:
        J = o["players"]["P1"]["J"]
        assert np.linalg.norm(J[PELVIS] - outs[40]["players"]["P1"]["J"][PELVIS]) < 2.0
        assert np.linalg.norm(J[LW] - ls.truth(players, "P1", o["ref_frame"])[9]) < 0.05


def test_player_missing_for_a_while_is_invalid_not_a_crash():
    _, _, outs, _ = _run(drop_steps=set(range(40, 70)))                          # Review Focus 1
    assert len(outs) == 80
    assert any(not o["players"]["P2"]["valid"] for o in outs[40:70])
    assert all(o["players"]["P1"]["valid"] for o in outs[30:])
    assert all(o["players"]["P2"]["interp"] for o in outs[40:70] if o["players"]["P2"]["valid"])


def test_missing_steps_are_filled_by_flush():
    _, _, outs, _ = _run(skip_steps={78, 79})                                    # Review Focus 2
    assert [o["step"] for o in outs] == list(range(80))
    assert outs[79]["quality"] == outputs.NONE


def test_lost_warmup_last_step_still_builds_models():
    _, _, outs, _ = _run(skip_steps={29})
    assert [o["step"] for o in outs] == list(range(80))
    assert all(o["players"]["P1"]["quality"] == outputs.FITTED for o in outs[30:])


def test_duplicate_and_out_of_range_steps_are_ignored():
    n = 40
    cams, players = ls.scene(F=2 * n + 4)
    cfg = ls.cfg(n_steps=n)
    s = StreamPose24(cfg, cams)
    outs = []
    for i, r in enumerate(cfg["ref_frames"]):
        fs = ls.frameset(i, r, cams, players)
        outs += s.push(fs)
        if i == 5:
            outs += s.push(fs)
    outs += s.push(dict(ls.frameset(0, 0, cams, players), step=n + 3))
    outs += s.flush()
    s.close()
    assert [o["step"] for o in outs] == list(range(n))


@pytest.mark.slow
def test_worker_pool_gives_the_same_poses_as_inline():
    _, _, a, _ = _run(n=70)
    _, _, b, _ = _run(n=70, workers=2)
    for x, y in zip(a[30:], b[30:]):
        np.testing.assert_allclose(x["players"]["P1"]["J"], y["players"]["P1"]["J"], atol=1e-9)


def test_fit_every_2_fits_every_other_window_and_falls_back_between():
    cfg, players, outs, events = _run(n=100, stream={"fit_every": 2})
    wins = [e for e in events if e["ev"] == "window"]
    st = [set(w["status"].values()) for w in wins]
    assert st[0] == {"fitted"} and st[1] == {"skipped"} and st[2] == {"fitted"}
    q = {o["step"]: o["quality"] for o in outs}
    assert q[wins[1]["start"]] == outputs.LATE and q[wins[2]["start"]] == outputs.FITTED
    assert [o["step"] for o in outs] == list(range(100))


def test_max_iter_reaches_the_solver(monkeypatch):
    from analysis.live import stream_fit
    seen = []
    real = stream_fit.fit.WindowProblem24.solve

    def spy(self, max_nfev=40, tol=None, lsmr_maxiter=None, info=None):
        seen.append((max_nfev, tol, lsmr_maxiter))
        return real(self, max_nfev=max_nfev, tol=tol, lsmr_maxiter=lsmr_maxiter, info=info)
    monkeypatch.setattr(stream_fit.fit.WindowProblem24, "solve", spy)
    _run(n=60, stream={"max_iter": 7})
    assert seen and set(seen) == {(7, 1e-4, 20)}                 # v0.8.1 defaults: tol 1e-4, LSMR cap 20
    seen.clear()
    _run(n=60, stream={"max_iter": 7, "solver_tol": None, "lsmr_maxiter": None})   # the offline solver
    assert seen and set(seen) == {(7, None, None)}
    seen.clear()
    _run(n=60, stream={"max_iter": 9, "solver_tol": 1e-4, "lsmr_maxiter": 25})   # v0.8.1 solver knobs
    assert seen and set(seen) == {(9, 1e-4, 25)}


def test_player_hidden_during_warmup_gets_a_late_body_model():
    cfg, players, outs, events = _run(n=110, drop_steps=tuple(range(0, 32)))
    assert any(e["ev"] == "late_model" and e["pid"] == "P2" for e in events)
    late_q = [o["players"]["P2"]["quality"] for o in outs[90:]]
    assert outputs.FITTED in late_q


def test_late_policy_use_keeps_late_fits():
    cfg, players, outs, events = _run(n=80, budget_s=0.0, stream={"late_policy": "use"})
    wins = [e for e in events if e["ev"] == "window"]
    assert wins and all(set(w["status"].values()) == {"late_used"} for w in wins)
    assert all(o["quality"] == outputs.FITTED for o in outs[30:])


def test_late_policy_behind_keeps_fits_only_while_the_stream_is_behind():
    _, _, outs, events = _run(n=80, budget_s=0.0, stream={"late_policy": "behind"}, capture_lag_s=10.0)
    wins = [e for e in events if e["ev"] == "window"]
    assert wins and all(set(w["status"].values()) == {"late_used"} for w in wins)       # behind: kept
    assert all(o["quality"] == outputs.FITTED for o in outs[30:])
    _, _, outs, events = _run(n=80, budget_s=0.0, stream={"late_policy": "behind"}, capture_lag_s=-10.0)
    wins = [e for e in events if e["ev"] == "window"]
    assert wins and all("late_used" not in w["status"].values() for w in wins)          # on time: dropped
