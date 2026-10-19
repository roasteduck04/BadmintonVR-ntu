import numpy as np

import pose24_synth as syn
from analysis import skeleton24 as sk
from analysis.live import fallback, outputs
from analysis.pose24 import kpset

VIEWS = ["a", "b", "c"]


def test_ema_smooths_holds_and_restarts():
    s = fallback.EmaSmoother(alpha=0.5, max_hold=2)
    nan = np.full((35, 3), np.nan)
    assert np.isnan(s.push(nan)).all()
    a = s.push(np.zeros((35, 3)))
    b = s.push(np.ones((35, 3)))
    assert a[0, 0] == 0.0 and b[0, 0] == 0.5
    assert s.push(nan)[0, 0] == 0.5 and s.push(nan)[0, 0] == 0.5      # held 2 steps
    assert np.isnan(s.push(nan)).all()                                # then dropped
    assert s.push(np.full((35, 3), 4.0))[0, 0] == 4.0                 # restarts from the new value


def test_regress_frames_on_truth_and_player_frame():
    cams = syn.rig(VIEWS)
    Xup = syn.motion(8)
    b = syn.bundle({"P1": Xup}, cams, VIEWS)
    uv, w = kpset.player_obs(b, "P1")
    Xdown = kpset.from133(Xup) * np.array([1.0, -1.0, 1.0])
    r = outputs.regress_frames(cams, VIEWS, Xdown, uv, w, "smpl-28")
    lw = sk.SMPL28_NAMES.index("left_wrist")
    np.testing.assert_allclose(r["J"][:, lw], Xup[:, 9], atol=1e-6)
    assert r["J"].shape == (8, 28, 3) and (r["C"][:, lw] > 0.3).all() and r["forward_sign"] in (-1.0, 1.0)
    pf = outputs.player_frame(r, 3, outputs.FITTED)
    assert pf["valid"] and pf["quality"] == outputs.FITTED and pf["J"].shape == (28, 3)
    Xnan = Xdown.copy()
    Xnan[3] = np.nan
    pf = outputs.player_frame(outputs.regress_frames(cams, VIEWS, Xnan, uv, w, "smpl-28"), 3, outputs.FITTED)
    assert not pf["valid"] and pf["quality"] == outputs.NONE


def test_frame_quality():
    q = outputs.frame_quality
    assert q({"P1": {"valid": True, "quality": 0}, "P2": {"valid": True, "quality": 2}}) == 2
    assert q({"P1": {"valid": True, "quality": 0}, "P2": {"valid": False, "quality": 3}}) == 0
    assert q({"P1": {"valid": False, "quality": 3}}) == 3
