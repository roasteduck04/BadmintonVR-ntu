# tools/tests/test_live_stream_fit.py
import numpy as np

import live_synth as ls
import pose24_synth as syn
from analysis import pose_v2 as pv
from analysis.live import stream_fit as sf
from analysis.pose24 import fit, kpset
from analysis.pose_fit import _fill_init

VIEWS = ["a", "b", "c"]


def _player(F=110, noise=0.3):
    cams = syn.rig(VIEWS)
    b = syn.bundle({"P1": syn.motion(F)}, cams, VIEWS, noise_px=noise)
    return cams, b


def test_stepper_schedule_matches_fit_segment():
    s = sf.WindowStepper(first=0, n_total=110, window=60, overlap=10)
    assert [s.next_window() for _ in range(3)] == [(0, 0, 60, 50), (48, 50, 110, 110), None]
    s = sf.WindowStepper(first=60, n_total=300, window=30, overlap=5)
    w = [s.next_window() for _ in range(11)]
    assert w[0] == (60, 60, 90, 85) and w[1] == (83, 85, 115, 110) and w[-2][2] == 300 and w[-1] is None


def test_stepped_windows_equal_offline_fit_segment():
    cams, b = _player()
    uv, w = kpset.player_obs(b, "P1")
    P = pv.proj_mats(cams, VIEWS)
    Xr, conf, inl = sf.triangulate35(cams, VIEWS, uv, w)
    ok = np.isfinite(Xr).all(-1)
    m = sf.warmup_model(cams, VIEWS, uv, w, ls.cfg()["stream"])
    init = _fill_init(Xr, ok, m.rest)
    contact = fit.detect_contact(Xr, 60.0)
    ref = fit._fit_segment(P, uv, w, init, m.edges, Xr, contact, m.sgn, 60.0, True)
    seg = init.copy()
    s = sf.WindowStepper(0, len(seg), 60, 10)
    while (win := s.next_window()) is not None:
        lo, start, stop, _ = win
        free = np.zeros(stop - lo, bool)
        free[start - lo:] = True
        r = sf.solve_window({"P": P, "uv": uv[lo:stop], "w": w[lo:stop], "init": seg[lo:stop], "free": free,
                             "edges": m.edges, "anchor": Xr[lo:stop], "contact": contact[lo:stop], "sgn": m.sgn,
                             "fps": 60.0})
        seg[lo:stop] = r["X"]
        assert r["solve_ns"] > 0 and r["t_done_ns"] > 0
    np.testing.assert_allclose(seg, ref, atol=1e-9)


def test_warmup_model_lengths_and_handedness():
    cams, b = _player(F=60, noise=0.0)
    uv, w = kpset.player_obs(b, "P1")
    m = sf.warmup_model(cams, VIEWS, uv, w, ls.cfg()["stream"])
    assert abs(m.lengths["edges"]["thigh"] - syn.LIMB_TRUE["thigh"]) < 0.02
    assert m.sgn == kpset.world_sign(cams)
    assert m.rest.shape == (35, 3) and np.isfinite(m.rest).all()


def test_warmup_model_without_hips_is_none():
    cams, b = _player(F=20)
    uv, w = kpset.player_obs(b, "P1")
    assert sf.warmup_model(cams, VIEWS, uv, np.zeros_like(w), ls.cfg()["stream"]) is None


def test_window_obs_applies_swaps():
    cams, b = _player(F=10)
    obs = b["obs"]["P1"]
    sw = np.zeros((3, 10), bool)
    sw[1, 2] = True
    uv0, _ = sf.window_obs(obs, VIEWS, 0, 10)
    uv1, _ = sf.window_obs(obs, VIEWS, 0, 10, sw)
    np.testing.assert_allclose(uv1[2, 9, 1], uv0[2, 10, 1])         # left wrist <-> right wrist in view b
    np.testing.assert_allclose(uv1[3], uv0[3])
