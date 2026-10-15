"""pose24 fit on a synthetic 4-camera clip: lengths, trunk bend, lag, jump, contact, gaps."""
import numpy as np
import pytest

import pose24_synth as syn
from analysis.pose24 import angles, fit, kpset

F = 150
VIEWS = ("a", "b", "c", "d")
QUIET = dict(log=lambda *a: None)


def _truth():
    return syn.motion(F, z0=-3.0, side_bend=20.0, jump=(40, 60), plant=(90, 130))


@pytest.fixture(scope="module")
def fitted():
    X = _truth()
    b = syn.bundle({"P1": X}, syn.rig(VIEWS), VIEWS, noise_px=1.0, seed=4)
    return X, fit.fit_player(b, "P1", **QUIET), fit.fit_player(b, "P1", use_contact=False, **QUIET)


def test_limb_lengths_and_stature(fitted):
    _, fr, _ = fitted
    for k, true in syn.LIMB_TRUE.items():
        assert fr["lengths"]["edges"][k] == pytest.approx(true, abs=0.01), k
    assert fr["lengths"]["stature"]["measured_m"] == pytest.approx(1.72, abs=0.08)


def test_no_frame_dropped_and_jump_kept(fitted):
    X, fr, _ = fitted
    assert np.isfinite(fr["X"]).all()
    assert (-fr["X"][:, 0, 1]).max() > 2.0


def test_trunk_side_bend_is_preserved(fitted):
    X, fr, _ = fitted
    true = angles.trunk_lateral_flexion(kpset.from133(X))
    got = angles.trunk_lateral_flexion(fr["X"])
    assert np.ptp(got) > 0.9 * np.ptp(true) and np.abs(got - true).mean() < 3.0


def test_zero_added_lag(fitted):
    X, fr, _ = fitted
    a = np.diff(kpset.from133(X)[:, 10, 0])
    b = np.diff(fr["X"][:, 10, 0])
    lags = range(-3, 4)
    corr = [np.dot(a[max(0, -L):len(a) - max(0, L)], b[max(0, L):len(b) - max(0, -L)]) for L in lags]
    assert list(lags)[int(np.argmax(corr))] == 0


def test_contact_term_reduces_foot_skate(fitted):
    _, with_c, without = fitted
    assert with_c["contact"][95:125, 2].mean() > 0.8          # left heel planted
    sk = [np.median(np.linalg.norm(np.diff(r["X"][92:128, 19][:, [0, 2]], axis=0), axis=-1)) for r in (with_c, without)]
    assert sk[0] < sk[1]


def test_long_gap_left_missing_short_gap_filled():
    X = syn.motion(90, z0=-3.0)
    b = syn.bundle({"P1": X}, syn.rig(VIEWS), VIEWS, noise_px=0.5)
    for v in VIEWS:
        b["obs"]["P1"][v][20:30] = np.nan                       # 10-frame gap: bridged
        b["obs"]["P1"][v][50:75] = np.nan                       # 25-frame gap (> MAX_GAP 20): missing
    fr = fit.fit_player(b, "P1", passes=1, **QUIET)
    assert np.isfinite(fr["X"][20:30]).all() and np.isnan(fr["X"][50:75]).all()
    assert not fr["inlier"][20:30].any()
    assert fit.support_segments(kpset.player_obs(b, "P1")[1]) == [(0, 50), (75, 90)]


def test_stature_prior_is_weak_with_many_frames():
    X = np.zeros((500, 35, 3))
    X[:, 13, 1] = 0.40                                         # thighs 0.40 m (implies 1.63 m)
    X[:, 14, 1] = 0.40
    L = fit.estimate_lengths(X, np.ones((500, 35), bool))
    assert L["edges"]["thigh"] == pytest.approx(0.40, abs=0.002)
    assert L["stature"]["prior_m"] == [1.70, 0.10]


def test_player_seen_by_one_view_only_is_reported():
    """singles_05x_2cam (real, 2026-09-28): the far player is out of frontright_051's frame, so no frame has
    2 views - the fit returns no pose and must say why, per view."""
    import pose24_synth as syn
    views = ("a", "b")
    b = syn.bundle({"P1": syn.motion(30, z0=-3.0)}, syn.rig(views), views)
    b["obs"]["P1"]["b"][:] = np.nan
    lines = []
    fr = fit.fit_player(b, "P1", passes=1, log=lines.append)
    assert not np.isfinite(fr["X"]).any() and fr["segments"] == []
    assert any("WARNING" in ln and "P1" in ln and "a 30" in ln and "b 0" in ln for ln in lines)
