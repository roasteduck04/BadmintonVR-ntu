"""pose24 left/right keypoint swap detection (Viterbi over per-view hypotheses)."""
import numpy as np
import pytest

import pose24_synth as syn
from analysis.pose24 import lrswap


def _bundle(views, F=90, noise=0.5):
    return syn.bundle({"P1": syn.motion(F, z0=-3.0)}, syn.rig(views), views, noise_px=noise)


def _inject(b, view, f0, f1):
    D = b["obs"]["P1"][view]
    D[f0:f1] = D[f0:f1][:, lrswap.PERM]


def test_perm_is_an_involution_that_swaps_left_right():
    assert (lrswap.PERM[lrswap.PERM] == np.arange(133)).all()
    assert lrswap.PERM[5] == 6 and lrswap.PERM[17] == 20 and lrswap.PERM[91] == 112 and lrswap.PERM[0] == 0
    assert lrswap.PERM[59] == 68 and lrswap.PERM[63] == 70 and lrswap.PERM[23] == 23   # eye rings; jaw not permuted


def test_viterbi_prefers_persistent_states():
    cost = np.zeros((20, 2))
    cost[:, 1] = 1.0
    cost[8:12] = [3.0, 0.0]
    cost[15] = [1.5, 0.0]
    s = lrswap.viterbi2(cost, switch=4.0)
    assert s[8:12].all() and not s[:8].any() and not s[15]   # a 1-frame blip isn't worth two switches


def test_three_views_recover_injected_swap():
    b = _bundle(("a", "b", "c"))
    _inject(b, "b", 30, 50)
    S = lrswap.detect_swaps(b, "P1")
    k = b["views"].index("b")
    assert S[k, 30:50].mean() > 0.9 and S[k, :28].sum() + S[k, 52:].sum() <= 2
    assert S[[i for i in range(3) if i != k]].sum() <= 2


def test_two_views_recover_injected_swap():
    b = _bundle(("a", "c"))
    _inject(b, "c", 30, 50)
    S = lrswap.detect_swaps(b, "P1")
    assert S[1, 30:50].mean() > 0.9 and S[1].sum() <= 22 and not S[0].any()


def test_apply_swaps_restores_observations():
    b = _bundle(("a", "b", "c"))
    clean = b["obs"]["P1"]["b"].copy()
    _inject(b, "b", 30, 50)
    S = lrswap.detect_swaps(b, "P1")
    counts = lrswap.apply_swaps(b, {"P1": S})
    k = b["views"].index("b")
    m = S[k, 30:50]
    assert counts["b"] >= 18 and counts["a"] <= 2
    assert np.allclose(b["obs"]["P1"]["b"][30:50][m], clean[30:50][m], equal_nan=True)


@pytest.mark.parametrize("views", [("a", "b", "c"), ("a", "c")])
def test_one_side_unseen_in_every_view_is_no_evidence(views):
    """Side-on player: right-side body scores below W_MIN everywhere -> neither hypothesis has evidence."""
    b = syn.bundle({"P1": syn.motion(150, z0=-3.0)}, syn.rig(views), views, noise_px=2.0)
    for v in views:
        b["obs"]["P1"][v][20:120, [6, 8, 10, 12, 14, 16], 2] = 0.2
    S = lrswap.detect_swaps(b, "P1")
    assert S.sum() <= 2


def _wrong_person(views, bad, other_at, F=120):
    """Views in `bad` track another player (at other_at = (x0, z0)) for frames 30-90."""
    cams = syn.rig(views)
    b = syn.bundle({"P1": syn.motion(F, z0=-3.0)}, cams, views, noise_px=0.5)
    other = syn.bundle({"P1": syn.motion(F, x0=other_at[0], z0=other_at[1])}, cams, views, noise_px=0.5, seed=3)
    for v in bad:
        b["obs"]["P1"][v][30:90] = other["obs"]["P1"][v][30:90]
    return b


@pytest.mark.parametrize("bad, other_at", [(("d",), (1.5, -1.0)), (("c", "d"), (1.0, -3.0)), (("b", "d"), (0.8, -2.0))])
def test_views_on_a_wrong_person_do_not_flip_good_views(bad, other_at):
    """Real doubles: a view (or two) tracks the partner for a stretch; the consensus of the other views then
    fits neither hypothesis of a good view -> no evidence, the good views stay as detected."""
    views = ("a", "b", "c", "d")
    b = _wrong_person(views, bad, other_at)
    S = lrswap.detect_swaps(b, "P1")
    good = [i for i, v in enumerate(views) if v not in bad]
    assert S[good].sum() <= 2
    if len(bad) == 1:                     # the wrong view itself: both hypotheses implausible -> not flagged
        assert S[views.index(bad[0])].sum() <= 2


@pytest.mark.parametrize("views", [("a", "b", "c", "d"), ("a", "c")])
def test_barely_seen_view_with_tied_costs_is_not_flagged(views):
    """Real singles far player in a front view: 4 frames of evidence with near-tied costs must not flip the
    whole track (no-evidence frames fall back to 'as detected')."""
    F = 150
    b = syn.bundle({"P1": syn.motion(F, z0=-3.0)}, syn.rig(views), views, noise_px=0.5)
    D = b["obs"]["P1"][views[-1]]
    seen = np.zeros(F, bool)
    seen[60:64] = True
    D[~seen, :, 2] = 0.0
    for l, r in lrswap._PAIRS:                # collapse each L/R pair to its midpoint, 1 px towards the mirror
        mid = 0.5 * (D[seen, l, :2] + D[seen, r, :2])
        u = D[seen, r, :2] - mid
        u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
        D[np.ix_(seen, [l], [0, 1])] = (mid + 1.0 * u)[:, None]
        D[np.ix_(seen, [r], [0, 1])] = (mid - 1.0 * u)[:, None]
    S = lrswap.detect_swaps(b, "P1")
    assert S.sum() <= 4


def test_one_view_and_zero_frames_flag_nothing():
    b = _bundle(("a",), F=20)
    S = lrswap.detect_swaps(b, "P1")
    assert S.shape == (1, 20) and not S.any()
    b = _bundle(("a", "b", "c"), F=20)
    for v in b["obs"]["P1"]:
        b["obs"]["P1"][v] = b["obs"]["P1"][v][:0]
    S = lrswap.detect_swaps(b, "P1")
    assert S.shape == (3, 0)
    assert lrswap.apply_swaps(b, {"P1": S}) == {"a": 0, "b": 0, "c": 0}
