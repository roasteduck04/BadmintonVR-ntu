# tools/tests/test_live_lrswap_lag.py
import numpy as np

import pose24_synth as syn
from analysis.live import lrswap_lag
from analysis.pose24 import lrswap


def test_window_swaps_equal_detect_swaps_on_the_span_and_find_an_injected_swap():
    views = ["a", "b", "c", "d"]
    cams = syn.rig(views)
    b = syn.bundle({"P1": syn.motion(90)}, cams, views, noise_px=0.3)
    D = b["obs"]["P1"]["b"]
    D[20:40] = D[20:40][:, lrswap.PERM]
    span = {"P1": {v: b["obs"]["P1"][v] for v in views}}
    S = lrswap_lag.window_swaps(cams, views, span, slice(30, 90))
    np.testing.assert_array_equal(S["P1"], lrswap.detect_swaps(b, "P1")[:, 30:90])
    assert S["P1"].shape == (4, 60)
    assert S["P1"][1, :10].all() and not S["P1"][1, 10:].any()
    assert not S["P1"][[0, 2, 3]].any()


def test_window_swaps_on_a_mid_clip_span_uses_span_relative_commit():
    views = ["a", "b", "c", "d"]
    cams = syn.rig(views)
    b = syn.bundle({"P1": syn.motion(90)}, cams, views, noise_px=0.3)
    D = b["obs"]["P1"]["b"]
    D[20:40] = D[20:40][:, lrswap.PERM]
    # span = absolute frames 10..90; commit is relative to the span start -> absolute frames 30..89
    span_obs = {"P1": {v: b["obs"]["P1"][v][10:90] for v in views}}
    span_bundle = dict(b, obs={"P1": {v: b["obs"]["P1"][v][10:90] for v in views}})
    S = lrswap_lag.window_swaps(cams, views, span_obs, slice(20, 80))
    np.testing.assert_array_equal(S["P1"], lrswap.detect_swaps(span_bundle, "P1")[:, 20:80])
    assert S["P1"].shape == (4, 60)
    # injected swap at absolute 20..39: absolute 30..39 is the first 10 committed frames
    assert S["P1"][1, :10].all() and not S["P1"][1, 10:].any()
    assert not S["P1"][[0, 2, 3]].any()
