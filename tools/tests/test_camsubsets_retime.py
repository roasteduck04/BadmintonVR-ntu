import numpy as np
import pytest

from analysis.pose24.camsubsets import retime


def test_frame_shift_median_and_spread():
    ref = {"a": 0.0, "b": 2.0, "c": -1.0}
    sub = {"b": 0.0, "c": -3.5}                       # sub timeline re-referenced on b
    s, sp = retime.frame_shift(ref, sub)
    assert s == pytest.approx(2.25) and sp == pytest.approx(0.5)
    assert retime.frame_shift(ref, {"z": 0.0}) == (None, None)


def test_resample_fractional_and_edges():
    a = np.arange(10, dtype=float)[:, None] * np.array([1.0, 10.0])
    out = retime.resample(a, 1.5, 10)
    np.testing.assert_allclose(out[0], [1.5, 15.0])
    assert np.isnan(out[-1]).all() and np.isfinite(out[7]).all()      # 8 + 1.5 = 9.5 > 9 -> NaN at f8, f9
    assert np.isnan(out[8]).all()
    neg = retime.resample(a, -2.0, 10)
    assert np.isnan(neg[:2]).all() and neg[2, 0] == 0.0
    b = a.copy()
    b[4] = np.nan
    assert np.isnan(retime.resample(b, 0.5, 10)[3]).all()             # NaN neighbour stays NaN
    long = retime.resample(a, 0.0, 14)
    assert long.shape == (14, 2) and np.isnan(long[10:]).all()


def test_resample_1d_bool_and_integer_shift_keeps_frame_before_nan():
    a = np.array([0.0, 2.0, 4.0, np.nan, 8.0])
    out = retime.resample(a, 0.5, 5)
    np.testing.assert_allclose(out[:2], [1.0, 3.0])
    assert np.isnan(out[2:]).all()
    np.testing.assert_allclose(retime.resample(a, 1.0, 2), [2.0, 4.0])   # exact frame: NaN next frame ignored
    m = retime.resample(np.array([True, False, True]), 1.0, 3)
    assert m.dtype == float and m[0] == 0.0 and m[1] == 1.0 and np.isnan(m[2])


def test_resample_empty():
    assert retime.resample(np.zeros((5, 3)), 0.0, 0).shape == (0, 3)
    e = retime.resample(np.zeros((0, 3)), 0.5, 4)
    assert e.shape == (4, 3) and np.isnan(e).all()
    assert retime.resample(np.zeros(0), 0.0, 0).shape == (0,)
