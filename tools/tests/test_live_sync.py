# tools/tests/test_live_sync.py
import numpy as np
import pytest

from analysis import pose_v2 as pv
from analysis.live import sync
from analysis.pose24 import scores


@pytest.mark.parametrize("ref, off, mode, frames, alpha", [
    (390, 0.0, "interp", [390], 0.0),
    (390, -0.7, "interp", [389, 390], 0.3),
    (390, -0.4, "interp", [389, 390], 0.6),
    (390, -0.7, "nearest", [389], 0.0),
    (390, -0.4, "nearest", [390], 0.0),
    (390, -0.5, "nearest", [390], 0.0),      # exact half -> later frame (video_io.view_frame rule)
])
def test_source_frames(ref, off, mode, frames, alpha):
    f, a = sync.source_frames(ref, off, mode)
    assert f == frames and a == pytest.approx(alpha)


def test_schedule_steps_and_views():
    cfg = {"ref_frames": [390, 392], "views": ["back_051", "backright_051"], "sync_mode": "interp",
           "offsets": {"back_051": 0.0, "backright_051": -0.7}}
    s = sync.schedule(cfg)
    assert [x["step"] for x in s] == [0, 1]
    assert s[1]["views"]["backright_051"] == {"frames": [391, 392], "alpha": pytest.approx(0.3)}
    assert s[0]["views"]["back_051"] == {"frames": [390], "alpha": 0.0}


def _det(cx, cy, raw=4.5):
    k = np.zeros((133, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = cx + np.arange(133) * 0.1, cy, raw
    return k


def test_calibrate_dets_gates_low_scores():
    params = scores.load_params()
    out, idx = sync.calibrate_dets([_det(0, 0, raw=1.0), _det(10, 10, raw=4.5)], params)
    assert idx == [1] and len(out) == 1
    assert out[0][:, 2].min() > pv.SCORE_GATE and out[0].dtype == np.float32


def test_partner_picks_nearest_body_centre_and_rejects_far():
    params = scores.load_params()
    a = sync.calibrate_dets([_det(100, 100)], params)[0][0]
    others = sync.calibrate_dets([_det(900, 900), _det(104, 101)], params)[0]
    assert sync.partner(a, others) == 1
    assert sync.partner(a, others[:1]) is None          # 800 px away: not the same person


def test_interp_rows_and_missing_partner():
    A, B = _det(0, 0), _det(10, 20)
    A[:, 2], B[:, 2] = 0.9, 0.8
    r = sync.interp_rows(A, B, 0.3)
    assert r[0, 0] == pytest.approx(3.0) and r[0, 1] == pytest.approx(6.0) and r[0, 2] == pytest.approx(0.8)
    np.testing.assert_array_equal(sync.interp_rows(A, None, 0.3), A)    # Review Focus 4
    np.testing.assert_array_equal(sync.interp_rows(None, B, 0.3), B)


def test_raw_det_array():
    d = {"kp": np.ones((133, 2)), "score": np.full(133, 4.0)}
    k = sync.raw_det_array(d)
    assert k.shape == (133, 3) and k[5, 2] == 4.0
