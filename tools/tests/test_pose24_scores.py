"""pose24 score calibration — defaults equal pose_v2, per-group params, loader parity."""
import json

import numpy as np
import pytest

from analysis import pose_v2 as pv
from analysis.pose24 import scores


def _kp2d(tmp_path, rng):
    frames = []
    for _ in range(6):
        dets = []
        for _ in range(3):
            k = np.c_[rng.uniform(0, 2000, (133, 2)), rng.uniform(0.5, 6.0, 133)]
            dets.append({"keypoints": k.tolist()})
        frames.append(dets)
    p = tmp_path / "v_kp2d.json"
    p.write_text(json.dumps({"view": "v", "fps": 50.0, "width": 2160, "height": 3840, "frames": frames}))
    return str(p)


def test_defaults_equal_pose_v2_and_are_monotone():
    params = scores.load_params(path=None)
    raw = np.tile(np.linspace(0, 8, 50)[:, None], (1, 133))
    p = scores.calibrate(raw, params)
    assert np.array_equal(p, pv.norm_score(raw))
    assert (np.diff(p, axis=0) > 0).all()
    assert params["source"].startswith("default")


def test_to_raw_inverts_calibrate():
    params = scores.load_params(path=None)
    raw = np.random.default_rng(0).uniform(1, 5, (10, 133))
    assert np.allclose(scores.to_raw(scores.calibrate(raw, params), params), raw, atol=1e-6)


def test_group_params_only_touch_their_group(tmp_path):
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"groups": {"hands": {"mid": 4.0, "width": 0.5}}}))
    params = scores.load_params(str(f))
    assert params["groups"]["hands"] == {"mid": 4.0, "width": 0.5}
    assert params["groups"]["body"] == scores.DEFAULT_PARAMS["body"] and params["source"] == "cal.json"
    p = scores.calibrate(np.full(133, 4.0), params)
    assert p[100] == pytest.approx(0.5) and p[0] == pytest.approx(pv.norm_score(4.0))


def test_load_kp2d24_matches_pose_v2_loader_with_defaults(tmp_path):
    path = _kp2d(tmp_path, np.random.default_rng(1))
    fr_a, fps_a, idx_a = pv.load_kp2d(path)
    fr_b, fps_b, idx_b, header = scores.load_kp2d24(path, scores.load_params(path=None))
    assert fps_b == fps_a == 50.0 and idx_a == idx_b and header["width"] == 2160 and "frames" not in header
    for a, b in zip(fr_a, fr_b):
        assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_load_kp2d24_reports_missing_fps(tmp_path):
    p = tmp_path / "n_kp2d.json"
    p.write_text(json.dumps({"frames": [[]]}))
    assert scores.load_kp2d24(str(p), scores.load_params(path=None))[1] is None


@pytest.mark.parametrize("groups", [{"body": {"mid": 3.0, "width": 0.0}}, {"feet": {"mid": 3.0, "width": -1}},
                                    {"hands": {"width": 0.3}}])
def test_load_params_rejects_unusable_groups(tmp_path, groups):
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"groups": groups}))
    with pytest.raises(ValueError):
        scores.load_params(str(f))


def test_load_kp2d24_drops_detections_below_the_score_gate(tmp_path):
    good = np.c_[np.full((133, 2), 100.0), np.full(133, 4.0)]       # calibrated body ~1
    weak = np.c_[np.full((133, 2), 300.0), np.full(133, 2.0)]       # calibrated body ~0.08 < SCORE_GATE
    p = tmp_path / "g_kp2d.json"
    p.write_text(json.dumps({"fps": 60, "frames": [[{"keypoints": weak.tolist()}, {"keypoints": good.tolist()}]]}))
    frames, _, idx, _ = scores.load_kp2d24(str(p), scores.load_params(path=None))
    assert idx == [[1]] and len(frames[0]) == 1 and frames[0][0][0, 0] == 100.0
