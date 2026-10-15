"""v0.7.5 monoprior: shared keypoint maps + the prior-file contract writer (priors/common.py)."""
import json

import numpy as np
import pytest

from analysis.pose24 import kpset, mono
from analysis.pose24.priors import common


def _pose133(F=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((F, 133, 3))
    x[:, :, :2] = rng.uniform(0, 1000, (F, 133, 2))
    x[:, :, 2] = rng.uniform(0.5, 1.0, (F, 133))
    return x


def test_coco133_to_h36m_follows_halpe2h36m():
    x = _pose133()
    y = common.coco133_to_h36m(x)
    assert y.shape == (3, 17, 3)
    xy, s = x[..., :2], x[..., 2]
    hip, neck, head = (xy[:, 11] + xy[:, 12]) / 2, (xy[:, 5] + xy[:, 6]) / 2, (xy[:, 3] + xy[:, 4]) / 2
    np.testing.assert_allclose(y[:, 0, :2], hip)
    np.testing.assert_allclose(y[:, 7, :2], (hip + neck) / 2)
    np.testing.assert_allclose(y[:, 8, :2], neck)
    np.testing.assert_allclose(y[:, 9, :2], xy[:, 0])
    np.testing.assert_allclose(y[:, 10, :2], head)
    for h, c in {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}.items():
        np.testing.assert_allclose(y[:, h], x[:, c])
    np.testing.assert_allclose(y[:, 0, 2], np.minimum(s[:, 11], s[:, 12]))
    np.testing.assert_allclose(y[:, 7, 2], s[:, [5, 6, 11, 12]].min(1))
    np.testing.assert_allclose(y[:, 8, 2], np.minimum(s[:, 5], s[:, 6]))
    np.testing.assert_allclose(y[:, 10, 2], np.minimum(s[:, 3], s[:, 4]))


def test_h36m_to_kp_pairs():
    assert len(common.H36M_TO_KP) == 13
    names = {h: kpset.NAMES[k] for h, k in common.H36M_TO_KP}
    assert names == {4: "left_hip", 1: "right_hip", 5: "left_knee", 2: "right_knee", 6: "left_ankle",
                     3: "right_ankle", 11: "left_shoulder", 14: "right_shoulder", 12: "left_elbow",
                     15: "right_elbow", 13: "left_wrist", 16: "right_wrist", 9: "nose"}


def test_missing_frames_get_zero_conf():
    x = _pose133(F=4)
    x[1] = 0.0                                   # a whole missing frame (score 0, coords 0)
    x[2, 12, 2] = 0.0                            # right hip missing -> root, spine missing too
    x[3, 0] = np.nan                             # NaN nose
    y = common.coco133_to_h36m(x)
    assert (y[1, :, 2] == 0).all() and np.isnan(y[1, :, :2]).all()
    for h in (0, 1, 7):
        assert y[2, h, 2] == 0 and np.isnan(y[2, h, :2]).all()
    assert y[2, 4, 2] > 0 and np.isfinite(y[2, 4, :2]).all()
    assert y[3, 9, 2] == 0 and np.isnan(y[3, 9, :2]).all()
    assert np.isfinite(y[0]).all()


def _players(F=5):
    rng = np.random.default_rng(1)
    kp = np.array([k for _, k in common.H36M_TO_KP])
    return {"P1": {"kp": kp, "rel_cam": rng.normal(size=(F, len(kp), 3)), "conf": rng.uniform(0, 1, F),
                   "contact": rng.uniform(0, 1, (F, 6))},
            "P2": {"kp": kp[:5], "rel_cam": rng.normal(size=(F, 5, 3)), "conf": rng.uniform(0, 1, F)}}


def _meta(F=5):
    return {"method": "motionbert", "clip": "c1", "view": "front", "timeline_of": None, "fps": 60.0,
            "frame_count": F, "build_id": "abc123"}


def test_prior_roundtrip_through_mono_load_prior(tmp_path):
    p = tmp_path / "sub" / "front_motionbert.npz"
    players = _players()
    common.write_prior(p, _meta(), players)
    assert p.exists() and not list(p.parent.glob("*.tmp*"))
    for pid, d in players.items():
        got = mono.load_prior(p, pid)
        np.testing.assert_array_equal(got["kp"], d["kp"])
        np.testing.assert_allclose(got["rel_cam"], d["rel_cam"])
        np.testing.assert_allclose(got["conf"], d["conf"])
        if "contact" in d:
            np.testing.assert_allclose(got["contact"], d["contact"])
        else:
            assert got["contact"] is None
        assert got["meta"]["build_id"] == "abc123" and got["meta"]["method"] == "motionbert"
    assert mono.load_prior(p, "P9") is None


def test_write_prior_validates(tmp_path):
    p = tmp_path / "x.npz"
    for key in ("method", "build_id", "frame_count"):
        m = _meta()
        del m[key]
        with pytest.raises(ValueError):
            common.write_prior(p, m, _players())
    bad = _players()
    bad["P1"]["rel_cam"] = bad["P1"]["rel_cam"][:4]
    with pytest.raises(ValueError):
        common.write_prior(p, _meta(), bad)
    bad = _players()
    bad["P1"]["conf"] = bad["P1"]["conf"][:, None]
    with pytest.raises(ValueError):
        common.write_prior(p, _meta(), bad)
    bad = _players()
    bad["P1"]["contact"] = bad["P1"]["contact"][:, :5]
    with pytest.raises(ValueError):
        common.write_prior(p, _meta(), bad)
    m = _meta()
    m["fps"] = float("nan")
    with pytest.raises(ValueError):
        common.write_prior(p, m, _players())
    assert not p.exists()


def test_view_image_size():
    cache = {"image_size": {"front": [1920, 1080], "back": [None, None]}}
    assert common.view_image_size(cache, "front") == (1920, 1080)
    for v in ("back", "side"):
        with pytest.raises(KeyError, match=v):
            common.view_image_size(cache, v)
    with pytest.raises(KeyError):
        common.view_image_size({}, "front")
