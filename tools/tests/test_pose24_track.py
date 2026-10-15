"""pose24 tracking/sync: shift recovery, detection indices, shared timeline, view health, errors."""
import json
import os

import numpy as np
import pytest

import pose24_synth as syn
from analysis.pose24 import config, scores, track

QUIET = dict(log=lambda *a: None)


def _clip(tmp_path, views=("a", "b", "c", "d"), shifts=None, F=150):
    cams = syn.rig(views)
    players = {"P1": syn.motion(F, x0=-0.5, z0=-3.0), "P2": syn.motion(F, x0=0.5, z0=3.0)}
    entry = syn.write_clip(str(tmp_path), "synclip", players, cams, views, shifts=shifts)
    return {"synclip": entry, "synclip_2cam": {"base": "synclip", "views": ["b", "c"], "timeline_of": "synclip"}}


def _cfg(tmp_path, reg, clip="synclip", **kw):
    return config.resolve(clip, registry=reg, data_dir=str(tmp_path / "data"), analysis_dir=str(tmp_path), **kw)


@pytest.fixture
def params():
    return scores.load_params(path=None)


def test_track_recovers_shift_and_detection_indices(tmp_path, monkeypatch, params):
    monkeypatch.chdir(tmp_path)
    reg = _clip(tmp_path, shifts={"c": 3})
    b = track.track_and_sync24(_cfg(tmp_path, reg), params, **QUIET)
    # pose_v2's sub-frame refinement is flat to ~+-0.3 f on this synthetic motion; the integer shift is exact
    assert b["shifts"]["c"] == 3 and b["offsets"]["c"] == pytest.approx(3.0, abs=0.35)
    assert b["time_ref_view"] == "a"
    assert b["view_health_checked"] and b["excluded"] == {} and set(b["obs"]) == {"P1", "P2"}
    k = b["views"].index("a")
    D = b["det_index"]["P1"]
    assert (D[k, 10:140] == 1).mean() > 0.95          # det 0 is the bystander, P1 is det 1
    raw = json.load(open(tmp_path / "data" / "pose2d" / "a_kp2d.json"))
    kp = np.array(raw["frames"][50 + b["shifts"]["a"]][D[k, 50]]["keypoints"])
    assert np.allclose(kp[:, :2], b["obs"]["P1"]["a"][50][:, :2], atol=1e-3)
    kc, Dc = b["views"].index("c"), b["det_index"]["P1"]           # shifted view: raw frame f + shifts["c"]
    raw_c = json.load(open(tmp_path / "data" / "pose2d" / "c_kp2d.json"))
    kp_c = np.array(raw_c["frames"][50 + b["shifts"]["c"]][Dc[kc, 50]]["keypoints"])
    gap = np.linalg.norm(kp_c[:17, :2] - b["obs"]["P1"]["c"][50][:17, :2], axis=-1)
    assert Dc[kc, 50] == 1 and np.median(gap) < 5.0                # obs is sub-frame interpolated


def test_subset_build_shares_the_full_timeline(tmp_path, monkeypatch, params):
    monkeypatch.chdir(tmp_path)
    reg = _clip(tmp_path, shifts={"c": 3})
    full = track.track_and_sync24(_cfg(tmp_path, reg), params, **QUIET)
    tl = {"clip": "synclip", "offsets": full["offsets_all"], "time_ref_view": full["time_ref_view"],
          "frame_count": full["frame_count"]}
    sub = track.track_and_sync24(_cfg(tmp_path, reg, "synclip_2cam"), params, tl, **QUIET)
    assert sub["time_ref_view"] == "a" and sub["frame_count"] == full["frame_count"]
    assert sub["offsets"]["c"] == full["offsets"]["c"] and sub["timeline_of"] == "synclip"
    assert sub["view_health_checked"] is False
    a, s = full["obs"]["P1"]["c"], sub["obs"]["P1"]["c"]
    m = np.isfinite(a[..., 0]).all(-1) & np.isfinite(s[..., 0]).all(-1)
    assert m.mean() > 0.9 and np.allclose(a[m], s[m])


def test_bad_calibration_view_is_dropped(tmp_path, monkeypatch, params):
    monkeypatch.chdir(tmp_path)
    reg = _clip(tmp_path)
    p = tmp_path / "data" / "calib" / "d_cam.json"
    cam = json.loads(p.read_text())
    cam["t"] = (np.array(cam["t"]) + [0.5, 0.0, 0.0]).tolist()
    p.write_text(json.dumps(cam))
    b = track.track_and_sync24(_cfg(tmp_path, reg), params, **QUIET)
    assert "d" in b["excluded"] and "d" not in b["views"] and "d" in b["cams_all"]
    assert "d" in b["offsets_all"] and b["view_health_checked"] is True


def test_dropped_sync_reference_rebases_the_timeline(tmp_path, monkeypatch, params):
    # a corrupted reference calibration also corrupts every sync offset against it (pose_v2 syncs pairwise
    # to the reference), so view health can't single it out; flag the reference directly instead
    monkeypatch.chdir(tmp_path)
    true = {"a": 0, "b": 5, "c": 3, "d": 0}                        # b (the next reference) is itself shifted
    reg = _clip(tmp_path, shifts=true)
    real = track.pv.view_health

    def health(cams, views, obs):
        h = real(cams, views, obs)
        return {**h, "a": 99.0} if "a" in h else h

    monkeypatch.setattr(track.pv, "view_health", health)
    b = track.track_and_sync24(_cfg(tmp_path, reg), params, **QUIET)
    ref, oa = b["time_ref_view"], b["offsets_all"]
    assert "a" in b["excluded"] and ref != "a" and oa[ref] == 0.0 and set(oa) == {"a", "b", "c", "d"}
    for u in true:                                                 # every entry (dropped a too) is on ref's timeline
        assert oa[u] == pytest.approx(true[u] - true[ref], abs=0.35), u


def test_drop_to_two_views_keeps_view_health_checked(tmp_path, monkeypatch, params):
    monkeypatch.chdir(tmp_path)
    reg = _clip(tmp_path)
    monkeypatch.setattr(track.pv, "view_health", lambda cams, views, obs: {v: (99.0 if v in "ab" else 2.0) for v in views})
    b = track.track_and_sync24(_cfg(tmp_path, reg), params, **QUIET)
    assert b["views"] == ["c", "d"] and set(b["excluded"]) == {"a", "b"}
    assert b["view_health_checked"] is True and set(b["view_health"]) == {"a", "b", "c", "d"}


def test_timeline_errors(tmp_path, monkeypatch, params):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="build synclip first"):
        track.load_timeline("synclip", str(tmp_path / "cache"))
    reg = _clip(tmp_path)
    tl = {"clip": "synclip", "offsets": {"a": 0.0, "b": 0.0}, "time_ref_view": "a", "frame_count": 150}
    with pytest.raises(ValueError, match="no offset"):
        track.track_and_sync24(_cfg(tmp_path, reg, "synclip_2cam"), params, tl, **QUIET)


def test_load_timeline_reads_cache_meta(tmp_path):
    os.makedirs(tmp_path / "cache")
    meta = {"offsets": {"a": 0.0}, "offsets_all": {"a": 0.0, "d": 1.5}, "time_ref_view": "a", "frame_count": 99}
    np.savez_compressed(track.cache_path("x", str(tmp_path / "cache")), meta=json.dumps(meta))
    tl = track.load_timeline("x", str(tmp_path / "cache"))
    assert tl == {"clip": "x", "offsets": {"a": 0.0, "d": 1.5}, "time_ref_view": "a", "frame_count": 99}
