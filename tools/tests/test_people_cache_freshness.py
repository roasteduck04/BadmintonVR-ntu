"""Appearance / OSNet caches: keyed to the kp2d they index (review 1 L11), file handles closed, OSNet weights path
resolved from the repo root (review 5 A3)."""
import json
import os

import numpy as np
import pytest

from analysis import appearance
from analysis.people import osnet


def _kp2d(path, n_dets):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"frames": [[{"keypoints": [[0.0, 0.0, 4.5]] * 133}] * n_dets]}, fh)


def _write_cache(path, views, sig=None):
    arrs = {}
    for v in views:
        arrs[f"{v}_idx"] = np.array([[0, 0], [2, 1]], np.int32)
        arrs[f"{v}_feat"] = np.ones((2, 4), np.float16)
        if sig is not None:
            arrs[f"{v}_kp2d_sha1"] = np.array(sig[v])
    np.savez_compressed(path, **arrs)


@pytest.mark.parametrize("kind", ["appearance", "osnet"])
def test_cache_is_dropped_per_view_when_its_kp2d_changed(kind, tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "pose2d").mkdir(parents=True)
    for v in ("a", "b"):
        _kp2d(data / "pose2d" / f"{v}_kp2d.json", 1)
    sig = {v: appearance.kp2d_sha1(str(data / "pose2d" / f"{v}_kp2d.json")) for v in ("a", "b")}
    monkeypatch.setattr(appearance, "CACHE_DIR", str(tmp_path))
    path = appearance.cache_path("clip") if kind == "appearance" else osnet.cache_path("clip", str(tmp_path))
    _write_cache(path, ("a", "b"), sig)
    _kp2d(data / "pose2d" / "b_kp2d.json", 2)                 # b's 2D re-run: its cached det indices are stale
    logs = []

    def load():
        if kind == "appearance":
            return appearance.load_clip("clip", data_dir=str(data), log=logs.append)
        return osnet.load_cache("clip", cache_dir=str(tmp_path), data_dir=str(data), log=logs.append)
    got = load()
    assert set(got) == {"a"} and got["a"][(2, 1)].shape == (4,)
    assert any("b" in s and "stale" in s for s in logs)
    os.remove(path)                                            # no handle left open (Windows refuses otherwise)


@pytest.mark.parametrize("kind", ["appearance", "osnet"])
def test_cache_without_a_key_is_used_with_a_note(kind, tmp_path, monkeypatch):
    monkeypatch.setattr(appearance, "CACHE_DIR", str(tmp_path))
    path = appearance.cache_path("clip") if kind == "appearance" else osnet.cache_path("clip", str(tmp_path))
    _write_cache(path, ("a",))
    logs = []
    got = (appearance.load_clip("clip", data_dir=str(tmp_path), log=logs.append) if kind == "appearance" else
           osnet.load_cache("clip", cache_dir=str(tmp_path), data_dir=str(tmp_path), log=logs.append))
    assert set(got) == {"a"} and any("no kp2d key" in s for s in logs)
    os.remove(path)


def test_osnet_weights_resolve_from_the_repo_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.isabs(osnet.WEIGHTS)
    assert os.path.normcase(osnet.WEIGHTS) == os.path.normcase(
        os.path.join(root, "models", "reid", "osnet_x0_25_msmt17.pth"))
