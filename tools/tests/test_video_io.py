import os
import shutil

import numpy as np
import pytest

import cvdebug_synth as cs
from analysis import video_io as vio

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gitignore_has_overlay_and_proxy_dirs():
    text = open(os.path.join(REPO, ".gitignore"), encoding="utf-8").read().splitlines()
    assert "data/overlays/" in text and "data/video_proxy/" in text


@pytest.mark.parametrize("f,off,idx,rem", [(10, 0.5, 11, -0.5), (10, -1.7, 8, 0.3), (0, -0.4, 0, -0.4),
                                           (0, -0.6, -1, 0.4), (5, 0.0, 5, 0.0)])
def test_view_frame_rounds_half_up(f, off, idx, rem):
    i, r = vio.view_frame(f, off)
    assert i == idx and r == pytest.approx(rem)


def test_reader_returns_exact_frames_forward_backward_and_stride(tmp_path):
    p = cs.write_numbered_video(str(tmp_path / "a.avi"), 40, 160, 96)
    with vio.VideoReader(p, n_frames=40) as r:
        assert cs.read_number(r.get(0)) == 0
        assert cs.read_number(r.get(17)) == 17          # forward skip (grab)
        assert cs.read_number(r.get(18)) == 18
        assert cs.read_number(r.get(5)) == 5            # backward: reopen, decode from 0
        assert [cs.read_number(r.get(k)) for k in range(20, 40, 7)] == [20, 27, 34]
        assert r.get(-1) is None and r.get(40) is None


def test_reader_scales_after_decode(tmp_path):
    p = cs.write_numbered_video(str(tmp_path / "a.avi"), 3, 160, 96)
    with vio.VideoReader(p, scale=0.5) as r:
        assert r.get(1).shape == (48, 80, 3)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_reader_exact_on_long_gop_h264(tmp_path):
    p = cs.write_numbered_video(str(tmp_path / "a.mp4"), 90, 160, 96, codec="h264")
    with vio.VideoReader(p) as r:
        assert [cs.read_number(r.get(k)) for k in (0, 29, 30, 31, 61, 89)] == [0, 29, 30, 31, 61, 89]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_probe_and_check_video(tmp_path):
    p = cs.write_numbered_video(str(tmp_path / "a.mp4"), 30, 160, 96, codec="h264")
    info = vio.probe(p)
    assert (info["width"], info["height"], info["frames"], info["vfr"]) == (160, 96, 30, False)
    assert vio.check_video("a", p, {"width": 160, "height": 96, "frame_count": 30})["frames"] == 30
    with pytest.raises(vio.VideoError, match="a: .*frame count 30 .* kp2d 31"):
        vio.check_video("a", p, {"width": 160, "height": 96, "frame_count": 31})
    with pytest.raises(vio.VideoError, match="a: .*96x160"):
        vio.check_video("a", p, {"width": 96, "height": 160, "frame_count": 30})


def test_check_video_missing_file_names_the_view(tmp_path):
    with pytest.raises(vio.VideoError, match="back_051: video not found"):
        vio.check_video("back_051", str(tmp_path / "nope.mov"), {"width": 1, "height": 1, "frame_count": 1})


def test_resolve_videos_uses_the_base_entry_for_subsets(tmp_path):
    reg = {"c": {"mode": "singles", "views": ["b", "a"], "videos": {"a": "v/a.mov", "b": None}},
           "c_2cam": {"base": "c", "views": ["a"]}}
    out = vio.resolve_videos("c_2cam", registry=reg, root=str(tmp_path))
    assert list(out) == ["a", "b"]
    assert out["a"] == os.path.join(str(tmp_path), "v/a.mov") and out["b"] is None
