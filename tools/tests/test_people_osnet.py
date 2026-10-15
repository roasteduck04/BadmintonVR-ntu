"""OSNet-x0.25: vendored definition loads the downloaded weights strictly; embeddings; crop; cache keying."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from analysis.people import osnet

HAVE = os.path.exists(osnet.WEIGHTS)


@pytest.mark.slow
@pytest.mark.needs_data
@pytest.mark.skipif(not HAVE, reason="models/reid/osnet_x0_25_msmt17.pth not present")
def test_strict_load_and_embedding():
    m = osnet.load_model()
    a = np.full((256, 128, 3), (0, 0, 200), np.uint8)
    b = np.full((256, 128, 3), (200, 0, 0), np.uint8)
    e = osnet.embed(m, [a, a.copy(), b])
    assert e.shape == (3, 512) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    assert e[0] @ e[1] > 0.999 and e[0] @ e[2] < e[0] @ e[1]


def test_crop():
    img = np.zeros((1000, 800, 3), np.uint8)
    k = np.zeros((133, 3), np.float32)
    k[:17, 0], k[:17, 1], k[:17, 2] = 400, np.linspace(200, 800, 17), 0.9
    c = osnet.crop(img, k)
    assert c is not None and c.shape[0] > c.shape[1]
    k[:, 2] = 0.0
    assert osnet.crop(img, k) is None


@pytest.mark.slow                                                       # fresh interpreter subprocess
def test_people_imports_without_torch():
    import subprocess
    import sys
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    code = ("import sys; sys.path.insert(0, %r); import analysis.people.osnet, analysis.people.__main__; "
            "print('torch' in sys.modules)" % tools)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _kp2d(path, n, w, h, dets_at):
    """kp2d.json of n frames; dets_at(f) -> list of (x centre, y top, y bottom) body boxes; raw SimCC score 6."""
    import json
    frames = []
    for f in range(n):
        fr = []
        for (x, y0, y1) in dets_at(f):
            k = np.zeros((133, 3))
            k[:17, 0], k[:17, 1], k[:17, 2] = x + np.linspace(-20, 20, 17), np.linspace(y0, y1, 17), 6.0
            fr.append({"keypoints": k.tolist()})
        frames.append(fr)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"width": w, "height": h, "frame_count": n, "frames": frames}, fh)


@pytest.mark.slow
@pytest.mark.needs_data
@pytest.mark.skipif(not HAVE, reason="models/reid/osnet_x0_25_msmt17.pth not present")
def test_extract_clip_cache_keying(tmp_path):
    import cvdebug_synth as cs
    n, w, h = 12, 320, 240
    os.makedirs(tmp_path / "pose2d")
    vid = cs.write_numbered_video(str(tmp_path / "a.avi"), n, w, h)
    # step 5 samples frames 0, 5, 10: two people, none, one; view b has no video
    _kp2d(str(tmp_path / "pose2d" / "a_kp2d.json"), n, w, h,
          lambda f: {0: [(80, 30, 200), (240, 40, 210)], 10: [(80, 30, 200)]}.get(f, []))
    reg = {"syn": {"views": ["a", "b"], "videos": {"a": vid}}}
    logs = []
    p = osnet.extract_clip("syn", ["a", "b"], registry=reg, step=5, batch=1, log=logs.append,
                           data_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert p == osnet.cache_path("syn", str(tmp_path / "cache")) and os.path.exists(p)
    c = osnet.load_cache("syn", cache_dir=str(tmp_path / "cache"))
    assert set(c) == {"a"} and sorted(c["a"]) == [(0, 0), (0, 1), (10, 0)]
    e = np.stack(list(c["a"].values()))
    assert e.shape == (3, 512) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-2)
    assert any("no video" in s for s in logs)
    osnet.extract_clip("syn", ["a"], registry=reg, log=logs.append, data_dir=str(tmp_path),
                       cache_dir=str(tmp_path / "cache"))
    assert logs[-1].endswith("cached")
    with pytest.raises(FileNotFoundError, match="reid-extract"):
        osnet.load_cache("nope", cache_dir=str(tmp_path / "cache"))


def test_reid_extract_cli(monkeypatch):
    from analysis.people import __main__ as cli
    from analysis.pose24 import config
    reg = {"syn": {"views": ["a", "b", "c"], "mode": "singles", "seeds": "syn"},
           "syn_2cam": {"base": "syn", "views": ["a", "b"]}}
    calls = []
    monkeypatch.setattr(config, "load_registry", lambda *a, **k: reg)
    monkeypatch.setattr(config, "all_clips", lambda r: ["syn", "syn_2cam"])
    monkeypatch.setattr(osnet, "extract_clip", lambda clip, views, **k: calls.append((clip, views, k["step"])) or "x")
    assert cli.main(["reid-extract", "--all"]) == 0
    assert calls == [("syn", ["a", "b", "c"], 5)]
    calls.clear()
    cli.main(["reid-extract", "--clip", "syn_2cam", "--step", "3"])
    assert calls == [("syn", ["a", "b", "c"], 3)]
