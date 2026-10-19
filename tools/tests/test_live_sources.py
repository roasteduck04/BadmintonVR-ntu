# tools/tests/test_live_sources.py
import json
import queue
import threading
import time
import types

import numpy as np
import pytest

from analysis.live import sources


def _kp2d(tmp_path, view, n=6):
    frames = []
    for f in range(n):
        k = np.zeros((133, 3))
        k[:, 0], k[:, 1], k[:, 2] = f, 2 * f, 5.0
        frames.append([{"track_id": 1, "keypoints": k.tolist()}] if f % 2 == 0 else [])
    p = tmp_path / f"{view}_kp2d.json"
    p.write_text(json.dumps({"view": view, "fps": 60.0, "frame_count": n, "width": 8, "height": 8,
                             "frames": frames}), encoding="utf-8")
    return str(p)


def test_cached_source_replays_kp2d_frames(tmp_path):
    cfg = {"views": ["a", "b"], "kp2d_paths": {v: _kp2d(tmp_path, v) for v in "ab"}, "source_fps": 60.0}
    s = sources.CachedSource(cfg, log=lambda *a: None)
    s.start({"a": [2, 3, 4], "b": [2, 3, 4]}, {"a": 2.0, "b": 2.0})
    s.begin(1_000_000_000)
    dets, tim = s.get("a", 4)
    assert len(dets) == 1 and dets[0].shape == (133, 3) and dets[0][0, 0] == 4 and dets[0][0, 2] == 5.0
    assert tim["t_avail_ns"] == 1_000_000_000 + int(2 / 60 * 1e9) and not tim["dropped"]
    assert s.get("b", 3)[0] == [] and s.get("b", 99)[0] == []


def test_plan_frames_collects_every_needed_source_frame():
    sched = [{"views": {"a": {"frames": [4, 5]}, "b": {"frames": [4]}}},
             {"views": {"a": {"frames": [6, 7]}, "b": {"frames": [6]}}}]
    assert sources.plan_frames(sched) == {"a": [4, 5, 6, 7], "b": [4, 6]}


def _video(tmp_path, n=8, w=64, h=48):
    cv2 = pytest.importorskip("cv2")
    p = str(tmp_path / "v.avi")
    wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (w, h))
    for f in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :, 2] = 20 * f                      # brightness encodes the frame number
        wr.write(img)
    wr.release()
    return p


class FakePose:
    def __init__(self):
        self.shapes = []

    def __call__(self, img):
        self.shapes.append(img.shape)
        f = round(float(img[..., 2].mean()) / 20)
        return np.full((1, 133, 2), float(f)), np.full((1, 133), 3.0)


def _run_worker(monkeypatch, tmp_path, frames, rcfg=None, realtime=False, t0=None):
    fake = FakePose()
    monkeypatch.setattr(sources, "build_model", lambda rc, dev: (fake, {"pose_model": ["FakeEP"]}))
    rc = {"mode": "performance", "det_every": 1, "downscale": 1.0, "model": "wholebody", "queue_depth": 8,
          "drop_behind_s": None}
    rc.update(rcfg or {})
    q, ev, go = queue.Queue(), threading.Event(), threading.Event()
    go.set()
    t0v = types.SimpleNamespace(value=t0 or time.monotonic_ns())
    sources._worker("a", _video(tmp_path), frames, rc, "cpu", 60.0, float(frames[0]), t0v, go, realtime, q, ev)
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    return fake, msgs


def test_worker_decodes_only_the_planned_frames_in_order(monkeypatch, tmp_path):
    fake, msgs = _run_worker(monkeypatch, tmp_path, [2, 3, 6])
    assert msgs[0][0] == "ready" and msgs[0][2]["providers"] == {"pose_model": ["FakeEP"]}
    fr = [m for m in msgs if m[0] == "frame"]
    assert [m[2] for m in fr] == [2, 3, 6]
    assert [int(round(m[3][0][0, 0])) for m in fr] == [2, 3, 6]          # the right image reached the model
    assert fr[0][4]["first_call"] and not fr[1][4]["first_call"]
    assert msgs[-1][0] == "done" and len(fake.shapes) == 3


def test_worker_downscale_maps_keypoints_back(monkeypatch, tmp_path):
    fake, msgs = _run_worker(monkeypatch, tmp_path, [4], {"downscale": 0.5})
    assert fake.shapes[0][:2] == (24, 32)
    f = [m for m in msgs if m[0] == "frame"][0]
    assert f[3][0][0, 0] == pytest.approx(4 / 0.5)                      # pixel coords back at full resolution


def test_worker_realtime_waits_for_capture_time_and_drops_late_frames(monkeypatch, tmp_path):
    t0 = time.monotonic_ns() - int(1e9)                                  # the "camera" started 1 s ago
    _, msgs = _run_worker(monkeypatch, tmp_path, [0, 1, 7], {"drop_behind_s": 0.5}, realtime=True, t0=t0)
    fr = {m[2]: m[4] for m in msgs if m[0] == "frame"}
    assert fr[0]["dropped"] and fr[1]["dropped"]                         # ~1 s old when reached
    _, msgs = _run_worker(monkeypatch, tmp_path, [0, 3], realtime=True, t0=time.monotonic_ns())
    fr = {m[2]: m[4] for m in msgs if m[0] == "frame"}
    assert fr[3]["wait_ns"] > 0.02e9 and not fr[3]["dropped"]          # waited ~50 ms for frame 3 to exist


def test_live_source_get_stashes_out_of_order_frames():
    s = sources.VideoRtmlibSource.__new__(sources.VideoRtmlibSource)
    q = queue.Queue()
    for f in (5, 6):
        q.put(("frame", "a", f, [f], {"t": f}))
    q.put(("done", "a", None))
    s.queues, s.stash, s.done = {"a": q}, {"a": {}}, set()
    assert s.get("a", 6) == ([6], {"t": 6})
    assert s.get("a", 5) == ([5], {"t": 5})
    dets, tim = s.get("a", 9)
    assert dets == [] and tim["dropped"]


def test_live_source_try_get_never_blocks_long():
    s = sources.VideoRtmlibSource.__new__(sources.VideoRtmlibSource)
    q = queue.Queue()
    s.queues, s.stash, s.done = {"a": q}, {"a": {}}, set()
    assert s.try_get("a", 3) is None
    q.put(("frame", "a", 4, [4], {"t": 4}))
    q.put(("frame", "a", 3, [3], {"t": 3}))
    assert s.try_get("a", 3) is None and s.try_get("a", 3) == ([3], {"t": 3})
    assert s.try_get("a", 4) == ([4], {"t": 4})
    q.put(("done", "a", None))
    assert s.try_get("a", 9) is None
    dets, tim = s.try_get("a", 9)
    assert dets == [] and tim["dropped"]


def test_live_source_serves_a_frame_shared_by_two_steps_twice():
    # 60 fps: step k needs source frames (9, 10) of a side camera, step k+1 needs (10, 11)
    s = sources.VideoRtmlibSource.__new__(sources.VideoRtmlibSource)
    q = queue.Queue()
    for f in (9, 10, 11):
        q.put(("frame", "a", f, [f], {"t": f}))
    s.queues, s.stash, s.done = {"a": q}, {"a": {}}, set()
    got = []
    for f in (9, 10, 10, 11):
        r = None
        while r is None:
            r = s.try_get("a", f)
        got.append(r[0])
    assert got == [[9], [10], [10], [11]]
    assert s.get("a", 10) == ([10], {"t": 10})
    for f in range(12, 30):
        q.put(("frame", "a", f, [f], {"t": f}))
    assert s.get("a", 29) == ([29], {"t": 29}) and min(s.stash["a"]) >= 25    # old frames pruned
