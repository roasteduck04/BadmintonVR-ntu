# tools/tests/test_live_fast2d.py
"""v0.8.1 fast live 2D: frame batching, frame sizes, court filter, bench plan / quality, config checks, readers."""
import glob
import os

import numpy as np
import pytest

from analysis.live import bench2d, config as live_config, pose2d_fast as pf


def test_target_size_shrinks_the_short_side_and_never_enlarges():
    assert pf.target_size(2160, 3840, None) == (2160, 3840, 1.0)
    assert pf.target_size(2160, 3840, 4000) == (2160, 3840, 1.0)
    w, h, s = pf.target_size(2160, 3840, 1080)
    assert (w, h) == (1080, 1920) and s == pytest.approx(0.5)
    w, h, s = pf.target_size(3840, 2160, 720)
    assert h == 720 and w % 2 == 0 and s == pytest.approx(w / 3840)


def test_schedule_batches_frames_of_one_output_step_across_cameras():
    plans = {"a": [10, 12, 14], "b": [10, 11, 12, 13, 14, 15]}
    origins = {"a": 10.0, "b": 10.0}
    b = pf.schedule(["a", "b"], plans, origins, fps=60.0, window_s=2 / 60 - 1e-6)
    assert [sorted(x[1]) for x in b[:2]] == [[("a", 10), ("b", 10), ("b", 11)], [("a", 12), ("b", 12), ("b", 13)]]
    assert b[0][0] == pytest.approx(1 / 60)                 # capture time of the batch's latest frame
    one = pf.schedule(["a"], {"a": [10, 12]}, {"a": 10.0}, fps=60.0, window_s=0.0)
    assert [x[1] for x in one] == [[("a", 10)], [("a", 12)]]
    # every frame appears exactly once, in each camera's plan order
    flat = [it for _, items in b for it in items]
    assert sorted(flat) == sorted((v, f) for v in plans for f in plans[v])
    for v in plans:
        assert [f for vv, f in flat if vv == v] == plans[v]


def _cam(z_back=10.0):
    """Camera behind the near baseline looking down the court (court world Y-down, floor Y = 0)."""
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    pitch = np.deg2rad(15)
    R = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    C = np.array([0.0, -3.0, -z_back])                      # 3 m up (Y-down), behind the court
    return K, R, -R @ C


def _project(K, R, t, X):
    p = K @ (R @ X + t)
    return p[:2] / p[2]


def test_court_filter_keeps_players_on_court_and_drops_bystanders():
    K, R, t = _cam()
    cf = pf.CourtFilter({"a": (K, R, t)}, margin_m=0.5)
    on = _project(K, R, t, np.array([1.0, 0.0, 2.0]))
    off = _project(K, R, t, np.array([9.0, 0.0, 2.0]))       # 9 m to the side: a spectator
    assert pf.floor_xz(on[0], on[1], K, R, t) == pytest.approx((1.0, 2.0), abs=1e-6)
    assert cf.on_court("a", *on) and not cf.on_court("a", *off)
    boxes = [np.array([on[0] - 20, on[1] - 200, on[0] + 20, on[1]]), np.array([off[0] - 20, off[1] - 200, off[0] + 20, off[1]])]
    kept = cf.keep("a", [b * 0.5 for b in boxes], inv_scale=2.0, src_h=1080)   # boxes of a half-size frame
    assert len(kept) == 1 and kept[0][0] == pytest.approx(boxes[0][0] * 0.5)
    cut = np.array([off[0] - 20, 900, off[0] + 20, 1079])    # feet cut by the bottom edge: cannot tell, keep
    assert len(cf.keep("a", [cut], 1.0, 1080)) == 1


def test_kp_map_puts_halpe26_feet_in_wholebody_slots():
    m = pf.kp_map("body_feet")
    assert list(m[:17]) == list(range(17)) and m[24] == 19 and m[25] == 22 and m[20] == 17 and m[17] == -1
    assert len(pf.kp_map("body")) == 17 and len(pf.kp_map("wholebody")) == 133


def _person(cx, cy, s=5.0):
    k = np.zeros((133, 3), np.float32)
    k[:, 0] = cx + np.arange(133) % 7
    k[:, 1] = cy + np.arange(133) // 7
    k[:, 2] = s
    return k


def test_match_quality_matches_persons_and_measures_error_per_part():
    off = {"a": [[{"keypoints": _person(100, 100).tolist()}, {"keypoints": _person(900, 100).tolist()}]]}
    live = _person(902, 101)                                  # only the second person, 2 / 1 px off
    q = bench2d.match_quality({("a", 0): [live]}, off)
    assert q["offline_persons"] == 2 and q["live_persons"] == 1 and q["matched"] == 1 and q["extra_live"] == 0
    assert q["parts"]["body"]["median_px"] == pytest.approx(np.hypot(2, 1), abs=1e-4)
    assert q["parts"]["hands"]["within_5px"] == 1.0
    far = _person(5000, 5000)                                 # nothing close: not matched
    assert bench2d.match_quality({("a", 0): [far]}, off)["matched"] == 0


def _cfg_stub(offsets_source="file", interp=True):
    return {"ref_frames": list(range(100, 120, 2)), "fps": 30.0, "source_fps": 60.0, "views": ["r", "x"],
            "time_ref": "r", "offsets_source": offsets_source, "offsets": {"r": 0.0, "x": 0.4},
            "sync": {"mode": "interp" if interp else "nearest", "estimate": {"max_offset_s": 0.05}},
            "source": {"r": {"frame_count": 1000}, "x": {"frame_count": 1000}}}


def test_bench_plan_matches_the_live_2d_load():
    plan, origins, n = bench2d.bench_plan(_cfg_stub("file"))
    assert n == 10 and plan["r"] == list(range(100, 120, 2))
    assert plan["x"] == sorted(set(range(100, 120, 2)) | {f + 1 for f in range(100, 120, 2)})
    assert origins["x"] == pytest.approx(100.4)
    auto, _, _ = bench2d.bench_plan(_cfg_stub("auto"))        # offsets unknown: every source frame +- the search range
    assert auto["r"] == list(range(100 - 4, 118 + 4 + 2)) and auto["x"] == auto["r"]   # = runner.plan_for
    short, _, n2 = bench2d.bench_plan(_cfg_stub("file"), seconds=0.1)
    assert n2 == 3 and short["r"] == [100, 102, 104]


def test_config_rejects_fast_only_settings_on_the_rtmlib_engine():
    base = live_config.load("singles_05x", preset="laptop-cached", probe=_probe)
    assert base["rtmpose"]["engine"] == "rtmlib" and base["rtmpose"]["decoder"] == "cv2"
    with pytest.raises(ValueError, match="engine=fast"):
        live_config.load("singles_05x", overrides=["rtmpose.batch_views=true"], probe=_probe)
    with pytest.raises(ValueError, match="engine=fast"):
        live_config.load("singles_05x", overrides=["rtmpose.court_filter_m=1"], probe=_probe)
    ok = live_config.load("singles_05x", overrides=["rtmpose.engine=fast", "rtmpose.decoder=pyav",
                                                    "rtmpose.decode_short_side=720", "rtmpose.model=body_feet"],
                          probe=_probe)
    assert ok["rtmpose"]["decode_short_side"] == 720
    with pytest.raises(ValueError, match="decoder"):
        live_config.load("singles_05x", overrides=["rtmpose.engine=fast", "rtmpose.decoder=gpu"], probe=_probe)
    with pytest.raises(ValueError, match="short_side"):
        live_config.load("singles_05x", overrides=["rtmpose.engine=fast", "rtmpose.decode_short_side=10"], probe=_probe)


def test_small_model_scores_are_mapped_onto_the_wholebody_scale():
    s = np.array([0.05, 0.3, 0.73, 0.9, 1.2], np.float32)
    for key, (xs, ys) in pf.SCORE_TO_WB.items():
        assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) >= 0), key
        m = pf.score_to_wb(*key.split("/"), s)
        assert np.all(np.diff(m) >= 0) and m[2] > 2.75, key     # a typical 0.73 lands above the calibration midpoint
    assert pf.score_to_wb("wholebody", "performance", s) is s
    with pytest.raises(ValueError, match="score map"):
        pf.score_to_wb("body", "lightweight", s)
    with pytest.raises(ValueError, match="score map"):
        live_config.load("singles_05x", overrides=["rtmpose.engine=fast", "rtmpose.model=body_feet",
                                                   "rtmpose.mode=lightweight"], probe=_probe)


def _probe(c):
    return {v: {"fps": 60.0, "frame_count": 1877, "width": 2160, "height": 3840} for v in c["views"]}


VIDEOS = sorted(glob.glob(os.path.join("data", "15-08", "singles", "0.5", "*back 0.5-1.mov")))


@pytest.mark.needs_data
@pytest.mark.skipif(not VIDEOS, reason="needs the singles 0.5x videos (gitignored)")
@pytest.mark.parametrize("kind", ["pyav", "ffmpeg"])
def test_readers_return_the_same_frame_as_opencv(kind):
    pytest.importorskip("av") if kind == "pyav" else None
    if kind == "ffmpeg" and pf.ffmpeg_exe() is None:
        pytest.skip("no ffmpeg")
    path = VIDEOS[0]
    ref = pf.open_reader("cv2", path, 400, 60.0, 540)
    frames = {f: ref.read(f).astype(np.int16) for f in range(400, 406)}
    ref.close()
    other = pf.open_reader(kind, path, 401, 60.0, 540)
    try:
        for f in (401, 403, 404):
            img = other.read(f)
            assert img is not None and img.shape == (960, 540, 3)
            diffs = {g: float(np.abs(frames[g] - img.astype(np.int16)).mean()) for g in frames}
            assert min(diffs, key=diffs.get) == f, f"{kind} frame {f} looks like OpenCV frame {min(diffs, key=diffs.get)}: {diffs}"
    finally:
        other.close()


class _FakeReader:
    path = "fake.mov"

    def __init__(self):
        self.read_frames = []

    def read(self, f):
        self.read_frames.append(f)
        return np.full((4, 4, 3), f, np.uint8)


def test_prefetch_skips_frames_outside_the_narrowed_plan_without_decoding_them():
    import threading
    r, holder = _FakeReader(), {"keep": {"a": {10, 12}}}
    p = pf.Prefetch(r, [10, 11, 12, 13], lambda f: None, depth=8, stop_ev=threading.Event(), view="a", holder=holder)
    p.run()
    got = [p.q.get() for _ in range(5)]
    assert [g[0] for g in got] == [10, 11, 12, 13, "end"]
    assert got[1][1] == pf.SKIP and got[3][1] == pf.SKIP and r.read_frames == [10, 12]


def test_pose_batches_are_padded_to_a_few_fixed_sizes():
    fp = pf.FastPose.__new__(pf.FastPose)       # only the sizing logic; no models
    fp.pad, fp.batch = True, 32
    assert [fp.bucket(n) for n in (1, 2, 3, 5, 8, 9, 17, 32)] == [1, 2, 4, 8, 8, 16, 32, 32]
    fp.pad = False
    assert fp.bucket(5) == 5
