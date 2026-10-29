# tools/tests/test_live_realtime.py
"""v0.8.5 real-time defaults and detector: thread limits + late fits kept by default, the live_30 setting, the detector
cut before its NMS (det_split, same boxes) and the detector in its own thread (det_async)."""
import os
import threading
import time

import numpy as np
import pytest

from analysis.live import config as live_config, det_split, pose2d_fast as pf
from tests.test_live_lab_tests import _probe

onnx = pytest.importorskip("onnx")
YOLOX_M = os.path.expanduser(os.path.join("~", ".cache", "rtmlib", "hub", "checkpoints",
                                          "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"))


# ------------------------------------------------------------------------------------------------ defaults + setting
def test_defaults_limit_threads_and_keep_late_fits_while_behind():
    c = live_config.load("singles_05x", preset="rtx5090", probe=_probe())
    assert c["cpu"]["limits"] is True and c["stream"]["late_policy"] == "behind"
    assert c["rtmpose"]["det_trt"] is False and c["rtmpose"]["det_async"] is False


def test_detector_options_need_the_fast_engine():
    for k in ("det_trt", "det_async"):
        with pytest.raises(ValueError, match=k):
            live_config.load("singles_05x", preset="rtx5090", overrides=["rtmpose.engine=rtmlib", "rtmpose.ep=cuda",
                                                                         f"rtmpose.{k}=true"],
                             probe=_probe())


def test_live_30_setting_loads_and_every_key_is_a_page_field():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline_server"))
    import settings
    d = settings.load_preset("live_30")
    assert d["output_fps"] == 30 and d["sync.mode"] == "nearest" and d["cpu.limits"] is True
    assert "live_30" in settings.preset_names() and "live_30" in settings.BUILTIN
    for k in ("rtmpose.det_trt", "rtmpose.det_async"):
        assert settings.BYKEY[k]["kind"] == "bool" and settings.BYKEY[k]["live_only"]
    over = settings.overrides({"rtmpose.det_async": True, "rtmpose.det_trt": True}, "live")
    assert settings.as_set_args(over) == ["rtmpose.det_async=true", "rtmpose.det_trt=true"]
    assert settings.overrides({"rtmpose.det_async": True}, "cached") == {}      # stored 2D: no detector


# ------------------------------------------------------------------------------------------------ det_split
def _toy_end2end(path, a=6, c=3):
    """A tiny graph shaped like the mmdeploy end2end tail: boxes / scores -> TopK -> Gather -> NonMaxSuppression."""
    from onnx import TensorProto, helper
    nodes = [
        helper.make_node("Identity", ["in_boxes"], ["boxes"]),
        helper.make_node("Identity", ["in_scores"], ["scores"]),
        helper.make_node("ReduceMax", ["scores"], ["smax"], axes=[2], keepdims=0),
        helper.make_node("Squeeze", ["smax", "ax0"], ["s1"]),
        helper.make_node("TopK", ["s1", "k_pre"], ["tv", "ti"], axis=0),
        helper.make_node("Gather", ["boxes", "ti"], ["gb"], axis=1),
        helper.make_node("Gather", ["scores", "ti"], ["gs"], axis=1),
        helper.make_node("Transpose", ["gs"], ["gst"], perm=[0, 2, 1]),
        helper.make_node("NonMaxSuppression", ["gb", "gst", "maxpc", "iou", "sthr"], ["sel"]),
    ]
    inits = [helper.make_tensor("ax0", TensorProto.INT64, [1], [0]),
             helper.make_tensor("k_pre", TensorProto.INT64, [1], [a]),
             helper.make_tensor("maxpc", TensorProto.INT64, [1], [200]),
             helper.make_tensor("iou", TensorProto.FLOAT, [1], [0.5]),
             helper.make_tensor("sthr", TensorProto.FLOAT, [1], [0.05])]
    g = helper.make_graph(nodes, "toy", [helper.make_tensor_value_info("in_boxes", TensorProto.FLOAT, [1, a, 4]),
                                         helper.make_tensor_value_info("in_scores", TensorProto.FLOAT, [1, a, c])],
                          [helper.make_tensor_value_info("sel", TensorProto.INT64, ["n", 3])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 7
    onnx.save(m, path)


def test_cut_nms_keeps_the_boxes_and_scores_and_reads_the_nms_parameters(tmp_path):
    import onnxruntime as ort
    src = str(tmp_path / "toy.onnx")
    _toy_end2end(src)
    out, params = det_split.cut_nms(src, str(tmp_path / "split"))
    assert params["iou_threshold"] == pytest.approx(0.5) and params["max_per_class"] == 200
    assert params["score_threshold"] == pytest.approx(0.05) and params["keep_top_k"] == 100
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    assert [o.name for o in s.get_outputs()] == ["boxes", "scores"]
    b = np.random.default_rng(0).random((1, 6, 4), np.float32)
    sc = np.random.default_rng(1).random((1, 6, 3), np.float32)
    rb, rs = s.run(None, {"in_boxes": b, "in_scores": sc})
    assert np.array_equal(rb, b) and np.array_equal(rs, sc)
    assert det_split.cut_nms(src, str(tmp_path / "split"))[0] == out          # cached


def test_nms_boxes_per_class_threshold_and_order():
    p = {"iou_threshold": 0.5, "max_per_class": 200, "score_threshold": 0.01, "keep_top_k": 100}
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10], [80, 80, 90, 90]], np.float32)
    scores = np.zeros((5, 2), np.float32)
    scores[0, 0], scores[1, 0], scores[2, 0] = 0.9, 0.8, 0.6     # box 1 overlaps box 0 (IoU 0.68): suppressed
    scores[3, 1] = 0.7                                            # same place, other class: kept (per-class NMS)
    scores[4, 0] = 0.2                                            # below 0.3: dropped
    out = det_split.nms_boxes(boxes, scores, p)
    assert out.tolist() == [[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]]   # best first: 0.9, 0.7, 0.6
    assert len(det_split.nms_boxes(boxes, scores, dict(p, keep_top_k=2))) == 2
    assert det_split.nms_boxes(boxes, np.zeros((5, 2), np.float32), p).shape == (0, 4)


@pytest.mark.needs_data
@pytest.mark.slow
def test_split_detector_gives_the_end2end_boxes(tmp_path):
    if not os.path.exists(YOLOX_M):
        pytest.skip("rtmlib YOLOX-m not downloaded")
    import onnxruntime as ort
    from rtmlib.tools.object_detection.yolox import YOLOX
    y = YOLOX(YOLOX_M, model_input_size=(640, 640), backend="onnxruntime", device="cpu")
    out, params = det_split.cut_nms(YOLOX_M, str(tmp_path))
    sd = det_split.SplitDetector(y, ort.InferenceSession(out, providers=["CPUExecutionProvider"]), params)
    rng = np.random.default_rng(3)
    img = (rng.random((540, 960, 3)) * 40 + 100).astype(np.uint8)
    for x0 in (100, 400, 700):                                    # three dark upright "people"
        img[150:450, x0:x0 + 90] = 30
        img[110:160, x0 + 25:x0 + 65] = 60
    a, b = np.asarray(y(img), np.float32).reshape(-1, 4), sd(img)
    assert a.shape == b.shape
    assert np.allclose(np.sort(a, 0), np.sort(b, 0), atol=1e-3)


# ------------------------------------------------------------------------------------------------ det_async
def test_merge_boxes_keeps_confirmed_tracks_adds_new_people_drops_unconfirmed():
    tracked = [np.array([0, 0, 100, 200], np.float32), np.array([500, 0, 600, 200], np.float32)]
    det = [np.array([10, 5, 105, 205], np.float32), np.array([900, 0, 980, 180], np.float32)]
    out = pf.merge_boxes(det, tracked)
    assert len(out) == 2
    assert out[0].tolist() == [0, 0, 100, 200]                  # the tracked (newer) box, confirmed by a detection
    assert out[1].tolist() == [900, 0, 980, 180]                # a new person
    assert pf.merge_boxes([], tracked) == []


def _engine(det_s=0.15, det_every=5, det_async=True):
    e = object.__new__(pf.FastPose)
    calls = []

    def det(img):
        calls.append(threading.current_thread().name)
        time.sleep(det_s)
        return [np.array([0, 0, 100, 200], np.float32), np.array([300, 0, 400, 200], np.float32)]
    e.det, e.det_every, e.det_async, e._det_pool, e.async_merges = det, det_every, det_async, None, 0
    e.court, e.geom, e.state, e.filtered = None, {}, {}, 0
    from rtmlib.tools.solution.pose_tracker import pose_to_bbox
    e._p2b = pose_to_bbox
    return e, calls


def test_async_detector_does_not_block_the_frames_that_have_tracked_boxes():
    e, calls = _engine()
    img = np.zeros((10, 10, 3), np.uint8)
    boxes, dt, detected = e.boxes("v", img)                     # nothing tracked yet: it must detect here
    assert len(boxes) == 2 and detected and dt >= 0.1e9 and calls == [threading.current_thread().name]
    e.finish("v", [np.array([[10, 10], [90, 190]], np.float32)])
    t = time.monotonic()
    detected_at = []
    for i in range(1, 30):
        _, dt, det_flag = e.boxes("v", img)
        if det_flag:
            detected_at.append(i)
            assert dt >= 0.1e9                                  # the detector's own time is reported on merge
        e.finish("v", [np.array([[10, 10], [90, 190]], np.float32)])
        time.sleep(0.02)
    wall = time.monotonic() - t
    assert wall < 29 * 0.02 + 0.15                              # frames never waited for a 150 ms detection
    assert len(calls) >= 3 and all(n.startswith("detector") for n in calls[1:])
    assert detected_at and e.async_merges == len(detected_at)


def test_sync_detector_unchanged_without_det_async():
    e, calls = _engine(det_s=0.0, det_async=False)
    img = np.zeros((10, 10, 3), np.uint8)
    flags = []
    for _ in range(11):
        _, _, d = e.boxes("v", img)
        flags.append(d)
        e.finish("v", [np.array([[10, 10], [90, 190]], np.float32)])
    assert flags == [True, False, False, False, False, True, False, False, False, False, True]
