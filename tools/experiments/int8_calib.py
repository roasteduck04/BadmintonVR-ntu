"""Round-5 experiment (test branch only): build an INT8 copy of the live pose model for TensorRT and measure what
INT8 costs in 2D accuracy before any stream runs.

    python tools/experiments/int8_calib.py [--clip singles_05x] [--mode performance] [--frames 40] [--method minmax]

1. Person crops exactly as the live fast engine cuts them (frames shrunk to 1080, rtmlib's detector, box x1.25, affine
   to the model input, mean / std), from frames spread over the clip's videos on its live cameras.
2. onnxruntime static quantization in QDQ format (Conv layers only by default: quantizing the attention-style head
   gave 150 px errors), symmetric INT8 (TensorRT needs zero-point 0), per-channel weights,
   calibrated on 3/4 of the crops -> ~/.cache/badmintonvr/int8/<pose model>.int8qdq.onnx (where rtmpose.exp_int8 finds
   it, see hooks.py).
3. The other 1/4: FP32 vs INT8 keypoints on the CPU (pixels in the model input) for pose24's 35 keypoints, written to
   <model>.int8qdq.json and printed. TensorRT's INT8 kernels round a little differently; the stream test is the judge.
Needs `onnx` (tools/requirements-pc.txt)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import hooks  # noqa: E402


def crops_from_clip(clip, mode, frames_per_view, device, short_side=1080, max_persons=4):
    import cv2
    from rtmlib import Wholebody
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
    from analysis.live import config as live_config
    cfg = live_config.load(clip, overrides=["backend=live", "rtmpose.engine=fast", "video_variant=null"])
    sol = Wholebody(mode=mode, backend="onnxruntime", device=device)
    w_in, h_in = (int(x) for x in sol.pose_model.model_input_size)
    mean = np.asarray(sol.pose_model.mean, np.float32).reshape(1, 1, 3)
    std = np.asarray(sol.pose_model.std, np.float32).reshape(1, 1, 3)
    out = []
    for v in cfg["views"]:
        cap = cv2.VideoCapture(cfg["videos"][v])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for f in np.linspace(0.05 * n, 0.95 * n, frames_per_view).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            if short_side and min(h, w) > short_side:
                s = short_side / min(h, w)
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            found = sol.det_model(img)
            boxes = sorted([np.asarray(b, np.float32) for b in (found if found is not None else [])],
                           key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:max_persons]
            for b in boxes:
                c, sc = bbox_xyxy2cs(b[:4], padding=1.25)
                crop, _ = top_down_affine((w_in, h_in), sc, c, img)
                x = ((crop.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
                out.append(np.ascontiguousarray(x, np.float32))
        cap.release()
    return sol.pose_model.onnx_model, out


def keypoints(sess, xs):
    from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
    name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    k = []
    for x in xs:
        sx, sy = sess.run(outs, {name: x})
        locs, _ = get_simcc_maximum(sx, sy)
        k.append(locs[0] / 2.0)
    return np.asarray(k, np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", default="singles_05x")
    ap.add_argument("--mode", default="performance", help="rtmlib wholebody mode of the live pose model")
    ap.add_argument("--frames", type=int, default=40, help="frames per camera (up to 4 persons each)")
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--device", default="cpu", help="device for the detector while collecting crops (cpu | cuda)")
    ap.add_argument("--out", help="output model (default: where rtmpose.exp_int8 looks)")
    ap.add_argument("--ops", default="Conv", help="op types to quantize (comma list; default Conv = the backbone, the "
                                                  "attention-style head stays in float: quantizing it broke the model)")
    a = ap.parse_args(argv)

    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    import onnxruntime as ort
    from analysis.pose24 import kpset

    t0 = time.time()
    model, xs = crops_from_clip(a.clip, a.mode, a.frames, a.device)
    if len(xs) < 8:
        raise SystemExit(f"only {len(xs)} person crops found - check the clip's videos")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(xs))
    n_cal = max(4, int(len(xs) * 0.75))
    cal, test = [xs[i] for i in idx[:n_cal]], [xs[i] for i in idx[n_cal:]]
    print(f"{len(xs)} crops ({len(cal)} calibration, {len(test)} check) in {time.time() - t0:.0f} s", flush=True)

    sess_in = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    in_name = sess_in.get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.it = iter(data)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {in_name: x}

    out = a.out or hooks.int8_path(model)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meth = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile}[a.method]
    t1 = time.time()
    import onnx
    from onnx import version_converter
    m = onnx.load(str(model))
    if max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) < 13:   # per-channel QDQ needs opset 13
        m = version_converter.convert_version(m, 13)
    src_model = out[:-len(".int8qdq.onnx")] + ".opset13.onnx"
    onnx.save(m, src_model)
    quantize_static(src_model, out, Reader(cal), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, calibrate_method=meth,
                    op_types_to_quantize=[x.strip() for x in a.ops.split(",") if x.strip()],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    print(f"INT8 model -> {out} ({os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t1:.0f} s)", flush=True)

    sess_q = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    k32, k8 = keypoints(sess_in, test), keypoints(sess_q, test)
    src = [s[0] for s in kpset.SOURCES if len(s) == 1]                # pose24's single-index keypoints (33 of 35)
    err = np.linalg.norm(k32[:, src] - k8[:, src], axis=-1)
    body = np.linalg.norm(k32[:, :17] - k8[:, :17], axis=-1)
    rep = {"model": os.path.basename(str(model)), "int8": out, "mode": a.mode, "method": a.method, "ops": a.ops,
           "clip": a.clip,
           "crops": len(xs), "calibration": len(cal), "check": len(test),
           "input_px_error": {"pose24_median": float(np.median(err)), "pose24_p90": float(np.percentile(err, 90)),
                              "body17_median": float(np.median(body)), "body17_p90": float(np.percentile(body, 90))}}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    e = rep["input_px_error"]
    print(f"FP32 vs INT8 on {len(test)} held-out crops (model-input pixels): pose24 keypoints median "
          f"{e['pose24_median']:.2f} / p90 {e['pose24_p90']:.2f} px, body median {e['body17_median']:.2f} px", flush=True)


if __name__ == "__main__":
    main()
