"""v0.8.5 split detector: the rtmlib YOLOX person detector without its built-in box filtering (NMS), so TensorRT can build
it (rtmpose.det_trt).

The exported detector (mmdeploy end2end ONNX) ends in TopK -> NonMaxSuppression -> TopK. TensorRT refuses its first
TopK (k = 5000 > TensorRT's 3840 limit), so in v0.8.3 the detector stayed on CUDA, where it costs ~65 ms per call on the
lab RTX 5090 against ~13 ms for the batched pose model (lab round 2). cut_nms() keeps the graph up to the decoded boxes
(1, A, 4; x1 y1 x2 y2 in model-input pixels) and the class scores (1, A, C) that feed the NMS, and reads the NMS
parameters from the graph; nms_boxes() does the same filtering in NumPy:

- per class, boxes scoring above KEEP_SCORE (0.3: rtmlib keeps only those of the end2end output), greedy NMS that
  suppresses IoU > the graph's iou_threshold (0.65), at most max_output_boxes_per_class (200) per class;
- all classes together, the keep_top_k (100) best, highest score first.

The graph's own pre-NMS score threshold (0.01) and pre-top-k (5000 of 8400 anchors) cannot change which boxes above 0.3
survive (greedy NMS never lets a lower score suppress a higher one), so the result equals the end2end model's boxes.
Needs the `onnx` package (cutting only; the cut model is cached next to the TensorRT engines)."""
from __future__ import annotations

import json
import os

import numpy as np

KEEP_SCORE = 0.3          # rtmlib YOLOX.postprocess keeps end2end boxes scoring above this
SPLIT_VERSION = 1


def _const_values(graph):
    from onnx import numpy_helper
    vals = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    vals[n.output[0]] = numpy_helper.to_array(a.t)
    return vals


def find_nms(model):
    """-> (boxes tensor, scores tensor, params) of an end2end detector's NonMaxSuppression; ValueError if absent."""
    g = model.graph
    nms = [n for n in g.node if n.op_type == "NonMaxSuppression"]
    if len(nms) != 1:
        raise ValueError(f"expected one NonMaxSuppression node, found {len(nms)}")
    nms = nms[0]
    prod = {o: n for n in g.node for o in n.output}
    consts = _const_values(g)

    def gather_data(t):                     # NMS inputs are the top-k Gather of the full tensors
        n = prod.get(t)
        while n is not None and n.op_type in ("Transpose", "Unsqueeze", "Squeeze", "Reshape"):
            n = prod.get(n.input[0])
        if n is None or n.op_type != "Gather":
            raise ValueError(f"NMS input {t} is not a Gather of the full tensor")
        return n.input[0]

    def const(i, default):
        if len(nms.input) <= i or not nms.input[i]:
            return default
        v = consts.get(nms.input[i])
        return default if v is None else float(np.asarray(v).ravel()[0])

    params = {"iou_threshold": const(3, 0.0), "max_per_class": int(const(2, 200)),
              "score_threshold": const(4, 0.0), "keep_top_k": 100}
    tk = [n for n in g.node if n.op_type == "TopK"]
    if len(tk) >= 2:                         # the last TopK caps the result (mmdeploy keep_top_k)
        k = consts.get(tk[-1].input[1])
        if k is None:                        # k = Reshape(Where(Less(keep_top_k, count), keep_top_k, count))
            w = prod.get(tk[-1].input[1])
            while w is not None and w.op_type == "Reshape":
                w = prod.get(w.input[0])
            if w is not None and w.op_type == "Where":
                cands = [consts[i] for i in w.input[1:] if i in consts and np.asarray(consts[i]).size == 1]
                k = cands[0] if cands else None
        if k is not None and np.asarray(k).size == 1 and int(np.asarray(k).ravel()[0]) > 0:
            params["keep_top_k"] = int(np.asarray(k).ravel()[0])
    return gather_data(nms.input[0]), gather_data(nms.input[1]), params


def cut_nms(onnx_in, out_dir):
    """Cut the detector before its NMS (once; cached) -> (cut model path, params)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(str(onnx_in)))[0]
    out = os.path.join(out_dir, f"{base}.nonms.onnx")
    meta = out + ".json"
    if os.path.exists(out) and os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("version") == SPLIT_VERSION and m.get("source_size") == os.path.getsize(onnx_in):
            return out, m["params"]
    import onnx
    import onnx.utils
    model = onnx.load(str(onnx_in))
    boxes_t, scores_t, params = find_nms(model)
    inp = [i.name for i in model.graph.input]
    onnx.utils.extract_model(str(onnx_in), out, inp, [boxes_t, scores_t])
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"version": SPLIT_VERSION, "source": os.path.basename(str(onnx_in)),
                   "source_size": os.path.getsize(onnx_in), "outputs": [boxes_t, scores_t], "params": params}, fh, indent=1)
    return out, params


def _nms(boxes, scores, iou_thr):
    """Greedy NMS (suppress IoU > iou_thr), highest score first -> kept indices."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, np.int64)


def nms_boxes(boxes, scores, params, keep_score=KEEP_SCORE):
    """boxes (A, 4), class scores (A, C) -> (N, 4) boxes the end2end detector keeps above keep_score, best first."""
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    thr = max(float(keep_score), float(params.get("score_threshold", 0.0)))
    ai, ci = np.nonzero(scores > thr)
    if ai.size == 0:
        return np.zeros((0, 4), np.float32)
    kb, ks = [], []
    for c in np.unique(ci):
        sel = ai[ci == c]
        s = scores[sel, c]
        k = _nms(boxes[sel], s, float(params.get("iou_threshold", 0.65)))[:int(params.get("max_per_class", 200))]
        kb.append(boxes[sel][k])
        ks.append(s[k])
    b, s = np.concatenate(kb), np.concatenate(ks)
    order = np.argsort(-s, kind="stable")[:int(params.get("keep_top_k", 100))]
    return b[order]


class SplitDetector:
    """Drop-in for rtmlib's YOLOX object (same __call__(image) -> (N, 4) boxes in image pixels): rtmlib's
    preprocessing, the cut model in `session` (TensorRT or any onnxruntime session), NMS in NumPy."""

    def __init__(self, yolox, session, params):
        self.yolox, self.session, self.params = yolox, session, params
        self.model_input_size = yolox.model_input_size
        self.onnx_model = getattr(session, "_model_path", None) or yolox.onnx_model
        self.in_name = session.get_inputs()[0].name
        self.out_names = [o.name for o in session.get_outputs()]

    def __call__(self, image):
        img, ratio = self.yolox.preprocess(image)
        x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
        boxes, scores = self.session.run(self.out_names, {self.in_name: x})
        return nms_boxes(boxes[0], scores[0], self.params) / ratio
