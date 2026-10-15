"""Cost table of the camera-subset study (spec § 5 "Cost"): per stored subset result, what the rig costs to run.
capture_mb = the subset's recorded video files (clips.json `videos`, repo-relative), gpu_2d_s = the 2D pose seconds of
its views from tools/cost/pose2d_gpu_timed.py (<clip>_gpu_timing.json), setup = n_cams x a given per-camera setup
time (not measured by default). CPU vs wall (R18): the study runs N single-BLAS-thread workers side by side, so a
job's wall seconds are inflated by the competition; cpu_* = time.process_time() of the job (the stored `cpu_s`:
oracle / mono = the fit job -> cpu_fit_s = cpu_total_s; e2e = the whole build + convert -> cpu_total_s only), None
when not recorded (results stored before cpu_s existed); wall_* = the stored wall seconds (oracle / mono: runtime_s;
e2e: the e2e doc's track / fit steps and runtime_s), `workers` = the competing worker count (None if unknown). A
"full" row carries the full build's own runtime_s over every recorded view (a serial single-process run: its step
seconds stand in for cpu_*, workers 1)."""
from __future__ import annotations

import csv
import json
import os

from analysis.pose24.camsubsets import report
from analysis.pose24.camsubsets import subsets as cs

COLUMNS = ("clip", "mode", "method", "subset", "n_cams", "capture_mb", "gpu_2d_s", "cpu_track_s", "cpu_fit_s",
           "cpu_total_s", "wall_track_s", "wall_fit_s", "wall_total_s", "workers", "setup_note")
MODE_ORDER = ("full", "oracle", "e2e")


def video_mb(registry, clip, data_dir=".") -> dict:
    """{view: MB (file size / 1e6)} of the clip's registry `videos`; None for a missing path or file."""
    e = registry[clip]
    vids = e.get("videos") or {}
    out = {}
    for v in e["views"]:
        p = vids.get(v)
        if p and not os.path.isabs(p):
            p = os.path.join(data_dir, p)
        out[v] = os.path.getsize(p) / 1e6 if p and os.path.isfile(p) else None
    return out


def load_timing(path, clip):
    """A <clip>_gpu_timing.json (path = the file, or a directory holding it); None when absent or for another clip."""
    if not path:
        return None
    p = os.path.join(path, f"{clip}_gpu_timing.json") if os.path.isdir(path) else path
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        t = json.load(fh)
    return t if t.get("clip") == clip else None


def _sum(vals):
    return None if not vals or any(v is None for v in vals) else round(sum(vals), 3)


def _num(v):
    return None if v is None else float(v)


def _setup(n, per_cam):
    return "not measured" if per_cam is None else f"{n * per_cam:g} min ({n} x {per_cam:g} min)"


def cost_rows(clip, registry, full_doc_runtime, subset_results, gpu_timing=None, data_dir=".",
              setup_min_per_cam=None) -> list:
    """subset_results {mode: {method: {subset key: stored result meta}}} (oracle.save_result's meta: runtime_s,
    e2e results also e2e_runtime_s); full_doc_runtime = the full doc's meta.pose24.runtime_s (None: no full row)."""
    mb = video_mb(registry, clip, data_dir)
    gpu = {v: t.get("seconds") for v, t in ((gpu_timing or {}).get("views") or {}).items()}

    def row(mode, method, views, cpu=(None, None, None), wall=(None, None, None), workers=None):
        return {"clip": clip, "mode": mode, "method": method, "subset": cs.key(views), "n_cams": len(views),
                "capture_mb": _sum([mb.get(v) for v in views]),
                "gpu_2d_s": _sum([gpu.get(v) for v in views]) if gpu_timing else None,
                **{f"cpu_{k}_s": _num(v) for k, v in zip(("track", "fit", "total"), cpu)},
                **{f"wall_{k}_s": _num(v) for k, v in zip(("track", "fit", "total"), wall)},
                "workers": None if workers is None else int(workers),
                "setup_note": _setup(len(views), setup_min_per_cam)}
    out = []
    if full_doc_runtime:
        rt = full_doc_runtime
        steps = (rt.get("track"), rt.get("fit"), rt.get("total"))
        out.append(row("full", "fit", registry[clip]["views"], cpu=steps, wall=steps, workers=1))
    for mode in sorted(subset_results, key=lambda m: (MODE_ORDER.index(m) if m in MODE_ORDER else 99, m)):
        for method in sorted(subset_results[mode]):
            res = subset_results[mode][method]
            for k in sorted(res, key=lambda k: (len(cs.parse_key(k)), k)):
                m = res[k]
                if mode == "e2e":
                    rt = m.get("e2e_runtime_s") or {}
                    out.append(row(mode, method, cs.parse_key(k), cpu=(None, None, m.get("cpu_s")),
                                   wall=(rt.get("track"), rt.get("fit"), m.get("runtime_s")), workers=m.get("workers")))
                else:
                    out.append(row(mode, method, cs.parse_key(k), cpu=(None, m.get("cpu_s"), m.get("cpu_s")),
                                   wall=(None, m.get("runtime_s"), m.get("runtime_s")), workers=m.get("workers")))
    return out


def write_csv(rows, path) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([report._cell(r.get(k)) for k in COLUMNS])
    return path


def table(rows) -> str:
    """A short fixed-width text table (the CSV's columns minus clip / setup)."""
    cols = [c for c in COLUMNS if c not in ("clip", "setup_note")]
    cells = [[report._cell(r.get(c)) or "-" for c in cols] for r in rows]
    wd = [max([len(c)] + [len(x[i]) for x in cells]) for i, c in enumerate(cols)]
    return "\n".join("  ".join(s.ljust(w) for s, w in zip(line, wd)) for line in [cols] + cells)
