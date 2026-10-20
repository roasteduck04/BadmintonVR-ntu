"""v0.8.1 2D bench: only the live 2D step, with the live run's own source code, cameras and frames, as fast as possible.

    python tools/live_run.py bench2d --clip <clip> --preset <machine> [--set rtmpose.engine=fast ...] [--seconds 10]
    python tools/live_run.py bench2d-compare runs/bench2d/<a> runs/bench2d/<b> ... [--out runs/bench2d/compare.md]

Frames: the run's section at its output fps; with sync.mode interp every camera except the time reference also needs
the next source frame (the interpolation partner), exactly the 2D load of a live run. Timing: the live source
(`sources.make`, pace asfast) delivers every frame; wall clock from begin() to the last frame. Quality: each frame's
detections against the offline 2D (data/pose2d/<view>_kp2d.json, full-resolution RTMPose Wholebody, detector on every
frame) - persons matched by body keypoints (Hungarian), then per-keypoint pixel error at full resolution on keypoints the
offline 2D is sure of (raw score >= score_thr), per body part.

Folder runs/bench2d/<date>_<time>_<host>[_tag]/: bench.json, bench.md, config.resolved.yaml, machine.json,
gpu_samples.csv, cpu_samples.csv, bench.log, dets.npz (v0.8.3: every frame's detections, quarter-pixel int16, so a
machine without the offline 2D - the lab PC - can be scored later with `live_run.py bench2d-rescore <folder>`)."""
from __future__ import annotations

import json
import os
import socket
import time

import numpy as np

PARTS = {"body": range(0, 17), "feet": range(17, 23), "face": range(23, 91), "hands": range(91, 133)}
SCORE_THR = 3.0          # raw SimCC score; ~0.7 calibrated probability (pose_v2.norm_score)
MATCH_PX = 150.0         # mean body-keypoint distance (full-res px) above which two persons are not the same


def bench_plan(cfg, seconds=None):
    """{view: frames}, origins, step count: the 2D load a live run asks for (runner.plan_for), for the section or its
    first `seconds`. offsets auto (unknown until the warm-up ends): every source frame of the section +- the offset
    search range; known offsets (file / config / zero): each step's frame (+ the interpolation partner)."""
    import math
    ref = list(cfg["ref_frames"])
    if seconds:
        ref = ref[:max(1, int(round(float(seconds) * cfg["fps"])))]
    n = {v: int(cfg["source"][v]["frame_count"]) for v in cfg["views"]}
    if cfg["offsets_source"] == "auto":
        m = int(math.ceil(float(cfg["sync"]["estimate"]["max_offset_s"]) * cfg["source_fps"])) + 1
        lo, hi = ref[0] - m, ref[-1] + m + 1
        plan = {v: list(range(max(0, lo), min(n[v], hi + 1))) for v in cfg["views"]}
    else:
        interp = cfg["sync"]["mode"] == "interp"
        plan = {}
        for v in cfg["views"]:
            off = float(cfg["offsets"].get(v, 0.0))
            fr = set()
            for r in ref:
                x = r + off
                fr.add(int(math.floor(x)) if interp else int(round(x)))
                if interp and v != cfg["time_ref"]:
                    fr.add(int(math.floor(x)) + 1)
            plan[v] = sorted(f for f in fr if 0 <= f < n[v])
    return plan, {v: float(ref[0]) + float(cfg["offsets"].get(v, 0.0)) for v in cfg["views"]}, len(ref)


def settings_label(cfg):
    r = cfg["rtmpose"]
    eng = r.get("engine", "rtmlib")
    parts = [f"{eng}", f"{r['model']}-{r['mode']}", f"det/{r['det_every']}", f"sync-{cfg['offsets_source']}"]
    if eng == "fast":
        parts += [r.get("decoder", "cv2"), f"short{r.get('decode_short_side') or 'full'}",
                  "1proc" if r.get("batch_views") else "proc/cam"]
        if r.get("court_filter_m") is not None:
            parts.append(f"court{r['court_filter_m']}")
        if r.get("ep", "cuda") == "trt":
            parts.append("trt-fp16" if r.get("trt_fp16", True) else "trt-fp32")
    if cfg.get("video_variant"):
        parts.append(f"video-{cfg['video_variant']}")
    elif float(r.get("downscale", 1.0)) != 1.0:
        parts.append(f"downscale{r['downscale']}")
    return " ".join(parts)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def _stats(vals):
    a = np.asarray([x for x in vals if x is not None], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": _pct(a, 50), "p90": _pct(a, 90), "max": float(a.max())}


def _persons(raw):
    return np.asarray([np.asarray(d["keypoints"], np.float32) for d in raw], np.float32).reshape(-1, 133, 3)


def _on_court(person, view, court, score_thr):
    """Offline person standing on the court (ankle midpoint on the floor within the court filter's limits)."""
    ank = [person[j, :2] for j in (15, 16) if person[j, 2] >= score_thr]
    if not ank:
        return False
    u, v = np.mean(ank, axis=0)
    return court.on_court(view, float(u), float(v))


def match_quality(results, kp2d, score_thr=SCORE_THR, match_px=MATCH_PX, court=None):
    """results {(view, frame): [ (133,3) ]}; kp2d {view: offline frames list}. Pixel errors per part + match counts;
    with `court` (pose2d_fast.CourtFilter) also how many of the offline persons standing on the court were matched."""
    from scipy.optimize import linear_sum_assignment
    errs = {p: [] for p in PARTS}
    n_off = n_live = n_match = n_court = n_court_match = 0
    body = np.arange(17)
    for (v, f), dets in results.items():
        off = _persons(kp2d[v][f]) if 0 <= f < len(kp2d[v]) else np.zeros((0, 133, 3), np.float32)
        live = np.asarray(dets, np.float32).reshape(-1, 133, 3)
        n_off += len(off)
        n_live += len(live)
        oc = [court is not None and _on_court(o, v, court, score_thr) for o in off]
        n_court += sum(oc)
        if not len(off) or not len(live):
            continue
        cost = np.full((len(off), len(live)), 1e9)
        for i, o in enumerate(off):
            okb = o[body, 2] >= score_thr
            if okb.sum() < 3:
                continue
            for j, lv in enumerate(live):
                d = np.linalg.norm(o[body, :2][okb] - lv[body, :2][okb], axis=1)
                cost[i, j] = float(d.mean())
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] > match_px:
                continue
            n_match += 1
            n_court_match += int(oc[i])
            o, lv = off[i], live[j]
            for p, idx in PARTS.items():
                idx = np.asarray(list(idx))
                ok = (o[idx, 2] >= score_thr) & (lv[idx, 2] > 0)
                if ok.any():
                    errs[p].extend(np.linalg.norm(o[idx, :2][ok] - lv[idx, :2][ok], axis=1).tolist())
    parts = {}
    for p, e in errs.items():
        a = np.asarray(e, float)
        parts[p] = ({"n": int(len(a)), "mean_px": float(a.mean()), "median_px": float(np.median(a)),
                     "p90_px": float(np.percentile(a, 90)), "within_5px": float((a <= 5).mean()),
                     "within_10px": float((a <= 10).mean())} if len(a) else {"n": 0})
    return {"offline_persons": n_off, "live_persons": n_live, "matched": n_match,
            "court_persons": n_court if court is not None else None,
            "court_matched_frac": (n_court_match / n_court if n_court else None) if court is not None else None,
            "matched_frac": n_match / n_off if n_off else None, "extra_live": n_live - n_match,
            "score_thr": score_thr, "match_px": match_px, "parts": parts}


def save_dets(path, results):
    """{(view, frame): [(133, 3) arrays]} -> npz: per key "<view>|<frame>" an int16 (n, 133, 3) array of x, y in quarter
    pixels and the raw score x 1000 (half the size of float32, 0.25 px steps)."""
    arrs = {}
    for (v, f), dets in results.items():
        a = np.zeros((len(dets), 133, 3), np.int16)
        for i, d in enumerate(dets):
            d = np.nan_to_num(np.asarray(d, np.float32))
            a[i, :, 0:2] = np.clip(np.round(d[:, 0:2] * 4), -32768, 32767)
            a[i, :, 2] = np.clip(np.round(d[:, 2] * 1000), -32768, 32767)
        arrs[f"{v}|{int(f)}"] = a
    np.savez_compressed(path, **arrs)


def load_dets(path) -> dict:
    out = {}
    with np.load(path) as z:
        for k in z.files:
            v, f = k.rsplit("|", 1)
            a = z[k].astype(np.float32)
            a[:, :, 0:2] /= 4.0
            a[:, :, 2] /= 1000.0
            out[(v, int(f))] = [x for x in a]
    return out


def score(cfg, results):
    """Quality of results against the offline 2D of cfg's views, or None when this machine has no offline 2D."""
    missing = [v for v in cfg["views"] if not os.path.exists(cfg["kp2d_paths"][v])]
    if missing:
        return None
    kp2d = {}
    for v in cfg["views"]:
        with open(cfg["kp2d_paths"][v], "r", encoding="utf-8") as fh:
            kp2d[v] = json.load(fh)["frames"]
    from .pose2d_fast import CourtFilter, load_cams
    court = CourtFilter(load_cams(cfg["cam_paths"]), cfg["rtmpose"].get("court_filter_m") or 1.0)
    return match_quality(results, kp2d, court=court)


def rescore(out_dir, log=print) -> dict:
    """Score a finished bench folder (dets.npz + config.resolved.yaml) against this machine's offline 2D; rewrites
    bench.json / bench.md."""
    import yaml
    with open(os.path.join(out_dir, "bench.json"), "r", encoding="utf-8") as fh:
        b = json.load(fh)
    with open(os.path.join(out_dir, "config.resolved.yaml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dp = os.path.join(out_dir, "dets.npz")
    if not os.path.exists(dp):
        raise FileNotFoundError(f"{dp} (bench made before v0.8.3 or without detections)")
    q = score(cfg, load_dets(dp))
    if q is None:
        raise FileNotFoundError(f"no offline 2D here for {cfg['views']} ({list(cfg['kp2d_paths'].values())[:1]}...)")
    b["quality"], b["quality_scored_on"] = q, socket.gethostname()
    with open(os.path.join(out_dir, "bench.json"), "w", encoding="utf-8") as fh:
        json.dump(b, fh, indent=1, default=str)
    with open(os.path.join(out_dir, "bench.md"), "w", encoding="utf-8") as fh:
        fh.write(render(b))
    log(summary_line(b))
    return b


def run(cfg, out_dir, seconds=None, log=print, quality=True):
    from . import config as live_config
    from . import machine, samplers, sources
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(cfg)
    cfg["pace"] = "asfast"
    live_config.dump(cfg, os.path.join(out_dir, "config.resolved.yaml"))
    m = machine.report()
    with open(os.path.join(out_dir, "machine.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, default=str)
    plan, origins, n_steps = bench_plan(cfg, seconds)
    n_frames = sum(len(x) for x in plan.values())
    label = settings_label(cfg)
    log(f"bench2d: {label}; {len(cfg['views'])} cameras, {n_steps} steps at {cfg['fps']:g} fps, {n_frames} 2D frames "
        f"(offsets {cfg['offsets_source']})")
    results, tims = {}, {}
    with samplers.GpuSampler(os.path.join(out_dir, "gpu_samples.csv")), \
            samplers.CpuSampler(os.path.join(out_dir, "cpu_samples.csv")):
        src = sources.make(cfg, log=log)
        t = time.perf_counter()
        src.start(plan, origins)
        setup_s = time.perf_counter() - t
        log(f"  source ready in {setup_s:.1f} s")
        order = sorted(((f - origins[v]), v, f) for v in cfg["views"] for f in plan[v])
        t0 = time.monotonic_ns()
        src.begin(t0)
        t_first = None
        try:
            for _, v, f in order:
                dets, tim = src.get(v, f)
                results[(v, f)] = [np.asarray(d, np.float32) for d in dets]
                tims[(v, f)] = tim
                if t_first is None:
                    t_first = (time.monotonic_ns() - t0) / 1e9
        finally:
            wall = (time.monotonic_ns() - t0) / 1e9
            src.close()
    info = getattr(src, "info", {})
    first_calls = [k for k, tm in tims.items() if tm.get("first_call")]
    steady = {k: tm for k, tm in tims.items() if k not in first_calls}
    ms = lambda key: [tm.get(key) / 1e6 for tm in steady.values() if tm.get(key) is not None]  # noqa: E731
    per_view = {}
    for v in cfg["views"]:
        vt = [tm for (vv, _), tm in steady.items() if vv == v]
        per_view[v] = {k: _stats([tm.get(k + "_ns", 0) / 1e6 for tm in vt]) for k in ("decode", "infer", "wait")}
        per_view[v]["persons"] = _stats([tm.get("persons") for tm in vt if tm.get("persons") is not None]) or \
            _stats([len(results[(vv, f)]) for (vv, f) in steady if vv == v])
    timing = {
        "wall_s": wall, "setup_s": setup_s, "first_frame_s": t_first, "steps": n_steps, "frames": n_frames,
        "steps_per_s": n_steps / wall if wall > 0 else None, "ms_per_step": 1000 * wall / n_steps,
        "frames_per_s": n_frames / wall if wall > 0 else None,
        "realtime_x": (n_steps / wall) / cfg["fps"] if wall > 0 else None,
        "decode_ms": _stats(ms("decode_ns")), "infer_ms": _stats(ms("infer_ns")), "wait_ms": _stats(ms("wait_ns")),
        "det_ms": _stats(ms("det_ns")), "pose_ms": _stats(ms("pose_ns")), "prep_ms": _stats(ms("prep_ns")),
        "batch_crops": _stats([tm.get("batch_crops") for tm in steady.values()]),
        "detector_runs_frac": (sum(1 for tm in steady.values() if tm.get("detected")) / len(steady)
                               if steady and any("detected" in tm for tm in steady.values()) else None),
        "dropped": sum(1 for tm in tims.values() if tm.get("dropped")), "per_view": per_view,
    }
    gpu = samplers.summarise_gpu(os.path.join(out_dir, "gpu_samples.csv"))
    cpu = samplers.summarise_cpu(os.path.join(out_dir, "cpu_samples.csv"))
    save_dets(os.path.join(out_dir, "dets.npz"), results)
    q = score(cfg, results) if quality else None
    if quality and q is None:
        log("  no offline 2D on this machine: quality left out (bench2d-rescore <folder> where data/pose2d is)")
    out = {"label": label, "rtmpose": cfg["rtmpose"], "device": cfg["device"], "views": cfg["views"],
           "video_variant": cfg.get("video_variant"), "video_to_calib": cfg.get("video_to_calib"),
           "fps": cfg["fps"], "source_fps": cfg["source_fps"], "section_s": cfg["section_s"], "seconds": seconds,
           "timing": timing, "gpu": gpu, "cpu": cpu, "quality": q, "workers": info, "machine": machine.short(m),
           "host": socket.gethostname()}
    with open(os.path.join(out_dir, "bench.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    md = render(out)
    with open(os.path.join(out_dir, "bench.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    log(summary_line(out))
    return out


def _f(x, fmt="{:.1f}"):
    return "–" if x is None else fmt.format(x)


def _p(d, key="p50"):
    return None if not d else d.get(key)


def summary_line(b):
    t, q, g = b["timing"], b.get("quality") or {}, b.get("gpu") or {}
    body = (q.get("parts") or {}).get("body") or {}
    return (f"  {b['label']}: {_f(t['steps_per_s'], '{:.2f}')} steps/s ({_f(t['ms_per_step'], '{:.0f}')} ms/step, "
            f"{_f(t['realtime_x'], '{:.3f}')}x real time), model p50 {_f(_p(t['infer_ms']), '{:.0f}')} ms, decode p50 "
            f"{_f(_p(t['decode_ms']), '{:.0f}')} ms, GPU {_f(g.get('util_mean'), '{:.0f}')} %, body error median "
            f"{_f(body.get('median_px'))} px")


def render(b):
    t, q, g, c = b["timing"], b.get("quality"), b.get("gpu") or {}, b.get("cpu") or {}
    L = [f"# 2D bench — {b['label']}", "", f"Machine: {b['machine']}", "",
         f"{len(b['views'])} cameras ({', '.join(b['views'])}), {t['steps']} steps at {b['fps']:g} fps "
         f"({t['frames']} 2D frames), section {b['section_s']}, device {b['device']}.", "",
         "## Speed", "", "| | value |", "|---|---|",
         f"| output steps per second (all cameras) | **{_f(t['steps_per_s'], '{:.2f}')}** |",
         f"| × real time | {_f(t['realtime_x'], '{:.3f}')} |",
         f"| ms per step | {_f(t['ms_per_step'], '{:.0f}')} |",
         f"| 2D frames per second (all cameras) | {_f(t['frames_per_s'], '{:.2f}')} |",
         f"| wall / setup / first frame (s) | {_f(t['wall_s'])} / {_f(t['setup_s'])} / {_f(t['first_frame_s'])} |",
         f"| dropped frames | {t['dropped']} |", "",
         "| stage (ms per frame, steady state) | p50 | p90 | mean | max |", "|---|---|---|---|---|"]
    for k, name in (("decode_ms", "decode (+ shrink)"), ("wait_ms", "waited for the decoder"),
                    ("infer_ms", "model total (detect + crops + pose)"), ("det_ms", "detector (or box from last pose)"),
                    ("prep_ms", "detect + crop, all frames of the batch"), ("pose_ms", "pose model, whole batch"),
                    ("batch_crops", "crops per pose call (count)")):
        s = t.get(k)
        if s:
            L.append(f"| {name} | {_f(s['p50'])} | {_f(s['p90'])} | {_f(s['mean'])} | {_f(s['max'])} |")
    if t.get("detector_runs_frac") is not None:
        L.append(f"\nDetector ran on {100 * t['detector_runs_frac']:.0f} % of frames.")
    L += ["", "| camera | decode p50 | model p50 | model p90 | persons p50 |", "|---|---|---|---|---|"]
    for v, pv in t["per_view"].items():
        L.append(f"| {v} | {_f(_p(pv['decode']))} | {_f(_p(pv['infer']))} | {_f(_p(pv['infer'], 'p90'))} | "
                 f"{_f(_p(pv['persons']), '{:.0f}')} |")
    L += ["", "## GPU / CPU", "",
          f"- GPU busy mean {_f(g.get('util_mean'), '{:.0f}')} % (max {_f(g.get('util_max'), '{:.0f}')} %, "
          f"> 20 % busy {_f(100 * g['util_busy_frac'] if g.get('util_busy_frac') is not None else None, '{:.0f}')} % "
          f"of samples), memory {_f(g.get('mem_used_min_mb'), '{:.0f}')} → {_f(g.get('mem_used_max_mb'), '{:.0f}')} MiB",
          f"- CPU mean {_f(c.get('cpu_mean'), '{:.0f}')} % (max {_f(c.get('cpu_max'), '{:.0f}')} %), "
          f"RSS max {_f(c.get('rss_max_mb'), '{:.0f}')} MB", ""]
    if q:
        L += ["## Quality vs the offline 2D", "",
              f"Persons: offline {q['offline_persons']}, live {q['live_persons']}, matched {q['matched']} "
              f"({_f(100 * q['matched_frac'] if q['matched_frac'] is not None else None, '{:.1f}')} % of offline), "
              f"extra live {q['extra_live']}. Offline persons standing on the court: {q.get('court_persons')}, matched "
              f"{_f(100 * q['court_matched_frac'] if q.get('court_matched_frac') is not None else None, '{:.1f}')} %. "
              f"Keypoints with offline raw score ≥ {q['score_thr']}.", "",
              "| part | keypoints | median px | mean px | p90 px | ≤ 5 px | ≤ 10 px |", "|---|---|---|---|---|---|---|"]
        for p, s in q["parts"].items():
            if s.get("n"):
                L.append(f"| {p} | {s['n']} | {_f(s['median_px'])} | {_f(s['mean_px'])} | {_f(s['p90_px'])} | "
                         f"{100 * s['within_5px']:.0f} % | {100 * s['within_10px']:.0f} % |")
            else:
                L.append(f"| {p} | 0 | – | – | – | – | – |")
    L += ["", "## Workers", ""] + [f"- {v}: {json.dumps(i, default=str)}" for v, i in (b.get("workers") or {}).items()]
    return "\n".join(L) + "\n"


def compare(dirs, out=None):
    rows = []
    for d in dirs:
        p = os.path.join(d, "bench.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                rows.append((os.path.basename(os.path.normpath(d)), json.load(fh)))
    base = rows[0][1]["timing"]["steps_per_s"] if rows else None
    L = ["# 2D bench comparison", "",
         "| run | settings | steps/s | speed-up | ms/step | model p50 ms | decode p50 ms | GPU % | CPU % | "
         "body median px | body ≤5 px | feet median px | hands median px | matched % | on-court matched % | extra |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in rows:
        t, q, g, c = b["timing"], b.get("quality") or {}, b.get("gpu") or {}, b.get("cpu") or {}
        pr = q.get("parts") or {}
        bd, ft, hd = pr.get("body") or {}, pr.get("feet") or {}, pr.get("hands") or {}
        L.append(f"| {name} | {b['label']} | {_f(t['steps_per_s'], '{:.2f}')} | "
                 f"{_f(t['steps_per_s'] / base if base else None, '{:.1f}')}× | {_f(t['ms_per_step'], '{:.0f}')} | "
                 f"{_f(_p(t['infer_ms']), '{:.0f}')} | {_f(_p(t['decode_ms']), '{:.0f}')} | "
                 f"{_f(g.get('util_mean'), '{:.0f}')} | {_f(c.get('cpu_mean'), '{:.0f}')} | {_f(bd.get('median_px'))} | "
                 f"{_f(100 * bd['within_5px'] if bd.get('within_5px') is not None else None, '{:.0f}')} % | "
                 f"{_f(ft.get('median_px'))} | {_f(hd.get('median_px'))} | "
                 f"{_f(100 * q['matched_frac'] if q.get('matched_frac') is not None else None, '{:.1f}')} | "
                 f"{_f(100 * q['court_matched_frac'] if q.get('court_matched_frac') is not None else None, '{:.1f}')} | "
                 f"{q.get('extra_live', '–')} |")
    md = "\n".join(L) + "\n"
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(md)
    return md
