"""Reports of live runs.

render_run(run_dir)          -> <run>/report.md + report.json + plots/ : machine, config, timing (per stage, latency,
                                achieved fps, windows, GPU / CPU), quality vs every reference build that exists
                                (whole-court, root-aligned and Procrustes MPJPE, pelvis, angles, per joint, identity
                                swaps; split by frame quality), reference-free checks (reprojection per view, jitter,
                                window-seam jumps, bone-length stability), sync, files.
render_compare(run_dirs, out) -> <out>/compare.md + compare.json + plots/ : one row per run, plus the offline
                                post-processing time of the same section (analysis.live.offline_timing)."""
from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from analysis import skeleton24 as sk  # noqa: E402
from analysis.live import framewriter, samplers, timing  # noqa: E402
from analysis.pose24 import build  # noqa: E402
from analysis.pose24 import doc as docmod  # noqa: E402
from analysis.pose24 import eval as ev  # noqa: E402

QN = {0: "fitted", 1: "warmup", 2: "late/fallback", 3: "none"}
GROUPS = [("after warm-up", (0, 2)), ("fitted", (0,)), ("late/fallback", (2,)), ("warmup", (1,)), ("all", (0, 1, 2))]
HEADLINE = "after warm-up"          # what a live viewer sees once the body model is fixed: fitted or fallback
COLORS = {0: "#4C78A8", 1: "#9D9D9D", 2: "#E45756", 3: "#000000"}


# ---------------------------------------------------------------------------------------------------- loading
def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_run(run_dir) -> dict:
    with open(os.path.join(run_dir, "config.resolved.yaml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["ref_frames"] = list(range(cfg["ref_frames"]["first"], cfg["ref_frames"]["last"] + 1, int(cfg["stride"])))
    r = {"dir": run_dir, "cfg": cfg, "machine": _json(os.path.join(run_dir, "machine.json")),
         "summary": _json(os.path.join(run_dir, "run_summary.json")), "events": timing.load(run_dir)}
    r["rows"] = timing.step_rows(r["events"])
    fd = r["summary"].get("final_doc")
    r["doc"] = _json(os.path.join(run_dir, fd)) if fd and os.path.exists(os.path.join(run_dir, fd)) else None
    cmd = os.path.join(run_dir, "command.json")
    r["command"] = _json(cmd) if os.path.exists(cmd) else None
    r["frames"] = [_json(f) for f in framewriter.frame_files(run_dir)]
    return r


def reference_docs(cfg) -> dict:
    out = {}
    for clip in cfg.get("reference_clips") or []:
        p = build.output_path(cfg["paths"]["analysis_dir"], clip)
        out[clip] = _json(p) if os.path.exists(p) else None
    return out


# ---------------------------------------------------------------------------------------------------- quality
def _names(a, b):
    return [n for n in sk.SMPL28_NAMES if n in a["skeleton"]["joint_names"] and n in b["skeleton"]["joint_names"]]


def _cm(x):
    return None if x is None else {k: (v * 100 if k != "n" else v) for k, v in x.items()}


def _st(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "n": int(len(x))} if len(x) else None


def identity(live, ref, pid_map, frames):
    """Frames where a live player sits nearer the other reference player than its matched one."""
    lp = {p["id"]: p for p in live["players"]}
    rp = {p["id"]: p for p in ref["players"]}
    res = {}
    for lid, rid in pid_map.items():
        others = [o for o in rp if o != rid]
        wrong, both, trans, prev = 0, 0, 0, None
        for f in frames:
            a = lp[lid]["frames"][f].get("position")
            b = rp[rid]["frames"][f].get("position") if f < len(rp[rid]["frames"]) else None
            if a is None or b is None:
                prev = None
                continue
            d_own = np.hypot(a[0] - b[0], a[1] - b[1])
            d_oth = min((np.hypot(a[0] - rp[o]["frames"][f]["position"][0], a[1] - rp[o]["frames"][f]["position"][1])
                         for o in others if rp[o]["frames"][f].get("position")), default=np.inf)
            bad = d_oth < d_own
            both += 1
            wrong += bad
            trans += prev is not None and bad != prev
            prev = bad
        res[lid] = {"ref": rid, "frames": both, "swapped_frames": wrong, "swap_transitions": trans}
    return res


def quality_vs(live, ref) -> dict:
    names = _names(live, ref)
    pid_map = docmod.match_players(ref, live)
    frames = live["meta"]["live"]["ref_frames"]
    out = {"pid_map": pid_map, "players": {}, "groups": {}, "per_joint_mm": {}, "per_step": {}}
    acc = {g: [] for g, _ in GROUPS}
    jacc = []
    for lid, rid in pid_map.items():
        A = ev.named_world(live, lid, names)[frames]
        B = ev.named_world(ref, rid, names)[frames]
        q = np.asarray(live["meta"]["live"]["quality"][lid])
        r = ev._compare_arrays(A, B, names)
        out["per_step"][lid] = {"w_mm": [None if not np.isfinite(x) else float(x * 1000) for x in r["w"]],
                                "quality": q.tolist()}
        for g, qs in GROUPS:
            m = np.isin(q, qs)
            if m.any():
                acc[g].append((lid, ev._compare_arrays(A, B, names, mask=m)))
        with np.errstate(invalid="ignore"):
            jacc.append(np.linalg.norm(A - B, axis=-1)[np.isin(q, (0, 2))] * 1000)
    for g, lst in acc.items():
        if not lst:
            continue
        cat = lambda k: np.concatenate([x[k] for _, x in lst]) if lst else np.array([])
        ang = {}
        for _, x in lst:
            for k, v in x["ang"].items():
                ang.setdefault(k, []).append(v)
        out["groups"][g] = {"n": int(sum(len(x["w"]) for _, x in lst)), "w_mpjpe_cm": _cm(_st(cat("w"))),
                            "root_mpjpe_cm": _cm(_st(cat("root"))), "pa_mpjpe_cm": _cm(_st(cat("pa"))),
                            "pelvis_xz_cm": _cm(_st(cat("pel"))), "pelvis_h_cm": _cm(_st(cat("pel_h"))),
                            "angle_mae_deg": {k: _st(np.concatenate(v)) for k, v in ang.items()}}
    if jacc:
        J = np.concatenate(jacc)
        out["per_joint_mm"] = {n: _st(J[:, j]) for j, n in enumerate(names)}
    out["identity"] = identity(live, ref, pid_map, frames)
    out["coverage"] = {lid: float(np.mean(np.asarray(live["meta"]["live"]["quality"][lid]) != 3)) for lid in pid_map}
    return out


def _world(doc, pid, frames, names):
    return ev.named_world(doc, pid, names)[frames]


def jitter(doc, pid, frames, names, seams=()):
    """Mean |second difference| of pelvis-relative joints (mm per step^2); seam vs non-seam step jumps (mm)."""
    X = _world(doc, pid, frames, names)
    pel = names.index("pelvis")
    R = X - X[:, pel:pel + 1]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        acc = np.nanmean(np.linalg.norm(R[2:] - 2 * R[1:-1] + R[:-2], axis=-1), axis=1) * 1000
        step = np.nanmean(np.linalg.norm(X[1:] - X[:-1], axis=-1), axis=1) * 1000
    seam = np.zeros(len(step), bool)
    for s in seams:
        if 1 <= s <= len(step):
            seam[s - 1] = True
    return {"accel_mm": _st(acc), "seam_jump_mm": _st(step[seam]), "other_jump_mm": _st(step[~seam])}


def bone_std(doc, pid, frames):
    names, parents = doc["skeleton"]["joint_names"], doc["skeleton"]["parents"]
    X = _world(doc, pid, frames, names)
    out = []
    for j, p in enumerate(parents):
        if p is None or p < 0:
            continue
        L = np.linalg.norm(X[:, j] - X[:, p], axis=-1) * 1000
        L = L[np.isfinite(L)]
        if len(L) > 5:
            out.append(float(L.std()))
    return float(np.mean(out)) if out else None


def reprojection(frames) -> dict:
    per = {}
    for fr in frames:
        for p in fr["players"]:
            if p.get("quality") not in (0, 2):
                continue
            for v, e in (p.get("reproj_px") or {}).items():
                if e is not None:
                    per.setdefault(v, []).append(e)
    return {v: _st(x) for v, x in per.items()}


def seedless_lines(evs, cfg) -> list:
    """How the run started without seeds: player acquisitions / drops, ankle plane, late body models, candidates."""
    tr = [e for e in evs if e.get("ev") == "tracker"]
    ank = next((e for e in evs if e.get("ev") == "ankle"), None)
    late = [e for e in evs if e.get("ev") == "late_model"]
    warm = int(cfg["warmup_steps"])
    nc = [e.get("n_cand") for e in evs if e.get("ev") == "step" and e.get("node") == "pose3d" and e.get("n_cand") is not None]
    L = ["## Seedless start", "", "P1 = near half, P2 = far half (no seeds). Candidates must be seen by >= "
         f"{cfg['warmup']['min_support_views']} cameras with a standing height in {cfg['warmup']['player_height_m']} m.", ""]
    for e in tr:
        if e["kind"] == "acquire":
            L.append(f"- step {e['step']}: **{e['pid']} acquired** at ({e['floor'][0]:+.2f}, {e['floor'][1]:+.2f}) m, "
                     f"{e['support']} views, height {_f(e['height_m'], 2)} m, {e['candidates']} candidates on court")
        else:
            L.append(f"- step {e['step']}: **{e['pid']} dropped** ({e['reason']}; height {_f(e['height_m'], 2)} m, "
                     f"{e['support']} views)")
    if not tr:
        L.append("- no acquisition recorded")
    if ank:
        L.append(f"- ankle plane {ank['ankle_h_m']:.3f} m from {ank['samples']} warm-up samples"
                 + (" (too few: default kept)" if ank["used_default"] else f" (default {ank['default_m']:.3f} m)"))
    else:
        L.append(f"- ankle plane fixed by config: {cfg.get('ankle_h_m')}")
    for e in late:
        L.append(f"- {e['pid']}: no warm-up body model, built late at step {e['step']}")
    if nc:
        a = np.asarray(nc[:warm], float)
        b = np.asarray(nc[warm:], float)
        L.append(f"- court candidates per step: warm-up median {_f(np.median(a) if len(a) else None, 0)}, "
                 f"after median {_f(np.median(b) if len(b) else None, 0)}, max {int(max(nc))}")
    return L + [""]


# ---------------------------------------------------------------------------------------------------- plots
def _plot_steps(rows, path):
    s = [r["step"] for r in rows if r["e2e"] is not None]
    e = [r["e2e"] for r in rows if r["e2e"] is not None]
    c = [COLORS.get(r["quality"], "#000") for r in rows if r["e2e"] is not None]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.scatter(s, e, c=c, s=6)
    ax.set_xlabel("step")
    ax.set_ylabel("capture → written (ms)")
    ax.set_title("End-to-end latency per step (blue fitted, red late/fallback, grey warm-up)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_error(qual, path, title):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for lid, d in qual["per_step"].items():
        ax.plot([np.nan if x is None else x for x in d["w_mm"]], lw=1, label=lid)
    ax.set_xlabel("step")
    ax.set_ylabel("mean joint error (mm)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_csv(path_csv, cols, path, title):
    import csv
    if not os.path.exists(path_csv):
        return False
    with open(path_csv, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    if len(rows) < 2:
        return False
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for col, lab in cols:
        y = []
        for r in rows:
            try:
                y.append(float(str(r.get(col, "")).strip()))
            except ValueError:
                y.append(np.nan)
        ax.plot(y, lw=1, label=lab)
    ax.set_xlabel("sample")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------------- text
def _f(x, nd=1):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _pct_row(label, p):
    if not p:
        return f"| {label} | – | – | – | – | – |"
    return f"| {label} | {_f(p['p50'])} | {_f(p['p90'])} | {_f(p['p99'])} | {_f(p['max'])} | {_f(p['mean'])} |"


def _machine_lines(m, src_info):
    g = m.get("gpus") or []
    lines = [f"- **Host** {m['host']} ({m['os']}{', WSL' if m.get('wsl') else ''}), Python {m['python']}",
             f"- **CPU** {m.get('cpu')} — {m.get('cpu_physical') or '?'} cores / {m.get('cpu_logical')} threads, "
             f"**RAM** {m.get('ram_gb')} GB",
             f"- **GPU** " + (", ".join(f"{x['name']} ({x['memory_mb']} MB, driver {x['driver']}, CUDA {x.get('cuda')})"
                                         for x in g) if g else "none (no nvidia-smi)"),
             f"- **onnxruntime** {m['onnxruntime'].get('version')} providers {m['onnxruntime'].get('providers')}",
             f"- **packages** " + ", ".join(f"{k} {v}" for k, v in sorted(m.get("packages", {}).items())),
             f"- **git** {m['git'].get('branch')} @ {str(m['git'].get('commit'))[:10]}"
             + (" (uncommitted changes)" if m["git"].get("dirty") else "")]
    if src_info:
        for v, i in src_info.items():
            if isinstance(i, dict) and "providers" in i:
                lines.append(f"- 2D worker {v}: model load {i['model_load_ns'] / 1e9:.1f} s, sessions {i['providers']}")
    return lines


def _quality_lines(name, q):
    L = [f"### vs `{name}`", "",
         f"Players matched: {q['pid_map']}; coverage " + ", ".join(f"{k} {v:.0%}" for k, v in q["coverage"].items()), "",
         "| frames | n | MPJPE (cm) | root-aligned (cm) | PA-MPJPE (cm) | pelvis XZ (cm) | pelvis height (cm) | "
         "knee flex MAE (°) | elbow flex MAE (°) |", "|---|---|---|---|---|---|---|---|---|"]
    for g, _ in GROUPS:
        r = q["groups"].get(g)
        if not r:
            continue
        a = r["angle_mae_deg"]
        knee = [a[k]["mean"] for k in ("left_knee_flexion", "right_knee_flexion") if a.get(k)]
        elb = [a[k]["mean"] for k in ("left_elbow_flexion", "right_elbow_flexion") if a.get(k)]
        med = lambda x: _f((x or {}).get("median"))
        L.append(f"| {g} | {r['n']} | {med(r['w_mpjpe_cm'])} | {med(r['root_mpjpe_cm'])} | {med(r['pa_mpjpe_cm'])} | "
                 f"{med(r['pelvis_xz_cm'])} | {med(r['pelvis_h_cm'])} | {_f(np.mean(knee) if knee else None)} | "
                 f"{_f(np.mean(elb) if elb else None)} |")
    L += ["", "(medians over frames; MPJPE = whole-court joint error, root-aligned = pelvis subtracted, "
          "PA = after the best similarity transform)", ""]
    allg = q["groups"].get("all")
    if allg:
        L += ["All joint angles (all frames, MAE ° / p90 °): " + ", ".join(
            f"{k} {_f(v['mean'])}/{_f(v['p90'])}" for k, v in sorted(allg["angle_mae_deg"].items()) if v), ""]
    if q["per_joint_mm"]:
        L += ["<details><summary>Per-joint error (fitted + late frames, mm: median / p90)</summary>", "",
              "| joint | median | p90 |", "|---|---|---|"]
        L += [f"| {n} | {_f(s['median'], 0)} | {_f(s['p90'], 0)} |" for n, s in q["per_joint_mm"].items() if s]
        L += ["", "</details>", ""]
    L += ["Identity: " + "; ".join(f"{k}→{v['ref']}: {v['swapped_frames']}/{v['frames']} frames nearer the other "
                                    f"player, {v['swap_transitions']} switches" for k, v in q["identity"].items()), ""]
    return L


def render_run(run_dir, log=print) -> str:
    R = load_run(run_dir)
    cfg, S, rows, evs = R["cfg"], R["summary"], R["rows"], R["events"]
    pdir = os.path.join(run_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    T = timing.summary(rows, evs, cfg)
    gpu = samplers.summarise_gpu(os.path.join(run_dir, "gpu_samples.csv"))
    cpu = samplers.summarise_cpu(os.path.join(run_dir, "cpu_samples.csv"))
    refs = reference_docs(cfg)
    qual = {}
    if R["doc"] is not None:
        for name, d in refs.items():
            if d is not None:
                qual[name] = quality_vs(R["doc"], d)
    seams = sorted({w["start"] for w in timing.windows(evs)})
    selfq = {}
    if R["doc"] is not None:
        frames = R["doc"]["meta"]["live"]["ref_frames"]
        names = [n for n in sk.SMPL28_NAMES if n in R["doc"]["skeleton"]["joint_names"]]
        for p in R["doc"]["players"]:
            selfq[p["id"]] = {"jitter": jitter(R["doc"], p["id"], frames, names, seams),
                              "bone_std_mm": bone_std(R["doc"], p["id"], frames)}
        prim = next((n for n in cfg.get("reference_clips") or [] if refs.get(n)), None)
        if prim:
            pm = qual[prim]["pid_map"]
            for lid, rid in pm.items():
                selfq[lid]["reference_jitter"] = jitter(refs[prim], rid, frames, names, seams)
                selfq[lid]["reference_bone_std_mm"] = bone_std(refs[prim], rid, frames)
    rep = reprojection(R["frames"])
    _plot_steps(rows, os.path.join(pdir, "latency_per_step.png"))
    for name, q in qual.items():
        _plot_error(q, os.path.join(pdir, f"error_vs_{name}.png"), f"Mean joint error per step vs {name}")
    has_gpu = _plot_csv(os.path.join(run_dir, "gpu_samples.csv"), [("utilization_gpu", "GPU %"),
                        ("utilization_memory", "memory bus %")], os.path.join(pdir, "gpu.png"), "GPU utilisation")
    has_cpu = _plot_csv(os.path.join(run_dir, "cpu_samples.csv"), [("cpu_percent", "CPU %")],
                        os.path.join(pdir, "cpu.png"), "System CPU")

    qc = T["quality_counts"]
    n_out = max(1, sum(qc.values()))
    post = [r for r in rows if r["quality"] in (0, 2)]
    late_frac = sum(r["quality"] == 2 for r in post) / len(post) if post else None
    sync = S.get("sync")
    prim = next((n for n in cfg.get("reference_clips") or [] if n in qual), None)
    L = [f"# Live run report — {os.path.basename(os.path.normpath(run_dir))}", "",
         f"_{time.strftime('%Y-%m-%d %H:%M')}; clip `{cfg['clip']}` on {len(cfg['views'])} cameras "
         f"({', '.join(cfg['views'])}), section {cfg['section_s'][0]}–{cfg['section_s'][1]} s, "
         f"{cfg['fps']:g} fps output, backend **{cfg['backend']}**, pace **{cfg['pace']}**_", "",
         "## Summary", "",
         f"- **Speed:** {T['written']}/{T['steps']} frames in {_f(T['wall_s'])} s for {_f(T['section_s'])} s of video "
         f"→ {_f(T['achieved_output_fps'])} fps achieved vs {T['target_fps']:g} target "
         f"({_f(T['realtime_factor'], 2)}× real time).",
         f"- **Latency** (capture → frame file): median {_f((T['stages']['e2e'] or {}).get('p50'), 0)} ms, "
         f"p90 {_f((T['stages']['e2e'] or {}).get('p90'), 0)} ms; first frame after {_f(T['first_frame_s'], 2)} s.",
         f"- **Frames:** " + ", ".join(f"{QN.get(k, k)} {v} ({v / n_out:.0%})" for k, v in sorted(qc.items()))
         + (f"; after warm-up {late_frac:.0%} late/fallback." if late_frac is not None else ""),
         ]
    if prim:
        g = qual[prim]["groups"].get(HEADLINE) or qual[prim]["groups"].get("all")
        if g:
            L.append(f"- **Quality vs `{prim}`** ({HEADLINE if HEADLINE in qual[prim]['groups'] else 'all'} frames): "
                     f"MPJPE {_f(g['w_mpjpe_cm']['median'])} cm, PA-MPJPE {_f(g['pa_mpjpe_cm']['median'])} cm, "
                     f"pelvis {_f(g['pelvis_xz_cm']['median'])} cm.")
    if gpu and gpu.get("samples"):
        L.append(f"- **GPU:** mean {gpu['util_mean']:.0f} %, max {gpu['util_max']:.0f} %, busy (>20 %) "
                 f"{gpu['util_busy_frac']:.0%} of samples, memory up to {_f(gpu['mem_used_max_mb'], 0)} MB.")
    if cpu and cpu.get("samples"):
        L.append(f"- **CPU:** mean {cpu['cpu_mean']:.0f} %, max {cpu['cpu_max']:.0f} %, this run's RSS up to "
                 f"{_f(cpu['rss_max_mb'], 0)} MB.")
    L += ["", "## Machine", ""] + _machine_lines(R["machine"], S.get("source_info")) + [""]
    L += ["## Settings", "",
          f"- sync **{cfg['sync']['mode']}**, offsets **{cfg['offsets_source']}** → "
          + ", ".join(f"{v} {o:+.2f}" for v, o in S["offsets"].items()) + " source frames",
          f"- warm-up {cfg['warmup']['seconds']} s ({cfg['warmup_steps']} steps); window {cfg['window']} / overlap "
          f"{cfg['overlap']} / history {cfg['history']} steps; fit budget {cfg['fit_budget_s']:.3f} s; "
          f"{cfg['workers']} fit workers",
          f"- 2D: {cfg['rtmpose']['model']} {cfg['rtmpose']['mode']}, detector every {cfg['rtmpose']['det_every']}, "
          f"downscale {cfg['rtmpose']['downscale']}, device {cfg['device']}, engine {cfg['rtmpose'].get('engine', 'rtmlib')}"
          + (f", decoder {cfg['rtmpose'].get('decoder')}, short side {cfg['rtmpose'].get('decode_short_side') or 'full'}, "
             f"{'one process for all cameras' if cfg['rtmpose'].get('batch_views') else 'one process per camera'}, "
             f"pose batch {cfg['rtmpose'].get('pose_batch')}"
             if cfg['rtmpose'].get('engine') == 'fast' else ""), ""]
    if sync:
        L += ["**Auto sync** (after warm-up): " + ", ".join(
            f"{v} {sync['offsets'][v]:+.2f} fr ({_f(sync['errors_px'][v])} px)" for v in sync["offsets"])
            + (f"; failed: {sync['failed_views']}" if sync["failed_views"] else ""), ""]
    L += seedless_lines(evs, cfg)
    L += ["## Timing", "", "| stage (ms per frame) | p50 | p90 | p99 | max | mean |", "|---|---|---|---|---|---|"]
    L += [_pct_row(lab, T["stages"][k]) for k, lab in timing.STAGES]
    L += ["", f"Setup {_f(T['setup_s'])} s (source ready {_f(T['source_ready_s'])} s), loop {_f(T['loop_s'])} s, "
          f"first 2D call {_f(T['first_2d_call_ms'], 0)} ms, dropped 2D frames {T['dropped_2d_frames']}.", "",
          f"Windows: {T['windows']}; status per player " + ", ".join(f"{p}: {s}" for p, s in T["window_status"].items())
          + "; solve ms " + ", ".join(f"{p} p50 {_f(s['p50'], 0)} / p90 {_f(s['p90'], 0)}"
                                      for p, s in T["solve_ms"].items() if s), "",
          f"Frame files: {S['writer']['frames']} × {_f(S['writer']['bytes_per_frame'], 0)} B; write "
          f"{_f((S['writer'].get('write_ms') or {}).get('median'), 2)} ms median.", "",
          "![latency](plots/latency_per_step.png)", ""]
    if has_gpu:
        L += ["![gpu](plots/gpu.png)", ""]
    if has_cpu:
        L += ["![cpu](plots/cpu.png)", ""]
    L += ["## Quality", ""]
    if R["doc"] is None:
        L += ["No final doc (output.final_doc off)."]
    for name, d in refs.items():
        if d is None:
            L += [f"Reference `{name}` not found in {cfg['paths']['analysis_dir']}.", ""]
            continue
        L += _quality_lines(name, qual[name]) + [f"![error](plots/error_vs_{name}.png)", ""]
    L += ["## Reference-free checks", "", "| player | joint accel (mm/step²) | ref accel | seam jump (mm) | "
          "other step (mm) | bone length sd (mm) | ref bone sd |", "|---|---|---|---|---|---|---|"]
    for pid, s in selfq.items():
        j, rj = s["jitter"], s.get("reference_jitter") or {}
        md = lambda x: _f((x or {}).get("median"))
        L.append(f"| {pid} | {md(j['accel_mm'])} | {md(rj.get('accel_mm'))} | {md(j['seam_jump_mm'])} | "
                 f"{md(j['other_jump_mm'])} | {_f(s['bone_std_mm'])} | {_f(s.get('reference_bone_std_mm'))} |")
    L += ["", "Reprojection error of fitted/late frames (px, median / p90): " + ", ".join(
        f"{v} {_f(s['median'])}/{_f(s['p90'])}" for v, s in rep.items() if s), ""]
    L += ["## Files", "", "- `unity_live_frames/manifest.json`, `unity_live_frames/frame_*.json`, `unity_live_frames/latest.json` — per-frame output",
          f"- `{S.get('final_doc')}` — final analysis file (Unity)",
          "- `timing.jsonl`, `gpu_samples.csv`, `cpu_samples.csv`, `run.log`, `run_summary.json`, `machine.json`",
          "- `config.resolved.yaml` — every parameter of this run", "",
          "<details><summary>Resolved config</summary>", "", "```yaml",
          open(os.path.join(run_dir, "config.resolved.yaml"), encoding="utf-8").read().rstrip(), "```", "</details>", ""]
    if R["command"]:
        L.insert(3, f"`python {' '.join(R['command']['argv'])}`")
        L.insert(4, "")
    md_path = os.path.join(run_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(docmod.json_safe({"timing": T, "gpu": gpu, "cpu": cpu, "quality": {k: {kk: vv for kk, vv in v.items()
                                    if kk != "per_step"} for k, v in qual.items()}, "self": selfq, "reprojection": rep,
                                    "sync": sync, "offsets": S["offsets"], "late_fraction_after_warmup": late_frac,
                                    "machine": R["machine"], "config": {k: cfg[k] for k in (
                                        "clip", "views", "section_s", "fps", "backend", "pace", "offsets_source")}}),
                  fh, indent=1)
    return md_path


# ---------------------------------------------------------------------------------------------------- compare
def _offline_timing(cfg, out_dir, log):
    from analysis.live import offline_timing
    ref = next((c for c in cfg.get("reference_clips") or [] if c != cfg["clip"]), None)
    if ref is None:
        return None
    a, b = cfg["ref_frames"][0], cfg["ref_frames"][-1] + int(cfg["stride"])
    cache = os.path.join(out_dir, f"offline_timing_{ref}_{a}_{b}.json")
    if os.path.exists(cache):
        return _json(cache)
    try:
        res = offline_timing.run(ref, (a, b), log=log)
    except (OSError, KeyError, ValueError) as e:
        log(f"  offline timing skipped: {type(e).__name__}: {e}")
        return None
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def render_compare(run_dirs, out_dir, log=print) -> str:
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    rows = []
    for rd in run_dirs:
        if not os.path.exists(os.path.join(rd, "report.json")):
            render_run(rd, log=log)
        rj = _json(os.path.join(rd, "report.json"))
        cfg = rj["config"]
        T = rj["timing"]
        prim = next(iter(rj["quality"]), None)
        g = (rj["quality"][prim]["groups"].get(HEADLINE) or rj["quality"][prim]["groups"].get("all")) if prim else None
        knee = None
        if g:
            ks = [g["angle_mae_deg"][k]["mean"] for k in ("left_knee_flexion", "right_knee_flexion")
                  if g["angle_mae_deg"].get(k)]
            knee = float(np.mean(ks)) if ks else None
        rep = [s["median"] for s in (rj.get("reprojection") or {}).values() if s]
        rows.append({"run": os.path.basename(os.path.normpath(rd)), "host": rj["machine"]["host"],
                     "gpu": (rj["machine"].get("gpus") or [{"name": "-"}])[0]["name"], "backend": cfg["backend"],
                     "fps": cfg["fps"], "pace": cfg["pace"], "offsets": cfg["offsets_source"],
                     "achieved_fps": T["achieved_output_fps"], "realtime_factor": T["realtime_factor"],
                     "e2e_p50": (T["stages"]["e2e"] or {}).get("p50"), "e2e_p90": (T["stages"]["e2e"] or {}).get("p90"),
                     "late_after_warmup": rj.get("late_fraction_after_warmup"),
                     "mpjpe_cm": g["w_mpjpe_cm"]["median"] if g else None,
                     "pa_cm": g["pa_mpjpe_cm"]["median"] if g else None, "knee_mae": knee,
                     "reproj_px": float(np.median(rep)) if rep else None,
                     "gpu_mean": (rj.get("gpu") or {}).get("util_mean"), "ref": prim})
    first_cfg = load_run(run_dirs[0])["cfg"]
    off = _offline_timing(first_cfg, out_dir, log)
    L = ["# Live runs compared", "", f"_{time.strftime('%Y-%m-%d %H:%M')}; quality vs `{rows[0]['ref']}` "
         f"({HEADLINE} frames = fitted + fallback, medians)_", "",
         "| run | machine | backend | fps | pace | offsets | achieved fps | × real time | latency p50/p90 (ms) | "
         "late after warm-up | MPJPE (cm) | PA (cm) | knee MAE (°) | reproj (px) | GPU % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        late = "–" if r["late_after_warmup"] is None else f"{r['late_after_warmup']:.0%}"
        L.append(f"| {r['run']} | {r['host']} / {r['gpu']} | {r['backend']} | {r['fps']:g} | {r['pace']} | "
                 f"{r['offsets']} | {_f(r['achieved_fps'])} | {_f(r['realtime_factor'], 2)} | "
                 f"{_f(r['e2e_p50'], 0)}/{_f(r['e2e_p90'], 0)} | {late} | "
                 f"{_f(r['mpjpe_cm'])} | {_f(r['pa_cm'])} | {_f(r['knee_mae'])} | {_f(r['reproj_px'])} | "
                 f"{_f(r['gpu_mean'], 0)} |")
    if off:
        L += ["", f"**Offline post-processing of the same section** (`{off['clip']}`, already tracked + synced, "
              "lrswap + 2-pass fit + regress):", "",
              "| stride | frames | fps | lrswap (s) | fit (s) | regress (s) | total (s) | ms per frame |",
              "|---|---|---|---|---|---|---|---|"]
        for s, x in off["runs"].items():
            L.append(f"| {s} | {x['frames']} | {x['fps']:g} | {x['lrswap_s']:.1f} | {x['fit_s']:.1f} | "
                     f"{x['regress_s']:.1f} | {x['total_s']:.1f} | {x['per_frame_ms']:.1f} |")
        if off.get("doc_runtime_s"):
            L.append(f"\nWhole-clip offline build runtime (from the doc): {off['doc_runtime_s']}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    lab = [f"{r['backend']}\n{r['fps']:g}fps {r['pace']}" for r in rows]
    axs[0].bar(range(len(rows)), [r["mpjpe_cm"] or 0 for r in rows], color="#4C78A8")
    axs[0].set_title("MPJPE vs reference (cm)", fontsize=9)
    axs[1].bar(range(len(rows)), [r["e2e_p50"] or 0 for r in rows], color="#E45756")
    axs[1].set_title("latency p50 (ms)", fontsize=9)
    for ax in axs:
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(lab, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plots", "compare.png"), dpi=110)
    plt.close(fig)
    L += ["", "![compare](plots/compare.png)", ""]
    p = os.path.join(out_dir, "compare.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(docmod.json_safe({"runs": rows, "offline": off}), fh, indent=1)
    return p
