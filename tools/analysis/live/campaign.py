"""v0.8.3 lab test campaign: run a list of named live tests one after another, then pack the results to analyse
elsewhere (the lab PC has no offline builds or offline 2D; the laptop scores quality).

    python tools/live_run.py campaign run --plan lab_5090 --group main [--unity-project PATH] [--only A] [--dry-run]
    python tools/live_run.py campaign run --plan lab_5090 --group unity --unity-project PATH   # Unity in Play mode
    python tools/live_run.py campaign pack <campaign folder>                                 # the zip to copy back
    python tools/live_run.py campaign report <zip or folder> [--out <md>]                    # on the laptop

Plan: configs/live/campaign/<name>.yaml (see lab_5090.yaml): clip, machine preset, settings defaults and tests. A test
is a bench2d (the 2D step alone, as fast as possible) or a run (a whole simulated-live stream), with a settings preset
(configs/live/settings/<name>.yaml, seconds), extra --set overrides, repeats, a group (main = Unity not playing, unity =
Unity in Play mode and following each stream) and requirements (tensorrt, PyNvVideoCodec, video:<tag>) - a test whose
requirement is missing is skipped with the reason, never failed.

Folder: <runs>/campaign/<plan>/ (<runs> = the Unity project's runs/ with --unity-project, else ./runs): one folder per
test (its run / bench folder: the usual files + campaign.log + test.json), plan.yaml, machine.json, campaign.json (every
test's status, time, error). Resumable: a test with a done test.json is skipped; touch <folder>/STOP to stop after the
current test. Each test is its own process (a crash or a hang - timeout - cannot take the campaign down).
Pack: <folder>.zip without the frame files (unity_live_frames/) - every report, log, timing, sample and final doc.
Report (laptop): re-renders every run report against the local offline builds (quality vs post-processing) and
rescores every bench against the local offline 2D, then one summary table per stage (summary.md / summary.json)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import yaml

from . import config as live_config

PLAN_DIR = os.path.join(live_config.CONFIG_DIR, "campaign")
SETTINGS_DIR = os.path.join(live_config.CONFIG_DIR, "settings")
UNITY_STALE_S = 3.0
TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------------------------------- plan
def plan_path(name_or_path):
    if os.path.exists(name_or_path):
        return name_or_path
    p = os.path.join(PLAN_DIR, f"{name_or_path}.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no campaign plan {name_or_path!r} ({p})")
    return p


def _flow(v):
    return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()


def settings_sets(name) -> list:
    """configs/live/settings/<name>.yaml -> ['key=value', ...] (the page's presets are flat config keys in seconds)."""
    if not name:
        return []
    with open(os.path.join(SETTINGS_DIR, f"{name}.yaml"), "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    d.pop("_about", None)
    return [f"{k}={_flow(v)}" for k, v in d.items()]


def load_plan(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        p = yaml.safe_load(fh) or {}
    for k in ("name", "clip", "machine", "tests"):
        if k not in p:
            raise ValueError(f"{path}: plan needs '{k}'")
    dflt = p.get("defaults") or {}
    tests, seen = [], set()
    for t in p["tests"]:
        if t.get("id") in seen or not t.get("id"):
            raise ValueError(f"{path}: test ids must be set and unique ({t.get('id')!r})")
        seen.add(t["id"])
        kind = t.get("kind", "run")
        if kind not in ("run", "bench2d"):
            raise ValueError(f"{t['id']}: kind must be run or bench2d")
        base = {"kind": kind, "group": t.get("group", "main"), "stage": t.get("stage", t["id"][:1]),
                "settings": t.get("settings", dflt.get("settings")),
                "set": list(dflt.get("set") or []) + list(t.get("set") or []),
                "seconds": t.get("seconds", dflt.get("seconds") if kind == "bench2d" else None),
                "requires": list(t.get("requires") or []), "timeout_s": float(t.get("timeout_s", dflt.get("timeout_s", 1800))),
                "note": t.get("note", ""), "compare_to": t.get("compare_to")}
        if base["group"] not in ("main", "unity"):
            raise ValueError(f"{t['id']}: group must be main or unity")
        n = int(t.get("repeat", 1))
        for r in range(n):
            tid = t["id"] if n == 1 else f"{t['id']}_r{r + 1}"
            tests.append({"id": tid, "base_id": t["id"], "repeat": r + 1, **base})
    p["expanded"] = tests
    return p


# ---------------------------------------------------------------------------------------------------- checks
def has_module(name) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_requirement(req, plan, test) -> str | None:
    """None when the requirement holds, else why the test is skipped."""
    if req == "tensorrt":
        if not (has_module("tensorrt_libs") or has_module("tensorrt")):
            return "TensorRT is not installed (pip install tensorrt-cu13)"
        return trt_check()
    if req == "PyNvVideoCodec":
        return None if has_module("PyNvVideoCodec") else "PyNvVideoCodec is not installed"
    if req.startswith("video:"):
        tag = req.split(":", 1)[1]
        try:
            live_config.load(plan["clip"], preset=plan["machine"], overrides=[f"video_variant={tag}"])
        except ValueError as e:
            return f"no {tag} videos ({str(e).splitlines()[0][:160]})"
        return None
    return f"unknown requirement {req!r}"


_TRT = {}


def trt_check() -> str | None:
    """Installed is not enough: onnxruntime must really start TensorRT (else it would quietly run on the CPU). Runs
    `gpu_check.py --trt` once per campaign; None = usable, else the reason."""
    if "result" not in _TRT:
        try:
            r = subprocess.run([sys.executable, os.path.join(TOOLS, "gpu_check.py"), "--trt"], capture_output=True,
                               text=True, timeout=3600, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            ok = r.returncode == 0
            last = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith(("FAIL", "gpu-check"))]
            _TRT["result"] = None if ok else f"TensorRT not usable: {(last[0] if last else 'gpu_check --trt failed')[:200]}"
        except (OSError, subprocess.SubprocessError) as e:
            _TRT["result"] = f"TensorRT check failed: {type(e).__name__}"
    return _TRT["result"]


def unity_status(unity_dir) -> dict:
    if not unity_dir:
        return {"known": False}
    p = os.path.join(unity_dir, "runs", "unity_status.json")
    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return {"known": True, "in_play": False, "seen": False}
    age = time.time() - float(st.get("t", 0))
    return {"known": True, "seen": True, "in_play": bool(st.get("playing")) and age < UNITY_STALE_S,
            "age_s": round(age, 1), "mode": st.get("mode")}


def point_unity(unity_dir, run_dir, title):
    """runs/now_playing.json -> this run (relative to runs/, as the live server writes it)."""
    runs = os.path.join(unity_dir, "runs")
    p = os.path.join(runs, "now_playing.json")
    seq = 0
    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            seq = int(json.load(fh).get("seq", 0))
    except (OSError, ValueError):
        pass
    rel = os.path.relpath(run_dir, runs).replace("\\", "/")
    obj = {"v": 1, "seq": seq + 1, "kind": "run", "run": rel, "frames": "unity_live_frames", "title": title,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------------------------------- run
def argv_for(test, plan, test_dir, python=None) -> list:
    py = python or sys.executable
    lr = os.path.join(TOOLS, "live_run.py")
    sets = settings_sets(test["settings"]) + list(test["set"])
    if test["kind"] == "bench2d":
        argv = [py, lr, "bench2d", "--clip", plan["clip"], "--preset", plan["machine"], "--run-dir", test_dir]
        if test.get("seconds"):
            argv += ["--seconds", str(test["seconds"])]
    else:
        argv = [py, lr, "run", "--clip", plan["clip"], "--preset", plan["machine"], "--run-dir", test_dir]
    for s in sets:
        argv += ["--set", s]
    return argv


def _read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _tail_error(log_path):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return None
    errs = [ln for ln in lines if "Error" in ln or "Exception" in ln or ln.startswith("error")]
    return (errs[-1] if errs else (lines[-1] if lines else ""))[:400]


def run_test(test, plan, out_dir, unity_dir=None, python=None, log=print) -> dict:
    test_dir = os.path.join(out_dir, test["id"])
    os.makedirs(test_dir, exist_ok=True)
    rec = {"id": test["id"], "base_id": test["base_id"], "kind": test["kind"], "group": test["group"],
           "stage": test["stage"], "repeat": test["repeat"], "note": test["note"], "settings": test["settings"],
           "set": test["set"], "compare_to": test.get("compare_to"), "unity_at_start": unity_status(unity_dir),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    argv = argv_for(test, plan, test_dir, python)
    rec["argv"] = argv
    if test["group"] == "unity" and not rec["unity_at_start"].get("in_play"):
        rec.update(status="skipped", reason="Unity is not in Play mode (press Play, then run the unity group)")
        return rec
    t0 = time.time()
    stop_watch = threading.Event()
    if test["group"] == "unity" and test["kind"] == "run":
        def watch():                         # point Unity at the run as soon as its first files exist
            mf = os.path.join(test_dir, "unity_live_frames", "manifest.json")
            while not stop_watch.wait(0.2):
                if os.path.exists(mf):
                    point_unity(unity_dir, test_dir, f"{plan['name']} · {test['id']}")
                    return
        threading.Thread(target=watch, daemon=True).start()
    with open(os.path.join(test_dir, "campaign.log"), "w", encoding="utf-8") as fh:
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen(argv, cwd=os.getcwd(), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"), **kw)
        try:
            rc = proc.wait(timeout=test["timeout_s"])
            status = "done" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            _kill(proc)
            rc, status = None, "timeout"
    stop_watch.set()
    rec.update(status=status, rc=rc, seconds=round(time.time() - t0, 1), ended=time.strftime("%Y-%m-%d %H:%M:%S"),
               unity_at_end=unity_status(unity_dir))
    if status != "done":
        rec["error"] = _tail_error(os.path.join(test_dir, "campaign.log"))
    with open(os.path.join(test_dir, "test.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def _kill(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def campaign_dir(plan, unity_dir=None):
    root = os.path.join(unity_dir, "runs") if unity_dir else "runs"
    return os.path.join(root, "campaign", plan["name"])


def run_campaign(plan_name, group="main", unity_dir=None, only=None, retry_failed=False, dry_run=False, python=None,
                 log=print) -> str:
    pp = plan_path(plan_name)
    plan = load_plan(pp)
    out = campaign_dir(plan, unity_dir)
    tests = [t for t in plan["expanded"] if (group == "all" or t["group"] == group)
             and (not only or t["stage"] in only or t["base_id"] in only or t["id"] in only)]
    if dry_run:
        for t in tests:
            miss = [m for m in (missing_requirement(r, plan, t) for r in t["requires"]) if m]
            log(f"{t['id']:<28} {t['kind']:<8} {t['group']:<6} {'SKIP: ' + miss[0] if miss else ''}")
            log("    " + " ".join(argv_for(t, plan, os.path.join(out, t["id"]), python)[2:]))
        return out
    os.makedirs(out, exist_ok=True)
    shutil.copy2(pp, os.path.join(out, "plan.yaml"))
    if not os.path.exists(os.path.join(out, "machine.json")):
        from . import machine
        with open(os.path.join(out, "machine.json"), "w", encoding="utf-8") as fh:
            json.dump(machine.report(), fh, indent=1, default=str)
    summary_p = os.path.join(out, "campaign.json")
    summary = _read_json(summary_p) or {"plan": plan["name"], "tests": {}}
    us = unity_status(unity_dir)
    if group == "main" and us.get("in_play"):
        log("WARNING: Unity is in Play mode - the main group measures the PC without Unity; stop Play for clean numbers")
    log(f"campaign {plan['name']} group {group}: {len(tests)} tests -> {out}")
    t_all = time.time()
    for i, t in enumerate(tests, 1):
        if os.path.exists(os.path.join(out, "STOP")):
            log("STOP file found: stopping (delete it and run again to resume)")
            break
        prev = _read_json(os.path.join(out, t["id"], "test.json"))
        if prev and (prev.get("status") == "done" or (prev.get("status") in ("failed", "timeout") and not retry_failed)):
            log(f"[{i}/{len(tests)}] {t['id']}: {prev['status']} before - kept")
            summary["tests"][t["id"]] = prev
            continue
        miss = [m for m in (missing_requirement(r, plan, t) for r in t["requires"]) if m]
        if miss:
            rec = {"id": t["id"], "base_id": t["base_id"], "kind": t["kind"], "group": t["group"], "stage": t["stage"],
                   "status": "skipped", "reason": miss[0], "note": t["note"]}
            log(f"[{i}/{len(tests)}] {t['id']}: SKIPPED - {miss[0]}")
        else:
            log(f"[{i}/{len(tests)}] {t['id']}: {t['kind']} {t['note']}")
            rec = run_test(t, plan, out, unity_dir, python, log)
            extra = f" - {rec.get('error') or rec.get('reason')}" if rec["status"] != "done" else ""
            log(f"[{i}/{len(tests)}] {t['id']}: {rec['status']} in {rec.get('seconds', 0):.0f} s{extra}")
        summary["tests"][t["id"]] = rec
        with open(summary_p, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    log(f"group {group} finished in {(time.time() - t_all) / 60:.1f} min")
    z = pack(out, log=log)
    log(f"results zip: {z}")
    return out


# ---------------------------------------------------------------------------------------------------- pack
SKIP_DIRS = {"unity_live_frames", "frames", "__pycache__"}


def pack(out_dir, log=print) -> str:
    out_dir = os.path.abspath(out_dir.rstrip("/\\"))
    z = out_dir + ".zip"
    n = 0
    with zipfile.ZipFile(z + ".part", "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith((".tmp", ".part")):
                    continue
                p = os.path.join(root, f)
                zf.write(p, os.path.join(os.path.basename(out_dir), os.path.relpath(p, out_dir)))
                n += 1
    os.replace(z + ".part", z)
    log(f"packed {n} files, {os.path.getsize(z) / 1048576:.1f} MB")
    return z


# ---------------------------------------------------------------------------------------------------- report
def _g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def run_row(test_dir, rec) -> dict:
    r = _read_json(os.path.join(test_dir, "report.json")) or {}
    t = r.get("timing") or {}
    row = {"id": rec["id"], "base_id": rec.get("base_id"), "note": rec.get("note"), "status": rec.get("status"),
           "unity": bool(_g(rec, "unity_at_start", "in_play")),
           "fps_target": t.get("target_fps"), "fps": t.get("achieved_output_fps"), "x_realtime": t.get("realtime_factor"),
           "setup_s": t.get("setup_s"), "delay_p50_ms": _g(t, "stages", "e2e", "p50"),
           "delay_p90_ms": _g(t, "stages", "e2e", "p90"), "late_after_warmup": r.get("late_fraction_after_warmup"),
           "gpu_mean": _g(r, "gpu", "util_mean"), "cpu_mean": _g(r, "cpu", "cpu_mean"), "offsets": r.get("offsets")}
    refs = list((r.get("quality") or {}).keys())                 # the clip config's reference_clips, primary first
    nv = len(_g(r, "config", "views") or [])
    same = next((x for x in refs if nv and x.endswith(f"_{nv}cam")), None)   # the offline build on the same cameras
    full = refs[0] if refs else None                                        # the clip's full-rig offline build
    row["ref_same"], row["ref_full"] = same, full
    for ref, key in ((same, "q3"), (full, "q5")):
        q = _g(r, "quality", ref) or {} if ref else {}
        g = _g(q, "groups", "after warm-up") or {}
        ang = g.get("angle_mae_deg") or {}
        knees = [ang[k]["mean"] for k in ("left_knee_flexion", "right_knee_flexion") if isinstance(ang.get(k), dict)]
        row.update({f"{key}_mpjpe_cm": _g(g, "w_mpjpe_cm", "median"), f"{key}_pa_cm": _g(g, "pa_mpjpe_cm", "median"),
                    f"{key}_knee_deg": sum(knees) / len(knees) if knees else None,
                    f"{key}_swaps": sum(int(v.get("swap_transitions", 0)) for v in (q.get("identity") or {}).values())
                    if q.get("identity") else None})
    return row


def bench_row(test_dir, rec) -> dict:
    b = _read_json(os.path.join(test_dir, "bench.json")) or {}
    t, q = b.get("timing") or {}, b.get("quality") or {}
    parts = q.get("parts") or {}
    return {"id": rec["id"], "base_id": rec.get("base_id"), "note": rec.get("note"), "status": rec.get("status"),
            "label": b.get("label"), "steps_per_s": t.get("steps_per_s"), "x_realtime": t.get("realtime_x"),
            "frames_per_s": t.get("frames_per_s"), "decode_ms": _g(t, "decode_ms", "p50"),
            "infer_ms": _g(t, "infer_ms", "p50"), "setup_s": t.get("setup_s"), "gpu_mean": _g(b, "gpu", "util_mean"),
            "cpu_mean": _g(b, "cpu", "cpu_mean"), "body_px": _g(parts, "body", "median_px"),
            "feet_px": _g(parts, "feet", "median_px"), "body_p90_px": _g(parts, "body", "p90_px"),
            "matched": q.get("court_matched_frac") if q.get("court_matched_frac") is not None else q.get("matched_frac"),
            "providers": {v: (w or {}).get("providers") for v, w in (b.get("workers") or {}).items()}}


def _f(x, d=1, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{d}f}"


def report(src, out=None, log=print) -> str:
    """src = the packed zip or an extracted campaign folder -> re-rendered reports + summary.md / summary.json."""
    if src.endswith(".zip"):
        dest = os.path.join("runs", "campaign_in")
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(dest)
        folder = os.path.join(dest, top)
    else:
        folder = src
    summ = _read_json(os.path.join(folder, "campaign.json")) or {"tests": {}}
    from . import bench2d
    from . import report as live_report
    benches, runs = [], []
    for tid, rec in summ["tests"].items():
        td = os.path.join(folder, tid)
        if rec.get("status") != "done":
            (benches if rec.get("kind") == "bench2d" else runs).append({"id": tid, "base_id": rec.get("base_id"),
                                                                         "note": rec.get("note"), "status": rec.get("status"),
                                                                         "reason": rec.get("reason") or rec.get("error")})
            continue
        try:
            if rec["kind"] == "bench2d":
                bench2d.rescore(td, log=lambda *a: None)
                benches.append(bench_row(td, rec))
            else:
                live_report.render_run(td, log=lambda *a: None)
                runs.append(run_row(td, rec))
            log(f"  {tid}: scored")
        except Exception as e:  # noqa: BLE001 - one broken test must not stop the summary
            log(f"  {tid}: could not score ({type(e).__name__}: {e})")
            row = (bench_row if rec["kind"] == "bench2d" else run_row)(td, rec)
            row["status"] = f"done, not scored: {type(e).__name__}"
            (benches if rec["kind"] == "bench2d" else runs).append(row)
    mach = _read_json(os.path.join(folder, "machine.json")) or {}
    L = [f"# Lab campaign {summ.get('plan')} — summary", "",
         f"Machine: {(mach.get('gpus') or [{}])[0].get('name', '?')}, {mach.get('cpu', '?')}, "
         f"onnxruntime {_g(mach, 'onnxruntime', 'version')}. Quality is the difference from the offline (post-processing) "
         "builds, not true error: q3 = the offline build on the same cameras (the fair comparison), q5 = the full-rig one.",
         "", "## Stage A — 2D only (as fast as possible, vs the offline 2D)", "",
         "| test | note | steps/s | × real time | decode ms | 2D ms | body px | feet px | body p90 | players found | GPU | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in benches:
        L.append(f"| {b['id']} | {b.get('note') or ''} | {_f(b.get('steps_per_s'))} | {_f(b.get('x_realtime'), 2)} | "
                 f"{_f(b.get('decode_ms'))} | {_f(b.get('infer_ms'))} | {_f(b.get('body_px'), 2)} | {_f(b.get('feet_px'), 2)} | "
                 f"{_f(b.get('body_p90_px'), 1)} | {_f(b.get('matched'), pct=True)} | {_f(b.get('gpu_mean'), 0)}% | "
                 f"{b.get('status')}{(' — ' + str(b['reason'])) if b.get('reason') else ''} |")
    L += ["", "## Stage B — whole streams", "",
          "| test | note | Unity | fps / target | × real time | setup s | delay p50 / p90 s | late | q3 MPJPE cm | q3 PA cm | "
          "q3 knee ° | q5 MPJPE cm | swaps | GPU | CPU | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        dl = lambda k: None if r.get(k) is None else r[k] / 1000.0  # noqa: E731
        L.append(f"| {r['id']} | {r.get('note') or ''} | {'yes' if r.get('unity') else ''} | "
                 f"{_f(r.get('fps'))} / {_f(r.get('fps_target'), 0)} | {_f(r.get('x_realtime'), 2)} | {_f(r.get('setup_s'))} | "
                 f"{_f(dl('delay_p50_ms'), 2)} / {_f(dl('delay_p90_ms'), 2)} | {_f(r.get('late_after_warmup'), pct=True)} | "
                 f"{_f(r.get('q3_mpjpe_cm'), 2)} | {_f(r.get('q3_pa_cm'), 2)} | {_f(r.get('q3_knee_deg'))} | "
                 f"{_f(r.get('q5_mpjpe_cm'), 2)} | {r.get('q3_swaps') if r.get('q3_swaps') is not None else '–'} | "
                 f"{_f(r.get('gpu_mean'), 0)}% | {_f(r.get('cpu_mean'), 0)}% | "
                 f"{r.get('status')}{(' — ' + str(r['reason'])) if r.get('reason') else ''} |")
    md = "\n".join(L) + "\n"
    out = out or os.path.join(folder, "summary.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"benches": benches, "runs": runs, "machine": mach}, fh, indent=1, default=str)
    log(f"summary: {out}")
    return out
