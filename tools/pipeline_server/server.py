"""BadmintonVR pipeline server (v0.8.2): one local page for the pipeline, Unity follows what it plays.

Spec: docs/superpowers/specs/2026-10-19-v0.8.2-pipeline-server-design.md. Run from the repository root:
    python tools/pipeline_server/server.py [--unity-project PATH] [--runs-dir PATH] [--machine rtx5090]
    -> http://127.0.0.1:8770
Tabs: 0 Source (v0.8.4: the clip's videos or live camera streams), 1 Calibrate, 2 Stream (every live setting; step 1
"Warm up" starts the live engine - models loaded, players warmed up - and step 2 "Start streaming to Unity" streams from
that warm state, as often as wanted, frame by frame). Unity's ServerFollower follows runs/now_playing.json
and writes a heartbeat (runs/unity_status.json) the server checks before streaming.
Loopback only unless --host 0.0.0.0 --allow-host <name> (same rule as serve_calibrator.py). The process works in
the repository root (config paths are relative to it); main() changes there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPO = TOOLS.parent
for p in (str(TOOLS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import calib as calibmod  # noqa: E402
import jobs as jobmod  # noqa: E402
import settings as st  # noqa: E402
from analysis.live import config as live_config  # noqa: E402
from analysis.live import framewriter as fw  # noqa: E402

STATIC = HERE / "static"
NOW_PLAYING = "now_playing.json"
UNITY_STATUS = "unity_status.json"
UNITY_STALE_S = 3.0              # no heartbeat for this long = Unity is not in Play mode
INPUT_FILE = "input.json"        # v0.8.4 Source tab (in the runs folder: machine-specific camera addresses)
# settings the warm-up depends on: changing one after the warm-up means warming up again
WARM_PREFIXES = ("backend", "rtmpose.", "cpu.", "warmup.", "sync.offsets", "video_variant", "input.", "device")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


def detect_machine() -> str:
    """Machine preset from the GPU: no NVIDIA GPU -> laptop-cached; RTX 5090 -> rtx5090; any other -> rtx3060."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "laptop-cached"
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "laptop-cached"
    if not out.strip():
        return "laptop-cached"
    return "rtx5090" if "5090" in out else "rtx3060"


def atomic_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: Unity is reading it
            time.sleep(0.05)
    os.replace(tmp, path)


def run_summary(run_dir) -> dict:
    """Short numbers of a finished run for the lists: speed, delay, quality vs the first reference, swaps."""
    out = {}
    rp = os.path.join(run_dir, "report.json")
    if not os.path.exists(rp):
        return out
    try:
        with open(rp, "r", encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return out
    t = r.get("timing") or {}
    out["achieved_fps"] = t.get("achieved_output_fps")
    out["realtime_factor"] = t.get("realtime_factor")
    out["late_after_warmup"] = r.get("late_fraction_after_warmup")
    out["setup_s"] = t.get("setup_s")
    out["delay_p50_ms"] = ((t.get("stages") or {}).get("e2e") or {}).get("p50")
    q = r.get("quality") or {}
    if q:
        ref = next(iter(q))
        g = ((q[ref].get("groups") or {}).get("after warm-up") or {})
        ang = g.get("angle_mae_deg") or {}
        knees = [ang[k]["mean"] for k in ("left_knee_flexion", "right_knee_flexion") if k in ang]
        ident = q[ref].get("identity") or {}
        out.update({"reference": ref, "mpjpe_cm": (g.get("w_mpjpe_cm") or {}).get("median"),
                    "pa_mpjpe_cm": (g.get("pa_mpjpe_cm") or {}).get("median"),
                    "knee_mae_deg": sum(knees) / len(knees) if knees else None,
                    "swaps": sum(int(v.get("swap_transitions", 0)) for v in ident.values())})
    return out


LOG_TS = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
STAGE_LABELS = [("decode", "2D: decode (slowest camera)"), ("infer", "2D: model (slowest camera)"),
                ("wait_2d", "captured -> all cameras' 2D in"), ("calib", "3D: score calibration"), ("track", "3D: tracker"),
                ("tri", "3D: triangulate + smoothing"), ("fit_share", "3D: body fit (per frame)"),
                ("hold", "waiting for the fit window"), ("write", "write the frame file"),
                ("e2e", "total delay: captured -> frame file")]


def _log_span(run_dir):
    """(seconds from the first line of run.log to its 'report' line or last line, report seconds): the whole job
    including setup and the report (run.log stamps are HH:MM:SS; a run past midnight wraps)."""
    try:
        with open(os.path.join(run_dir, "run.log"), "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh.read().splitlines() if LOG_TS.match(ln)]
    except OSError:
        return None, None

    def t(ln):
        h, m, s_ = (int(x) for x in LOG_TS.match(ln).groups())
        return h * 3600 + m * 60 + s_
    if len(lines) < 2:
        return None, None
    span = lambda a, b: (t(b) - t(a)) % 86400                     # noqa: E731
    done = next((ln for ln in lines if " run done in " in ln), None)
    rep_ = next((ln for ln in reversed(lines) if re.search(r"^\S+ report \S", ln)), None)
    total = span(lines[0], rep_ or lines[-1])
    return total, (span(done, rep_) if done and rep_ else None)


def run_timing(run_dir) -> dict:
    """The time report of a run for the page: whole job, setup, streaming speed, delay, per-stage table."""
    out = {"name": os.path.basename(run_dir), "has_report": False}
    total, report_s = _log_span(run_dir)
    out["job_s"], out["report_s"] = total, report_s
    rp = os.path.join(run_dir, "report.json")
    try:
        with open(rp, "r", encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return out
    t = r.get("timing") or {}
    cfg = r.get("config") or {}
    st_ = t.get("stages") or {}
    out.update(has_report=True, steps=t.get("steps"), written=t.get("written"), section_s=t.get("section_s"),
               target_fps=t.get("target_fps"), achieved_fps=t.get("achieved_output_fps"),
               realtime_factor=t.get("realtime_factor"), wall_s=t.get("wall_s"), loop_s=t.get("loop_s"),
               setup_s=t.get("setup_s"), source_ready_s=t.get("source_ready_s"), first_frame_s=t.get("first_frame_s"),
               first_2d_call_ms=t.get("first_2d_call_ms"), dropped_2d_frames=t.get("dropped_2d_frames"),
               late_after_warmup=r.get("late_fraction_after_warmup"), pace=cfg.get("pace"),
               backend=cfg.get("backend"), solve_ms=t.get("solve_ms"),
               e2e=st_.get("e2e"),
               stages=[{"key": k, "label": lab, **{q: (st_.get(k) or {}).get(q) for q in ("p50", "p90", "p99", "max", "mean")}}
                       for k, lab in STAGE_LABELS
                       if st_.get(k) and not (cfg.get("backend") == "cached" and k in ("decode", "infer"))])
    return out


def win_path(p) -> str:
    """The path as Windows Explorer shows it (in WSL converted with wslpath -w)."""
    p = str(p)
    if os.name == "nt" or not shutil.which("wslpath"):
        return p
    try:
        return subprocess.run(["wslpath", "-w", p], capture_output=True, text=True, timeout=5).stdout.strip() or p
    except (OSError, subprocess.SubprocessError):
        return p


def open_in_explorer(p) -> bool:
    p = str(p)
    try:
        if os.name == "nt":
            os.startfile(p)                     # noqa: S606 - a local folder of this server's own runs
            return True
        if shutil.which("explorer.exe"):        # WSL
            subprocess.Popen(["explorer.exe", win_path(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except OSError:
        pass
    return False


def flat_settings(run_dir) -> dict:
    """The page's setting values recorded in a run's config.resolved.yaml (for the compare view)."""
    p = os.path.join(run_dir, "config.resolved.yaml")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            vals = st.values_from(yaml.safe_load(fh) or {})
    except (OSError, ValueError, yaml.YAMLError):
        return {}

    def num(v):              # "1e-4" from a command-line --set (YAML 1.1 reads it as text) compares as 0.0001
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        return v
    return {k: num(v) for k, v in vals.items()}


class Server:
    def __init__(self, unity_project=None, runs_dir=None, machine=None, repo=REPO):
        self.repo = Path(repo).resolve()
        self.unity = Path(unity_project).resolve() if unity_project else self.repo
        self.runs = Path(runs_dir).resolve() if runs_dir else self.unity / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.machine = machine or detect_machine()
        self.jobs = jobmod.JobManager()
        self._np_lock = threading.Lock()
        self.warm_request = None         # {clip, warm: {key: value}} of the last warm-up sent to the engine

    # ------------------------------------------------------------------ catalogue
    @property
    def default_settings(self) -> str:
        names = st.preset_names()
        want = "laptop_realtime" if self.machine == "laptop-cached" else "best_live"
        return want if want in names else (names[0] if names else "")

    def clips(self) -> list:
        """Clips with a live config (configs/live/<clip>.yaml): cameras, section, and what exists for them."""
        cdir = self.repo / live_config.CONFIG_DIR
        out = []
        for f in sorted(cdir.glob("*.yaml")):
            if f.name == "defaults.yaml":
                continue
            with open(f, "r", encoding="utf-8") as fh:
                c = yaml.safe_load(fh) or {}
            name = c.get("clip") or f.stem
            views = list(c.get("views") or [])
            data = self.repo / "data"
            out.append({"name": name, "views": views, "section_s": c.get("section_s"),
                        "calibrated": all((data / "calib" / f"{v}_cam.json").exists() for v in views),
                        "stored_2d": all((data / "pose2d" / f"{v}_kp2d.json").exists() for v in views),
                        "sync_file": (data / "calib" / f"{name}_sync.json").exists()})
        return out

    def merged(self, clip, preset=None) -> dict:
        over = st.load_preset(preset) if preset else {}
        return live_config.layered(clip, self.machine, over)   # config paths are relative to the repo (main() cds there)

    def runs_list(self) -> list:
        out = []
        for d in sorted(self.runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            fd = fw.frames_dir(str(d))
            if fd is None:
                continue
            n = len(fw.frame_files(str(d)))
            out.append({"name": d.name, "frames": n, "finished": any(d.glob("*.analysis.json")),
                        "modified": d.stat().st_mtime, "layout": os.path.basename(fd),
                        "has_report": (d / "report.md").exists(), "summary": run_summary(str(d))})
        return out

    def builds(self) -> list:
        a = self.unity / "Assets" / "StreamingAssets" / "analysis"
        if not a.is_dir():
            return []
        return [{"name": f.name, "mb": round(f.stat().st_size / 1048576, 1), "modified": f.stat().st_mtime}
                for f in sorted(a.glob("*.analysis.json"))]

    def run_path(self, name) -> Path:
        if not SAFE_NAME.match(name or "") or ".." in name:
            raise HTTPException(400, "bad run name")
        p = self.runs / name
        if not p.is_dir():
            raise HTTPException(404, f"no run {name}")
        return p

    def compare(self, names) -> dict:
        """Side-by-side rows (first = baseline) with the settings that differ between them."""
        rows = []
        for n in names:
            d = self.run_path(n)
            rows.append({"name": n, "summary": run_summary(str(d)), "settings": flat_settings(str(d))})
        keys = [f["key"] for f in st.FIELDS]
        diff = [k for k in keys if len({json.dumps(r["settings"].get(k)) for r in rows}) > 1]
        labels = {f["key"]: f["label"] for f in st.FIELDS}
        return {"rows": rows, "differs": [{"key": k, "label": labels[k]} for k in diff]}

    # ------------------------------------------------------------------ Unity pointer
    def now_playing(self) -> dict | None:
        p = self.runs / NOW_PLAYING
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def play(self, kind, name=None, title=None) -> dict:
        """Point Unity at a live / finished run folder (kind run) or a build in StreamingAssets (kind clip);
        kind stop = Unity keeps the current view. Paths are names relative to runs/ or StreamingAssets/analysis, so a
        server in WSL and Unity on Windows agree without path translation."""
        if kind not in ("run", "clip", "stop"):
            raise HTTPException(400, f"bad kind {kind!r}")
        with self._np_lock:
            prev = self.now_playing() or {}
            obj = {"v": 1, "seq": int(prev.get("seq", 0)) + 1, "kind": kind,
                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if kind == "run":
                d = self.run_path(name)
                fd = fw.frames_dir(str(d))
                if fd is None:
                    raise HTTPException(400, f"{name} has no {fw.FRAMES_DIR}/manifest.json yet")
                obj.update({"run": name, "frames": os.path.basename(fd), "title": title or name})
            elif kind == "clip":
                if not SAFE_NAME.match(name or "") or not name.endswith(".analysis.json"):
                    raise HTTPException(400, "bad build name")
                if not (self.unity / "Assets" / "StreamingAssets" / "analysis" / name).exists():
                    raise HTTPException(404, f"no build {name}")
                obj.update({"file": name, "title": title or name.replace(".analysis.json", "")})
            atomic_json(self.runs / NOW_PLAYING, obj)
            return obj

    # ------------------------------------------------------------------ Unity heartbeat
    def unity_status(self) -> dict:
        """Unity's heartbeat (runs/unity_status.json, written every second in Play mode by ServerFollower)."""
        p = self.runs / UNITY_STATUS
        try:
            with open(p, "r", encoding="utf-8") as fh:
                st_ = json.load(fh)
        except (OSError, ValueError):
            return {"in_play": False, "seen": False}
        age = time.time() - float(st_.get("t", 0))
        st_.update(seen=True, age_s=round(age, 1), in_play=bool(st_.get("playing")) and age < UNITY_STALE_S)
        return st_

    # ------------------------------------------------------------------ stream
    def start_stream(self, clip, preset=None, values=None, force=False) -> dict:
        if clip not in [c["name"] for c in self.clips()]:
            raise HTTPException(400, f"no live config for clip {clip!r}")
        if not force and not self.unity_status().get("in_play"):
            raise HTTPException(409, {"code": "unity_not_playing",
                                      "message": "Unity is not in Play mode. Press Play in Unity, or start anyway "
                                                 "(the frames are still written and Unity can join later)."})
        cfg = self.merged(clip, preset)
        # settings the request leaves out keep their seconds at the requested frame rate (frames are per output frame)
        base = dict(cfg, output_fps=(values or {}).get("output_fps", cfg.get("output_fps")))
        vals = st.values_from(base)
        vals.update(values or {})
        try:
            over = st.overrides(vals, vals.get("backend") or "cached", base_cfg=base)
        except ValueError as e:
            raise HTTPException(400, str(e))
        fps = over.get("output_fps", cfg.get("output_fps"))
        src = "live2d" if over.get("backend") == "live" else "stored2d"
        name = "_".join(["live", clip, f"{int(fps)}fps", src, over.get("pace", "realtime"),
                         time.strftime("%Y-%m-%d_%H%M%S")])
        run_dir = self.runs / name
        argv = jobmod.python_cmd(str(self.repo / "tools" / "live_run.py"), "run", "--clip", clip,
                                 "--preset", self.machine, "--run-dir", str(run_dir))
        for s in st.as_set_args(over):
            argv += ["--set", s]
        wu_s = over.get("warmup.seconds", st.get_key(cfg, "warmup.seconds")) or 0
        job = jobmod.Job("stream", argv, cwd=str(self.repo), run_dir=str(run_dir),
                         title=f"{clip} · {int(fps)} fps · {'2D from video' if src == 'live2d' else 'stored 2D'} · "
                               f"{over.get('pace', 'realtime')}",
                         meta={"views": len(cfg.get("views") or []), "warmup_steps": int(round(float(wu_s) * float(fps))),
                               "backend": over.get("backend", cfg.get("backend")), "fps": float(fps),
                               "pace": over.get("pace", cfg.get("pace")), "run": name,
                               "frames_dir": win_path(run_dir / fw.FRAMES_DIR)})
        try:
            self.jobs.start(job)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        threading.Thread(target=self._point_unity_when_ready, args=(job, name), daemon=True).start()
        return job.status()

    # ------------------------------------------------------------------ v0.8.4 Source tab
    def get_input(self) -> dict:
        p = self.runs / INPUT_FILE
        d = {"kind": "video", "fps": 60, "max_minutes": 5, "cameras": {}, "video_variant": None}
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d.update(json.load(fh) or {})
        except (OSError, ValueError):
            pass
        return d

    def set_input(self, d) -> dict:
        kind = d.get("kind", "video")
        if kind not in ("video", "camera"):
            raise HTTPException(400, "kind must be video or camera")
        cams = {str(k): str(v).strip() for k, v in (d.get("cameras") or {}).items() if str(v).strip()}
        bad = [v for v, u in cams.items() if "://" not in u]
        if bad:
            raise HTTPException(400, f"not a stream address (rtsp://, http://, tcp://, udp://...): {bad}")
        try:
            fps, mins = float(d.get("fps", 60)), float(d.get("max_minutes", 5))
        except (TypeError, ValueError):
            raise HTTPException(400, "fps and max_minutes must be numbers")
        if not (1 <= fps <= 240 and 0.1 <= mins <= 60):
            raise HTTPException(400, "fps 1-240, max_minutes 0.1-60")
        vv = d.get("video_variant") or None
        if vv is not None and not re.match(r"^[A-Za-z0-9_-]{1,20}$", str(vv)):
            raise HTTPException(400, "bad video_variant")
        out = {"kind": kind, "fps": fps, "max_minutes": mins, "cameras": cams, "video_variant": vv}
        atomic_json(self.runs / INPUT_FILE, out)
        return out

    def video_variants(self, clip) -> list:
        """Tags of the smaller video copies (`live_run.py transcode`) that exist for every live view of the clip."""
        import glob
        try:
            cfg = live_config.load(clip, overrides=["backend=live", "rtmpose.engine=fast", "video_variant=null"],
                                   probe=lambda c: {v: dict(zip(("width", "height"), live_config.calib_size(
                                       c["cam_paths"][v]) or (1, 1)), fps=60.0, frame_count=10 ** 6) for v in c["views"]})
        except (ValueError, FileNotFoundError):
            return []
        vids = [cfg["videos"][v] for v in cfg["views"] if cfg["videos"].get(v)]
        if not vids:
            return []
        d0 = os.path.dirname(vids[0])
        tags = sorted({os.path.basename(p)[len(os.path.basename(d0)) + 1:] for p in glob.glob(d0 + "_*") if os.path.isdir(p)})
        return [t for t in tags if t and all(os.path.exists(live_config.variant_video(v, t)) for v in vids)]

    def input_sets(self) -> list:
        d = self.get_input()
        if d["kind"] == "camera":
            return ["input.kind=camera", "input.cameras=" + json.dumps(d["cameras"]), f"input.fps={d['fps']}",
                    f"input.max_minutes={d['max_minutes']}", "video_variant=null"]
        return ["input.kind=video"] + ([f"video_variant={d['video_variant']}"] if d.get("video_variant") else [])

    def test_cameras(self, cams) -> dict:
        """One frame from every camera address: {view: {ok, size | error, ms}}."""
        import cv2
        out = {}
        for v, url in (cams or {}).items():
            t = time.perf_counter()
            if "://" not in str(url):
                out[v] = {"ok": False, "error": "not a stream address"}
                continue
            cap = cv2.VideoCapture(str(url), cv2.CAP_FFMPEG)
            try:
                ok, img = cap.read() if cap.isOpened() else (False, None)
            finally:
                cap.release()
            out[v] = ({"ok": True, "size": [int(img.shape[1]), int(img.shape[0])],
                       "ms": round(1000 * (time.perf_counter() - t))} if ok and img is not None
                      else {"ok": False, "error": "no frames (address, port, firewall?)"})
        return out

    # ------------------------------------------------------------------ v0.8.4 live engine
    def engine_job(self):
        j = self.jobs.current
        return j if isinstance(j, jobmod.EngineJob) and j.running else None

    def _ensure_engine(self, clip):
        j = self.engine_job()
        if j is not None and j.clip == clip and j.preset == self.machine:
            return j
        if j is not None:
            j.stop()
        cmd_dir = self.runs / "engine"
        cmd_dir.mkdir(parents=True, exist_ok=True)
        cmds = cmd_dir / f"commands_{time.strftime('%Y-%m-%d_%H%M%S')}.jsonl"
        cmds.write_text("", encoding="utf-8")
        argv = jobmod.python_cmd(str(self.repo / "tools" / "live_run.py"), "engine", "--clip", clip, "--preset",
                                 self.machine, "--runs-dir", str(self.runs), "--commands", str(cmds))
        j = jobmod.EngineJob(argv, cwd=str(self.repo), clip=clip, preset=self.machine,
                             title=f"live engine · {clip} · {self.machine}", commands=str(cmds))
        try:
            self.jobs.start(j)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return j

    def _request(self, clip, preset, values):
        """(cfg, overrides, fps, src) of a page request, frames converted to seconds (as start_stream does)."""
        if clip not in [c["name"] for c in self.clips()]:
            raise HTTPException(400, f"no live config for clip {clip!r}")
        cfg = self.merged(clip, preset)
        base = dict(cfg, output_fps=(values or {}).get("output_fps", cfg.get("output_fps")))
        vals = st.values_from(base)
        vals.update(values or {})
        try:
            over = st.overrides(vals, vals.get("backend") or "cached", base_cfg=base)
        except ValueError as e:
            raise HTTPException(400, str(e))
        fps = over.get("output_fps", cfg.get("output_fps"))
        src = "live2d" if over.get("backend") == "live" else "stored2d"
        return cfg, over, fps, src

    @staticmethod
    def _warm_part(sets) -> dict:
        out = {}
        for a in sets:
            k, _, v = a.partition("=")
            if k.startswith(WARM_PREFIXES):
                out[k] = v
        return out

    def warmup(self, clip, preset=None, values=None) -> dict:
        cfg, over, fps, src = self._request(clip, preset, values)
        sets = st.as_set_args(over) + self.input_sets()
        if self.get_input()["kind"] == "camera" and src != "live2d":
            raise HTTPException(400, "camera input needs the 2D from the video (backend live, GPU PC)")
        j = self._ensure_engine(clip)
        if j.engine.get("state") == "streaming":
            raise HTTPException(409, "a stream is running: stop it first")
        j.send({"cmd": "warmup", "set": sets})
        self.warm_request = {"clip": clip, "warm": self._warm_part(sets), "fps": fps}
        return j.status()

    def stream_engine(self, clip, preset=None, values=None, force=False) -> dict:
        j = self.engine_job()
        if j is None or j.clip != clip or not j.engine.get("warm"):
            raise HTTPException(409, {"code": "not_warm", "message": "Warm up first (step 1): it loads the models and "
                                                                      "picks the players once for every stream after it."})
        if j.engine.get("state") in ("loading", "warming"):
            raise HTTPException(409, {"code": "warming", "message": "Still warming up: wait for Ready."})
        if j.engine.get("state") == "streaming":
            raise HTTPException(409, "a stream is already running")
        cfg, over, fps, src = self._request(clip, preset, values)
        sets = st.as_set_args(over) + self.input_sets()
        changed = sorted(k for k in set(self._warm_part(sets)) | set((self.warm_request or {}).get("warm", {}))
                         if self._warm_part(sets).get(k) != (self.warm_request or {}).get("warm", {}).get(k))
        if changed:
            raise HTTPException(409, {"code": "rewarm", "message": "These settings changed since the warm-up: "
                                      + ", ".join(changed) + ". Warm up again (step 1)."})
        if not force and not self.unity_status().get("in_play"):
            raise HTTPException(409, {"code": "unity_not_playing",
                                      "message": "Unity is not in Play mode. Press Play in Unity, or start anyway "
                                                 "(the frames are still written and Unity can join later)."})
        kind = self.get_input()["kind"]
        name = "_".join(["live", clip, f"{int(fps)}fps", "camera" if kind == "camera" else src,
                         over.get("pace", "realtime"), time.strftime("%Y-%m-%d_%H%M%S")])
        run_dir = self.runs / name
        title = f"{clip} · {int(fps)} fps · {'cameras' if kind == 'camera' else ('2D from video' if src == 'live2d' else 'stored 2D')} · warm"
        j.meta.update(fps=float(fps), pace=over.get("pace", cfg.get("pace")), backend=over.get("backend"),
                      views=len(cfg.get("views") or []), frames_dir=win_path(run_dir / fw.FRAMES_DIR), title=title)
        j.send({"cmd": "stream", "set": sets, "run_dir": str(run_dir), "title": title})
        threading.Thread(target=self._point_unity_when_ready, args=(j, name, str(run_dir), title), daemon=True).start()
        return j.status()

    def stop_stream(self) -> dict:
        j = self.engine_job()
        if j is not None:
            if j.engine.get("state") == "streaming":
                j.send({"cmd": "stop"})
            return j.status()
        job = self.jobs.stop()
        return job.status() if job else {"state": "idle"}

    def _point_unity_when_ready(self, job, name, run_dir=None, title=None):
        """As soon as the run's manifest exists, Unity follows it (the first frames are then picked up live)."""
        rd = run_dir or job.run_dir
        t_end = time.time() + 600
        while job.running and time.time() < t_end:
            if rd and fw.frames_dir(rd) is not None:
                try:
                    self.play("run", name, title=title or job.title)
                except HTTPException:
                    pass
                return
            time.sleep(0.2)


def create_app(server: Server, allowed_hosts=("127.0.0.1", "localhost"), port=8770) -> FastAPI:
    app = FastAPI(title="BadmintonVR pipeline server")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts) + ["testserver"])
    origins = {f"http://{h}:{port}" for h in allowed_hosts}

    @app.middleware("http")
    async def reject_cross_site_posts(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin is not None and origin not in origins:
            return JSONResponse({"detail": "cross-site request rejected"}, status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in ("/", "/calibrate") or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(calibmod.router(calibmod.Calib(server.repo, server.unity, input_fn=server.get_input), TOOLS))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"machine": server.machine, "default_settings": server.default_settings,
                "unity_project": str(server.unity), "runs_dir": str(server.runs), "repo": str(server.repo),
                "now_playing": server.now_playing()}

    @app.get("/api/clips")
    def clips():
        return server.clips()

    @app.get("/api/schema")
    def schema():
        return {"groups": [{"id": g, "label": lab, "help": h} for g, lab, h in st.GROUPS], "fields": st.FIELDS}

    @app.get("/api/presets")
    def presets():
        out = []
        for n in st.preset_names():
            about = ""
            try:
                with open(os.path.join(st.SETTINGS_DIR, f"{n}.yaml"), "r", encoding="utf-8") as fh:
                    about = (yaml.safe_load(fh) or {}).get("_about", "")
            except OSError:
                pass
            out.append({"name": n, "about": about, "builtin": n in st.BUILTIN})
        return out

    @app.post("/api/presets")
    async def save_preset(request: Request):
        b = await request.json()
        try:
            st.save_preset(b.get("name", ""), b.get("values") or {}, b.get("about", ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"saved": b.get("name")}

    @app.get("/api/settings")
    def settings_values(clip: str, preset: str = ""):
        try:
            cfg = server.merged(clip, preset or None)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return {"values": st.values_from(cfg), "machine": server.machine}

    @app.post("/api/stream/start")
    async def stream_start(request: Request):
        b = await request.json()
        if b.get("oneshot"):                    # the v0.8.2 path: one process loads, warms up and streams once
            return server.start_stream(b.get("clip"), b.get("preset") or None, b.get("values") or {}, bool(b.get("force")))
        return server.stream_engine(b.get("clip"), b.get("preset") or None, b.get("values") or {}, bool(b.get("force")))

    @app.post("/api/stream/stop")
    def stream_stop():
        return server.stop_stream()

    @app.post("/api/engine/warmup")
    async def engine_warmup(request: Request):
        b = await request.json()
        return server.warmup(b.get("clip"), b.get("preset") or None, b.get("values") or {})

    @app.post("/api/engine/stop")
    def engine_stop():
        job = server.jobs.stop()
        server.warm_request = None
        return job.status() if job else {"state": "idle"}

    @app.get("/api/input")
    def input_get():
        return server.get_input()

    @app.post("/api/input")
    async def input_set(request: Request):
        return server.set_input(await request.json())

    @app.get("/api/input/variants")
    def input_variants(clip: str):
        return {"variants": server.video_variants(clip)}

    @app.post("/api/input/test")
    async def input_test(request: Request):
        b = await request.json()
        return server.test_cameras(b.get("cameras") or server.get_input().get("cameras") or {})

    @app.get("/api/job")
    def job_status():
        j = server.jobs.current
        return {"current": j.status() if j else None, "history": list(server.jobs.history)}

    @app.get("/api/runs")
    def runs():
        return server.runs_list()

    @app.get("/api/runs/{name}/timing")
    def run_timing_ep(name: str):
        d = server.run_path(name)
        out = run_timing(str(d))
        fd = fw.frames_dir(str(d))
        out["frames_dir"] = win_path(fd) if fd else None
        return out

    @app.post("/api/open_folder")
    async def open_folder(request: Request):
        b = await request.json()
        d = server.run_path(b.get("run", ""))
        fd = fw.frames_dir(str(d)) or str(d)
        return {"opened": open_in_explorer(fd), "path": win_path(fd)}

    @app.get("/api/runs/{name}/report", response_class=PlainTextResponse)
    def run_report(name: str):
        p = server.run_path(name) / "report.md"
        if not p.exists():
            raise HTTPException(404, "no report yet")
        return p.read_text(encoding="utf-8")

    @app.get("/api/builds")
    def builds():
        return server.builds()

    @app.post("/api/play")
    async def play(request: Request):
        b = await request.json()
        return server.play(b.get("kind"), b.get("name"), b.get("title"))

    @app.post("/api/job/stop")
    def job_stop():
        job = server.jobs.stop()
        return job.status() if job else {"state": "idle"}

    @app.get("/api/compare")
    def compare(runs: str):
        names = [n for n in runs.split(",") if n]
        if not 1 <= len(names) <= 8:
            raise HTTPException(400, "compare 1-8 runs")
        return server.compare(names)

    @app.get("/api/unity")
    def unity():
        return server.unity_status()

    @app.get("/api/now_playing")
    def now_playing():
        return server.now_playing() or {}

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--allow-host", action="append", default=[], help="extra Host name to accept (LAN access)")
    ap.add_argument("--unity-project", help="Unity project folder (default: this repository)")
    ap.add_argument("--runs-dir", help="where runs and now_playing.json go (default: <unity project>/runs)")
    ap.add_argument("--machine", choices=["laptop-cached", "rtx3060", "rtx5090"],
                    help="machine preset (default: detected from the GPU)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost") and not a.allow_host:
        sys.exit("refusing a non-loopback --host without --allow-host <name> (the name LAN browsers use)")
    os.chdir(REPO)
    server = Server(a.unity_project, a.runs_dir, a.machine)
    hosts = ["127.0.0.1", "localhost"] + [h.lower() for h in a.allow_host]
    app = create_app(server, hosts, a.port)
    print(f"pipeline server: http://127.0.0.1:{a.port}  (machine preset {server.machine}, runs in {server.runs})")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
