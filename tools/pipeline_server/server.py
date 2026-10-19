"""BadmintonVR pipeline server (v0.8.2): one local page for the pipeline, Unity follows what it plays.

Spec: docs/superpowers/specs/2026-10-19-v0.8.2-pipeline-server-design.md. Run from the repository root:
    python tools/pipeline_server/server.py [--unity-project PATH] [--runs-dir PATH] [--machine rtx5090]
    -> http://127.0.0.1:8770
Two tabs: 1 Calibrate, 2 Stream (every live setting; "Start streaming to Unity" goes straight from the videos - or the
stored 2D on a machine without a GPU - to Unity, frame by frame). Unity's ServerFollower follows runs/now_playing.json
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
        vals = st.values_from(cfg)
        vals.update(values or {})
        try:
            over = st.overrides(vals, vals.get("backend") or "cached")
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
        job = jobmod.Job("stream", argv, cwd=str(self.repo), run_dir=str(run_dir),
                         title=f"{clip} · {int(fps)} fps · {'2D from video' if src == 'live2d' else 'stored 2D'} · "
                               f"{over.get('pace', 'realtime')}")
        try:
            self.jobs.start(job)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        threading.Thread(target=self._point_unity_when_ready, args=(job, name), daemon=True).start()
        return job.status()

    def _point_unity_when_ready(self, job, name):
        """As soon as the run's manifest exists, Unity follows it (the first frames are then picked up live)."""
        while job.running:
            if fw.frames_dir(job.run_dir) is not None:
                try:
                    self.play("run", name, title=job.title)
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
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(calibmod.router(calibmod.Calib(server.repo, server.unity), TOOLS))   # tab 1 Calibrate

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
            out.append({"name": n, "about": about, "builtin": n in ("best_live", "laptop_realtime")})
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
        return server.start_stream(b.get("clip"), b.get("preset") or None, b.get("values") or {}, bool(b.get("force")))

    @app.post("/api/stream/stop")
    def stream_stop():
        job = server.jobs.stop()
        return job.status() if job else {"state": "idle"}

    @app.get("/api/job")
    def job_status():
        j = server.jobs.current
        return {"current": j.status() if j else None, "history": list(server.jobs.history)}

    @app.get("/api/runs")
    def runs():
        return server.runs_list()

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
