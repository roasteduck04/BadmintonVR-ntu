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
THUMB_W = 240                    # Test cameras preview width (px)
TEST_CAM_PORT = 8090             # v0.8.7 one-click test cameras (tools/fake_camera.py)
TESTCAMS_PID = "testcams.pid"    # in the runs folder: the test cameras' process, stopped by the next start
USB_CAM_PORT = 8091              # v0.8.7 USB cameras on this PC: their relay (tools/fake_camera.py --webcam)
USBCAMS_FILE = "usbcams.json"    # in the runs folder: the relay's pid / address / stop token, stopped by the next start
USB_SIZE = "1920x1080"           # asked of every USB camera (it picks its nearest mode; the 2D shrinks to 1080 anyway)
USB_MAX_DEVICE = 9
USB_LOST_RE = re.compile(r"^camera (\S+): (no frames|frames again)\b")
CAM_TEST_S = 10.0                # Test cameras: seconds each camera gets to open and to send a frame
WSL_NET_HINT = (". The server runs in WSL: a camera that plays in VLC on Windows can still be out of reach from WSL - "
                "in a WSL terminal try  ping <camera address>")
USB_WSL_HELP = ("USB cameras: this server runs in WSL, which cannot see USB cameras, so they are served from Windows - "
                "install Python on Windows (python.org, tick 'Add to PATH'), run  py -m pip install opencv-python  in a "
                "Windows terminal, then try again (or set BVR_WINDOWS_PYTHON to a Windows python.exe that has OpenCV)")
# settings the warm-up depends on: changing one after the warm-up means warming up again
WARM_PREFIXES = ("backend", "rtmpose.", "cpu.", "warmup.", "sync.offsets", "video_variant", "input.", "device")
WARM_PREFIXES += ("views",)       # other angles ticked = other 2D processes: warm up again
LOGS_DIR = "server_logs"
ENGINE_PID = "engine.pid"        # runs/engine/engine.pid: the engine this server started (stopped by the next server)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
CAM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")         # live camera names name their calibration files
CLIP_LABELS = {"singles_1x": "Singles 1x", "singles_05x": "Singles 0.5x", "doubles_1side": "Doubles, one side",
               "doubles_2sides": "Doubles, two sides", "doubles_2sides_4angle": "Doubles, two sides (4 angles)"}


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
               e2e=st_.get("e2e"), lag_2d_ms_start_end=t.get("lag_2d_ms_start_end"),
               e2e_ms_start_end=t.get("e2e_ms_start_end"), keeps_up=t.get("keeps_up"),
               keeps_up_2d=t.get("keeps_up_2d"),
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
        jobmod.LOG_DIR = str(self.runs / LOGS_DIR)       # every job's whole output, kept after the window closes
        self._np_lock = threading.Lock()
        self.warm_request = None         # {clip, warm: {key: value}} of the last warm-up sent to the engine
        self._beat = {"t": None, "mono": 0.0}   # last heartbeat stamp seen + when (this process's monotonic clock)
        self.testcams = None             # v0.8.7 test cameras job (runs next to the engine / stream job)
        self.usbcams = None              # v0.8.7 USB camera relay job (USB rows of the live camera table)
        self._usb_lock = threading.RLock()

    # ------------------------------------------------------------------ catalogue
    @property
    def default_settings(self) -> str:
        names = st.preset_names()
        want = "laptop_realtime" if self.machine == "laptop-cached" else "best_live"
        return want if want in names else (names[0] if names else "")

    def clips(self) -> list:
        """v0.8.6: every recorded clip of the pose24 registry (the Source step's dropdown): its cameras, the live config's
        default cameras and section when it has one (configs/live/<clip>.yaml = streamable), what exists for it."""
        from analysis.pose24 import config as p24config
        cdir = self.repo / live_config.CONFIG_DIR
        data = self.repo / "data"
        out = []
        for name, e in p24config.load_registry().items():
            if "base" in e:
                continue
            views = list(e.get("views") or [])
            cf = cdir / f"{name}.yaml"
            c = {}
            if cf.exists():
                with open(cf, "r", encoding="utf-8") as fh:
                    c = yaml.safe_load(fh) or {}
            mode = e.get("mode", "singles")
            out.append({"name": name, "label": CLIP_LABELS.get(name, name), "mode": mode, "views": views,
                        "live_views": list(c.get("views") or views), "section_s": c.get("section_s"),
                        "streamable": bool(c) and (mode == "singles" or c.get("mode") == mode),
                        "calibrated": {v: (data / "calib" / f"{v}_cam.json").exists() for v in views},
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
        """Unity's heartbeat (runs/unity_status.json, written every second in Play mode by ServerFollower).

        Its age is measured by the stamp CHANGING, on this process's monotonic clock - not by comparing Unity's stamp
        (Windows clock) with this clock: in WSL the two drift apart (seconds to minutes after the PC sleeps), which made a
        live Unity look stopped. Only the first reading falls back to the wall clocks; a stamp that changes is fresh."""
        p = self.runs / UNITY_STATUS
        try:
            with open(p, "r", encoding="utf-8-sig") as fh:
                st_ = json.load(fh)
        except (OSError, ValueError):
            return {"in_play": False, "seen": False, "path": str(p)}
        t = float(st_.get("t", 0) or 0)
        now = time.monotonic()
        wall_age = time.time() - t
        if t != self._beat["t"]:
            self._beat = {"t": t, "mono": now, "changed": self._beat["t"] is not None}
        if self._beat.get("changed"):            # seen it change: its age on this process's clock, whatever the clocks
            age = now - self._beat["mono"]
            st_.update(age_s=round(age, 1), clock_offset_s=round(wall_age - age, 1))
        else:                                    # not changed since this server started: only the wall clocks can tell
            age = abs(wall_age)                  # a stamp far in the "future" is as unknown as an old one
            st_.update(age_s=round(wall_age, 1), clock_offset_s=None)
        st_.update(seen=True, in_play=bool(st_.get("playing")) and age < UNITY_STALE_S, path=str(p))
        return st_

    # ------------------------------------------------------------------ stream
    def start_stream(self, clip, preset=None, values=None, force=False) -> dict:
        c = next((x for x in self.clips() if x["name"] == clip), None)
        if c is None or not c["streamable"]:
            raise HTTPException(400, f"{clip!r} cannot stream yet (no live config for it; doubles need the doubles "
                                     f"tracker): it can still be calibrated")
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
                                 "--preset", self.machine, "--run-dir", str(run_dir),
                                 "--set", f"paths.runs_dir={self.runs}")   # 1 camera: the engine's warm state is here
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
        """The Source step: {kind video|camera, clip, views (chosen angles of the clip), video_variant, cameras [{name,
        url, use} + type usb, device for a USB camera on this PC (v0.8.7; url = its relay address)] (live broadcast;
        names are the user's, default cam1, cam2, ...), fps, max_minutes}. v0.8.4 files
        ({view: url} cameras, no clip) are read too."""
        p = self.runs / INPUT_FILE
        d = {"kind": "video", "clip": None, "views": [], "fps": 60, "max_minutes": 5, "cameras": [], "video_variant": None}
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d.update(json.load(fh) or {})
        except (OSError, ValueError):
            pass
        if isinstance(d.get("cameras"), dict):
            d["cameras"] = [{"name": k, "url": v, "use": True} for k, v in d["cameras"].items()]
        names = [c["name"] for c in self.clips()]
        if d.get("clip") not in names:
            d["clip"] = "singles_05x" if "singles_05x" in names else (names[0] if names else None)
        return d

    def used_cameras(self, d=None) -> dict:
        d = d or self.get_input()
        return {c["name"]: live_config.with_rotate(c["url"], c.get("rotate"))
                for c in d.get("cameras") or [] if c.get("use", True) and c.get("url")}

    @staticmethod
    def _camera_rows(raw):
        """The live camera table, checked: [{name, url, use}] for network cameras, [{name, url, use, type: usb,
        device}] for USB cameras on this PC (url = their relay address, filled when it runs); + {name: device} of the
        USB cameras in use."""
        if isinstance(raw, dict):
            raw = [{"name": k, "url": v, "use": True} for k, v in raw.items()]
        cams, seen, devs = [], set(), {}
        for c in raw or []:
            usb = str(c.get("type") or "network") == "usb"
            name, url = str(c.get("name") or "").strip(), str(c.get("url") or "").strip()
            if not name and not url and not usb:
                continue
            if not CAM_NAME.match(name):
                raise HTTPException(400, f"camera name {name!r}: letters, digits, _ or - (it names the calibration file)")
            if name in seen:
                raise HTTPException(400, f"camera name {name!r} is used twice")
            seen.add(name)
            rot = {"rotate": 180} if str(c.get("rotate")) in ("180", "True", "true", "1") else {}   # upside down
            if usb:
                try:
                    dev = int(c.get("device", 0))
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{name}: the USB camera number must be 0-{USB_MAX_DEVICE}")
                if not 0 <= dev <= USB_MAX_DEVICE:
                    raise HTTPException(400, f"{name}: the USB camera number must be 0-{USB_MAX_DEVICE}")
                use = bool(c.get("use", True))
                other = next((k for k, v in devs.items() if v == dev), None)
                if use and other:
                    raise HTTPException(400, f"{name}: USB camera {dev} is already {other}")
                if use:
                    devs[name] = dev
                cams.append({"name": name, "url": "", "use": use, "type": "usb", "device": dev, **rot})
                continue
            if url and "://" not in url:
                raise HTTPException(400, f"{name}: not a stream address (rtsp://, http://, tcp://, udp://...)")
            cams.append({"name": name, "url": url, "use": bool(c.get("use", True)) and bool(url), **rot})
        return cams, devs

    def set_input(self, d) -> dict:
        kind = d.get("kind", "video")
        if kind not in ("video", "camera"):
            raise HTTPException(400, "kind must be video or camera")
        clips = {c["name"]: c for c in self.clips()}
        clip = d.get("clip") or self.get_input()["clip"]
        if clip not in clips:
            raise HTTPException(400, f"unknown clip {clip!r} (one of {sorted(clips)})")
        views = [str(v) for v in d.get("views") or []]
        bad = [v for v in views if v not in clips[clip]["views"]]
        if bad:
            raise HTTPException(400, f"{bad} are not angles of {clip}")
        if kind == "video" and not views:
            raise HTTPException(400, "pick at least 1 angle (1 = the 1-camera mode, 2+ = triangulated 3D)")
        cams, devs = self._camera_rows(d.get("cameras") or [])
        if kind == "camera" and not any(c["use"] for c in cams):
            raise HTTPException(400, "use at least 1 camera with a stream address or a USB camera (1 = the 1-camera mode)")
        try:
            fps, mins = float(d.get("fps", 60)), float(d.get("max_minutes", 5))
        except (TypeError, ValueError):
            raise HTTPException(400, "fps and max_minutes must be numbers")
        if not (1 <= fps <= 240 and 0.1 <= mins <= 60):
            raise HTTPException(400, "fps 1-240, max_minutes 0.1-60")
        vv = d.get("video_variant") or None
        if vv is not None and not re.match(r"^[A-Za-z0-9_-]{1,20}$", str(vv)):
            raise HTTPException(400, "bad video_variant")
        if kind == "camera" and devs:                    # v0.8.7: USB rows stream through the relay (started here)
            urls = self.usb_start(devs, fps)
            for c in cams:
                if c.get("type") == "usb" and c["use"]:
                    c["url"] = urls[c["name"]]
        else:
            self.usb_stop()                              # frees the USB cameras for other apps
        out = {"kind": kind, "clip": clip, "views": views, "fps": fps, "max_minutes": mins, "cameras": cams,
               "video_variant": vv}
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
        """--set arguments for the Source step: the chosen angles (video) or the live cameras, named by the user, with
        their offsets estimated in the warm-up (a clip's sync file is for its recorded videos only)."""
        d = self.get_input()
        if d["kind"] == "camera":
            d = self.usb_ensure(d)                       # USB rows: their relay runs (after a server restart too)
            cams = self.used_cameras(d)
            return ["input.kind=camera", "input.cameras=" + json.dumps(cams), f"views=[{', '.join(cams)}]",
                    f"input.fps={d['fps']}", f"input.max_minutes={d['max_minutes']}", "video_variant=null",
                    "sync.offsets=auto"]
        out = ["input.kind=video"] + ([f"video_variant={d['video_variant']}"] if d.get("video_variant") else [])
        return out + ([f"views=[{', '.join(d['views'])}]"] if d.get("views") else [])

    def test_cameras(self, cams, timeout_s=CAM_TEST_S) -> dict:
        """One frame from every camera address (all at once, each given timeout_s to open and to send a frame):
        {view: {ok, size, ms, thumb (JPEG data URL) | error}}."""
        from concurrent.futures import ThreadPoolExecutor
        cams = dict(cams or {})
        if not cams:
            return {}
        with ThreadPoolExecutor(max_workers=len(cams)) as ex:
            return dict(zip(cams, ex.map(lambda u: self._test_camera(u, timeout_s), cams.values())))

    def test_rows(self, rows, fps=None, timeout_s=CAM_TEST_S) -> dict:
        """v0.8.7 Test cameras with the table's rows (network and USB): USB rows start their relay first; each
        result of a USB row carries its relay address (url)."""
        cams, devs = self._camera_rows(rows)
        urls = self.usb_start(devs, float(fps or self.get_input().get("fps") or 60)) if devs else {}
        out = self.test_cameras({c["name"]: live_config.with_rotate(urls.get(c["name"]) or c["url"], c.get("rotate"))
                                 for c in cams if c["url"] or c["name"] in urls}, timeout_s)
        for nm, u in urls.items():
            out[nm]["url"] = u
        return out

    @staticmethod
    def _test_camera(url, timeout_s) -> dict:
        import base64
        import cv2
        t = time.perf_counter()
        if "://" not in str(url):
            return {"ok": False, "error": "not a stream address"}
        url, rot = live_config.split_rotate(url)
        where = "WSL" if in_wsl() else "this PC"
        reach = stream_reachable(url)                    # the network first: it tells a wrong address from a wrong path
        if reach is False:
            host, port = stream_host_port(url)
            return {"ok": False, "error": f"cannot connect to {host}:{port} from {where} (camera off, wrong address "
                                          f"or port, or a firewall)" + (WSL_NET_HINT if where == "WSL" else "")}
        ms = int(timeout_s * 1000)
        cap = cv2.VideoCapture(str(url), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms])
        try:
            opened = cap.isOpened()
            ok, img = cap.read() if opened else (False, None)
        finally:
            cap.release()
        if not ok or img is None:
            if reach and not opened:
                return {"ok": False, "error": f"the camera answers but the stream did not open in {timeout_s:g} s "
                                              "(stream path, user name / password, or a codec this PC cannot read)"}
            if reach:
                return {"ok": False, "error": f"the stream opened but sent no picture in {timeout_s:g} s"}
            return {"ok": False, "error": f"no frame in {timeout_s:g} s (address, port, firewall?)"}
        img = live_config.upright(img, rot)
        h, w = img.shape[:2]
        ms = round(1000 * (time.perf_counter() - t))
        s = THUMB_W / w
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (THUMB_W, max(1, round(h * s))), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
        return {"ok": True, "size": [int(w), int(h)], "ms": ms,
                "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None}

    # ------------------------------------------------------------------ v0.8.7 one-click test cameras
    def testcams_status(self) -> dict:
        j = self.testcams
        on = j is not None and j.running
        m = j.meta if j is not None else {}
        err = None
        if j is not None and not on and not j._stop_requested:
            err = j.error or (list(j.lines)[-1] if j.lines else f"exit code {j.returncode}")
        return {"running": on, "clip": m.get("clip"), "views": list(m.get("views") or []), "start_s": m.get("start_s"),
                "urls": dict(m.get("urls") or {}), "port": m.get("port"), "variant": m.get("variant"), "error": err}

    def testcams_start(self, clip=None, views=None, port=TEST_CAM_PORT, ready_s=30.0, variant=None) -> dict:
        """Play the clip's chosen angles as looping live cameras (tools/fake_camera.py, http://localhost:<port>/<angle>.mjpg,
        from the clip's live section) and switch the Source step to them. The cameras are named after the angles, so
        the clip's calibrations apply; the next warm-up measures their offsets as for real cameras."""
        clips = {c["name"]: c for c in self.clips()}
        inp = self.get_input()
        clip = clip or inp["clip"]
        if clip not in clips:
            raise HTTPException(400, f"unknown clip {clip!r}")
        c = clips[clip]
        same = inp.get("clip") == clip and inp.get("views")
        views = [str(v) for v in (views or (inp["views"] if same else c["live_views"]))]
        bad = [v for v in views if v not in c["views"]]
        if bad or not views:
            raise HTTPException(400, f"{bad} are not angles of {clip}" if bad else "pick at least 1 angle")
        self.testcams_stop()
        self._stop_stale_testcams()
        if port_open(port):
            raise HTTPException(409, f"port {port} is in use (a fake_camera.py started by hand?): stop it, then try again")
        start_s = float((c.get("section_s") or [0.0])[0] or 0.0)
        if variant and variant not in self.video_variants(clip):
            raise HTTPException(400, f"no {variant!r} copies of {clip}")
        argv = jobmod.python_cmd(str(self.repo / "tools" / "fake_camera.py"), "--clip", clip, "--views", ",".join(views),
                                 "--port", str(port), "--host", "127.0.0.1", "--start-s", f"{start_s:g}")
        if variant:                                      # lighter to decode (a laptop cannot decode three 4K videos)
            argv += ["--variant", str(variant)]
        urls = {v: f"http://localhost:{port}/{v}.mjpg" for v in views}
        j = jobmod.Job("testcams", argv, cwd=str(self.repo), title=f"test cameras · {clip}",
                       meta={"clip": clip, "views": views, "start_s": start_s, "urls": urls, "port": port,
                             "variant": variant})
        j.start()
        self.testcams = j
        (self.runs / TESTCAMS_PID).write_text(str(j.proc.pid), encoding="utf-8")
        t_end = time.monotonic() + ready_s
        while not port_open(port):                       # the cameras open their videos before the server listens
            if not j.running:
                raise HTTPException(500, f"the test cameras stopped: {self.testcams_status()['error']}")
            if time.monotonic() > t_end:
                self.testcams_stop()
                raise HTTPException(504, f"the test cameras did not start in {ready_s:.0f} s")
            time.sleep(0.3)
        saved = self.set_input({"kind": "camera", "clip": clip, "views": inp["views"] if same else views,
                                "cameras": [{"name": v, "url": urls[v], "use": True} for v in views],
                                "fps": video_fps(self.repo, clip, views[0]), "max_minutes": inp.get("max_minutes", 5),
                                "video_variant": None})
        return {"status": self.testcams_status(), "input": saved}

    def testcams_stop(self) -> dict:
        if self.testcams is not None and self.testcams.running:
            self.testcams.stop()
        pf = self.runs / TESTCAMS_PID
        if pf.exists() and (self.testcams is None or self.testcams.proc is None
                            or pf.read_text(encoding="utf-8").strip() == str(self.testcams.proc.pid)):
            pf.unlink()
        return self.testcams_status()

    def _stop_stale_testcams(self):
        """Test cameras left by an earlier server (its window closed without stopping them) still hold the port."""
        pf = self.runs / TESTCAMS_PID
        if not pf.exists():
            return
        try:
            import psutil
            pr = psutil.Process(int(pf.read_text(encoding="utf-8").strip()))
            if any("fake_camera.py" in str(a) for a in pr.cmdline()):
                for ch in pr.children(recursive=True):
                    ch.kill()
                pr.kill()
                pr.wait(5)
        except Exception:  # noqa: BLE001 - gone already / not ours / psutil missing
            pass
        pf.unlink(missing_ok=True)

    # ------------------------------------------------------------------ v0.8.7 USB cameras on this PC
    def usb_runner(self) -> dict:
        """How the relay (tools/fake_camera.py) runs next to the USB cameras: with this Python, or - when the server
        runs in WSL, which cannot see USB cameras - with a Windows Python that has OpenCV, through WSL interop
        (listening on every address: WSL reaches Windows through its virtual network unless networking is mirrored)."""
        script = str(self.repo / "tools" / "fake_camera.py")
        if not in_wsl():
            return {"where": "this PC", "argv": jobmod.python_cmd("-u", script), "bind": "127.0.0.1",
                    "hosts": ["127.0.0.1"]}
        exe = find_windows_python()
        if not exe:
            raise HTTPException(400, USB_WSL_HELP)
        return {"where": "Windows", "argv": [*exe, "-u", win_path(script)], "bind": "0.0.0.0",
                "hosts": wsl_windows_hosts()}

    def usb_status(self) -> dict:
        j = self.usbcams
        on = j is not None and j.running
        m = j.meta if j is not None else {}
        err = None
        if j is not None and not on and not j._stop_requested:
            err = j.error or next((ln for ln in reversed(list(j.lines)) if ln.strip()), None) or f"exit code {j.returncode}"
        lost = {}
        for ln in (list(j.lines) if on else []):         # fake_camera.py: "camera <name>: no frames" / "frames again"
            mm = USB_LOST_RE.match(ln)
            if mm:
                lost[mm.group(1)] = mm.group(2) == "no frames"
        lost = sorted(k for k, v in lost.items() if v)
        if lost:
            err = "no frames from " + ", ".join(lost) + " (USB cable unplugged? it is tried again every few seconds)"
        return {"running": on, "lost": lost, "devices": dict(m.get("devices") or {}), "urls": dict(m.get("urls") or {}),
                "where": m.get("where"), "host": m.get("host"), "port": m.get("port"), "fps": m.get("fps"), "error": err}

    def usb_find(self, n=6, fps=60.0, timeout_s=120.0) -> dict:
        """Which USB cameras open (0..n-1), each with its size and a preview frame - the ones the relay already serves
        are listed as in use (a USB camera opens in one program at a time)."""
        run = self.usb_runner()
        st = self.usb_status()
        busy = {int(i): nm for nm, i in st["devices"].items()} if st["running"] else {}
        argv = run["argv"] + ["--probe", str(int(n)), "--webcam-size", USB_SIZE, "--fps", f"{float(fps):g}"]
        if busy:
            argv += ["--skip", ",".join(str(i) for i in sorted(busy))]
        try:
            r = subprocess.run(argv, cwd=str(self.repo), capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"the USB camera search took over {timeout_s:.0f} s")
        rows = []
        for ln in r.stdout.splitlines():
            if ln.startswith("{"):
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
        if not any(x.get("done") for x in rows):
            tail = next((ln for ln in reversed((r.stderr or r.stdout or "").splitlines()) if ln.strip()), "no output")
            raise HTTPException(500, f"the USB camera search failed ({run['where']}): {tail}")
        named = {x["index"]: x.get("name") for x in rows if x.get("skipped")}
        cams = [x for x in rows if "index" in x and not x.get("skipped")]
        cams += [{"index": i, "name": named.get(i), "in_use": nm} for i, nm in busy.items()]
        return {"where": run["where"], "cameras": sorted(cams, key=lambda x: x["index"])}

    def usb_start(self, devices, fps=60.0, ready_s=45.0) -> dict:
        """Serve the USB cameras {name: device number} as http://<address>:8091/<name>.mjpg (the relay keeps running
        while they stay the same); returns {name: address}."""
        devices = {str(k): int(v) for k, v in (devices or {}).items()}
        with self._usb_lock:
            st = self.usb_status()
            if st["running"] and st["devices"] == devices and st["fps"] == float(fps):
                return st["urls"]
            if st["running"] and self.cameras_in_use():  # a restart would cut the cameras under the engine
                raise HTTPException(409, "The engine is using the cameras, and this would restart the USB cameras "
                                         "(other cameras or frame rate). Shut down the engine in step 3 first.")
            self.usb_stop()
            self._stop_stale_usbcams()
            run = self.usb_runner()
            port = USB_CAM_PORT
            busy = next((h for h in run["hosts"] if port_open(port, h)), None)
            if busy:
                raise HTTPException(409, f"port {port} is in use on {busy} (a fake_camera.py started by hand?): "
                                         "stop it, then try again")
            import secrets
            token = secrets.token_hex(8)
            argv = run["argv"] + ["--webcam", ",".join(str(i) for i in devices.values()),
                                  "--webcam-names", ",".join(devices), "--webcam-size", USB_SIZE,
                                  "--fps", f"{float(fps):g}", "--port", str(port), "--host", run["bind"],
                                  "--quit-token", token]
            j = jobmod.Job("usbcams", argv, cwd=str(self.repo), title="USB cameras",
                           meta={"devices": devices, "where": run["where"], "port": port, "fps": float(fps),
                                 "urls": {}, "host": None})
            j.start()
            self.usbcams = j
            info = {"pid": j.proc.pid, "token": token, "port": port, "hosts": run["hosts"], "where": run["where"]}
            atomic_json(self.runs / USBCAMS_FILE, info)
            t_end = time.monotonic() + ready_s
            while not any(str(ln).startswith("fake cameras on port") for ln in list(j.lines)):   # every camera opened
                if not j.running:
                    err = self.usb_status()["error"]
                    self.usb_stop()
                    raise HTTPException(500, f"the USB cameras did not start ({run['where']}): {err}")
                if time.monotonic() > t_end:
                    self.usb_stop()
                    raise HTTPException(504, f"the USB cameras did not start in {ready_s:.0f} s")
                time.sleep(0.2)
            host = None
            for _ in range(10):                          # it listens right after that line
                host = next((h for h in run["hosts"] if port_open(port, h, timeout_s=1.0)), None)
                if host:
                    break
                time.sleep(0.3)
            if host is None:
                self.usb_stop()
                raise HTTPException(502, f"the USB cameras run on {run['where']} but this server cannot reach port "
                                         f"{port} (tried {', '.join(run['hosts'])}): allow Python through Windows "
                                         "Firewall (Private and Public networks), then try again")
            urls = {nm: f"http://{host}:{port}/{nm}.mjpg" for nm in devices}
            j.meta.update(host=host, urls=urls)
            atomic_json(self.runs / USBCAMS_FILE, dict(info, host=host))
            return urls

    def usb_ensure(self, d=None) -> dict:
        """A saved source with USB rows: start their relay if it is not running (e.g. after a server restart) and
        update their addresses when they moved (WSL's address of Windows changes with a reboot)."""
        d = d or self.get_input()
        devs = {c["name"]: int(c.get("device", 0)) for c in d.get("cameras") or []
                if c.get("type") == "usb" and c.get("use", True)}
        if d.get("kind") != "camera" or not devs:
            return d
        urls = self.usb_start(devs, float(d.get("fps") or 60))
        if any(c.get("url") != urls[c["name"]] for c in d["cameras"] if c["name"] in urls):
            d = dict(d, cameras=[dict(c, url=urls[c["name"]]) if c["name"] in urls else c for c in d["cameras"]])
            atomic_json(self.runs / INPUT_FILE, d)
        return d

    def usb_autostart(self):
        """At server start: a saved source with USB cameras gets its relay back (calibration frames need it too)."""
        try:
            self.usb_ensure()
        except HTTPException as e:
            print(f"USB cameras not started: {e.detail}", flush=True)
        except Exception as e:  # noqa: BLE001 - never stop the server over the cameras
            print(f"USB cameras not started: {e}", flush=True)

    def usb_stop(self) -> dict:
        with self._usb_lock:
            j = self.usbcams
            pf = self.runs / USBCAMS_FILE
            info = read_json(pf) or {}
            ours = j is not None and j.proc is not None and str(info.get("pid")) == str(j.proc.pid)
            if j is not None and j.running:
                j._stop_requested = True
                _quit_relay(info, [j.meta.get("host")] if j.meta.get("host") else info.get("hosts") or [])
                if j.wait(5.0) != "stopped" and j.running:
                    j.stop()                             # the relay did not answer /quit
            if ours:
                pf.unlink(missing_ok=True)
            elif info:
                self._stop_stale_usbcams()               # an earlier server's relay: it still holds the cameras
            return self.usb_status()

    def _stop_stale_usbcams(self):
        """A relay left by an earlier server (its window closed) still holds the USB cameras and the port."""
        pf = self.runs / USBCAMS_FILE
        info = read_json(pf)
        if not info:
            return
        hosts = [info["host"]] if info.get("host") else info.get("hosts") or []
        if _quit_relay(info, hosts):                     # it lets the port go a moment after answering
            port, t_end = int(info.get("port") or USB_CAM_PORT), time.monotonic() + 5.0
            while time.monotonic() < t_end and any(port_open(port, h) for h in hosts):
                time.sleep(0.2)
        if info.get("where") == "this PC":
            try:
                import psutil
                pr = psutil.Process(int(info["pid"]))
                if any("fake_camera.py" in str(a) for a in pr.cmdline()):
                    pr.kill()
                    pr.wait(5)
            except Exception:  # noqa: BLE001 - gone already / not ours / psutil missing
                pass
        pf.unlink(missing_ok=True)

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
        try:
            (cmd_dir / ENGINE_PID).write_text(str(j.proc.pid), encoding="utf-8")
        except (OSError, AttributeError):
            pass
        return j

    def engine_shutdown(self):
        """The server ends: so does the engine or stream job it started (else it keeps the GPU and the cameras)."""
        try:
            self.jobs.stop()
        finally:
            (self.runs / "engine" / ENGINE_PID).unlink(missing_ok=True)

    def stop_stale_engine(self):
        """An engine left by an earlier server (its window was closed): it still holds GPU memory and the cameras."""
        pf = self.runs / "engine" / ENGINE_PID
        if not pf.exists():
            return False
        killed = False
        try:
            import psutil
            pr = psutil.Process(int(pf.read_text(encoding="utf-8").strip()))
            cl = [str(a) for a in pr.cmdline()]
            if any("live_run.py" in a for a in cl) and "engine" in cl:
                for ch in pr.children(recursive=True):
                    ch.kill()
                pr.kill()
                pr.wait(5)
                killed = True
        except Exception:  # noqa: BLE001 - gone already / not ours / psutil missing
            pass
        pf.unlink(missing_ok=True)
        return killed

    def input_mtime(self) -> float:
        """When the source was last saved (a live camera's calibration frame older than that is grabbed again)."""
        try:
            return (self.runs / "input.json").stat().st_mtime
        except OSError:
            return 0.0

    def cameras_in_use(self) -> bool:
        """The engine is reading the live cameras (warming up, warm, or streaming)."""
        j = self.engine_job()
        if j is None or self.get_input().get("kind") != "camera":
            return False
        return bool(j.engine.get("warm")) or j.engine.get("state") in ("loading", "warming", "streaming")

    def _request(self, clip, preset, values):
        """(cfg, overrides, fps, src) of a page request, frames converted to seconds (as start_stream does)."""
        c = next((x for x in self.clips() if x["name"] == clip), None)
        if c is None or not c["streamable"]:
            raise HTTPException(400, f"{clip!r} cannot stream yet (no live config for it; doubles need the doubles "
                                     f"tracker): it can still be calibrated")
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
        j.error, j.engine["error"] = None, None           # a new stream: the last error is history
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
            if getattr(job, "engine", None) and job.engine.get("error"):
                return                                   # the engine refused the stream: nothing to follow
            if rd and fw.frames_dir(rd) is not None:
                try:
                    self.play("run", name, title=title or job.title)
                except HTTPException:
                    pass
                return
            time.sleep(0.2)


def stream_host_port(url):
    """(host, port) of a stream address (the scheme's usual port when none is given), or (None, None)."""
    from urllib.parse import urlsplit
    try:
        u = urlsplit(str(url))
        port = u.port or {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443, "rtmp": 1935}.get(u.scheme.lower())
        return (u.hostname, port) if u.hostname and port else (None, None)
    except ValueError:
        return None, None


def stream_reachable(url, timeout_s=3.0):
    """True / False: a TCP connection to the stream's host and port opens; None when that cannot be told (udp://,
    no port known)."""
    host, port = stream_host_port(url)
    if not host or str(url).lower().startswith(("udp:", "rtp:", "srt:")):
        return None
    try:
        return port_open(port, host, timeout_s)
    except OSError:
        return False


def port_open(port, host="127.0.0.1", timeout_s=0.3) -> bool:
    """Something listens on host:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.settimeout(timeout_s)
        return sk.connect_ex((host, int(port))) == 0


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def in_wsl() -> bool:
    """This server runs in WSL (Linux on Windows: no USB cameras, but Windows programs start through interop)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


_WIN_PY = []                     # the Windows Python found once (with OpenCV), as an argv prefix


def find_windows_python() -> list:
    """From WSL: a Windows Python that imports OpenCV - BVR_WINDOWS_PYTHON, else the py launcher, else python.exe
    (the Microsoft Store stand-in fails the import check). [] when there is none; a miss is checked again next time."""
    if _WIN_PY:
        return list(_WIN_PY)
    cands = []
    env = os.environ.get("BVR_WINDOWS_PYTHON")
    if env:
        cands.append([env])
    for exe, extra in (("py.exe", ["-3"]), ("python.exe", [])):
        p = shutil.which(exe)
        if p:
            cands.append([p, *extra])
    for c in cands:
        try:
            r = subprocess.run([*c, "-c", "import cv2"], capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            _WIN_PY[:] = c
            return list(c)
    return []


def wsl_windows_hosts() -> list:
    """The addresses WSL may reach Windows on: localhost (mirrored networking), else the default gateway and the
    resolv.conf name server (NAT networking: Windows' address on WSL's virtual network)."""
    hosts = ["127.0.0.1"]
    try:
        out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=5).stdout
        hosts += re.findall(r"default via (\S+)", out)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as fh:
            hosts += re.findall(r"^nameserver\s+(\d+\.\d+\.\d+\.\d+)", fh.read(), re.M)
    except OSError:
        pass
    return list(dict.fromkeys(hosts))


def _quit_relay(info, hosts):
    """Ask a USB camera relay to stop (GET /quit with its token: it may run on Windows, out of reach of a kill)."""
    import urllib.request
    if not info or not info.get("token"):
        return False
    for h in hosts:
        try:
            with urllib.request.urlopen(f"http://{h}:{int(info.get('port') or USB_CAM_PORT)}/quit?token={info['token']}",
                                        timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - not there / already gone
            continue
    return False


def video_fps(repo, clip, view) -> float:
    """The frame rate the test cameras send: the clip's video of that angle (60 if it cannot be read)."""
    try:
        import cv2
        from analysis.pose24 import config as p24config
        reg = p24config.load_registry()
        e = reg[clip]
        vids = (reg.get(e.get("base"), e) if e.get("base") else e).get("videos") or {}
        cap = cv2.VideoCapture(str(Path(repo) / vids[view]) if not os.path.isabs(vids[view]) else vids[view])
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return round(fps, 3) if fps > 0 else 60.0
    except Exception:  # noqa: BLE001
        return 60.0


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
    app.include_router(calibmod.router(calibmod.Calib(server.repo, server.unity, input_fn=server.get_input, input_mtime_fn=server.input_mtime), TOOLS))

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
        from starlette.concurrency import run_in_threadpool     # a USB source may restart its relay first (seconds)
        return await run_in_threadpool(server.stream_engine, b.get("clip"), b.get("preset") or None,
                                       b.get("values") or {}, bool(b.get("force")))

    @app.post("/api/stream/stop")
    def stream_stop():
        return server.stop_stream()

    @app.post("/api/engine/warmup")
    async def engine_warmup(request: Request):
        from starlette.concurrency import run_in_threadpool     # a USB camera source may start its relay first
        b = await request.json()
        return await run_in_threadpool(server.warmup, b.get("clip"), b.get("preset") or None, b.get("values") or {})

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
        from starlette.concurrency import run_in_threadpool     # USB rows start their relay (seconds)
        b = await request.json()
        return await run_in_threadpool(server.set_input, b)

    @app.get("/api/input/variants")
    def input_variants(clip: str):
        return {"variants": server.video_variants(clip)}

    @app.post("/api/input/test")
    async def input_test(request: Request):
        from starlette.concurrency import run_in_threadpool     # off the event loop: the page keeps answering
        b = await request.json()
        if b.get("rows") is not None:                            # v0.8.7: the table's rows (network + USB)
            return await run_in_threadpool(server.test_rows, b["rows"], b.get("fps"))
        return await run_in_threadpool(server.test_cameras, b.get("cameras") or server.used_cameras())

    @app.get("/api/usb")
    def usb_get():
        return server.usb_status()

    @app.post("/api/usb/find")
    async def usb_find(request: Request):
        from starlette.concurrency import run_in_threadpool
        b = await request.json()
        return await run_in_threadpool(lambda: server.usb_find(fps=float(b.get("fps") or 60)))

    @app.post("/api/usb/stop")
    def usb_stop():
        if server.usb_status()["running"] and server.cameras_in_use():
            raise HTTPException(409, "The engine is using the cameras. Shut down the engine in step 3 first.")
        return server.usb_stop()

    @app.get("/api/testcams")
    def testcams_get():
        return server.testcams_status()

    @app.post("/api/testcams/start")
    async def testcams_start(request: Request):
        from starlette.concurrency import run_in_threadpool
        b = await request.json()
        return await run_in_threadpool(lambda: server.testcams_start(b.get("clip"), b.get("views"),
                                                                     variant=b.get("variant") or None))

    @app.post("/api/testcams/stop")
    def testcams_stop():
        return server.testcams_stop()

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
    import atexit
    if server.stop_stale_engine():
        print("stopped a live engine left by an earlier server")
    atexit.register(server.engine_shutdown)              # the engine / stream job ends with the server
    atexit.register(server.testcams_stop)                # test cameras end with the server
    atexit.register(server.usb_stop)                     # so does the USB camera relay
    threading.Thread(target=server.usb_autostart, name="usb-autostart", daemon=True).start()
    hosts = ["127.0.0.1", "localhost"] + [h.lower() for h in a.allow_host]
    app = create_app(server, hosts, a.port)
    print(f"pipeline server: http://127.0.0.1:{a.port}  (machine preset {server.machine}, runs in {server.runs})")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
