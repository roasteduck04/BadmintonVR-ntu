"""v0.8.4 live engine: one long-lived process that keeps the 2D models loaded between streams.

    python tools/live_run.py engine --clip <clip> --preset <machine> [--set ...] [--runs-dir runs]

Commands, one JSON object per line - appended to the --commands file (the live server's way: the engine runs with no
stdin, because on Windows a thread blocked reading a stdin pipe stalls the child processes the engine starts), or on
stdin when no --commands file is given (by hand / tests):
    {"cmd": "warmup", "set": ["key=value", ...]}   load / reuse the models, then the player warm-up
    {"cmd": "stream", "set": [...], "run_dir": "<folder>", "title": "..."}   one stream from the warm state
    {"cmd": "stop"}                                end the current stream (its frames so far are finalised)
    {"cmd": "quit"}
Status lines on stdout: `ENGINE {"ev": ..., ...}` (state changes, results, errors) next to the usual run log lines.

Warm-up = (1) the 2D processes (models loaded and warmed on the GPU, TensorRT engines built) - kept until a setting
they depend on changes (warm_key: backend, device, cameras / videos, rtmpose.*, CPU threads) - and (2) the player
warm-up: a short run over the first warmup.seconds of the chosen part (video) or of what the cameras send now, which
picks P1 / P2, fixes their body models, the ankle plane, the facing signs and (sync.offsets auto) the camera offsets.
A stream then starts warm: no model loading, no warm-up frames - the tracker re-finds P1 / P2 by court half. A video
stream plays the chosen part from its start every time; a camera stream runs until stopped or input.max_minutes.
The warm state is also saved to <runs>/engine/warm_state.json."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

from . import config as live_config


def emit(**ev):
    print("ENGINE " + json.dumps(ev, default=str), flush=True)


def warm_key(cfg) -> str:
    """Settings the 2D processes depend on: a change means loading the models again."""
    rp = {k: v for k, v in cfg["rtmpose"].items() if not k.startswith("_")}
    key = {"backend": cfg["backend"], "device": cfg["device"], "views": list(cfg["views"]), "videos": cfg["videos"],
           "rtmpose": rp, "cpu": cfg.get("cpu_budget"), "video_variant": cfg.get("video_variant"),
           "fps": cfg["fps"] if rp.get("batch_views") else None}
    return json.dumps(key, sort_keys=True, default=str)


def player_key(cfg) -> str:
    """Settings the player warm-up depends on (a change means warming the players up again, models kept)."""
    return json.dumps({"warmup": cfg.get("warmup"), "offsets": cfg["sync"].get("offsets"), "input": cfg.get("input"),
                       "section_start": cfg["section_s"][0], "tracker": cfg.get("tracker")}, sort_keys=True, default=str)


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        self.fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def close(self):
        self.fh.close()


class LiveEngine:
    def __init__(self, clip, preset=None, base_sets=None, runs_dir="runs", log=print):
        self.clip, self.preset, self.base = clip, preset, list(base_sets or [])
        self.runs_dir, self.log = runs_dir, log
        self.src, self.src_key, self.src_cfg = None, None, None
        self.warm_state, self.player_key = None, None
        self.stop_ev = threading.Event()
        self.state = "cold"

    # ------------------------------------------------------------------ helpers
    def load_cfg(self, sets):
        return live_config.load(self.clip, preset=self.preset, overrides=self.base + list(sets or []))

    def _set_state(self, state, **kw):
        self.state = state
        emit(ev="state", state=state, **kw)

    def _ensure_source(self, cfg, log):
        from . import sources
        key = warm_key(cfg)
        if self.src is not None and key == self.src_key:
            log("  2D models already loaded and warm (reused)")
            return False
        if self.src is not None:
            log("  2D settings changed: loading the models again")
            self.src.close()
            self.src = None
        self._set_state("loading")
        t = time.perf_counter()
        src = sources.make(cfg, log=log)
        src.open()
        self.src, self.src_key, self.src_cfg = src, key, cfg
        self.warm_state, self.player_key = None, None            # new models / cameras: the players warm up again
        emit(ev="models", seconds=round(time.perf_counter() - t, 1), info=getattr(src, "info", {}))
        return True

    # ------------------------------------------------------------------ commands
    def warmup(self, sets=None):
        from .runner import Runner
        cfg = self.load_cfg(sets)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        rd = os.path.join(self.runs_dir, "engine", f"warmup_{cfg['clip']}_{stamp}")
        log = Tee(os.path.join(rd, "run.log"))
        try:
            log(f"engine warm-up: {cfg['clip']}, {len(cfg['views'])} cameras, input {cfg['input']['kind']}")
            log(f"run folder {rd}")
            self._ensure_source(cfg, log)
            self._set_state("warming")
            w = int(cfg["warmup_steps"])
            a = float(cfg["section_s"][0])
            b = a + (w + 2) / float(cfg["fps"])           # the warm-up steps plus a little: the first frames of the part
            wcfg = self.load_cfg(list(sets or []) + [f"section_s=[{a}, {b}]"])
            t = time.perf_counter()
            res = Runner(wcfg, rd, log=log, source=self.src, session=True, export_warm=True).run()
            ws = res.get("warm_state") or {}
            ws.update(clip=cfg["clip"], made=time.strftime("%Y-%m-%d %H:%M:%S"), input=cfg["input"]["kind"],
                      warmup_steps=w, seconds=round(time.perf_counter() - t, 1),
                      players={p: m is not None for p, m in (ws.get("models") or {}).items()})
            self.warm_state, self.player_key = ws, player_key(cfg)
            os.makedirs(os.path.join(self.runs_dir, "engine"), exist_ok=True)
            with open(os.path.join(self.runs_dir, "engine", "warm_state.json"), "w", encoding="utf-8") as fh:
                json.dump(ws, fh, indent=1, default=str)
            log(f"warm-up done in {ws['seconds']} s: players {ws['players']}, ankle {ws.get('ankle_h_m', 0):.3f} m, "
                f"offsets {ws.get('offsets')}")
            self._set_state("ready", warm={k: ws.get(k) for k in ("players", "offsets", "ankle_h_m", "signs", "made",
                                                                   "warmup_steps", "seconds", "input")})
        finally:
            log.close()

    def stream(self, sets=None, run_dir=None, title=None):
        from . import report
        from .runner import Runner
        cfg = self.load_cfg(sets)
        if self.src is None or self.warm_state is None:
            raise RuntimeError("not warm: run the warm-up first")
        if warm_key(cfg) != self.src_key:
            raise RuntimeError("a 2D setting changed since the warm-up (model, TensorRT, video size, detector, cameras, "
                               "CPU threads): warm up again")
        if player_key(cfg) != self.player_key:
            raise RuntimeError("a warm-up setting changed (warm-up length, camera offsets, input, part of the video): "
                               "warm up again")
        ws = self.warm_state
        cfg["warmup_steps"] = 0                          # warm already: every frame is a stream frame
        cfg["offsets"] = {v: float(ws["offsets"].get(v, 0.0)) for v in cfg["views"]}
        cfg["offsets_source"] = f"warm-up ({ws.get('offsets_source')})"
        cfg["offsets"][cfg["time_ref"]] = 0.0
        rd = run_dir or os.path.join(self.runs_dir, "_".join(
            ["live", cfg["clip"], f"{int(cfg['fps'])}fps", "live2d" if cfg["backend"] == "live" else "stored2d",
             cfg["pace"], time.strftime("%Y-%m-%d_%H%M%S")]))
        log = Tee(os.path.join(rd, "run.log"))
        self.stop_ev.clear()
        try:
            log(f"engine stream: {title or cfg['clip']} (warm since {ws.get('made')})")
            log(f"run folder {rd}")
            emit(ev="stream_start", run_dir=rd, steps=len(cfg["ref_frames"]), title=title)
            self._set_state("streaming", run_dir=rd)
            t = time.perf_counter()
            res = Runner(cfg, rd, log=log, source=self.src, session=True, warm_state=ws, stop_event=self.stop_ev).run()
            log(f"run done in {time.perf_counter() - t:.1f} s: {res['written']}/{res['steps']} frames")
            try:
                p = report.render_run(rd, log=log)
                log(f"report {p}")
            except Exception as e:  # noqa: BLE001 - the stream itself is complete
                log(f"  report failed: {type(e).__name__}: {e}")
            emit(ev="stream_done", run_dir=rd, written=res["written"], steps=res["steps"],
                 stopped=self.stop_ev.is_set())
        finally:
            log.close()
            self._set_state("ready")

    def stop(self):
        self.stop_ev.set()

    def close(self):
        if self.src is not None:
            self.src.close()
            self.src = None


def follow(path, stop_ev, poll_s=0.1):
    """Every line of `path` (created if missing; a fresh file per engine, so from its start) and every line appended
    later, until stop_ev."""
    open(path, "a", encoding="utf-8").close()
    with open(path, "r", encoding="utf-8") as fh:
        buf = ""
        while not stop_ev.is_set():
            chunk = fh.read()
            if not chunk:
                time.sleep(poll_s)
                continue
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line


def serve(clip, preset=None, base_sets=None, runs_dir="runs", stdin=None, commands=None):
    """The engine command loop (JSON lines from the `commands` file, else stdin). A reader thread handles stop at once;
    warm-up / stream run on the main thread one after another."""
    import queue
    eng = LiveEngine(clip, preset, base_sets, runs_dir)
    q = queue.Queue()
    done = threading.Event()
    inp = follow(commands, done) if commands else (stdin or sys.stdin)

    def reader():
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                emit(ev="error", message=f"not JSON: {line[:80]}")
                continue
            if msg.get("cmd") == "stop":
                eng.stop()
                emit(ev="stop_requested")
                continue
            q.put(msg)
            if msg.get("cmd") == "quit":
                return
        q.put({"cmd": "quit"})
    threading.Thread(target=reader, daemon=True, name="engine-stdin").start()
    emit(ev="state", state="cold", clip=clip, preset=preset)
    try:
        while True:
            msg = q.get()
            cmd = msg.get("cmd")
            if cmd == "quit":
                break
            try:
                if cmd == "warmup":
                    eng.warmup(msg.get("set"))
                elif cmd == "stream":
                    eng.stream(msg.get("set"), msg.get("run_dir"), msg.get("title"))
                else:
                    emit(ev="error", message=f"unknown command {cmd!r}")
            except Exception as e:  # noqa: BLE001 - reported, the engine keeps running
                emit(ev="error", cmd=cmd, message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
                eng._set_state("ready" if eng.warm_state is not None else ("loaded" if eng.src else "cold"))
    finally:
        done.set()
        eng.close()
        emit(ev="state", state="off")
