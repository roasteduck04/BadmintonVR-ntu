"""v0.8.4 fake cameras: play a clip's videos as live network cameras (HTTP MJPEG), to test the live server's camera
source without Raspberry Pis. Run from the repository root:

    python tools/fake_camera.py --clip singles_05x [--port 8090] [--short-side 1080] [--start-s 6.5] [--variant 720p]
    -> http://<this machine>:8090/<view>.mjpg for every live view of the clip (http://localhost:8090/ lists them)

Every camera runs on one shared clock from the moment the server starts, like real cameras that are already
recording: a viewer that connects later gets the frame of "now", not the first frame; at the end of the video it loops.
A camera behind the clock skips frames but still sends one at least every MIN_EMIT_S (v0.8.7: on a laptop the three
4K videos never caught up after the seek to --start-s, so the skip loop never sent a single frame).
Frames are shrunk to --short-side (default 1080) and JPEG-encoded (--quality); the live run rescales the 2D to the
4K calibration (config video_to_calib), as for a real camera of another resolution. --variant reads the
`live_run.py transcode` copies (e.g. 720p: much lighter to decode than the 4K originals on a laptop).

--webcam serves a camera attached to this machine instead (laptop webcam, USB camera), like a tablet or a Pi would:

    python tools/fake_camera.py --webcam 0 [--webcam-size 1280x720] [--fps 30]
    -> http://<this machine>:8090/webcam.mjpg
    python tools/fake_camera.py --webcam 0,1 --webcam-names back,side    -> /back.mjpg, /side.mjpg
    python tools/fake_camera.py --probe 6      one JSON line per USB camera that opens (index, size, fps, preview)

The server page's "USB on this PC" camera rows start this as their relay (v0.8.7; on Windows when the server runs in
WSL, which cannot see USB cameras) and stop it with GET /quit?token=<--quit-token>.

On a Raspberry Pi the same idea serves the Pi camera instead of a file (a later version adds capture timestamps)."""
from __future__ import annotations

import argparse
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOUNDARY = "bvrframe"
MIN_EMIT_S = 0.25                # a camera behind the clock still sends a frame this often
LOST_S = 3.0                     # a USB camera with no frame for this long is reported and reopened
REOPEN_S = 2.0


class Camera(threading.Thread):
    """Decodes one video on the shared clock and keeps the newest JPEG."""

    def __init__(self, view, path, t_start, start_s, short_side, quality):
        super().__init__(daemon=True, name=f"fake-{view}")
        import cv2
        self.cv2, self.view, self.path = cv2, view, path
        self.t_start, self.start_s, self.short, self.quality = t_start, float(start_s), short_side, int(quality)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.size = None, 0, None
        self.running = True

    def _size(self, w, h):
        if not self.short or min(w, h) <= self.short:
            return w, h
        s = self.short / min(w, h)
        return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2

    def run(self):
        cv2 = self.cv2
        first = int(round(self.start_s * self.fps)) % self.n
        while self.running:
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            f = first
            last_emit = 0.0
            while self.running:
                due = self.t_start + (f - first) / self.fps                  # this frame's moment on the shared clock
                now = time.monotonic()
                ahead = due - now
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -1.0 / self.fps and now - last_emit < MIN_EMIT_S:   # behind: skip, like a live camera
                    if not cap.grab():
                        break
                    f += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                h, w = img.shape[:2]
                tw, th = self._size(w, h)
                if (tw, th) != (w, h):
                    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    with self.cond:
                        self.jpeg, self.seq, self.size = buf.tobytes(), self.seq + 1, (tw, th)
                        self.cond.notify_all()
                    last_emit = time.monotonic()
                f += 1
            cap.release()
            self.t_start, first = time.monotonic(), 0                        # loop from the start of the video

    def wait_next(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq or not self.running, timeout=timeout)
            return self.jpeg, self.seq


class Webcam(Camera):
    """A camera attached to this machine (index for OpenCV): every captured frame becomes the newest JPEG."""

    def __init__(self, view, index, short_side, quality, size=None, fps=30.0, fourcc="MJPG"):
        threading.Thread.__init__(self, daemon=True, name=f"webcam-{view}")
        import cv2
        self.cv2, self.view, self.path = cv2, view, f"webcam {index}"
        self.short, self.quality = short_side, int(quality)
        self.index, self.want = index, (size, fps, fourcc)
        self.cap = open_webcam(cv2, index, size, fps, fourcc)
        if self.cap is None:
            raise RuntimeError(f"USB camera {index} does not open (in use by another app, or no such camera)")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or float(fps)        # DirectShow often reports 0
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.size = None, 0, None
        self.running = True

    def run(self):
        cv2 = self.cv2
        bad_since, warned = None, False
        try:
            while self.running:
                ok, img = self.cap.read()                                    # blocks until the camera's next frame
                if not ok or img is None:
                    now = time.monotonic()
                    bad_since = bad_since or now
                    if now - bad_since < LOST_S:
                        time.sleep(0.05)
                        continue
                    if not warned:                       # the server reads this line (usb_status): cable unplugged?
                        print(f"camera {self.view}: no frames from {self.path}", flush=True)
                        warned = True
                    self.cap.release()                   # try the camera again every REOPEN_S (plugged back in)
                    time.sleep(REOPEN_S)
                    self.cap = open_webcam(cv2, self.index, *self.want) or cv2.VideoCapture()
                    continue
                if warned:
                    print(f"camera {self.view}: frames again from {self.path}", flush=True)
                bad_since, warned = None, False
                h, w = img.shape[:2]
                tw, th = self._size(w, h)
                if (tw, th) != (w, h):
                    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    with self.cond:
                        self.jpeg, self.seq, self.size = buf.tobytes(), self.seq + 1, (tw, th)
                        self.cond.notify_all()
        finally:
            self.cap.release()


def open_webcam(cv2, index, size=None, fps=30.0, fourcc="MJPG"):
    """A USB camera by OpenCV index, asked for MJPG (else most webcams give 1080p at 5 fps uncompressed over USB),
    the size and the frame rate - the camera picks its nearest mode. None when it does not open."""
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY           # MSMF often fails to open on Windows
    cap = cv2.VideoCapture(int(index), api)
    if not cap.isOpened():
        cap.release()
        return None
    if fourcc:                                                                # before the size: DirectShow order
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)[:4].ljust(4)))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def dshow_names() -> list:
    """Windows camera names in DirectShow order (= OpenCV's indices), from ffmpeg when it is installed; else []."""
    import re
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg") if sys.platform == "win32" else None
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r'\]\s+"([^"]+)"\s+\(video\)', r.stderr or "")


def probe_webcams(n=6, skip=(), size=None, fps=30.0, fourcc="MJPG", thumb_w=240, out=None, measure_s=1.0):
    """Try USB cameras 0..n-1 (as the relay would open them) and print one JSON line per camera that sends a frame:
    {"index", "name" (ffmpeg's, or null), "size": [w, h], "fps" (measured over measure_s), "thumb": JPEG data URL};
    then {"done": true, "found": k}."""
    import base64
    import json
    import cv2
    out = out or sys.stdout
    names = dshow_names()
    found = 0
    for i in range(int(n)):
        if i in skip:                                                         # in use: named, not opened
            print(json.dumps({"index": i, "name": names[i] if i < len(names) else None, "skipped": True}),
                  file=out, flush=True)
            continue
        cap = open_webcam(cv2, i, size, fps, fourcc)
        if cap is None:
            continue
        try:
            img = None
            for _ in range(15):                                               # the first reads can come back empty
                ok, img = cap.read()
                if ok and img is not None:
                    break
                time.sleep(0.05)
            if img is None:
                continue
            n, t0, t_end = 0, time.monotonic(), time.monotonic() + measure_s     # the real rate: DirectShow
            while time.monotonic() < t_end:                                   # echoes the one asked for
                ok, last = cap.read()
                if ok and last is not None:
                    n, img = n + 1, last
            rate = n / (time.monotonic() - t0) if n >= 2 else None
            h, w = img.shape[:2]
            th = max(1, round(h * thumb_w / w))
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA),
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            print(json.dumps({"index": i, "name": names[i] if i < len(names) else None,
                              "size": [int(w), int(h)], "fps": round(rate, 1) if rate else None,
                              "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None}),
                  file=out, flush=True)
            found += 1
        finally:
            cap.release()
    print(json.dumps({"done": True, "found": found}), file=out, flush=True)
    return found


def make_handler(cams, quit_token=None, on_quit=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            name = self.path.strip("/").split("?")[0]
            if name == "quit":                       # the server page stops its USB relay (it may run on Windows)
                from urllib.parse import parse_qs, urlparse
                tok = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
                if not quit_token or tok != quit_token:
                    self.send_error(403, "wrong token")
                    return
                self.send_response(200)
                self.send_header("Content-Length", "3")
                self.end_headers()
                self.wfile.write(b"bye")
                if on_quit is not None:
                    on_quit()
                return
            if name in ("", "index.html"):
                rows = "".join(f'<li><a href="/{v}.mjpg">/{v}.mjpg</a> {c.size[0] if c.size else "?"}x'
                               f'{c.size[1] if c.size else "?"} at {c.fps:g} fps</li>' for v, c in cams.items())
                body = f"<h1>fake cameras</h1><ul>{rows}</ul>".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            view = name[:-5] if name.endswith(".mjpg") else name
            cam = cams.get(view)
            if cam is None:
                self.send_error(404, f"no camera {view!r} (have {sorted(cams)})")
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seq = 0
            try:
                while True:
                    jpeg, new = cam.wait_next(seq)
                    if not cam.running:                  # the camera was stopped: end this stream
                        return
                    if jpeg is None or new == seq:       # nothing new (a camera that stopped): never resend the old frame
                        continue
                    seq = new
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                                     .encode() + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
    return Handler


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def videos_for(clip, variant=None, views=None):
    from analysis.live import config as live_config
    cfg = live_config.load(clip, overrides=["backend=live", "rtmpose.engine=fast", "video_variant=null"])
    out = {}
    for v in (views or cfg["views"]):
        p = cfg["videos"][v]
        out[v] = live_config.variant_video(p, variant) if variant else p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", default="singles_05x", help="configs/live/<clip>.yaml: its live views and videos")
    ap.add_argument("--views", help="comma-separated views (default: the clip's live views)")
    ap.add_argument("--variant", help="stream the transcoded copies <video folder>_<variant>/ (e.g. 720p)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from WSL / other PCs; 127.0.0.1 = local")
    ap.add_argument("--short-side", type=int, default=1080, help="shrink frames to this shorter side (0 = as recorded)")
    ap.add_argument("--start-s", type=float, default=0.0, help="start this far into the videos")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--webcam", help="serve camera(s) attached to this machine instead of videos: OpenCV index, e.g. 0 "
                                     "or 0,1 (names webcam, webcam1, ...)")
    ap.add_argument("--webcam-size", help="ask the webcam for this size, e.g. 1280x720 (default: its own)")
    ap.add_argument("--webcam-names", help="stream names for the --webcam cameras, e.g. back,side (default webcam, "
                                           "webcam1, ...)")
    ap.add_argument("--webcam-fourcc", default="MJPG", help="ask the webcams for this format ('' = their default)")
    ap.add_argument("--fps", type=float, default=30.0, help="ask the webcam for this frame rate")
    ap.add_argument("--probe", type=int, metavar="N", help="list the USB cameras 0..N-1 that open (JSON lines), then exit")
    ap.add_argument("--skip", default="", help="--probe: camera indices to leave alone (in use), e.g. 0,2")
    ap.add_argument("--quit-token", help="GET /quit?token=<this> stops the cameras (the server page's USB relay)")
    a = ap.parse_args(argv)
    size = tuple(int(x) for x in a.webcam_size.lower().split("x")) if a.webcam_size else None
    if a.probe is not None:
        probe_webcams(a.probe, {int(x) for x in a.skip.split(",") if x.strip()}, size, a.fps, a.webcam_fourcc)
        return
    if a.webcam is not None:
        idx = [int(x) for x in str(a.webcam).split(",")]
        names = [x.strip() for x in a.webcam_names.split(",")] if a.webcam_names else \
            ["webcam" if k == 0 else f"webcam{k}" for k in range(len(idx))]
        if len(names) != len(idx) or len(set(names)) != len(names):
            ap.error("--webcam-names needs one different name per --webcam index")
        cams = {nm: Webcam(nm, i, a.short_side or None, a.quality, size, a.fps, a.webcam_fourcc)
                for nm, i in zip(names, idx)}
        paths = {v: c.path for v, c in cams.items()}
    else:
        paths = videos_for(a.clip, a.variant, a.views.split(",") if a.views else None)
        t0 = time.monotonic() + 0.5
        cams = {v: Camera(v, p, t0, a.start_s, a.short_side or None, a.quality) for v, p in paths.items()}
    for c in cams.values():
        c.start()
    srv = ThreadingServer((a.host, a.port), make_handler(       # /quit: shutdown from another thread than serve_forever
        cams, a.quit_token, lambda: threading.Thread(target=srv.shutdown, daemon=True).start()))
    print(f"fake cameras on port {a.port}" + ("" if a.webcam is not None else
          f" (loop, shared clock, start {a.start_s:g} s)") + ":", flush=True)
    for v, p in paths.items():
        print(f"  {v}: http://localhost:{a.port}/{v}.mjpg  <- {p}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams.values():
            c.running = False
        srv.server_close()


if __name__ == "__main__":
    main()
