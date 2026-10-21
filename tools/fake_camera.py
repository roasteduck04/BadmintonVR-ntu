"""v0.8.4 fake cameras: play a clip's videos as live network cameras (HTTP MJPEG), to test the live server's camera
source without Raspberry Pis. Run from the repository root:

    python tools/fake_camera.py --clip singles_05x [--port 8090] [--short-side 1080] [--start-s 6.5] [--variant 720p]
    -> http://<this machine>:8090/<view>.mjpg for every live view of the clip (http://localhost:8090/ lists them)

Every camera runs on one shared clock from the moment the server starts, like real cameras that are already
recording: a viewer that connects later gets the frame of "now", not the first frame; at the end of the video it loops.
Frames are shrunk to --short-side (default 1080) and JPEG-encoded (--quality); the live run rescales the 2D to the
4K calibration (config video_to_calib), as for a real camera of another resolution. --variant reads the
`live_run.py transcode` copies (e.g. 720p: much lighter to decode than the 4K originals on a laptop).

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
            while self.running:
                due = self.t_start + (f - first) / self.fps                  # this frame's moment on the shared clock
                ahead = due - time.monotonic()
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -1.0 / self.fps:                                # behind: skip, like a live camera
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
                f += 1
            cap.release()
            self.t_start, first = time.monotonic(), 0                        # loop from the start of the video

    def wait_next(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq or not self.running, timeout=timeout)
            return self.jpeg, self.seq


def make_handler(cams):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            name = self.path.strip("/").split("?")[0]
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
                    jpeg, seq = cam.wait_next(seq)
                    if jpeg is None:
                        continue
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
    a = ap.parse_args(argv)
    paths = videos_for(a.clip, a.variant, a.views.split(",") if a.views else None)
    t0 = time.monotonic() + 0.5
    cams = {v: Camera(v, p, t0, a.start_s, a.short_side or None, a.quality) for v, p in paths.items()}
    for c in cams.values():
        c.start()
    srv = ThreadingServer((a.host, a.port), make_handler(cams))
    print(f"fake cameras on port {a.port} (loop, shared clock, start {a.start_s:g} s):", flush=True)
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
