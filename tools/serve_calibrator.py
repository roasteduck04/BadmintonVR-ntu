"""Browser-based camera calibrator for multi-view court PnP.

Extracts one frame per clip at t=1s, serves a browser UI for clicking court
points, computes PnP calibration on save. Also serves the identity labeler (/labeler).

Run:  python tools/serve_calibrator.py [--port 8001] [--no-browser]
Open: http://localhost:8001

Security: binds 127.0.0.1 only (the API overwrites tracked calibrations and serves video
frames of people). `--host 0.0.0.0 --allow-host <lan-name-or-ip>` opts in to LAN access.
Requests with a foreign Host header (DNS rebinding) or a foreign Origin (CSRF) are refused,
POST bodies must be application/json (no "simple" cross-site requests) and are size-capped,
and only the two pages, the API and data/calib/frames are served (no static file server).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, unquote, urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

import cv2
import numpy as np

from calibrate_court import COURT_POINTS
from analysis import multiview as mv

FRAMES_DIR = os.path.join(ROOT, "data", "calib", "frames")
CALIB_DIR = os.path.join(ROOT, "data", "calib")
LABELS = None          # analysis.people.labels.LabelService for /labeler, created on first use
MAX_BODY = 1024 * 1024                       # POST bodies are small JSON (points, seeds, labels)
TRUSTED_HOSTS = {"127.0.0.1", "localhost"}   # Host header names accepted (main() adds --allow-host)


class BadRequest(Exception):
    """Client error: answered as JSON {"error": msg} with the given status."""

    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


def _labels():
    global LABELS
    if LABELS is None:
        from analysis.people.labels import LabelService
        LABELS = LabelService(root=ROOT)
    return LABELS


def _safe_path(base_dir, *parts):
    """Join base_dir with parts and ensure the result stays within base_dir.

    Returns the resolved real path, or None if the requested path would
    escape base_dir (e.g. via '..' path traversal in a user-supplied
    clip id or filename).
    """
    base_real = os.path.realpath(base_dir)
    candidate = os.path.realpath(os.path.join(base_dir, *parts))
    if candidate != base_real and not candidate.startswith(base_real + os.sep):
        return None
    return candidate

CLIPS = {
    "Singles 0.5x": {"clip": "singles_05x", "mode": "singles"},
    "Singles 1x":   {"clip": "singles_1x",  "mode": "singles"},
    "Doubles 1-side":  {"clip": "doubles_1side",  "mode": "doubles"},
    "Doubles 2-sides": {"clip": "doubles_2sides", "mode": "doubles"},
    "Doubles 2-sides (4 angle)": {"clip": "doubles_2sides_4angle", "mode": "doubles"},
}

CAMERAS = [
    {"view": "back_051", "video": "data/15-08/singles/0.5/15.8 back 0.5-1.mov",
     "group": "Singles 0.5x", "camera": "back", "phone": "iPhone 16", "fov_deg": 120},
    {"view": "backleft_051", "video": "data/15-08/singles/0.5/15.8 back left 0.5-1.mov",
     "group": "Singles 0.5x", "camera": "backleft", "phone": "iPhone 16 Pro Max", "fov_deg": 120},
    {"view": "backright_051", "video": "data/15-08/singles/0.5/15.8 back right 0.5-1.mov",
     "group": "Singles 0.5x", "camera": "backright", "phone": "iPhone 16", "fov_deg": 120},
    {"view": "frontleft_051", "video": "data/15-08/singles/0.5/15.8 front left 0.5-1.mov",
     "group": "Singles 0.5x", "camera": "frontleft", "phone": "iPhone 16 Pro", "fov_deg": 120},
    {"view": "frontright_051", "video": "data/15-08/singles/0.5/15.8 front right 0.5-1.mov",
     "group": "Singles 0.5x", "camera": "frontright", "phone": "iPhone 13 Pro Max", "fov_deg": 120},

    {"view": "back_11", "video": "data/15-08/singles/1/15.8 back 1-1.mov",
     "group": "Singles 1x", "camera": "back", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "backleft_11", "video": "data/15-08/singles/1/15.8 back left 1-1.mov",
     "group": "Singles 1x", "camera": "backleft", "phone": "iPhone 16 Pro Max", "fov_deg": 82},
    {"view": "backright_11", "video": "data/15-08/singles/1/15.8 back right 1-1.mov",
     "group": "Singles 1x", "camera": "backright", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "frontleft_11", "video": "data/15-08/singles/1/15.8 front left 1-1.mov",
     "group": "Singles 1x", "camera": "frontleft", "phone": "iPhone 16 Pro", "fov_deg": 82},
    {"view": "frontright_11", "video": "data/15-08/singles/1/15.8 front right 1-1.mov",
     "group": "Singles 1x", "camera": "frontright", "phone": "iPhone 13 Pro Max", "fov_deg": 78},

    {"view": "back_21", "video": "data/15-08/doubles/1 side/15.8 back 2-1.mov",
     "group": "Doubles 1-side", "camera": "back", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "backleft_21", "video": "data/15-08/doubles/1 side/15.8 back left 2-1.mov",
     "group": "Doubles 1-side", "camera": "backleft", "phone": "iPhone 16 Pro Max", "fov_deg": 82},
    {"view": "backright_21", "video": "data/15-08/doubles/1 side/15.8 back right 2-1.mov",
     "group": "Doubles 1-side", "camera": "backright", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "frontleft_21", "video": "data/15-08/doubles/1 side/15.8 front left 2-1.mov",
     "group": "Doubles 1-side", "camera": "frontleft", "phone": "iPhone 16 Pro", "fov_deg": 82},
    {"view": "frontright_21", "video": "data/15-08/doubles/1 side/15.8 front right 2-1.mov",
     "group": "Doubles 1-side", "camera": "frontright", "phone": "iPhone 13 Pro Max", "fov_deg": 78},

    {"view": "back_31", "video": "data/15-08/doubles/2 sides/15.8 back 3-1.mov",
     "group": "Doubles 2-sides", "camera": "back", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "backleft_31", "video": "data/15-08/doubles/2 sides/15.8 back left 3-1.mov",
     "group": "Doubles 2-sides", "camera": "backleft", "phone": "iPhone 16 Pro Max", "fov_deg": 82},
    {"view": "backright_31", "video": "data/15-08/doubles/2 sides/15.8 back right 3-1.mov",
     "group": "Doubles 2-sides", "camera": "backright", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "frontleft_31", "video": "data/15-08/doubles/2 sides/15.8 front left 3-1.mov",
     "group": "Doubles 2-sides", "camera": "frontleft", "phone": "iPhone 16 Pro", "fov_deg": 82},
    {"view": "frontright_31", "video": "data/15-08/doubles/2 sides/15.8 front right 3-1.mov",
     "group": "Doubles 2-sides", "camera": "frontright", "phone": "Xiaomi 13", "fov_deg": 75},

    {"view": "back_41", "video": "data/15-08/doubles/2 sides (4 angle)/15.8 back 4-1.mov",
     "group": "Doubles 2-sides (4 angle)", "camera": "back", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "backright_41", "video": "data/15-08/doubles/2 sides (4 angle)/15.8 back right 4-1.mov",
     "group": "Doubles 2-sides (4 angle)", "camera": "backright", "phone": "iPhone 16", "fov_deg": 78},
    {"view": "frontleft_41", "video": "data/15-08/doubles/2 sides (4 angle)/15.8 front left 4-1.mov",
     "group": "Doubles 2-sides (4 angle)", "camera": "frontleft", "phone": "iPhone 16 Pro", "fov_deg": 82},
    {"view": "frontright_41", "video": "data/15-08/doubles/2 sides (4 angle)/15.8 front right 4-1.mov",
     "group": "Doubles 2-sides (4 angle)", "camera": "frontright", "phone": "iPhone 13 Pro Max", "fov_deg": 78},
]


def extract_frame(cam, frame_time=1.0):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out = os.path.join(FRAMES_DIR, cam["view"] + ".jpg")
    if os.path.exists(out):
        return True
    video = os.path.join(ROOT, cam["video"])
    if not os.path.exists(video):
        print(f"  SKIP {cam['view']}: video not found")
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(frame_time), "-i", video,
             "-frames:v", "1", "-q:v", "2", out],
            capture_output=True, check=True,
        )
        print(f"  OK {cam['view']}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  FAIL {cam['view']}: {e}")
        return False


def extract_temp_frame(cam, frame_time):
    """Extract a frame at an arbitrary time for additional-seed clicking,
    without touching the cached primary frame for this view."""
    safe_t = str(frame_time).replace(".", "_").replace("-", "neg")
    fname = f"{cam['view']}_add_{safe_t}.jpg"
    out = os.path.join(FRAMES_DIR, fname)
    video = os.path.join(ROOT, cam["video"])
    if not os.path.exists(video):
        print(f"  SKIP {cam['view']}@{frame_time}: video not found")
        return None
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(frame_time), "-i", video,
             "-frames:v", "1", "-q:v", "2", out],
            capture_output=True, check=True,
        )
        return fname
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  FAIL {cam['view']}@{frame_time}: {e}")
        return None


def _camera(view):
    cam = next((c for c in CAMERAS if c["view"] == view), None)
    if cam is None:
        raise BadRequest(f"unknown view {view!r}")
    return cam


def _points(body):
    """Validated clicked points -> (names, image_px Nx2, court_xz Nx2)."""
    points = body.get("points")
    if not isinstance(points, list):
        raise BadRequest("missing points")
    if len(points) < 4:
        raise BadRequest("need >= 4 points")
    names = []
    for p in points:
        if not isinstance(p, dict) or p.get("name") not in COURT_POINTS:
            name = p.get("name") if isinstance(p, dict) else p
            raise BadRequest(f"unknown court point {name!r}")
        names.append(p["name"])
    image_px = np.array([[float(p["px"]), float(p["py"])] for p in points], dtype=np.float64)
    court_xz = np.array([COURT_POINTS[n] for n in names], dtype=np.float64)
    return names, image_px, court_xz


def _frame_size(view):
    img = cv2.imread(os.path.join(FRAMES_DIR, view + ".jpg"))
    if img is None:
        raise BadRequest("frame not found", 404)
    return img.shape[1], img.shape[0]


class Handler(BaseHTTPRequestHandler):
    """Explicit routes only: no static file server, so no HEAD / directory listing of the repo."""

    # ---- request guards ----

    def _host_ok(self):
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):                       # [::1]:8001
            name = host[1:host.find("]")] if "]" in host else host
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in TRUSTED_HOSTS or name == "::1"

    def _origin_ok(self):
        """A page on another site can POST here (CSRF); browsers send its Origin."""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        u = urlparse(origin)
        port = self.server.server_address[1]
        return (u.scheme == "http" and (u.hostname in TRUSTED_HOSTS or u.hostname == "::1")
                and (u.port or 80) == port)

    def _guard(self, post=False):
        if not self._host_ok():
            self._json_response({"error": "untrusted Host header"}, 403)
            return False
        if post:
            if not self._origin_ok():
                self._json_response({"error": "cross-site request rejected"}, 403)
                return False
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._json_response({"error": "Content-Type must be application/json"}, 415)
                return False
        return True

    def _dispatch(self, route):
        try:
            route()
        except BadRequest as e:
            self._json_response({"error": str(e)}, e.code)
        except (KeyError, ValueError, TypeError, AttributeError) as e:  # missing/ill-typed fields, bad JSON
            self._json_response({"error": f"bad request: {type(e).__name__}: {e}"}, 400)
        except Exception as e:  # noqa: BLE001 - answer instead of dropping the connection
            traceback.print_exc()
            self._json_response({"error": f"server error: {type(e).__name__}: {e}"}, 500)

    def do_GET(self):
        if self._guard():
            self._dispatch(self._get)

    def do_POST(self):
        if self._guard(post=True):
            self._dispatch(self._post)

    # ---- routes ----

    def _get(self):
        path = urlparse(self.path).path
        if path in ("/", "/calibrator.html"):
            return self._serve_file(
                os.path.join(TOOLS_DIR, "calibrator.html"), "text/html; charset=utf-8")
        if path == "/labeler":
            return self._serve_file(os.path.join(TOOLS_DIR, "labeler.html"), "text/html; charset=utf-8")
        if path == "/api/cameras":
            result = []
            for c in CAMERAS:
                frame_ok = os.path.exists(
                    os.path.join(FRAMES_DIR, c["view"] + ".jpg"))
                calib_path = os.path.join(CALIB_DIR, c["view"] + "_cam.json")
                clip_info = CLIPS.get(c["group"], {})
                entry = {**c, "has_frame": frame_ok,
                         "calibrated": os.path.exists(calib_path),
                         "clip": clip_info.get("clip"),
                         "mode": clip_info.get("mode")}
                if entry["calibrated"]:
                    with open(calib_path) as f:
                        cd = json.load(f)
                    entry["rms"] = cd.get("rms_px")
                    entry["saved_fov"] = cd.get("fov_deg")
                    entry["saved_points"] = {
                        n: info["px"]
                        for n, info in cd.get("court_points", {}).items()
                    }
                result.append(entry)
            return self._json_response(result)
        if path == "/api/clips":
            return self._json_response(CLIPS)
        if path == "/api/court-points":
            return self._json_response(
                {k: list(v) for k, v in COURT_POINTS.items()})
        if path.startswith("/api/seeds/"):
            clip_id = unquote(path[len("/api/seeds/"):])
            seeds_path = _safe_path(CALIB_DIR, f"{clip_id}_seeds.json")
            if seeds_path is None:
                raise BadRequest("invalid clip id")
            if os.path.isfile(seeds_path):
                with open(seeds_path, encoding="utf-8") as f:
                    return self._json_response(json.load(f))
            return self._json_response({})
        if path.startswith("/frames/"):
            fname = unquote(path[len("/frames/"):])
            fpath = _safe_path(FRAMES_DIR, fname)
            if fpath is None:
                raise BadRequest("invalid filename")
            if os.path.isfile(fpath) and os.path.splitext(fpath)[1].lower() in (".jpg", ".jpeg", ".png"):
                return self._serve_image(fpath)
            raise BadRequest("not found", 404)
        if path.startswith("/api/label/"):
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            svc = _labels()
            if path == "/api/label/clips":
                return self._json_response(svc.clips())
            if path == "/api/label/frames":
                return self._json_response(svc.frames(q["clip"]))
            if path == "/api/label/frame":
                return self._json_response(svc.frame(q["clip"], q["view"], int(q["raw"])))
            if path == "/api/label/seeds":
                return self._json_response(svc.seeds(q["clip"], q["view"]))
            if path == "/api/label/image":
                data = svc.image(q["clip"], q["view"], int(q["raw"]))
                if data is None:
                    raise BadRequest("no video frame", 404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        raise BadRequest("not found", 404)

    def _post(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")

        if path == "/api/label/save":
            n = _labels().save(body["clip"], body["view"], int(body["raw"]), body["labels"])
            return self._json_response({"saved": n})

        if path == "/api/seeds":
            clip_id = body.get("clip")
            if not clip_id or not isinstance(clip_id, str):
                raise BadRequest("missing clip")
            seeds_path = _safe_path(CALIB_DIR, f"{clip_id}_seeds.json")
            if seeds_path is None:
                raise BadRequest("invalid clip id")
            os.makedirs(os.path.dirname(seeds_path), exist_ok=True)
            with open(seeds_path, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2)
            print(f"wrote seeds: {seeds_path}")
            return self._json_response({"ok": True, "path": seeds_path})

        if path == "/api/auto-fov":
            view = _camera(body.get("view"))["view"]
            fov_on = body.get("fov_on", "long")
            _, image_px, court_xz = _points(body)
            w, h = _frame_size(view)
            best_fov, best_rms = mv.find_best_fov(
                w, h, court_xz, image_px, on=fov_on)
            print(f"auto-fov {view}: best={best_fov:.1f}° rms={best_rms:.2f}px")
            return self._json_response(
                {"fov_deg": best_fov, "rms_px": best_rms})

        if path == "/api/calibrate":
            cam_info = _camera(body.get("view"))
            view = cam_info["view"]
            fov_deg = float(body["fov_deg"])
            fov_on = body.get("fov_on", "long")
            names, image_px, court_xz = _points(body)
            w, h = _frame_size(view)
            K = mv.intrinsics_from_fov(w, h, fov_deg, on=fov_on)
            R, t_vec = mv.solve_extrinsics(court_xz, image_px, K)
            rms = mv.reprojection_rms(court_xz, image_px, K, R, t_vec)
            out = {
                "view": view,
                "source": os.path.basename(cam_info["video"]),
                "width": w, "height": h,
                "fov_deg": fov_deg, "fov_on": fov_on,
                "focal_px": float(K[0, 0]),
                "K": K.tolist(), "R": R.tolist(), "t": t_vec.tolist(),
                "rms_px": float(rms),
                "court_points": {
                    n: {"px": [float(x), float(y)], "court_xz": list(COURT_POINTS[n])}
                    for n, (x, y) in zip(names, image_px)
                },
            }
            out_path = os.path.join(CALIB_DIR, f"{view}_cam.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            status = f"wrote {out_path}: rms {rms:.2f}px"
            if rms > 15:
                status += " WARNING: rms > 15px"
            print(status)
            return self._json_response(
                {"rms_px": float(rms), "path": out_path, "warning": bool(rms > 15)})

        if path == "/api/extract-frame":
            cam = _camera(body.get("view"))
            frame_time = float(body.get("time", 1.0))
            fname = extract_temp_frame(cam, frame_time)
            if not fname:
                return self._json_response({"error": "extract failed"}, 500)
            return self._json_response({"ok": True, "filename": fname})

        if path == "/api/re-extract":
            cam = _camera(body.get("view"))
            frame_time = float(body.get("time", 1.0))
            out = os.path.join(FRAMES_DIR, cam["view"] + ".jpg")
            if os.path.exists(out):
                os.remove(out)
            ok = extract_frame(cam, frame_time)
            return self._json_response({"ok": ok})

        raise BadRequest("not found", 404)

    # ---- helpers ----

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise BadRequest("bad Content-Length")
        if length < 0:
            raise BadRequest("bad Content-Length")
        if length > MAX_BODY:
            self.close_connection = True
            if length <= 64 * MAX_BODY:          # drain so the client reads the 413, not a reset
                self.rfile.read(length)
            raise BadRequest(f"body larger than {MAX_BODY} bytes", 413)
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise BadRequest(f"malformed JSON: {e}")

    def _serve_file(self, fpath, content_type):
        if not os.path.isfile(fpath):
            raise BadRequest("not found", 404)
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_image(self, fpath):
        ext = os.path.splitext(fpath)[1].lower()
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".png": "image/png"}.get(ext, "application/octet-stream")
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = str(args[0]) if args else ""
        if "/frames/" not in msg:
            super().log_message(fmt, *args)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; 0.0.0.0 exposes the API to the LAN)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--allow-host", action="append", default=[],
                    help="extra Host header name to accept (needed for LAN access), repeatable")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    TRUSTED_HOSTS.update(h.lower() for h in args.allow_host)
    if args.host not in ("", "0.0.0.0", "::"):
        TRUSTED_HOSTS.add(args.host.lower())
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {args.host or 'all interfaces'}: any host on the network can overwrite "
              "calibrations and read video frames. Use --allow-host <name> for the Host header LAN browsers send.")
    print("Extracting calibration frames (t=1.0s)...")
    for cam in CAMERAS:
        extract_frame(cam)
    print(f"\nFrames in {FRAMES_DIR}")

    server = ThreadedServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Calibrator running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
