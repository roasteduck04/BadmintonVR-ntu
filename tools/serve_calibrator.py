"""Browser-based camera calibrator for multi-view court PnP.

Extracts one frame per clip at t=1s, serves a browser UI for clicking court
points, computes PnP calibration on save.

Run:  tools\\.venv\\Scripts\\python.exe tools\\serve_calibrator.py
Open: http://localhost:8001
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

import cv2
import numpy as np

from calibrate_court import COURT_POINTS
from analysis import multiview as mv

FRAMES_DIR = os.path.join(ROOT, "data", "calib", "frames")
CALIB_DIR = os.path.join(ROOT, "data", "calib")


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


class Handler(SimpleHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/calibrator.html"):
            return self._serve_file(
                os.path.join(TOOLS_DIR, "calibrator.html"), "text/html")
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
                self.send_error(400, "invalid clip id")
                return
            if os.path.isfile(seeds_path):
                with open(seeds_path, encoding="utf-8") as f:
                    return self._json_response(json.load(f))
            return self._json_response({})
        if path.startswith("/frames/"):
            fname = unquote(path[len("/frames/"):])
            fpath = _safe_path(FRAMES_DIR, fname)
            if fpath is None:
                self.send_error(400, "invalid filename")
                return
            if os.path.isfile(fpath):
                return self._serve_image(fpath)
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/seeds":
            body = self._read_body()
            clip_id = body.get("clip")
            if not clip_id:
                return self._json_response({"error": "missing clip"}, 400)
            seeds_path = _safe_path(CALIB_DIR, f"{clip_id}_seeds.json")
            if seeds_path is None:
                return self._json_response({"error": "invalid clip id"}, 400)
            os.makedirs(os.path.dirname(seeds_path), exist_ok=True)
            with open(seeds_path, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2)
            print(f"wrote seeds: {seeds_path}")
            return self._json_response({"ok": True, "path": seeds_path})

        if path == "/api/auto-fov":
            body = self._read_body()
            view = body["view"]
            fov_on = body.get("fov_on", "long")
            points = body["points"]
            if len(points) < 4:
                return self._json_response(
                    {"error": "need >= 4 points"}, 400)
            frame_path = os.path.join(FRAMES_DIR, view + ".jpg")
            img = cv2.imread(frame_path)
            if img is None:
                return self._json_response(
                    {"error": "frame not found"}, 404)
            h, w = img.shape[:2]
            names = [p["name"] for p in points]
            image_px = np.array(
                [[p["px"], p["py"]] for p in points], dtype=np.float64)
            court_xz = np.array(
                [COURT_POINTS[n] for n in names], dtype=np.float64)
            best_fov, best_rms = mv.find_best_fov(
                w, h, court_xz, image_px, on=fov_on)
            print(f"auto-fov {view}: best={best_fov:.1f}° rms={best_rms:.2f}px")
            return self._json_response(
                {"fov_deg": best_fov, "rms_px": best_rms})

        if path == "/api/calibrate":
            body = self._read_body()
            view = body["view"]
            fov_deg = float(body["fov_deg"])
            fov_on = body.get("fov_on", "long")
            points = body["points"]
            if len(points) < 4:
                return self._json_response(
                    {"error": "need >= 4 points"}, 400)
            frame_path = os.path.join(FRAMES_DIR, view + ".jpg")
            img = cv2.imread(frame_path)
            if img is None:
                return self._json_response(
                    {"error": "frame not found"}, 404)
            h, w = img.shape[:2]
            names = [p["name"] for p in points]
            image_px = np.array(
                [[p["px"], p["py"]] for p in points], dtype=np.float64)
            court_xz = np.array(
                [COURT_POINTS[n] for n in names], dtype=np.float64)
            K = mv.intrinsics_from_fov(w, h, fov_deg, on=fov_on)
            R, t_vec = mv.solve_extrinsics(court_xz, image_px, K)
            rms = mv.reprojection_rms(court_xz, image_px, K, R, t_vec)
            cam_info = next(c for c in CAMERAS if c["view"] == view)
            out = {
                "view": view,
                "source": os.path.basename(cam_info["video"]),
                "width": w, "height": h,
                "fov_deg": fov_deg, "fov_on": fov_on,
                "focal_px": float(K[0, 0]),
                "K": K.tolist(), "R": R.tolist(), "t": t_vec.tolist(),
                "rms_px": float(rms),
                "court_points": {
                    p["name"]: {
                        "px": [p["px"], p["py"]],
                        "court_xz": list(COURT_POINTS[p["name"]]),
                    }
                    for p in points
                },
            }
            path = os.path.join(CALIB_DIR, f"{view}_cam.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            status = f"wrote {path}: rms {rms:.2f}px"
            if rms > 15:
                status += " WARNING: rms > 15px"
            print(status)
            return self._json_response(
                {"rms_px": float(rms), "path": path, "warning": rms > 15})

        if path == "/api/extract-frame":
            body = self._read_body()
            view = body["view"]
            frame_time = float(body.get("time", 1.0))
            cam = next((c for c in CAMERAS if c["view"] == view), None)
            if not cam:
                return self._json_response({"error": "unknown view"}, 404)
            fname = extract_temp_frame(cam, frame_time)
            if not fname:
                return self._json_response({"error": "extract failed"}, 500)
            return self._json_response({"ok": True, "filename": fname})

        if path == "/api/re-extract":
            body = self._read_body()
            view = body["view"]
            frame_time = float(body.get("time", 1.0))
            cam = next((c for c in CAMERAS if c["view"] == view), None)
            if not cam:
                return self._json_response({"error": "unknown view"}, 404)
            out = os.path.join(FRAMES_DIR, cam["view"] + ".jpg")
            if os.path.exists(out):
                os.remove(out)
            ok = extract_frame(cam, frame_time)
            return self._json_response({"ok": ok})

        self.send_error(404)

    # ---- helpers ----

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _serve_file(self, fpath, content_type):
        if not os.path.isfile(fpath):
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", os.path.getsize(fpath))
        self.end_headers()
        with open(fpath, "rb") as f:
            self.wfile.write(f.read())

    def _serve_image(self, fpath):
        self.send_response(200)
        ext = os.path.splitext(fpath)[1].lower()
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".png": "image/png"}.get(ext, "application/octet-stream")
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", os.path.getsize(fpath))
        self.end_headers()
        with open(fpath, "rb") as f:
            self.wfile.write(f.read())

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = str(args[0]) if args else ""
        if "/frames/" not in msg:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print("Extracting calibration frames (t=1.0s)...")
    for cam in CAMERAS:
        extract_frame(cam)
    print(f"\nFrames in {FRAMES_DIR}")

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    port = 8001
    server = ThreadedServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Calibrator running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
