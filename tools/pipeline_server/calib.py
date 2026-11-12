"""Tab 1 Calibrate of the live server: the court calibrator (tools/calibrator.html) served under /calib, driven by the
pose24 clip registry instead of serve_calibrator.py's hard-coded camera table, plus the sync file for fixed cameras.

    /calibrate                  the calibrator page (its /api and /frames paths rewritten to /calib/...)
    /calib/api/cameras          every camera of every full registry clip (?clip= one clip): video, frame, calibration
    /calib/api/source-cameras   v0.8.6: only the cameras the Source step chose (ticked angles / live cameras in use)
    /calib/thumbs/<view>.jpg    a small copy of the primary frame
    /calib/api/clips            {group: {clip, mode}}         /calib/api/court-points   {name: [x, z]}
    /calib/frames/<file>        a frame image (grabbed with OpenCV at t = 1 s the first time; WSL has no ffmpeg)
    POST /calib/api/auto-fov    best field of view for the clicked points
    POST /calib/api/calibrate   best field of view + PnP -> data/calib/<view>_cam.json (the previous one is archived first)
    POST /calib/api/extract-frame, /calib/api/re-extract     another frame number (or time) / a new primary frame
    GET|POST /calib/api/seeds   offline-build seeds (the live stream needs none)
    /api/sync, POST /api/sync   camera offsets of a clip: data/calib/<clip>_sync.json, made from its offline build
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from analysis import multiview as mv
from analysis.live import config as live_config
from analysis.pose24 import config as p24config

SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
DEFAULT_FOV = 80.0
RMS_WARN_PX = 15.0


def _court_points():
    from calibrate_court import COURT_POINTS     # loads data/calib/court_geometry.json
    return COURT_POINTS


LIVE_GRAB_MS = 10000             # a live camera gets this long to open and to send its frame


class Calib:
    def __init__(self, repo, unity, input_fn=None, input_mtime_fn=None):
        self.repo, self.unity = str(repo), str(unity)
        self.input_mtime_fn = input_mtime_fn or (lambda: 0.0)       # when the source was last saved
        self.input_fn = input_fn or (lambda: {"kind": "video"})     # v0.8.4: the Source tab's choice
        self.calib_dir = os.path.join(self.repo, "data", "calib")
        self.frames_dir = os.path.join(self.calib_dir, "frames")

    # ------------------------------------------------------------------ cameras
    def registry(self):
        return {c: e for c, e in p24config.load_registry().items() if "base" not in e}

    def entry(self, view, video, clip, mode, group=None, url=None):
        """One camera for the page: its frame source, the saved calibration (points, where they land, FOV, RMS)."""
        calib = os.path.join(self.calib_dir, f"{view}_cam.json")
        cam = {"view": view, "video": url or video, "group": group or clip, "clip": clip, "mode": mode,
               "camera": view.rsplit("_", 1)[0] if url is None else view, "phone": "", "fov_deg": DEFAULT_FOV,
               "has_video": bool(url) or (bool(video) and os.path.exists(os.path.join(self.repo, video))),
               "has_frame": os.path.exists(os.path.join(self.frames_dir, f"{view}.jpg")),
               "calibrated": os.path.exists(calib)}
        if url:
            cam["input"] = "camera"
        if cam["calibrated"]:
            try:
                with open(calib, "r", encoding="utf-8") as fh:
                    cd = json.load(fh)
                cam.update(rms=cd.get("rms_px"), saved_fov=cd.get("fov_deg"), fov_deg=cd.get("fov_deg") or DEFAULT_FOV,
                           saved_points={n: i["px"] for n, i in (cd.get("court_points") or {}).items()},
                           saved_proj=self.reprojected(cd), made=cd.get("made"))
            except (OSError, ValueError, KeyError, TypeError):
                cam["calibrated"] = False
        return cam

    def cameras(self, clip=None):
        """Every camera of every recorded clip (or of one clip), as the stand-alone calibrator lists them."""
        out = []
        for c, e in self.registry().items():
            if clip and c != clip:
                continue
            for view in e.get("views") or []:
                out.append(self.entry(view, (e.get("videos") or {}).get(view), c, e.get("mode", "singles")))
        return out

    def source_cameras(self):
        """v0.8.6: the cameras the Source step chose - the clip's ticked angles (recorded videos), or the live broadcast
        cameras in use (named by the user; frames from their streams, calibration data/calib/<name>_cam.json)."""
        inp = self.input_fn() or {}
        clip = inp.get("clip")
        e = self.registry().get(clip) or {}
        mode = e.get("mode", "singles")
        if inp.get("kind") == "camera":
            cams = inp.get("cameras") or []
            if isinstance(cams, dict):
                cams = [{"name": k, "url": v, "use": True} for k, v in cams.items()]
            return [self.entry(c["name"], None, clip, mode, group="broadcast", url=live_config.with_rotate(c["url"], c.get("rotate")))
                    for c in cams if c.get("use", True) and c.get("url") and SAFE.match(c.get("name") or "")]
        chosen = inp.get("views") or self.live_views(clip) or e.get("views") or []
        views = [v for v in (e.get("views") or []) if v in chosen]
        return [self.entry(v, (e.get("videos") or {}).get(v), clip, mode) for v in views]

    def live_views(self, clip):
        """The clip's default live cameras (configs/live/<clip>.yaml), used until the Source step picks angles."""
        p = os.path.join(self.repo, "configs", "live", f"{clip}.yaml")
        if not clip or not os.path.exists(p):
            return []
        import yaml
        with open(p, "r", encoding="utf-8") as fh:
            return list((yaml.safe_load(fh) or {}).get("views") or [])

    @staticmethod
    def reprojected(cd):
        """{point: [x, y, err_px]} where each clicked court point lands with a saved calibration (shown on the frame)."""
        pts = cd.get("court_points") or {}
        if not pts or "K" not in cd:
            return {}
        names = list(pts)
        world = np.array([[pts[n]["court_xz"][0], 0.0, pts[n]["court_xz"][1]] for n in names], dtype=np.float64)
        P = mv.projection_matrix(np.array(cd["K"]), np.array(cd["R"]), np.array(cd["t"]))
        hom = (P @ np.hstack([world, np.ones((len(world), 1))]).T).T
        uv = hom[:, :2] / hom[:, 2:3]
        clicks = np.array([pts[n]["px"] for n in names], dtype=np.float64)
        err = np.linalg.norm(uv - clicks, axis=1)
        return {n: [round(float(a), 1), round(float(b), 1), round(float(e), 2)] for n, (a, b), e in zip(names, uv, err)}

    def camera(self, view):
        cam = next((c for c in self.cameras() if c["view"] == view), None) or \
            next((c for c in self.source_cameras() if c["view"] == view), None)
        if cam is None:
            raise HTTPException(400, f"unknown camera {view!r}")
        return cam

    def thumb_path(self, view, width=360):
        """A small copy of the camera's primary frame (the Source step's cards)."""
        if not SAFE.match(view or ""):
            raise HTTPException(400, "bad camera name")
        src = self.frame_path(f"{view}.jpg")
        tdir = os.path.join(self.frames_dir, "thumbs")
        dst = os.path.join(tdir, f"{view}.jpg")
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            img = cv2.imread(src)
            if img is None:
                raise HTTPException(404, "frame not readable")
            h, w = img.shape[:2]
            os.makedirs(tdir, exist_ok=True)
            cv2.imwrite(dst, cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        return dst

    # ------------------------------------------------------------------ frames
    def grab(self, cam, t_s, name, frame=None):
        """One video frame (at t_s seconds, or frame number `frame` of this camera's video) -> data/calib/frames/<name>
        (JPEG). OpenCV, so no ffmpeg is needed."""
        if not cam.get("has_video"):
            raise HTTPException(404, f"video of {cam['view']} not found ({cam.get('video')})")
        os.makedirs(self.frames_dir, exist_ok=True)
        is_url = "://" in str(cam["video"])
        src, rot = live_config.split_rotate(cam["video"]) if is_url else (cam["video"], 0)
        cap = (cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, LIVE_GRAB_MS,
                                                               cv2.CAP_PROP_READ_TIMEOUT_MSEC, LIVE_GRAB_MS])
               if is_url else cv2.VideoCapture(os.path.join(self.repo, cam["video"])))
        try:
            if is_url:                                    # a live camera: the frame it sends now
                pass
            elif frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame)))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_s)) * 1000.0)
            ok, img = cap.read()
        finally:
            cap.release()
        if not ok or img is None:
            at = f"frame {frame}" if frame is not None else f"{t_s} s"
            raise HTTPException(500, f"could not read {cam['view']} at {at}")
        img = live_config.upright(img, rot)                   # an upside-down camera: calibrated the right way up
        cv2.imwrite(os.path.join(self.frames_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return name

    def frame_path(self, fname):
        if not SAFE.match(fname or "") or os.path.splitext(fname)[1].lower() not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(400, "bad frame name")
        p = os.path.join(self.frames_dir, fname)
        if not os.path.exists(p):
            view = os.path.splitext(fname)[0]
            if "_add_" in view:
                raise HTTPException(404, "frame not found")
            self.grab(self.camera(view), 1.0, fname)              # first visit: the primary frame at t = 1 s
        elif "_add_" not in os.path.splitext(fname)[0] and self._live_frame_is_old(os.path.splitext(fname)[0], p):
            try:                                                  # cam1 is another camera since the source was saved
                self.grab(self.camera(os.path.splitext(fname)[0]), 1.0, fname)
            except HTTPException:                                 # camera off: the old picture is better than none
                pass
        return p

    def _live_frame_is_old(self, view, path) -> bool:
        """A live camera's saved frame from before the source was last saved (names like cam1 are reused)."""
        try:
            if os.path.getmtime(path) >= float(self.input_mtime_fn()):
                return False
            return "://" in str(self.camera(view).get("video"))
        except (OSError, HTTPException, ValueError, TypeError):
            return False

    def frame_size(self, view):
        img = cv2.imread(self.frame_path(f"{view}.jpg"))
        if img is None:
            raise HTTPException(404, "frame not readable")
        return img.shape[1], img.shape[0]

    # ------------------------------------------------------------------ solve
    @staticmethod
    def points(body):
        cp = _court_points()
        pts = body.get("points")
        if not isinstance(pts, list) or len(pts) < 4:
            raise HTTPException(400, "need at least 4 clicked court points")
        names = []
        for p in pts:
            if not isinstance(p, dict) or p.get("name") not in cp:
                raise HTTPException(400, f"unknown court point {p.get('name') if isinstance(p, dict) else p!r}")
            names.append(p["name"])
        image_px = np.array([[float(p["px"]), float(p["py"])] for p in pts], dtype=np.float64)
        court_xz = np.array([cp[n] for n in names], dtype=np.float64)
        return names, image_px, court_xz

    def auto_fov(self, body):
        view = self.camera(body.get("view"))["view"]
        _, image_px, court_xz = self.points(body)
        w, h = self.frame_size(view)
        fov, rms = mv.find_best_fov(w, h, court_xz, image_px, on=body.get("fov_on", "long"))
        return {"fov_deg": float(fov), "rms_px": float(rms)}

    def calibrate(self, body):
        """Save a calibration. Without fov_deg (the page's Save) the best field of view is found first."""
        cam = self.camera(body.get("view"))
        view = cam["view"]
        fov_on = body.get("fov_on", "long")
        fov = body.get("fov_deg")
        if fov is not None:
            try:
                fov = float(fov)
            except (TypeError, ValueError):
                raise HTTPException(400, "fov_deg must be a number")
        names, image_px, court_xz = self.points(body)
        w, h = self.frame_size(view)
        try:
            s = mv.solve_camera(w, h, court_xz, image_px, fov_deg=fov, on=fov_on)
        except (RuntimeError, ValueError, cv2.error) as e:
            raise HTTPException(400, f"calibration failed: {e}")
        K, R, t, rms = s["K"], s["R"], s["t"], s["rms_px"]
        cp = _court_points()
        out = {"view": view, "source": os.path.basename(cam["video"] or ""), "width": w, "height": h,
               "fov_deg": s["fov_deg"], "fov_on": fov_on, "fov_auto": fov is None, "focal_px": float(K[0, 0]),
               "K": K.tolist(), "R": R.tolist(), "t": t.tolist(), "rms_px": rms,
               "court_points": {n: {"px": [float(x), float(y)], "court_xz": list(cp[n]), "err_px": round(float(e), 2)}
                                for n, (x, y), e in zip(names, image_px, s["err_px"])},
               "made": time.strftime("%Y-%m-%d %H:%M"), "tool": "pipeline_server calibrate"}
        path = os.path.join(self.calib_dir, f"{view}_cam.json")
        archived = self.archive(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        worst = int(np.argmax(s["err_px"]))
        return {"rms_px": rms, "fov_deg": s["fov_deg"], "path": os.path.relpath(path, self.repo),
                "warning": rms > RMS_WARN_PX, "archived": archived, "notes": s["notes"],
                "points": [{"name": n, "proj": [float(a), float(b)], "err_px": float(e)}
                           for n, (a, b), e in zip(names, s["proj"], s["err_px"])],
                "worst": {"name": names[worst], "err_px": float(s["err_px"][worst])}}

    def archive(self, path):
        """Copy a file about to be replaced into the housekeeping archive (gitignored); returns its new path."""
        if not os.path.exists(path):
            return None
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        dst_dir = os.path.join(self.repo, "_archive", f"data-{stamp[:10]}", "calib")
        os.makedirs(dst_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(path))
        dst = os.path.join(dst_dir, f"{base}_{stamp}{ext}")
        shutil.copy2(path, dst)
        return os.path.relpath(dst, self.repo)

    # ------------------------------------------------------------------ seeds (offline builds only)
    def seeds_path(self, clip):
        if not SAFE.match(clip or ""):
            raise HTTPException(400, "bad clip id")
        return os.path.join(self.calib_dir, f"{clip}_seeds.json")

    # ------------------------------------------------------------------ sync file
    def sync_status(self):
        out = []
        for clip, e in self.registry().items():
            p = os.path.join(self.calib_dir, f"{clip}_sync.json")
            item = {"clip": clip, "views": e.get("views") or [], "has_build": self.offline_build(clip) is not None,
                    "sync": None}
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        item["sync"] = json.load(fh)
                except (OSError, ValueError):
                    pass
            out.append(item)
        return out

    def offline_build(self, clip):
        """The clip's offline pose24 build: StreamingAssets, data/analysis, or the archive folder the analysis files
        were moved to (~/OneDrive/Desktop/archive on the laptop). None if there is none."""
        name = f"{clip}_pose24.analysis.json"
        for d in (os.path.join(self.unity, "Assets", "StreamingAssets", "analysis"), os.path.join(self.repo, "data", "analysis"),
                  os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "archive")):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    def make_sync(self, clip):
        if clip not in self.registry():
            raise HTTPException(400, f"no registry clip {clip!r}")
        doc = self.offline_build(clip)
        if doc is None:
            raise HTTPException(404, f"{clip} has no offline build ({clip}_pose24.analysis.json) to take the offsets from")
        with open(doc, "r", encoding="utf-8") as fh:
            meta = json.load(fh).get("meta") or {}
        pvar = meta.get("pose_variant") or {}
        offs = pvar.get("offsets")
        if not offs:
            raise HTTPException(400, f"{os.path.basename(doc)} records no camera offsets")
        out = {"clip": clip, "time_ref": pvar.get("time_ref_view"), "offsets": {v: float(o) for v, o in offs.items()},
               "source": os.path.basename(doc), "made": time.strftime("%Y-%m-%d %H:%M")}
        path = os.path.join(self.calib_dir, f"{clip}_sync.json")
        archived = self.archive(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        return {**out, "archived": archived}


def _frame_no(v):
    try:
        f = int(v)
    except (TypeError, ValueError):
        raise HTTPException(400, "frame must be a whole number")
    if f < 0:
        raise HTTPException(400, "frame must be 0 or more")
    return f


def router(calib: Calib, tools_dir) -> APIRouter:
    r = APIRouter()
    page = os.path.join(str(tools_dir), "calibrator.html")

    @r.get("/calibrate", response_class=HTMLResponse)
    def calibrator_page():
        with open(page, "r", encoding="utf-8") as fh:
            html = fh.read()
        # the page calls /api/... and /frames/...: here they live under /calib/
        html = html.replace("'/api/", "'/calib/api/").replace("'/frames/", "'/calib/frames/")
        return html.replace("<head>", "<head><script>window.EMBED_LIVE_SERVER = true;</script>", 1)

    @r.get("/calib/api/cameras")
    def cameras(clip: str = ""):
        return calib.cameras(clip or None)

    @r.get("/calib/api/source-cameras")
    def source_cameras():
        return calib.source_cameras()

    @r.get("/calib/thumbs/{view}.jpg")
    def thumb(view: str):
        return FileResponse(calib.thumb_path(view), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @r.get("/calib/api/clips")
    def clips():
        return {c: {"clip": c, "mode": e.get("mode", "singles")} for c, e in calib.registry().items()}

    @r.get("/calib/api/court-points")
    def court_points():
        return {k: list(v) for k, v in _court_points().items()}

    @r.get("/calib/frames/{fname}")
    def frame(fname: str):
        return FileResponse(calib.frame_path(fname), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @r.post("/calib/api/auto-fov")
    async def auto_fov(request: Request):
        return await run_in_threadpool(calib.auto_fov, await request.json())

    @r.post("/calib/api/calibrate")
    async def calibrate(request: Request):
        return await run_in_threadpool(calib.calibrate, await request.json())

    @r.post("/calib/api/extract-frame")
    async def extract_frame(request: Request):
        b = await request.json()
        cam = calib.camera(b.get("view"))
        if b.get("frame") is not None:
            f = _frame_no(b["frame"])
            return {"ok": True, "filename": await run_in_threadpool(
                lambda: calib.grab(cam, None, f"{cam['view']}_add_f{f}.jpg", frame=f))}
        t = float(b.get("time", 1.0))
        name = f"{cam['view']}_add_{str(t).replace('.', '_').replace('-', 'neg')}.jpg"
        return {"ok": True, "filename": await run_in_threadpool(calib.grab, cam, t, name)}

    @r.post("/calib/api/re-extract")
    async def re_extract(request: Request):
        b = await request.json()
        cam = calib.camera(b.get("view"))
        if b.get("frame") is not None:
            f = _frame_no(b["frame"])
            await run_in_threadpool(lambda: calib.grab(cam, None, f"{cam['view']}.jpg", frame=f))
        else:
            await run_in_threadpool(calib.grab, cam, float(b.get("time", 1.0)), f"{cam['view']}.jpg")
        return {"ok": True}

    @r.get("/calib/api/seeds/{clip}")
    def seeds(clip: str):
        p = calib.seeds_path(clip)
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @r.post("/calib/api/seeds")
    async def save_seeds(request: Request):
        b = await request.json()
        p = calib.seeds_path(b.get("clip"))
        calib.archive(p)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(b, fh, indent=2)
        return {"ok": True, "path": os.path.relpath(p, calib.repo)}

    @r.get("/api/sync")
    def sync_status():
        return calib.sync_status()

    @r.post("/api/sync")
    async def make_sync(request: Request):
        return calib.make_sync((await request.json()).get("clip"))

    return r
