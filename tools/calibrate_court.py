"""
calibrate_court.py — Phase 2 of the BadmintonVR video->twin pipeline.

Maps court line intersections seen in a (static-camera) video frame to their
known regulation-court coordinates and solves the ground-plane homography
(image plane -> court XZ plane). extract_skeleton.py later pushes the player's
foot pixel through this homography to get `root_court_xz`.

Court coordinate convention (same as Unity / CourtBuilder / skeleton.json):
  origin at court center, meters, X = width (+X = right when standing at the
  camera looking down the court), Z = length (+Z = away from the camera).
  Doubles court: X in [-3.05, 3.05], Z in [-6.70, 6.70]. Camera side is -Z.

Usage:
  # interactive: click the 4 doubles corners (near-left, near-right,
  # far-right, far-left) on a frame shown in a window
  python tools/calibrate_court.py data/raw/clip.mp4

  # non-interactive: give named points directly (any 4+ from --list-points)
  python tools/calibrate_court.py data/raw/clip.mp4 --frame-time 12 \
      --point corner_nl=333,975 --point corner_nr=1497,905 \
      --point corner_fl=705,470 --point corner_fr=1150,455

Output:
  data/calib/<clip>_court.json   (points, homography, errors)
  data/calib/<clip>_overlay.png  (full court grid reprojected onto the frame —
                                  LOOK AT THIS to verify the calibration)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# --- regulation court geometry (meters) ---
XD = 3.05    # doubles sideline
XS = 2.59    # singles sideline
ZB = 6.70    # baseline
ZL = 6.70 - 0.76   # doubles long service line = 5.94
ZS = 1.98    # short service line
NET = 0.0

# Named reference points: every useful line intersection.
# Naming: n=near (camera side, -Z), f=far (+Z); l=left (-X), r=right (+X).
COURT_POINTS = {
    # doubles corners
    "corner_nl": (-XD, -ZB), "corner_nr": (XD, -ZB),
    "corner_fl": (-XD, ZB),  "corner_fr": (XD, ZB),
    # singles sideline x baseline
    "sing_bl_nl": (-XS, -ZB), "sing_bl_nr": (XS, -ZB),
    "sing_bl_fl": (-XS, ZB),  "sing_bl_fr": (XS, ZB),
    # center line x baseline
    "ctr_bl_n": (0.0, -ZB), "ctr_bl_f": (0.0, ZB),
    # doubles long service line x doubles sideline
    "lsl_nl": (-XD, -ZL), "lsl_nr": (XD, -ZL),
    "lsl_fl": (-XD, ZL),  "lsl_fr": (XD, ZL),
    # doubles long service line x singles sideline
    "lsl_sing_nl": (-XS, -ZL), "lsl_sing_nr": (XS, -ZL),
    "lsl_sing_fl": (-XS, ZL),  "lsl_sing_fr": (XS, ZL),
    # doubles long service line x center line
    "lsl_ctr_n": (0.0, -ZL), "lsl_ctr_f": (0.0, ZL),
    # short service line x sidelines
    "ssl_nl": (-XD, -ZS), "ssl_nr": (XD, -ZS),
    "ssl_fl": (-XD, ZS),  "ssl_fr": (XD, ZS),
    "ssl_sing_nl": (-XS, -ZS), "ssl_sing_nr": (XS, -ZS),
    "ssl_sing_fl": (-XS, ZS),  "ssl_sing_fr": (XS, ZS),
    # short service line x center line
    "ssl_ctr_n": (0.0, -ZS), "ssl_ctr_f": (0.0, ZS),
    # net line x sidelines (only if a line is painted under the net)
    "net_l": (-XD, NET), "net_r": (XD, NET),
}

# Single source of truth: if Unity's CourtBuilder has exported the court
# geometry (Tools > Badminton > Build Court writes data/calib/court_geometry.json),
# load the corner coordinates from there so the calibration and the Unity floor
# can never drift apart. Falls back to the constants above if the file is absent.
_GEOMETRY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "calib",
                              "court_geometry.json")


def _load_shared_geometry():
    try:
        with open(_GEOMETRY_PATH) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return
    pts = doc.get("points")
    if not isinstance(pts, dict):
        return
    loaded = 0
    for name, xz in pts.items():
        if isinstance(xz, (list, tuple)) and len(xz) == 2:
            COURT_POINTS[name] = (float(xz[0]), float(xz[1]))
            loaded += 1
    if loaded:
        print(f"  court geometry: {loaded} points from {os.path.normpath(_GEOMETRY_PATH)}")


_load_shared_geometry()

DEFAULT_CLICK_ORDER = ["corner_nl", "corner_nr", "corner_fr", "corner_fl"]

# Court line segments for the verification overlay (court XZ space).
# half="far"/"near" draws only that half (for half-court paint, e.g. a void
# deck where only SSL..baseline of one side is marked).
def court_segments(half="full"):
    zmin = 0.0 if half == "far" else -ZB
    zmax = 0.0 if half == "near" else ZB
    segs = []
    # lines along Z (sidelines + center line pieces)
    for x in (-XD, XD, -XS, XS):
        segs.append(((x, zmin), (x, zmax)))
    if half != "far":
        segs.append(((0.0, -ZB), (0.0, -ZS)))   # center line near half
    if half != "near":
        segs.append(((0.0, ZS), (0.0, ZB)))     # center line far half
    # lines along X
    zs = [z for z in (-ZB, ZB, -ZL, ZL, -ZS, ZS) if zmin <= z <= zmax]
    for z in zs:
        segs.append(((-XD, z), (XD, z)))
    segs.append(((-XD, NET), (XD, NET)))    # net line (may not be painted)
    return segs


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def grab_frame(video_path, t_sec):
    # A static calibration can also come from a single still photo (a corner
    # placement shot once); accept image files directly, ignoring --frame-time.
    if os.path.splitext(video_path)[1].lower() in _IMAGE_EXTS:
        frame = cv2.imread(video_path)
        if frame is None:
            sys.exit(f"ERROR: could not read image: {video_path}")
        return frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"ERROR: could not read frame at t={t_sec}s")
    return frame


def video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps) if n else 0.0


def refine_named(frame, named_px, do_refine):
    """Snap each clicked point to the nearest line corner (unless disabled).
    Off-frame estimates (negative or past the image size) have no pixels to snap
    to, so they are left exactly as clicked."""
    if not do_refine:
        return named_px
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for lb, pt in list(named_px.items()):
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue  # clicked in the margin — nothing to refine against
        snapped = refine_point(gray, pt)
        d = np.hypot(snapped[0] - pt[0], snapped[1] - pt[1])
        if d > 0.5:
            print(f"    refined {lb}: ({pt[0]:.0f},{pt[1]:.0f}) -> "
                  f"({snapped[0]:.1f},{snapped[1]:.1f})  [{d:.1f}px]")
        named_px[lb] = snapped
    return named_px


def refine_point(gray, pt, win=18):
    """Snap a rough click to the nearest strong corner on the white line mask."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = gray.shape
    x0, y0 = max(x - win, 0), max(y - win, 0)
    x1, y1 = min(x + win, w - 1), min(y + win, h - 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return pt
    # white court lines are much brighter than the floor
    _, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=5, qualityLevel=0.05,
                                      minDistance=5)
    if corners is None:
        return pt
    best, best_d = None, 1e9
    for c in corners.reshape(-1, 2):
        cx, cy = c[0] + x0, c[1] + y0
        d = (cx - pt[0]) ** 2 + (cy - pt[1]) ** 2
        if d < best_d:
            best, best_d = (float(cx), float(cy)), d
    return best if best is not None else pt


def click_points(frame, labels, scale, banner=None, pad=0.0):
    """Interactive: click each labeled point in order. Returns {label: (x,y)} in
    ORIGINAL image pixels.

    With pad>0 the frame is centered on a larger gray canvas so corners that fell
    OUTSIDE the frame (the camera panned past them) can still be clicked/estimated.
    Returned coords may be negative or exceed the image size — the homography is
    happy with that; only the auto-snap refine step skips off-frame points.
    """
    h, w = frame.shape[:2]
    ox = int(round(w * pad))
    oy = int(round(h * pad))
    if ox or oy:
        canvas = np.full((h + 2 * oy, w + 2 * ox, 3), 50, np.uint8)
        canvas[oy:oy + h, ox:ox + w] = frame
        cv2.rectangle(canvas, (ox, oy), (ox + w - 1, oy + h - 1), (0, 255, 0), 1)
        base_full = canvas
    else:
        base_full = frame
    disp_base = cv2.resize(base_full, None, fx=scale, fy=scale)
    clicks = []  # (label, (img_x, img_y)) in ORIGINAL image pixels

    def redraw():
        disp = disp_base.copy()
        for lb, (px, py) in clicks:
            p = (int((px + ox) * scale), int((py + oy) * scale))
            cv2.drawMarker(disp, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(disp, lb, (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        nxt = labels[len(clicks)] if len(clicks) < len(labels) else "done - press ENTER"
        cv2.putText(disp, f"click: {nxt}   (u=undo, ESC=cancel)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if banner:
            cv2.putText(disp, banner, (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
        if ox or oy:
            cv2.putText(disp, "green box = real frame edge; click in the gray margin to estimate off-frame corners",
                        (12, disp.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("calibrate_court", disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(labels):
            clicks.append((labels[len(clicks)], (x / scale - ox, y / scale - oy)))
            redraw()

    cv2.namedWindow("calibrate_court", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calibrate_court", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("cancelled")
        if k in (ord("u"), ord("U")) and clicks:
            clicks.pop()
            redraw()
        if k in (13, 10) and len(clicks) == len(labels):
            break
    cv2.destroyAllWindows()
    return dict(clicks)


def solve_homography(named_px, refine_src=None):
    """named_px: {label: (px, py)}. Returns (H_img_to_court, per-point errors)."""
    labels = list(named_px.keys())
    img_pts = np.array([named_px[lb] for lb in labels], dtype=np.float64)
    court_pts = np.array([COURT_POINTS[lb] for lb in labels], dtype=np.float64)

    if len(labels) < 4:
        sys.exit("ERROR: need at least 4 points for a homography")

    # least-squares over ALL points: with a handful of hand-picked points we
    # want lens-distortion errors averaged, not points discarded (bad points
    # show up in the reported per-point errors instead)
    H, _ = cv2.findHomography(img_pts, court_pts, 0)
    if H is None:
        sys.exit("ERROR: homography solve failed (degenerate points?)")

    # reprojection errors
    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_m = np.linalg.norm(proj - court_pts, axis=1)
    errors = {lb: float(e) for lb, e in zip(labels, err_m)}
    return H, errors


def draw_overlay(frame, H, named_px, out_png, half="full"):
    Hinv = np.linalg.inv(H)
    img = frame.copy()

    def to_img(cx, cz):
        p = cv2.perspectiveTransform(
            np.array([[[cx, cz]]], dtype=np.float64), Hinv).reshape(2)
        return int(round(p[0])), int(round(p[1]))

    h, w = img.shape[:2]
    for (a, b) in court_segments(half):
        # sample the segment so perspective curvature of any lens error shows
        n = 40
        pts = []
        for i in range(n + 1):
            t = i / n
            cx = a[0] + (b[0] - a[0]) * t
            cz = a[1] + (b[1] - a[1]) * t
            px, py = to_img(cx, cz)
            if -2 * w < px < 3 * w and -2 * h < py < 3 * h:
                pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                          (0, 255, 255), 2, cv2.LINE_AA)

    for lb, (px, py) in named_px.items():
        p = (int(round(px)), int(round(py)))
        cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, lb, (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imwrite(out_png, img)
    print(f"  overlay -> {out_png}   (open it: yellow grid must sit on the real lines)")


def calibrate_multi(args):
    """Moving-camera calibration: click the same corners at several timestamps.

    Writes a schema-2.0 json with a list of keyframes (time + clicked corners +
    per-keyframe homography). extract_skeleton.py interpolates the corner pixels
    between keyframes for every frame, so the mapping follows a panning/drifting
    camera instead of baking in one static homography.
    """
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    for lb in labels:
        if lb not in COURT_POINTS:
            sys.exit(f"ERROR: unknown point '{lb}' (use --list-points)")

    if args.times:
        times = [float(t) for t in args.times.split(",") if t.strip()]
    else:
        n = max(args.multi, 2)
        dur = video_duration(args.video)
        if dur <= 0:
            sys.exit("ERROR: could not determine clip duration; pass --times explicitly.")
        pad = min(0.3, dur * 0.03)
        times = list(np.linspace(pad, dur - pad, n))
    times = sorted(times)
    print(f"Multi-keyframe calibration: {len(times)} timestamps "
          f"[{', '.join(f'{t:.2f}s' for t in times)}]")
    print(f"  click these {len(labels)} corners at EACH timestamp, same order: {', '.join(labels)}")

    base = os.path.basename(args.video)
    stem = base
    while os.path.splitext(stem)[1]:
        stem = os.path.splitext(stem)[0]
    out_path = args.out or os.path.join("data", "calib", stem + "_court.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay_dir = os.path.join(os.path.dirname(out_path), stem + "_multi_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    image_size = None
    keyframes = []
    for k, t in enumerate(times):
        frame = grab_frame(args.video, t)
        if image_size is None:
            image_size = [frame.shape[1], frame.shape[0]]
        banner = f"keyframe {k + 1}/{len(times)}  @ t={t:.2f}s"
        named_px = click_points(frame, labels, args.scale, banner=banner, pad=args.pad)
        named_px = refine_named(frame, named_px, not args.no_refine)
        H, errors = solve_homography(named_px)
        worst = max(errors.values())
        flag = "  <-- CHECK" if worst > 0.15 else ""
        print(f"  keyframe {k + 1}/{len(times)} @ {t:.2f}s: worst reproj {worst:.3f} m{flag}")
        overlay_png = os.path.join(overlay_dir, f"kf{k + 1:02d}_t{t:.1f}.png")
        draw_overlay(frame, H, named_px, overlay_png, half=args.half)
        keyframes.append({
            "frame_time": round(float(t), 3),
            "points": {lb: {"px": [round(p[0], 2), round(p[1], 2)],
                            "court_xz": list(COURT_POINTS[lb])}
                       for lb, p in named_px.items()},
            "reprojection_error_m": {lb: round(e, 4) for lb, e in errors.items()},
            "homography_img_to_court": H.tolist(),
        })

    doc = {
        "schema_version": "2.0",
        "video": base,
        "multi_keyframe": True,
        "image_size": image_size,
        "convention": "court XZ, meters, origin center; +Z away from camera, +X right of camera; camera side is -Z",
        "half": args.half,
        "keyframes": keyframes,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  wrote {out_path}  ({len(keyframes)} keyframes)")
    print(f"  overlays -> {overlay_dir}  (open each: yellow grid must sit on the paint)")


def main():
    ap = argparse.ArgumentParser(description="Solve the court ground-plane homography for a static-camera clip.")
    ap.add_argument("video")
    ap.add_argument("--frame-time", type=float, default=1.0, help="seconds into the clip to grab the calibration frame")
    ap.add_argument("--point", action="append", default=[], metavar="NAME=PX,PY",
                    help="named image point, e.g. corner_nl=333,975 (repeatable; skips the click UI)")
    ap.add_argument("--labels", default=",".join(DEFAULT_CLICK_ORDER),
                    help="comma-separated point names to click in interactive mode")
    ap.add_argument("--list-points", action="store_true", help="print all valid point names and exit")
    ap.add_argument("--no-refine", action="store_true", help="don't snap points to detected line corners")
    ap.add_argument("--half", choices=["full", "near", "far"], default="full",
                    help="how much court is painted/visible (controls the overlay grid; recorded in the json)")
    ap.add_argument("--scale", type=float, default=0.8, help="display scale for the click window")
    ap.add_argument("--multi", type=int, default=0, metavar="N",
                    help="moving-camera mode: click the corners at N timestamps; "
                         "extract_skeleton interpolates the corners between them per frame")
    ap.add_argument("--times", default=None, metavar="t1,t2,...",
                    help="explicit keyframe times (seconds) for --multi; "
                         "default = N evenly spaced across the clip")
    ap.add_argument("--pad", type=float, default=0.0, metavar="FRAC",
                    help="gray margin around the frame (fraction of w/h) so you can "
                         "click corners that panned OFF-screen, e.g. 0.4")
    ap.add_argument("--out", default=None, help="output json (default data/calib/<name>_court.json)")
    args = ap.parse_args()

    if args.list_points:
        for k, v in COURT_POINTS.items():
            print(f"  {k:14s} -> X={v[0]:+.2f}, Z={v[1]:+.2f}")
        return

    if args.multi or args.times:
        return calibrate_multi(args)

    frame = grab_frame(args.video, args.frame_time)

    if args.point:
        named_px = {}
        for spec in args.point:
            name, _, xy = spec.partition("=")
            if name not in COURT_POINTS:
                sys.exit(f"ERROR: unknown point '{name}' (use --list-points)")
            x, _, y = xy.partition(",")
            named_px[name] = (float(x), float(y))
    else:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
        for lb in labels:
            if lb not in COURT_POINTS:
                sys.exit(f"ERROR: unknown point '{lb}' (use --list-points)")
        named_px = click_points(frame, labels, args.scale)

    if not args.no_refine:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for lb, pt in list(named_px.items()):
            snapped = refine_point(gray, pt)
            d = np.hypot(snapped[0] - pt[0], snapped[1] - pt[1])
            if d > 0.5:
                print(f"  refined {lb}: ({pt[0]:.0f},{pt[1]:.0f}) -> "
                      f"({snapped[0]:.1f},{snapped[1]:.1f})  [{d:.1f}px]")
            named_px[lb] = snapped

    H, errors = solve_homography(named_px)
    worst = max(errors.values())
    print("  reprojection error per point (meters):")
    for lb, e in errors.items():
        print(f"    {lb:14s} {e:.3f}")
    if worst > 0.15:
        print(f"  WARNING: worst error {worst:.2f} m - a point is probably misplaced.")

    base = os.path.basename(args.video)
    stem = base
    while os.path.splitext(stem)[1]:
        stem = os.path.splitext(stem)[0]
    out_path = args.out or os.path.join("data", "calib", stem + "_court.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    doc = {
        "schema_version": "1.0",
        "video": base,
        "frame_time": args.frame_time,
        "image_size": [frame.shape[1], frame.shape[0]],
        "convention": "court XZ, meters, origin center; +Z away from camera, +X right of camera; camera side is -Z",
        "half": args.half,
        "points": {lb: {"px": [round(p[0], 2), round(p[1], 2)],
                        "court_xz": list(COURT_POINTS[lb])}
                   for lb, p in named_px.items()},
        "reprojection_error_m": {lb: round(e, 4) for lb, e in errors.items()},
        "homography_img_to_court": H.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  wrote {out_path}")

    draw_overlay(frame, H, named_px,
                 os.path.splitext(out_path)[0].replace("_court", "") + "_overlay.png",
                 half=args.half)


if __name__ == "__main__":
    main()
