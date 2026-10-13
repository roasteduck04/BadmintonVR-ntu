"""
check_position.py — diagnose WHERE a court-position mismatch comes from.

The pipeline has layers; each can be wrong independently:
  paint -> clicked corners -> homography -> foot pixel -> root_court_xz -> Unity
This tool tests the video-side layers with EVIDENCE, no hardcoding:

  1) default: back-projects the extracted trajectory onto the video
     (a red dot must sit on the player's feet in every panel) and draws a
     top-down court map with the same trajectory. If the dot rides the feet
     but the map looks wrong, the problem is the calibration, not extraction.

  2) --probe: interactive. Click ANY floor spot in the frame -> prints its
     court XZ through the homography; consecutive clicks print the real-world
     distance between them. Click known spots (line intersections, or two ends
     of a painted line whose length you know) and check the numbers:
       - SSL -> baseline is 4.72 m, doubles width is 6.10 m, singles 5.18 m
       - a line crossing should print its --list-points coordinate
     If probed distances are off, the PAINT is not regulation-sized (or the
     homography is biased) — that shifts every position.

Compare the top-down map with Unity's floor (Tools > Badminton > Debug >
Draw Clip Path draws the same path in Unity): if the two maps agree, the
mismatch is NOT in Unity; if they disagree, it is.

Usage:
  tools/.venv/Scripts/python tools/check_position.py data/raw/test_5.mp4
  tools/.venv/Scripts/python tools/check_position.py data/raw/test_5.mp4 --probe
  tools/.venv/Scripts/python tools/check_position.py data/raw/test_5.mp4 --video

Outputs (default mode), in data/calib/:
  <clip>_check_sheet.png    panels: video frame + foot dot + trail + inset map
  <clip>_check_topdown.png  the full trajectory on a to-scale court map
  <clip>_check.mp4          (with --video) the same, every frame
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# reuse the court model + shared-geometry loading (court_geometry.json)
from calibrate_court import COURT_POINTS, court_segments, grab_frame, XD, ZB, ZS


# ---------------------------------------------------------------- loading

def default_paths(video_path):
    stem = os.path.basename(video_path)
    while os.path.splitext(stem)[1]:
        stem = os.path.splitext(stem)[0]
    return (os.path.join("data", "calib", stem + "_court.json"),
            os.path.join("data", "skeleton", stem + ".json"),
            stem)


def load_calib(path):
    with open(path) as f:
        return json.load(f)


def build_H_series(doc, times):
    """Homography (img->court) for each time in `times`. Returns (H[N,3,3],
    Hinv[N,3,3]). v1 calib: one constant homography broadcast to all frames.
    v2 (moving camera): interpolate the clicked corner PIXELS between keyframes
    and solve per time — same math extract_skeleton uses, so the back-projected
    dot lands where the extractor placed it."""
    times = np.asarray(times, dtype=np.float64)
    N = len(times)
    if "keyframes" not in doc:
        H = np.array(doc["homography_img_to_court"], dtype=np.float64)
        Hs = np.repeat(H[None, :, :], N, axis=0)
    else:
        kfs = sorted(doc["keyframes"], key=lambda k: k["frame_time"])
        kt = np.array([k["frame_time"] for k in kfs], dtype=np.float64)
        labels = list(kfs[0]["points"].keys())
        court = np.array([kfs[0]["points"][lb]["court_xz"] for lb in labels],
                         dtype=np.float64)
        px = np.array([[kf["points"][lb]["px"] for lb in labels] for kf in kfs],
                      dtype=np.float64)
        L = len(labels)
        Hs = np.empty((N, 3, 3), dtype=np.float64)
        for i, t in enumerate(times):
            ip = np.empty((L, 2), dtype=np.float64)
            for a in range(2):
                for l in range(L):
                    ip[l, a] = np.interp(t, kt, px[:, l, a])
            Hs[i], _ = cv2.findHomography(ip, court, 0)
    return Hs, np.linalg.inv(Hs)


def load_trajectory(path):
    """Returns (times, xz Nx2 array, conf) for frames that carry a root."""
    with open(path) as f:
        doc = json.load(f)
    ts, xz, conf = [], [], []
    for fr in doc["frames"]:
        r = fr.get("root_court_xz")
        if r is None:
            continue
        ts.append(float(fr.get("time", len(ts))))
        xz.append((float(r[0]), float(r[1])))
        conf.append(float(fr.get("root_confidence", 1.0)))
    if not xz:
        sys.exit(f"ERROR: no root_court_xz in {path} — extract with --court first")
    return np.array(ts), np.array(xz), np.array(conf)


def to_img(Hinv, cx, cz):
    p = cv2.perspectiveTransform(
        np.array([[[cx, cz]]], dtype=np.float64), Hinv).reshape(2)
    return int(round(p[0])), int(round(p[1]))


def to_court(H, px, py):
    p = cv2.perspectiveTransform(
        np.array([[[px, py]]], dtype=np.float64), H).reshape(2)
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------- top-down map

class CourtMap:
    """To-scale top-down map of the court (same data the Unity floor uses)."""

    def __init__(self, half="far", px_per_m=60, margin_m=1.2):
        self.ppm = px_per_m
        zmin = (0.0 if half == "far" else -ZB) - margin_m
        zmax = (0.0 if half == "near" else ZB) + margin_m
        xmin, xmax = -XD - margin_m, XD + margin_m
        self.xmin, self.zmin = xmin, zmin
        self.w = int((xmax - xmin) * px_per_m)
        self.h = int((zmax - zmin) * px_per_m)
        self.half = half

    def pt(self, cx, cz):
        # +X right; +Z goes UP the image (far side at the top, like Unity top view)
        px = int(round((cx - self.xmin) * self.ppm))
        py = self.h - int(round((cz - self.zmin) * self.ppm))
        return px, py

    def render(self, xz, upto=None, dot=None):
        img = np.full((self.h, self.w, 3), (40, 90, 40), dtype=np.uint8)
        for (a, b) in court_segments(self.half):
            cv2.line(img, self.pt(*a), self.pt(*b), (255, 255, 255), 2, cv2.LINE_AA)
        # trajectory (time-colored: blue start -> red end)
        n = len(xz) if upto is None else max(1, min(upto + 1, len(xz)))
        for i in range(1, n):
            t = i / max(1, len(xz) - 1)
            color = (255 - int(255 * t), 60, int(255 * t))  # BGR blue->red
            cv2.line(img, self.pt(*xz[i - 1]), self.pt(*xz[i]), color, 2, cv2.LINE_AA)
        cv2.circle(img, self.pt(*xz[0]), 6, (0, 255, 0), -1)          # start
        cv2.circle(img, self.pt(*xz[n - 1]), 6, (0, 0, 255), -1)      # latest/end
        if dot is not None:
            cv2.circle(img, self.pt(*dot), 8, (0, 255, 255), 2)
        return img


# ---------------------------------------------------------------- default mode

def draw_frame_overlay(frame, Hinv, xz, upto, trail=90):
    """Foot dot + recent trail back-projected onto the video frame."""
    img = frame.copy()
    lo = max(0, upto - trail)
    pts = [to_img(Hinv, x, z) for x, z in xz[lo:upto + 1]]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                      (0, 255, 255), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], 10, (0, 0, 255), 3)
    return img


def frame_at(cap, t, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def idx_at(ts, t):
    return int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))


def make_sheet(video, Hinv_series, ts, xz, cmap, panels, out_png):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = ts[-1]
    times = [dur * (i + 0.5) / panels for i in range(panels)]
    cells = []
    for t in times:
        frame = frame_at(cap, t, fps)
        if frame is None:
            continue
        i = idx_at(ts, t)
        img = draw_frame_overlay(frame, Hinv_series[i], xz, i)
        # inset map, bottom-right
        m = cmap.render(xz, upto=i)
        mh = img.shape[0] // 3
        mw = int(m.shape[1] * mh / m.shape[0])
        m = cv2.resize(m, (mw, mh))
        img[-mh - 10:-10, -mw - 10:-10] = m
        cv2.putText(img, f"t={t:.1f}s  xz=({xz[i][0]:+.2f},{xz[i][1]:+.2f})",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cells.append(cv2.resize(img, None, fx=0.5, fy=0.5))
    cap.release()
    cols = 2
    rows = (len(cells) + cols - 1) // cols
    ch, cw = cells[0].shape[:2]
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for k, c in enumerate(cells):
        r, col = divmod(k, cols)
        sheet[r * ch:(r + 1) * ch, col * cw:(col + 1) * cw] = c
    cv2.imwrite(out_png, sheet)
    print(f"  sheet   -> {out_png}   (red dot must sit on the FEET in every panel)")


def make_video(video, Hinv_series, ts, xz, cmap, out_mp4):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        i = idx_at(ts, t)
        img = draw_frame_overlay(frame, Hinv_series[i], xz, i)
        m = cmap.render(xz, upto=i)
        mh = h // 3
        mw = int(m.shape[1] * mh / m.shape[0])
        img[-mh - 10:-10, -mw - 10:-10] = cv2.resize(m, (mw, mh))
        vw.write(img)
        n += 1
    cap.release()
    vw.release()
    print(f"  video   -> {out_mp4}")


def print_stats(ts, xz, conf, half):
    x, z = xz[:, 0], xz[:, 1]
    zlo, zhi = (ZS, ZB) if half in ("far", "near") else (-ZB, ZB)
    inbox = np.mean((np.abs(x) <= XD) & (z >= zlo) & (z <= zhi)) * 100
    print(f"  trajectory: {len(xz)} frames, {ts[-1]:.1f}s")
    print(f"    start ({x[0]:+.2f},{z[0]:+.2f})  end ({x[-1]:+.2f},{z[-1]:+.2f})")
    print(f"    X [{x.min():+.2f},{x.max():+.2f}]  Z [{z.min():+.2f},{z.max():+.2f}]  "
          f"conf mean {conf.mean():.2f}")
    print(f"    inside tracked box (|X|<={XD}, {zlo}<=Z<={zhi}): {inbox:.0f}%")


# ---------------------------------------------------------------- route mode

def route_check(video, Hinv_series, ts, xz, names, out_dir, stem):
    """The user names the intersections they actually walked to (ground truth).
    For each, find the trajectory's closest approach and render that video
    frame with BOTH the extracted dot (red) and the true point (magenta).
      - dot on the FEET but feet visibly NOT at the magenta marker
          -> the person really wasn't there, or the homography maps that
             pixel to the wrong coordinate (probe the spot to confirm)
      - dot NOT on the feet -> foot-pixel/extraction bias
    Prints the gap in meters AND pixels: a small pixel gap that is a large
    metric gap = perspective leverage at range, not a wrong click."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # project each trajectory sample to pixels IN ITS OWN FRAME (per-frame
    # homography for a moving camera; constant for a static one)
    px_all = np.array([
        cv2.perspectiveTransform(xz[k].reshape(1, 1, 2).astype(np.float64),
                                 Hinv_series[k]).reshape(2)
        for k in range(len(xz))])
    print(f"  route check ({len(names)} points):")
    print(f"    {'point':14s} {'sel':5s} {'t':>6s} {'extracted':>16s} {'true':>16s} "
          f"{'dX':>6s} {'dZ':>6s} {'gap_m':>6s} {'gap_px':>7s}")

    def render(name, target, i, tag):
        t = ts[i]
        ex, ez = xz[i]
        dx, dz = ex - target[0], ez - target[1]
        gap_m = float(np.hypot(dx, dz))
        p_true = to_img(Hinv_series[i], *target)
        gap_px = float(np.hypot(px_all[i][0] - p_true[0], px_all[i][1] - p_true[1]))
        print(f"    {name:14s} {tag:5s} {t:5.1f}s ({ex:+6.2f},{ez:+6.2f}) "
              f"({target[0]:+6.2f},{target[1]:+6.2f}) {dx:+6.2f} {dz:+6.2f} "
              f"{gap_m:6.2f} {gap_px:6.0f}px")
        frame = frame_at(cap, t, fps)
        if frame is None:
            return
        img = draw_frame_overlay(frame, Hinv_series[i], xz, i)
        cv2.drawMarker(img, p_true, (255, 0, 255), cv2.MARKER_CROSS, 34, 3)
        cv2.putText(img, name, (p_true[0] + 12, p_true[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.putText(img, f"t={t:.1f}s [{tag}]  extracted ({ex:+.2f},{ez:+.2f})  "
                         f"true {name} ({target[0]:+.2f},{target[1]:+.2f})  "
                         f"gap {gap_m:.2f}m / {gap_px:.0f}px",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        suffix = "" if tag == "court" else "_px"
        out = os.path.join(out_dir, f"{stem}_route_{name}{suffix}.png")
        cv2.imwrite(out, img)
        print(f"      -> {out}")

    for name in names:
        if name not in COURT_POINTS:
            print(f"    {name:14s} UNKNOWN (see --list-points)")
            continue
        target = np.array(COURT_POINTS[name])
        # closest approach in court space (what the trajectory claims)...
        i = int(np.argmin(np.linalg.norm(xz - target, axis=1)))
        render(name, target, i, "court")
        # ...and in IMAGE space (the moment the feet look closest to the
        # point). If these disagree, the extraction is biased at that spot.
        # The true point's pixel moves with the camera, so project it per-frame.
        p_true_all = np.array([to_img(Hinv_series[k], *target)
                               for k in range(len(xz))], dtype=np.float64)
        j = int(np.argmin(np.linalg.norm(px_all - p_true_all, axis=1)))
        if abs(ts[j] - ts[i]) > 0.3:
            render(name, target, j, "image")
    cap.release()
    print("    read it: RED dot = what was extracted (must ride the feet);")
    print("    MAGENTA cross = where the true point is in the image.")


# ---------------------------------------------------------------- probe mode

def probe(video, calib_doc, H, Hinv, frame_time, scale):
    frame = grab_frame(video, frame_time)
    # faint calibration grid for context
    ctx = frame.copy()
    for (a, b) in court_segments(calib_doc.get("half", "full")):
        n = 30
        pts = []
        for i in range(n + 1):
            t = i / n
            px, py = to_img(Hinv, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if -frame.shape[1] < px < 2 * frame.shape[1] and -frame.shape[0] < py < 2 * frame.shape[0]:
                pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(ctx, [np.array(pts, dtype=np.int32)], False,
                          (120, 200, 200), 1, cv2.LINE_AA)
    clicks = []

    def redraw():
        disp = cv2.resize(ctx, None, fx=scale, fy=scale).copy()
        for i, (px, py, cx, cz) in enumerate(clicks):
            p = (int(px * scale), int(py * scale))
            cv2.drawMarker(disp, p, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(disp, f"({cx:+.2f},{cz:+.2f})", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if i > 0:
                q = (int(clicks[i - 1][0] * scale), int(clicks[i - 1][1] * scale))
                cv2.line(disp, q, p, (255, 180, 0), 1, cv2.LINE_AA)
                d = np.hypot(cx - clicks[i - 1][2], cz - clicks[i - 1][3])
                mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
                cv2.putText(disp, f"{d:.2f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        cv2.putText(disp, "click floor spots -> court XZ + distances  "
                          "(c=clear, ESC=quit)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("probe", disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = x / scale, y / scale
            cx, cz = to_court(H, px, py)
            clicks.append((px, py, cx, cz))
            line = f"  ({px:6.0f},{py:6.0f}) px -> court X={cx:+.2f}  Z={cz:+.2f}"
            if len(clicks) > 1:
                d = np.hypot(cx - clicks[-2][2], cz - clicks[-2][3])
                line += f"   dist from previous: {d:.3f} m"
            print(line)
            redraw()

    print("PROBE: click known floor spots and compare with reality.")
    print("  regulation checks: SSL->baseline 4.72 m, doubles width 6.10 m,")
    print("  singles width 5.18 m, LSL->baseline 0.76 m")
    cv2.namedWindow("probe", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("probe", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k in (ord("c"), ord("C")):
            clicks.clear()
            redraw()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Diagnose court-position mismatches (video side).")
    ap.add_argument("video")
    ap.add_argument("--calib", default=None, help="calibration json (default data/calib/<clip>_court.json)")
    ap.add_argument("--skeleton", default=None, help="skeleton json (default data/skeleton/<clip>.json)")
    ap.add_argument("--probe", action="store_true", help="interactive click-to-measure mode")
    ap.add_argument("--route", default=None, metavar="NAME,NAME,...",
                    help="named intersections the player actually visited (ground "
                         "truth); renders the closest-approach frame for each")
    ap.add_argument("--video", dest="write_video", action="store_true", help="also write a per-frame check mp4")
    ap.add_argument("--panels", type=int, default=6, help="contact-sheet panel count")
    ap.add_argument("--frame-time", type=float, default=1.0, help="probe frame time (s)")
    ap.add_argument("--scale", type=float, default=0.8, help="probe display scale")
    args = ap.parse_args()

    calib_path, skel_path, stem = default_paths(args.video)
    calib_path = args.calib or calib_path
    skel_path = args.skeleton or skel_path

    calib_doc = load_calib(calib_path)
    half = calib_doc.get("half", "full")
    kind = f"{len(calib_doc['keyframes'])} keyframes" if "keyframes" in calib_doc else "static"
    print(f"  calib   <- {calib_path} (half={half}, {kind})")

    if args.probe:
        Hs, Hinv = build_H_series(calib_doc, [args.frame_time])
        probe(args.video, calib_doc, Hs[0], Hinv[0], args.frame_time, args.scale)
        return

    ts, xz, conf = load_trajectory(skel_path)
    print(f"  clip    <- {skel_path}")
    print_stats(ts, xz, conf, half)

    _, Hinv_series = build_H_series(calib_doc, ts)

    out_dir = os.path.dirname(calib_path) or "."
    if args.route:
        names = [s.strip() for s in args.route.split(",") if s.strip()]
        route_check(args.video, Hinv_series, ts, xz, names, out_dir, stem)
        return

    cmap = CourtMap(half=half)
    top = cmap.render(xz)
    top_png = os.path.join(out_dir, stem + "_check_topdown.png")
    cv2.imwrite(top_png, top)
    print(f"  topdown -> {top_png}   (compare with Unity: Tools > Badminton > Debug > Draw Clip Path)")

    make_sheet(args.video, Hinv_series, ts, xz, cmap, args.panels,
               os.path.join(out_dir, stem + "_check_sheet.png"))
    if args.write_video:
        make_video(args.video, Hinv_series, ts, xz, cmap,
                   os.path.join(out_dir, stem + "_check.mp4"))


if __name__ == "__main__":
    main()
