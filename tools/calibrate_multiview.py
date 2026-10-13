# tools/calibrate_multiview.py
"""Close-up camera calibration for triangulation.

Click >=4 named court points on ONE frame of a close-up view. With an assumed
focal length (FOV guess; same phone for all close-ups) this solves the camera's
3D pose (PnP) in the Y-up court world frame, writing data/calib/<view>_cam.json.
The reprojection rms it prints is the per-camera quality signal.

Usage:
  PYTHONPATH=tools tools/.venv/Scripts/python.exe tools/calibrate_multiview.py \
      "data/3cams + 2 cctv/front left 3-1.mov" --view frontleft31 \
      --frame-time 6 --fov-deg 67 --scale 0.35

  # non-interactive (replay saved clicks):
  ... --point corner_fl=812,1533 --point corner_fr=1290,1470 \
      --point ssl_fl=770,980 --point ssl_fr=1220,960
Run `--list-points` to see the named court points.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from calibrate_court import COURT_POINTS, grab_frame
from analysis import multiview as mv


def _click_points(frame, scale):
    disp = cv2.resize(frame, None, fx=scale, fy=scale)
    picks = []            # (name, px, py)
    pending = {"name": None}
    print("Type a point name in the terminal prompt, then click it. "
          "ESC in the window when done.")

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and pending["name"]:
            picks.append((pending["name"], x / scale, y / scale))
            cv2.drawMarker(disp, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(disp, pending["name"], (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("calib", disp)
            print(f"  set {pending['name']} = {picks[-1][1]:.0f},{picks[-1][2]:.0f}")
            pending["name"] = None

    cv2.namedWindow("calib", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calib", on_mouse)
    cv2.imshow("calib", disp)
    while True:
        if pending["name"] is None:
            name = input("point name (blank to finish): ").strip()
            if not name:
                break
            if name not in COURT_POINTS:
                print(f"  unknown '{name}' (see --list-points)"); continue
            pending["name"] = name
        if (cv2.waitKey(30) & 0xFF) == 27:
            break
    cv2.destroyAllWindows()
    return picks


def main(argv=None):
    ap = argparse.ArgumentParser(description="Close-up court PnP calibrator.")
    ap.add_argument("video")
    ap.add_argument("--view", required=True, help="short id, e.g. frontleft31")
    ap.add_argument("--frame-time", type=float, default=1.0)
    ap.add_argument("--fov-deg", type=float, default=67.0)
    ap.add_argument("--fov-on", choices=["long", "short"], default="long")
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--point", action="append", default=[],
                    metavar="name=px,py", help="non-interactive court point")
    ap.add_argument("--list-points", action="store_true")
    args = ap.parse_args(argv)

    if args.list_points:
        for k, v in COURT_POINTS.items():
            print(f"  {k:16s} court {v}")
        return 0

    frame = grab_frame(args.video, args.frame_time)
    h, w = frame.shape[:2]

    if args.point:
        picks = []
        for spec in args.point:
            parts = spec.split("=")
            if len(parts) != 2:
                raise SystemExit(f"bad --point '{spec}', expected name=px,py")
            name, px = parts
            xy = px.split(",")
            if len(xy) != 2:
                raise SystemExit(f"bad --point '{spec}', expected name=px,py")
            x, y = xy
            if name not in COURT_POINTS:
                raise SystemExit(f"unknown point {name}")
            picks.append((name, float(x), float(y)))
    else:
        picks = _click_points(frame, args.scale)

    if len(picks) < 4:
        raise SystemExit(f"need >= 4 points, got {len(picks)}")

    names = [p[0] for p in picks]
    image_px = np.array([[p[1], p[2]] for p in picks], dtype=np.float64)
    court_xz = np.array([COURT_POINTS[n] for n in names], dtype=np.float64)

    K = mv.intrinsics_from_fov(w, h, args.fov_deg, on=args.fov_on)
    R, t = mv.solve_extrinsics(court_xz, image_px, K)
    rms = mv.reprojection_rms(court_xz, image_px, K, R, t)

    out = {
        "view": args.view, "source": os.path.basename(args.video),
        "width": w, "height": h, "fov_deg": args.fov_deg,
        "focal_px": float(K[0, 0]),
        "K": K.tolist(), "R": R.tolist(), "t": t.tolist(),
        "rms_px": rms,
        "court_points": {p[0]: {"px": [p[1], p[2]], "court_xz": COURT_POINTS[p[0]]}
                         for p in picks},
    }
    path = os.path.join("data", "calib", f"{args.view}_cam.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}: focal {K[0,0]:.0f}px, reprojection rms {rms:.2f}px "
          f"over {len(picks)} points")
    if rms > 15:
        print("  WARNING: rms > 15px - check clicks or --fov-deg; this camera "
              "may be too oblique to triangulate from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
