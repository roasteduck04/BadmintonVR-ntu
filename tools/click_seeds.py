# tools/click_seeds.py
"""Click the target player(s) in frame 0 of each view to seed the Colab tracker.

Close-ups: one click on the red player. CCTV: click red, then navy. Writes
data/calib/3-1_seeds.json; the Colab notebook picks the detected track whose
frame-0 box contains each seed.

Usage:
  PYTHONPATH=tools tools/.venv/Scripts/python.exe tools/click_seeds.py \
    --clip 3-1 --frame-time 0 --scale 0.35 \
    --view frontleft31="data/3cams + 2 cctv/front left 3-1.mov":red \
    --view backleft31="data/3cams + 2 cctv/back left 3-1.mov":red \
    --view backright31="data/3cams + 2 cctv/back right 3-1.mov":red \
    --view cctv31="data/3cams + 2 cctv/cctv 3-1.mov":red,navy
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

from calibrate_court import grab_frame


def _click_labels(frame, labels, scale):
    disp = cv2.resize(frame, None, fx=scale, fy=scale)
    got = {}
    order = list(labels)
    print(f"  click in order: {', '.join(order)} (ESC to skip remaining)")

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and order:
            lab = order.pop(0)
            got[lab] = [x / scale, y / scale]
            cv2.drawMarker(disp, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(disp, lab, (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("seed", disp)
            print(f"    {lab} = {got[lab][0]:.0f},{got[lab][1]:.0f}")

    cv2.namedWindow("seed", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("seed", on_mouse)
    cv2.imshow("seed", disp)
    while order and (cv2.waitKey(30) & 0xFF) != 27:
        pass
    cv2.destroyAllWindows()
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seed clicker for the Colab tracker.")
    ap.add_argument("--clip", default="3-1")
    ap.add_argument("--frame-time", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--view", action="append", default=[], required=True,
                    metavar="id=path:labels", help="e.g. cctv31=clip.mov:red,navy")
    ap.add_argument("--additional-frame", action="append", default=[],
                    metavar="time:labels",
                    help="Additional seed at a different time, e.g. 5.0:P3,P4")
    args = ap.parse_args(argv)

    views = {}
    for spec in args.view:
        vid, rest = spec.split("=", 1)
        path, labels = rest.rsplit(":", 1)
        frame = grab_frame(path, args.frame_time)
        seeds = _click_labels(frame, labels.split(","), args.scale)
        views[vid] = {"source": os.path.basename(path),
                      "frame_time": args.frame_time, "seeds": seeds}

    # Process additional seed frames
    for af_spec in args.additional_frame:
        af_time, af_labels = af_spec.split(":", 1)
        af_time = float(af_time)
        af_label_list = af_labels.split(",")

        for spec in args.view:
            vid, rest = spec.split("=", 1)
            path, _ = rest.rsplit(":", 1)
            frame = grab_frame(path, af_time)
            seeds = _click_labels(frame, af_label_list, args.scale)
            if seeds:
                views[vid].setdefault("additional_seeds", []).append(
                    {"frame_time": af_time, "seeds": seeds})

    out = {"clip": args.clip, "views": views}
    path = os.path.join("data", "calib", f"{args.clip}_seeds.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}: {len(views)} views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
