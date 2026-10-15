"""Draw racket variants and hand estimates onto the real camera frames for visual QA.

    PYTHONPATH=tools python -m analysis.racket_overlay --clip singles_1x --frame 600 --player P1 \
        --out racket_overlay.png

Legend: grey = raw RacketVision candidates, orange = v1, cyan = rigid, magenta = bootstrap,
yellow dot = raw triangulated hand, blue dot = analysis (smoothed) hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from analysis import racket_rigidfit as rf
from analysis import racket_variants as rv

COLORS = {"v1": (0, 140, 255), "rigid": (255, 220, 0), "bootstrap": (230, 60, 230)}
SKEL = [(rf.HANDLE, rf.BOTTOM), (rf.BOTTOM, rf.LEFT), (rf.LEFT, rf.TOP), (rf.TOP, rf.RIGHT), (rf.RIGHT, rf.BOTTOM)]


def _v1_points(racket, pos):
    off = np.array([pos[0], 0.0, pos[1]])
    return {k: np.array(racket[k]) + off for k in ("grip", "head", "side")}


def _variant_keypoints(sc, pid_idx, f, pos):
    r = sc["players"][pid_idx]["frames"][f]
    if not r:
        return None
    tpl = rf.RacketTemplate(**sc["template"])
    z = np.array(r["axis"])
    x = np.array(r["side"]) - np.array(r["head"])
    x -= (x @ z) * z
    x /= max(np.linalg.norm(x), 1e-9)
    R = np.column_stack([x, np.cross(z, x), z])
    return np.array(r["grip"]) + np.array([pos[0], 0.0, pos[1]]) + tpl.keypoints() @ R.T


def render(clip, frame, pid, analysis_dir, racket2d_dir, pose2d_dir, crop=700):
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from racket2d_gpu import CLIP_CAMERAS
    from racket_pseudolabel import VIDEO_DIRS

    src = os.path.join(analysis_dir, f"{clip}.analysis.json")
    with open(src, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    inp = rv.prepare_inputs(doc, racket2d_dir, pose2d_dir, log=lambda *a: None)
    pids = inp["pids"]
    pi = pids.index(pid)
    pf = doc["players"][pi]["frames"][frame]
    pos = pf.get("position")
    sidecars = {}
    for name in ("rigid", "bootstrap"):
        p = rv.sidecar_path(src, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                sidecars[name] = json.load(fh)
    hands_an = rv.player_world_hands(doc)[pid][0][frame]
    hands_raw = inp["world_hands"][pid][frame]

    tiles = []
    for v, cam in inp["cams"].items():
        g = frame + inp["offsets"].get(v, 0)
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIRS[clip], CLIP_CAMERAS[clip][v]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, g)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        center = None
        if np.isfinite(hands_raw).all(axis=1).any():
            center = cam.project(hands_raw[np.isfinite(hands_raw).all(axis=1)].mean(axis=0))[0]
        elif pos is not None:
            center = cam.project(np.array([pos[0], 1.2, pos[1]]))[0]
        if center is None or not np.isfinite(center).all():
            continue
        rk = rv.load_racket2d(os.path.join(racket2d_dir, f"{v}_racket2d.json"))
        if 0 <= g < len(rk):
            for c in rk[g]:
                for a, b in SKEL:
                    cv2.line(img, tuple(map(int, c[a, :2])), tuple(map(int, c[b, :2])), (160, 160, 160), 2)
        assigned = inp["frames"][pid][frame].get(v)
        if assigned:
            for a, b in SKEL:
                cv2.line(img, tuple(map(int, assigned[0][a, :2])), tuple(map(int, assigned[0][b, :2])), (255, 255, 255), 3)
        if pf.get("racket") and pos is not None:
            pts = _v1_points(pf["racket"], pos)
            uv, _ = cam.project(np.stack([pts["grip"], pts["head"], pts["side"]]))
            cv2.line(img, tuple(map(int, uv[0])), tuple(map(int, uv[1])), COLORS["v1"], 4)
            cv2.line(img, tuple(map(int, uv[1])), tuple(map(int, uv[2])), COLORS["v1"], 4)
        for name, sc in sidecars.items():
            kp = _variant_keypoints(sc, pi, frame, pos) if pos is not None else None
            if kp is not None:
                uv, _ = cam.project(kp)
                for a, b in SKEL:
                    cv2.line(img, tuple(map(int, uv[a])), tuple(map(int, uv[b])), COLORS[name], 4)
        for H, col in ((hands_an, (255, 80, 0)), (hands_raw, (0, 255, 255))):
            for h in H:
                if np.isfinite(h).all():
                    cv2.circle(img, tuple(map(int, cam.project(h)[0])), 12, col, -1)
        x1 = int(np.clip(center[0] - crop / 2, 0, img.shape[1] - crop))
        y1 = int(np.clip(center[1] - crop / 2, 0, img.shape[0] - crop))
        tile = img[y1:y1 + crop, x1:x1 + crop].copy()
        cv2.putText(tile, f"{v} raw#{g}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    return np.vstack(rows)


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description="Racket variants drawn on the real camera frames.")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--player", default="P1")
    ap.add_argument("--analysis-dir", default=rv.DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--racket2d-dir", default=rv.DEFAULT_RACKET2D_DIR)
    ap.add_argument("--pose2d-dir", default=rv.DEFAULT_POSE2D_DIR)
    ap.add_argument("--crop", type=int, default=700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    img = render(args.clip, args.frame, args.player, args.analysis_dir, args.racket2d_dir,
                 args.pose2d_dir, args.crop)
    if img is None:
        raise SystemExit("nothing to draw")
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
