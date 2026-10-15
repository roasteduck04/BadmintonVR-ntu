"""Drawing primitives + every colour of the CV overlays (SP2 spec § 4.2). BGR throughout (OpenCV). Player
colours are Unity's AnalysisPlayer.playerColors, so the video overlay and the twin agree. Text passed to
cv2.putText must be ASCII: the Hershey fonts cannot draw anything else."""
from __future__ import annotations

import textwrap

import cv2
import numpy as np


def unity_bgr(r, g, b):
    return tuple(int(round(c * 255)) for c in (b, g, r))


PLAYER_BGR = [unity_bgr(0.2, 0.7, 1.0), unity_bgr(1.0, 0.5, 0.2), unity_bgr(0.3, 0.9, 0.4), unity_bgr(0.9, 0.3, 0.7)]
GREY, WHITE, BLACK = (150, 150, 150), (255, 255, 255), (0, 0, 0)
RED, YELLOW, MAGENTA, CYAN = (0, 0, 255), (0, 230, 255), (255, 0, 255), (255, 255, 0)
SHUTTLE_BGR = (255, 90, 150)                 # violet: white is the --compare skeleton
STATE_BGR = {2: YELLOW, 3: MAGENTA}          # gap-filled, prior-only (measured = the player's colour)
FONT = cv2.FONT_HERSHEY_SIMPLEX

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
              (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]


def player_bgr(i):
    return PLAYER_BGR[int(i) % len(PLAYER_BGR)]


def err_bgr(err_px, max_px=40.0):
    if not np.isfinite(err_px):
        return GREY                          # unknown error
    a = float(np.clip(err_px / max_px, 0.0, 1.0))
    return (0, int(round(255 * (1 - a))), int(round(255 * a)))


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def _ok(p):
    return p is not None and np.isfinite(p).all()


def draw_edges_dashed(img, uv, edges, bgr, thick=1):
    """draw_edges with dashed segments (SP3: 'unknown' persons)."""
    for a, b in edges:
        if _ok(uv[a]) and _ok(uv[b]):
            dashed_line(img, uv[a], uv[b], bgr, thick)
    return img


def draw_skeleton(img, uv, parents, bgr, thick=2, joint_bgr=None, radius=3):
    for j, par in enumerate(parents):
        if par >= 0 and _ok(uv[j]) and _ok(uv[par]):
            cv2.line(img, _pt(uv[j]), _pt(uv[par]), bgr, thick, cv2.LINE_AA)
    for j in range(len(uv)):
        if _ok(uv[j]):
            cv2.circle(img, _pt(uv[j]), radius, joint_bgr[j] if joint_bgr is not None else bgr, -1, cv2.LINE_AA)
    return img


def draw_edges(img, uv, edges, bgr, thick=1):
    for a, b in edges:
        if _ok(uv[a]) and _ok(uv[b]):
            cv2.line(img, _pt(uv[a]), _pt(uv[b]), bgr, thick, cv2.LINE_AA)
    return img


def draw_keypoints(img, uv, p, bgr, w_min=0.3, radius=4):
    """Confident keypoints filled (blended by their mean score), the rest hollow."""
    p = np.asarray(p, float)
    over = img.copy()
    for k in range(len(uv)):
        if not _ok(uv[k]):
            continue
        if p[k] < w_min:
            cv2.circle(img, _pt(uv[k]), radius, bgr, 1, cv2.LINE_AA)
        else:
            cv2.circle(over, _pt(uv[k]), radius, bgr, -1, cv2.LINE_AA)
    confident = p[p >= w_min]
    alpha = float(np.clip(confident.mean(), 0.3, 1.0)) if len(confident) else 0.3
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)
    return img


def dashed_line(img, a, b, bgr, thick=1, dash=6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.linalg.norm(b - a) // dash))
    for i in range(0, n, 2):
        p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
        cv2.line(img, _pt(p), _pt(q), bgr, thick, cv2.LINE_AA)
    return img


def draw_residual(img, a, b, err_px, rejected):
    """Detection -> reprojection segment coloured by error; dashed when the view was rejected."""
    if not (_ok(np.asarray(a, float)) and _ok(np.asarray(b, float))):
        return img
    if rejected:
        dashed_line(img, a, b, err_bgr(err_px), 1)
    else:
        cv2.line(img, _pt(a), _pt(b), err_bgr(err_px), 1, cv2.LINE_AA)
    return img


def draw_bbox_tag(img, uv, bgr, text):
    pts = np.asarray(uv, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return img
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    cv2.rectangle(img, _pt((x0, y0)), _pt((x1, y1)), bgr, 1, cv2.LINE_AA)
    cv2.putText(img, text, _pt((x0, max(12, y0 - 4))), FONT, 0.45, bgr, 1, cv2.LINE_AA)
    return img


def banner_height(img):
    return max(22, img.shape[0] // 24)


def draw_banner(img, text, bgr=RED):
    h = banner_height(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), bgr, -1)
    cv2.putText(img, text, (6, int(h * 0.72)), FONT, h / 40, WHITE, 1, cv2.LINE_AA)
    return img


def wrap_lines(lines, width_chars=46):
    out = []
    for s in lines:
        out += textwrap.wrap(s, width_chars) or [""]
    return out


def draw_text_panel(img, lines, width_chars=46, origin=(6, 6), scale=0.42):
    lines = wrap_lines(lines, width_chars)
    lh = int(18 * scale / 0.42)
    x, y = origin
    w = int(width_chars * 7.6 * scale / 0.42)
    over = img.copy()
    cv2.rectangle(over, (x, y), (min(img.shape[1] - 1, x + w), min(img.shape[0] - 1, y + lh * len(lines) + 6)),
                  BLACK, -1)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x + 4, y + lh * (i + 1)), FONT, scale, WHITE, 1, cv2.LINE_AA)
    return img


def legend_strip(width):
    img = np.full((28, width, 3), 30, np.uint8)
    items = [("P1", PLAYER_BGR[0]), ("P2", PLAYER_BGR[1]), ("P3", PLAYER_BGR[2]), ("P4", PLAYER_BGR[3]),
             ("other", GREY), ("filled", YELLOW), ("prior", MAGENTA), ("err 0", err_bgr(0)), ("40px", err_bgr(40)),
             ("compare", WHITE), ("shuttle", SHUTTLE_BGR)]
    x = 6
    for name, c in items:
        if x > width - 40:
            break
        cv2.rectangle(img, (x, 8), (x + 12, 20), c, -1)
        cv2.putText(img, name, (x + 16, 19), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
        x += 22 + 8 * len(name)
    return img
