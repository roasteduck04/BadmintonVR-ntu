"""Annotated original camera video (SP2 spec § 4): what the CV pipeline saw and concluded, per camera.

    PYTHONPATH=tools python -m analysis.overlay_video render --clip singles_05x --variant pose24 \
        [--compare posefit] --views all --layers dets,kp,skel,resid,bbox,court,calib,text[,aux,racket=rigid,shuttle=tracker] \
        --frames 300:600 --mode mosaic|views|still [--frame 450] --scale 0.25 [--crop player:P1] [--preview] \
        [--source original|proxy] [--tile 960x540] --out data/overlays/singles_05x/
    PYTHONPATH=tools python -m analysis.overlay_video proxies --clip singles_05x

Facts come from the cvdebug/1 sidecar (built in memory when missing or stale), so this and Unity agree. Frames are
drawn at ctx.scale: every 3D point goes through the full-resolution camera, then is scaled; residuals and
calibration errors are reported in full-resolution pixels. Text on images is ASCII (Hershey fonts)."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction

import cv2
import numpy as np

from analysis import cvdebug as cd
from analysis import draw
from analysis import pose_v2 as pv
from analysis import racket_rigidfit as rf
from analysis import skeleton24 as sk
from analysis import video_io
from analysis.pose24 import config, kpset
from analysis.pose24 import doc as docmod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibrate_court import XD, ZB, court_segments  # noqa: E402
from check_position import CourtMap  # noqa: E402

LAYERS = ("dets", "kp", "skel", "resid", "bbox", "court", "calib", "text", "aux", "racket", "shuttle")
DEFAULT_LAYERS = "dets,kp,skel,resid,bbox,court,calib,text"
NET_POST_M = 1.55
STATE_NAMES = ("missing", "measured", "gap-filled", "prior-only")
PROXY_DIR = os.path.join("data", "video_proxy")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WHOLE_CLIP_WARN = 600
TIMELINE_TOL = 0.05                   # --compare: offsets further apart than this (frames) are another timeline


def parse_layers(s):
    out = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        name, _, opt = part.partition("=")
        if name not in LAYERS:
            raise ValueError(f"unknown layer {name!r}; choose from {', '.join(LAYERS)}")
        out[name] = opt or None
    return out


def parse_frames(s, n):
    a, _, b = s.partition(":")
    a, b = max(0, int(a or 0)), min(n, int(b) if b else n)
    return list(range(a, b))


_ASCII = str.maketrans({"·": "|", "—": "-", "–": "-", "→": "->", "σ": "sigma"})


def ascii_text(s):
    """Hershey fonts draw ASCII only: map the doc's usual symbols, anything else becomes '?'."""
    return str(s).translate(_ASCII).encode("ascii", "replace").decode("ascii")


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class OverlayContext:
    """Everything a frame render needs, loaded once: doc, sidecar, full-res cameras, world joints, raw kp2d."""

    def __init__(self, clip, variant, compare=None, analysis_dir=config.DEFAULT_ANALYSIS_DIR, cache_dir=pv.CACHE_DIR,
                 data_dir="data", registry=None, views="all", scale=0.25, source="original", log=print):
        self.clip, self.variant, self.scale, self.source, self.log = clip, variant, float(scale), source, log
        self.reg = registry if registry is not None else config.load_registry()
        self.analysis_dir, self.data_dir = analysis_dir, data_dir
        with open(cd.doc_path(analysis_dir, clip, variant), "rb") as fh:
            raw = fh.read()
        self.doc = json.loads(raw)
        self.sc = self._sidecar(raw, cache_dir)
        self.names = [c["name"] for c in self.sc["cameras"]]
        self.cams = [rf.Camera(np.reshape(c["K"], (3, 3)), np.reshape(c["R"], (3, 3)), c["t"])
                     for c in self.sc["cameras"]]
        self.views = self._select_views(views)
        self.world = cd.doc_world_joints(self.doc)
        self.parents = self._parents(self.doc)
        self.cmp_world, self.cmp_parents, self.cmp_label = {}, None, None
        if compare:
            cclip, _, cvar = compare.rpartition(":") if ":" in compare else (clip, "", compare)
            cdoc = _load(cd.doc_path(analysis_dir, cclip or clip, cvar))
            self.cmp_label = f"{cclip or clip} {cvar}"
            # by court position, not id: a variant / subset doc may label the same people differently (L7)
            to_cmp = docmod.match_players(cdoc, self.doc)
            cw = cd.doc_world_joints(cdoc)
            self.cmp_world = {pid: cw[cid] for pid, cid in to_cmp.items() if cid in cw}
            self.cmp_parents = self._parents(cdoc)
            missed = sorted(set(cw) - set(to_cmp.values()))
            if missed:
                self.log(f"    --compare {self.cmp_label}: {', '.join(missed)} match no {clip} {variant} player "
                         "by position: not drawn")
            why = timeline_mismatch(self.doc, cdoc)
            if why:
                self.log(f"    WARNING: --compare {self.cmp_label} is on another timeline than {clip} {variant} "
                         f"({why}): the white skeleton is misaligned in time")
        self.raw, self.headers, self.court_pts = {}, {}, {}
        for v in self.names:
            p = os.path.join(data_dir, "pose2d", f"{v}_kp2d.json")
            self.raw[v], self.headers[v] = cd.load_raw_kp2d(p) if os.path.exists(p) else ([], {})
            cp = os.path.join(data_dir, "calib", f"{v}_cam.json")
            pts = (_load(cp).get("court_points") or {}) if os.path.exists(cp) else {}
            self.court_pts[v] = [(np.array(q["px"], float), q["court_xz"]) for q in pts.values()]
        self.pids = [p["id"] for p in self.doc["players"]]
        self.frames = {p["id"]: p["frames"] for p in self.doc["players"]}
        self.splayer = {p["id"]: p for p in self.sc["players"]}
        self.pmap = person_map(self.sc)
        self.F = int(self.sc["frame_count"])
        self.J = len(self.sc["joint_names"])
        self.resid_sources = self._resid_sources()
        contact_names = [kpset.NAMES[i] for i in kpset.CONTACT]
        self.aux_contact = [contact_names.index(n) if n in contact_names else -1
                            for n in self.sc.get("aux_names") or sk.AUX_NAMES]
        self._racket, self._shuttle = {}, {}

    def _select_views(self, views):
        if views == "all":
            return list(range(len(self.names)))
        out = []
        for v in (x.strip() for x in views.split(",") if x.strip()):
            if v not in self.names:
                raise ValueError(f"unknown view {v!r} in --views; cameras of {self.clip}: {', '.join(self.names)}")
            out.append(self.names.index(v))
        return out

    def _sidecar(self, raw, cache_dir):
        path = cd.sidecar_path(self.analysis_dir, self.clip, self.variant)
        name, want = os.path.basename(path), cd._doc_build_id(json.loads(raw), raw)
        if not os.path.exists(path):
            self.log(f"    no {name}: building in memory")
        else:
            sc = _load(path)
            if sc.get("source_build_id") != want:
                self.log(f"    {name} is stale (build {sc.get('source_build_id')} != {want}): rebuilding in memory")
            elif not all("det_shift" in c for c in sc.get("cameras", [])):
                self.log(f"    {name} is an older cvdebug/1 (no det_shift): rebuilding in memory")
            else:
                return sc
        return cd.build(self.clip, self.variant, analysis_dir=self.analysis_dir, cache_dir=cache_dir,
                        data_dir=self.data_dir, registry=self.reg, write=False, log=self.log)[1]

    @staticmethod
    def _parents(doc):
        """skeleton.parents, else by skeleton.type (smpl-24 / smpl-28 / schema.SKELETON_TYPE = smplx-58), else by
        the joint count (24 / 28 / 58)."""
        from analysis import schema
        skel = doc["skeleton"]
        names = skel["joint_names"]
        par = skel.get("parents")
        if par is not None:
            return list(par)
        by_type = {k: list(v[1]) for k, v in sk.TYPES.items()}
        by_type[schema.SKELETON_TYPE] = list(schema.PARENTS)
        par = by_type.get(skel.get("type"))
        if par is None:
            par = {len(v): v for v in by_type.values()}.get(len(names))
        if par is None or len(par) != len(names):
            raise ValueError(f"skeleton type {skel.get('type')!r} with {len(names)} joints and no parents: "
                             "can't tell the bones")
        return par

    def _resid_sources(self):
        """Per sidecar joint: 133-layout keypoint indices the residual starts from (None = no residual)."""
        names = self.sc["joint_names"]
        if self.sc.get("variant") == "pose24":
            _, err_src = cd.pose24_joint_sources(names)
            return [sorted({i for k in s for i in kpset.SOURCES[k]}) if s else None for s in err_src]
        return [ks or None for ks in cd.variant_joint_keypoints(names)]

    def readers(self):
        """{view index: VideoReader at ctx.scale, or None (no video)}; videos are checked against the kp2d header."""
        out = {}
        for ci in self.views:
            c = self.sc["cameras"][ci]
            n = (self.headers.get(c["name"]) or {}).get("frame_count")
            path, scale, proxy = c["video"], self.scale, False
            if self.source == "proxy":
                base = self.reg[self.clip].get("base", self.clip)
                pp = os.path.join(PROXY_DIR, base, f"{c['name']}.mp4")
                if os.path.exists(pp):
                    path, proxy = pp, True
                    scale = self.scale * c["width"] / max(1, video_io.probe(pp)["width"])
                else:
                    self.log(f"    {c['name']}: no proxy ({pp}) - using the original video")
            if not path:
                out[ci] = None
                continue
            if not os.path.exists(path):
                self.log(f"    {c['name']}: video not found ({path})")
                out[ci] = None
                continue
            if not proxy:
                video_io.check_video(c["name"], path, self.headers.get(c["name"]) or {}, log=self.log)
            out[ci] = video_io.VideoReader(path, scale=scale, n_frames=int(n) if n is not None else None)
        return out


def timeline_mismatch(doc, cdoc, tol=TIMELINE_TOL):
    """'' when both docs share a timeline (pose_variant time_ref_view and the offsets of common views), else why."""
    a, b = doc["meta"].get("pose_variant") or {}, cdoc["meta"].get("pose_variant") or {}
    why = []
    if (a.get("time_ref_view") or "") != (b.get("time_ref_view") or ""):
        why.append(f"time_ref_view {a.get('time_ref_view') or '-'} vs {b.get('time_ref_view') or '-'}")
    oa, ob = a.get("offsets") or {}, b.get("offsets") or {}
    diff = [f"{v} {float(oa[v]):+.2f} vs {float(ob[v]):+.2f}" for v in sorted(set(oa) & set(ob))
            if abs(float(oa[v]) - float(ob[v])) > tol]
    if diff:
        why.append("offsets " + ", ".join(diff))
    return "; ".join(why)


def raw_index(ctx, f, ci):
    """Video frame of analysis frame f in view ci: (floor(f + offset + 0.5), remainder); an exact half takes the
    tracker's raw frame f + det_shift when it is one of the two equidistant frames (video_io.view_frame)."""
    cam = ctx.sc["cameras"][ci]
    return video_io.view_frame(f, cam["offset"], cam.get("det_shift"))


def det_frame(ctx, f, ci):
    """Raw kp2d frame the sidecar's det_index points into: f + det_shift (pose24: the tracker's integer shift,
    which can differ from the video frame, e.g. offset exactly 0.5)."""
    return f + int(ctx.sc["cameras"][ci]["det_shift"])


def blank_tile(size, text):
    w, h = size
    img = np.full((h, w, 3), 70, np.uint8)
    cv2.putText(img, text, (10, h // 2), draw.FONT, 0.6, draw.WHITE, 1, cv2.LINE_AA)
    return img


def _uv(ctx, ci, X):
    return cd.project(ctx.cams[ci], X) * ctx.scale


def _masks(ctx, sp, f, V):
    J = ctx.J
    return (cd.unpack_bits(np.array(sp["seen_mask"][f * J:(f + 1) * J]), V),
            cd.unpack_bits(np.array(sp["view_mask"][f * J:(f + 1) * J]), V))


def _draw_court(ctx, img, ci):
    segs = court_segments("full")
    X = np.array([[[x0, 0.0, z0], [x1, 0.0, z1]] for (x0, z0), (x1, z1) in segs])
    posts = np.array([[[x, 0.0, 0.0], [x, NET_POST_M, 0.0]] for x in (-XD, XD)])
    for lines, thick in ((_uv(ctx, ci, X), 1), (_uv(ctx, ci, posts), 2)):
        for a, b in lines:
            if np.isfinite(a).all() and np.isfinite(b).all():
                cv2.line(img, draw._pt(a), draw._pt(b), draw.CYAN, thick, cv2.LINE_AA)


def _draw_calib(ctx, img, ci):
    s = ctx.scale
    for px, (cx, cz) in ctx.court_pts.get(ctx.names[ci], []):
        a = px * s
        cv2.drawMarker(img, draw._pt(a), draw.WHITE, cv2.MARKER_CROSS, 10, 1)
        q = _uv(ctx, ci, np.array([cx, 0.0, cz]))
        if np.isfinite(q).all():
            e = float(np.linalg.norm(px - q / s))                          # full-resolution px
            cv2.line(img, draw._pt(a), draw._pt(q), draw.err_bgr(e), 1, cv2.LINE_AA)
            cv2.putText(img, f"{e:.0f}", draw._pt(a + [4, -4]), draw.FONT, 0.35, draw.err_bgr(e), 1, cv2.LINE_AA)


def _detections(ctx, f, ci):
    """Raw kp2d detections of the frame det_index refers to (f + det_shift) + {detection: player index}."""
    ridx = det_frame(ctx, f, ci)
    frames = ctx.raw.get(ctx.names[ci]) or []
    dets = frames[ridx] if 0 <= ridx < len(frames) else np.zeros((0, 133, 3), np.float32)
    assigned = {}
    for k, pid in enumerate(ctx.pids):
        sp = ctx.splayer.get(pid)
        di = sp["det_index"][ci * ctx.F + f] if sp else -1
        if 0 <= di < len(dets):
            assigned[di] = k
    return dets, assigned


def person_map(sc):
    """{(camera index, f): {raw det index: (label, track_id, player_id)}} from the sidecar's persons[]."""
    out, V = {}, len(sc["cameras"])
    for t in sc.get("persons") or []:
        L = t["last_frame"] - t["first_frame"] + 1
        D = np.asarray(t["det_index"]).reshape(V, L)
        for ci in range(V):
            for k in np.nonzero(D[ci] >= 0)[0]:
                out.setdefault((ci, t["first_frame"] + int(k)), {})[int(D[ci, k])] = (
                    t["label"], int(t["track_id"]), t["player_id"])
    return out


def _draw_dets(ctx, img, dets, assigned, persons=None):
    """Player detections in the player colour; persons[] non-players grey with 'N<track>', unknowns dashed grey
    with '?<track>' (SP3); detections no track holds stay plain grey (SP2 behaviour)."""
    persons = persons or {}
    for i, d in enumerate(dets):
        uv = np.where(d[:, 2:3] >= pv.W_MIN, d[:, :2] * ctx.scale, np.nan)
        if i in assigned:
            draw.draw_edges(img, uv, draw.COCO_EDGES, draw.player_bgr(assigned[i]), 1)
            continue
        lab = persons.get(i)
        if lab is None:
            draw.draw_edges(img, uv, draw.COCO_EDGES, draw.GREY, 1)
            continue
        label, tid, _ = lab
        if label == "unknown":
            draw.draw_edges_dashed(img, uv, draw.COCO_EDGES, draw.GREY, 1)
            tag = f"?{tid}"
        else:
            draw.draw_edges(img, uv, draw.COCO_EDGES, draw.GREY, 1)
            tag = f"N{tid}" if label == "nonplayer" else f"T{tid}"
        ok = uv[np.isfinite(uv).all(axis=1)]
        if len(ok):
            cv2.putText(img, tag, draw._pt(ok[np.argmin(ok[:, 1])] + [4, -6]), draw.FONT, 0.4, draw.GREY, 1, cv2.LINE_AA)


def _draw_residuals(ctx, img, ci, sp, f, kp, uvj):
    s, V = ctx.scale, len(ctx.names)
    seen, view = _masks(ctx, sp, f, V)
    for j, ks in enumerate(ctx.resid_sources):
        if ks is None or j >= len(uvj) or not seen[j, ci]:
            continue
        pts = kp[ks]
        pts = pts[pts[:, 2] >= pv.W_MIN, :2]
        if not len(pts):
            continue
        a = pts.mean(axis=0) * s
        e = float(np.linalg.norm(a - uvj[j]) / s) if np.isfinite(uvj[j]).all() else float("nan")
        draw.draw_residual(img, a, uvj[j], e, rejected=not view[j, ci])
    if sp["lr_swapped"][ci * ctx.F + f] and np.isfinite(uvj).all(axis=1).any():
        cv2.putText(img, "LR", draw._pt(np.nanmin(uvj, axis=0) - [0, 6]), draw.FONT, 0.5, draw.YELLOW, 2, cv2.LINE_AA)


def _draw_players(ctx, img, f, ci, layers):
    s = ctx.scale
    dets, assigned = _detections(ctx, f, ci)
    if "dets" in layers:
        _draw_dets(ctx, img, dets, assigned, ctx.pmap.get((ci, f), {}))
    by_player = {k: d for d, k in assigned.items()}
    for k, pid in enumerate(ctx.pids):
        col, sp = draw.player_bgr(k), ctx.splayer.get(pid)
        state = sp["frame_state"][f] if sp else 0
        Jw = ctx.world[pid][0][f]
        uvj = _uv(ctx, ci, Jw) if state else np.full((len(Jw), 2), np.nan)
        kp = dets[by_player[k]] if k in by_player else None
        if kp is not None and "kp" in layers:
            draw.draw_keypoints(img, kp[:23, :2] * s, kp[:23, 2], col, w_min=pv.W_MIN)
        if kp is not None and "bbox" in layers:
            ok = kp[:23, 2] >= pv.W_MIN
            draw.draw_bbox_tag(img, kp[:23][ok, :2] * s, col, f"{pid} p {float(kp[:17, 2].mean()):.2f}")
        if state and "skel" in layers:
            draw.draw_skeleton(img, uvj, ctx.parents[:len(uvj)], col, 2, [draw.STATE_BGR.get(state, col)] * len(uvj))
        if kp is not None and state and "resid" in layers:
            _draw_residuals(ctx, img, ci, sp, f, kp, uvj)
        if "aux" in layers:
            _draw_aux(ctx, img, ci, pid, f, col)
        if pid in ctx.cmp_world:
            Jc = ctx.cmp_world[pid][0]
            if f < len(Jc):
                draw.draw_skeleton(img, _uv(ctx, ci, Jc[f]), ctx.cmp_parents[:len(Jc[f])], draw.WHITE, 1, radius=1)


def _text_lines(ctx, f, ci):
    cam, V = ctx.sc["cameras"][ci], len(ctx.names)
    ridx, rem = raw_index(ctx, f, ci)
    label = ctx.doc["meta"].get("label") or ctx.sc["skeleton_type"]
    lines = [f"{ctx.clip} | {ctx.variant} | {label}",
             f"{cam['name']}: f {f} -> raw {ridx} (offset {cam['offset']:+.2f} {cam['offset_source']}, rem {rem:+.2f})"]
    if det_frame(ctx, f, ci) != ridx:
        lines.append(f"{cam['name']}: 2D from raw {det_frame(ctx, f, ci)} (tracker shift {cam['det_shift']:+d})")
    for pid in ctx.pids:
        sp = ctx.splayer.get(pid)
        if sp is None:
            continue
        st = sp["frame_state"][f]
        vm = _masks(ctx, sp, f, V)[1].any(axis=0)
        used = ",".join(n for n, u in zip(ctx.names, vm) if u) or "-"
        fe = sp["frame_err_px"][f]
        sig = [x for x in sp["sigma_mm"][f * ctx.J:(f + 1) * ctx.J] if x >= 0]
        lines.append(f"{pid} {STATE_NAMES[st]} | err {'-' if fe < 0 else f'{fe / 10:.1f}'} px | "
                     f"sigma {'-' if not sig else f'{np.mean(sig):.0f}'} mm | views {used}")
    if ctx.sc.get("banner"):
        lines.append(ctx.sc["banner"])
    if ctx.cmp_label:
        lines.append(f"white = {ctx.cmp_label}")
    return [ascii_text(s) for s in lines]


def render_view(ctx, f, ci, img, layers):
    """Draw the selected layers of analysis frame f onto view ci's image (already at ctx.scale)."""
    cam = ctx.sc["cameras"][ci]
    if "court" in layers:
        _draw_court(ctx, img, ci)
    if "calib" in layers:
        _draw_calib(ctx, img, ci)
    _draw_players(ctx, img, f, ci, layers)
    if "racket" in layers:
        _draw_racket(ctx, img, ci, f, layers["racket"] or "rigid")
    if "shuttle" in layers:
        _draw_shuttle(ctx, img, ci, f, layers["shuttle"] or "tracker")
    if cam["excluded"]:
        draw.draw_banner(img, f"EXCLUDED (health {cam['health_px']:.0f} px)")
    elif cam["held_out"]:
        draw.draw_banner(img, "HELD OUT (not used by this build)", draw.MAGENTA)
    top = draw.banner_height(img) + 6 if (cam["excluded"] or cam["held_out"]) else 6
    if "text" in layers:
        draw.draw_text_panel(img, _text_lines(ctx, f, ci), width_chars=max(30, img.shape[1] // 9), origin=(6, top))
    return img


def _draw_aux(ctx, img, ci, pid, f, col):
    fr = ctx.frames[pid][f]
    a, ps = fr.get("aux_flat"), fr.get("position")
    if not a or not ps:
        return
    A = np.asarray(a, float).reshape(-1, 4)
    X = A[:, :3] + np.array([ps[0], 0.0, ps[1]])
    X[A[:, 3] <= 0] = np.nan
    sp = ctx.splayer.get(pid) or {}
    flags = sp.get("contact", [])[f * len(kpset.CONTACT):(f + 1) * len(kpset.CONTACT)]
    for a_i, p in enumerate(_uv(ctx, ci, X)):
        if not np.isfinite(p).all():
            continue
        c = ctx.aux_contact[a_i] if a_i < len(ctx.aux_contact) else -1
        if 0 <= c < len(flags) and flags[c]:
            cv2.circle(img, draw._pt(p), 3, col, -1, cv2.LINE_AA)          # planted foot point
        else:
            cv2.circle(img, draw._pt(p), 2, col, 1, cv2.LINE_AA)


def _draw_racket(ctx, img, ci, f, variant):
    from analysis import racket_overlay as ro
    if variant not in ctx._racket:
        p = cd.doc_path(ctx.analysis_dir, ctx.clip, ctx.variant).replace(".analysis.json", f".racket_{variant}.json")
        ctx._racket[variant] = _load(p) if os.path.exists(p) else None
    sc = ctx._racket[variant]
    if not sc:
        return
    for k, sp in enumerate(sc["players"]):
        frames = ctx.frames.get(sp["id"])
        pos = frames[f].get("position") if frames and f < len(frames) else None
        if pos is None or f >= len(sp["frames"]):
            continue
        X = ro._variant_keypoints(sc, k, f, pos)
        if X is None:
            continue
        uv = _uv(ctx, ci, X)
        for a, b in ro.SKEL:
            if np.isfinite(uv[[a, b]]).all():
                cv2.line(img, draw._pt(uv[a]), draw._pt(uv[b]), ro.COLORS.get(variant, draw.YELLOW), 1, cv2.LINE_AA)


def _shuttle_sidecar(ctx, variant):
    """Next to the doc (shuttle_variants.sidecar_path), else next to the base clip's doc."""
    doc = cd.doc_path(ctx.analysis_dir, ctx.clip, ctx.variant)
    base = ctx.reg[ctx.clip].get("base", ctx.clip)
    for p in (doc.replace(".analysis.json", f".shuttle_{variant}.json"),
              os.path.join(ctx.analysis_dir, f"{base}.shuttle_{variant}.json")):
        if os.path.exists(p):
            return _load(p)
    return None


def _draw_shuttle(ctx, img, ci, f, variant, trail=10):
    if variant not in ctx._shuttle:
        ctx._shuttle[variant] = _shuttle_sidecar(ctx, variant)
    sc = ctx._shuttle[variant]
    if not sc:
        return
    pts = [sc["frames"][g].get("position") for g in range(max(0, f - trail), min(f + 1, len(sc["frames"])))]
    uv = [(_uv(ctx, ci, np.array(p, float)) if p else None) for p in pts]
    for a, b in zip(uv, uv[1:]):
        if a is not None and b is not None and np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img, draw._pt(a), draw._pt(b), draw.SHUTTLE_BGR, 1, cv2.LINE_AA)
    if uv and uv[-1] is not None and np.isfinite(uv[-1]).all():
        cv2.circle(img, draw._pt(uv[-1]), 4, draw.SHUTTLE_BGR, -1, cv2.LINE_AA)


def view_tile(ctx, f, ci, layers, readers, size=None):
    """View ci's synced raw frame (grey tile when there is none) with the layers drawn."""
    cam = ctx.sc["cameras"][ci]
    w, h = size or (int(round(cam["width"] * ctx.scale)), int(round(cam["height"] * ctx.scale)))
    ridx, _ = raw_index(ctx, f, ci)
    r = readers.get(ci)
    img = r.get(ridx) if r is not None else None
    if img is None:
        img = blank_tile((w, h), ascii_text(f"{cam['name']}: no frame (raw {ridx})"))
    elif img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return render_view(ctx, f, ci, img, layers)


def _fit(img, size, bg=30):
    w, h = size
    s = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), bg, np.uint8)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def court_tile(ctx, f, size, trail=120):
    """Top-down court (CourtMap geometry, far side up): pelvis tracks of the last `trail` frames and the cameras."""
    cpos = np.array([c["unity_pos"] for c in ctx.sc["cameras"]], float).reshape(-1, 3)
    need = max([1.2] + [max(abs(x) - XD, abs(z) - ZB) + 0.8 for x, _, z in cpos])
    cm = CourtMap(half="full", px_per_m=40, margin_m=min(need, 8.0))
    img = np.full((cm.h, cm.w, 3), (40, 90, 40), np.uint8)
    for a, b in court_segments("full"):
        cv2.line(img, cm.pt(*a), cm.pt(*b), draw.WHITE, 2, cv2.LINE_AA)
    for k, pid in enumerate(ctx.pids):
        pts = [cm.pt(*fr["position"]) for fr in ctx.frames[pid][max(0, f - trail):f + 1] if fr.get("position")]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, draw.player_bgr(k), 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 7, draw.player_bgr(k), -1)
    lim = cm.xmin + 0.3
    for cam, (x, _, z) in zip(ctx.sc["cameras"], cpos):
        p = cm.pt(float(np.clip(x, lim, -lim)), float(np.clip(z, cm.zmin + 0.3, -cm.zmin - 0.3)))
        colour = draw.RED if cam["excluded"] else (draw.MAGENTA if cam["held_out"] else draw.CYAN)
        cv2.drawMarker(img, p, colour, cv2.MARKER_TRIANGLE_UP, 14, 2)
        cv2.putText(img, cam["name"], (p[0] + 8, p[1]), draw.FONT, 0.45, colour, 1, cv2.LINE_AA)
    return _fit(img, size)


def stats_tile(ctx, f, size):
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    lines = [f"{ctx.clip} {ctx.variant} f {f}/{ctx.F}"]
    for v in ctx.sc["stats"]["views"]:
        cam = ctx.sc["cameras"][ctx.names.index(v["name"])]
        tag = " EXCLUDED" if cam["excluded"] else (" held-out" if cam["held_out"] else "")
        lines.append(f"{v['name']}: used {100 * v['used_frac']:.0f}% | health {cam['health_px']:.1f} px{tag}")
    for pid, c in zip([p["id"] for p in ctx.sc["players"]], ctx.sc["stats"]["coverage_per_player"]):
        lines.append(f"{pid}: coverage {100 * c:.1f}%")
    draw.draw_text_panel(img, [ascii_text(s) for s in lines], width_chars=max(30, w // 8))
    return img


def mosaic_layout(n_views):
    """(rows, cols): the court tile always fits; a stats tile fills a free slot (4 views -> 2x3 with stats)."""
    cols = 3 if n_views + 1 > 4 else 2
    return math.ceil((n_views + 1) / cols), cols


def compose_mosaic(tiles, extra, layout, tile_size):
    rows, cols = layout
    w, h = tile_size
    out = np.full((rows * h, cols * w, 3), 30, np.uint8)
    for i, t in enumerate(list(tiles) + list(extra)):
        if i >= rows * cols:
            break
        r, c = divmod(i, cols)
        if t.shape[:2] != (h, w):
            t = _fit(t, (w, h))                                # letterbox: a tile of another aspect is not stretched
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return np.vstack([out, draw.legend_strip(out.shape[1])])


def crop_box(points, img_wh, margin=0.35, aspect=(9, 16)):
    """Box of the given aspect around the finite points (+ margin), clipped to the image; whole image if none."""
    W, H = img_wh
    p = np.asarray(points, float).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        return 0, 0, W, H
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(y1 - y0, (x1 - x0) * aspect[1] / aspect[0], 8.0) * (1 + 2 * margin)
    h = min(h, H, W * aspect[1] / aspect[0])
    w = h * aspect[0] / aspect[1]
    xa = int(np.clip(cx - w / 2, 0, W - w))
    ya = int(np.clip(cy - h / 2, 0, H - h))
    return xa, ya, int(xa + w), int(ya + h)


def check_out_dir(path):
    """Privacy (spec § 5): frame-bearing images never go under the repo's Assets/ or docs/."""
    p = os.path.normcase(os.path.realpath(path))
    for bad in ("Assets", "docs"):
        root = os.path.normcase(os.path.realpath(os.path.join(REPO, bad)))
        if p == root or p.startswith(root + os.sep):
            raise ValueError(f"output dir {path} not allowed (frame-bearing images must stay out of {bad}/): "
                             f"use data/overlays/<clip>/")


class FFmpegWriter:
    """BGR frames piped to ffmpeg (libx264 yuv420p crf 20); odd sizes are padded to even."""

    def __init__(self, path, fps, size):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH (needed to write mp4; still mode needs no ffmpeg)")
        w, h = size
        self.path, self.size, self.n = path, (w + w % 2, h + h % 2), 0
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
               f"{self.size[0]}x{self.size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "20", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _failed(self):
        return RuntimeError(f"ffmpeg failed (exit {self.p.wait()}) writing {self.path}")

    def write(self, img):
        w, h = self.size
        if img.shape[:2] != (h, w):
            img = img[:h, :w]
            pad = np.zeros((h, w, 3), np.uint8)
            pad[:img.shape[0], :img.shape[1]] = img
            img = pad
        try:
            self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        except OSError:
            raise self._failed() from None
        self.n += 1

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        if self.p.wait() != 0:
            raise self._failed()


PREVIEW_KEYS = ("space play/pause | a/d or arrows step | 1-9 toggle the first nine layers (racket/shuttle via "
                "--layers only) | v cycle views | q/Esc quit")
_LEFT, _RIGHT = (2424832, 65361, 81), (2555904, 65363, 83)     # waitKeyEx arrows: Windows, GTK, Qt/mac low byte


def handle_key(state, key, n_frames, layer_names, n_views):
    """Pure preview state transition (the unit-tested part of the window loop); never mutates `state`."""
    st = dict(state, layers=set(state["layers"]))
    if key in (ord("q"), 27):
        st["quit"] = True
    elif key == ord(" "):
        st["playing"] = not st["playing"]
    elif key in (ord("d"),) + _RIGHT:
        st["f"] = min(n_frames - 1, st["f"] + 1)
    elif key in (ord("a"),) + _LEFT:
        st["f"] = max(0, st["f"] - 1)
    elif ord("1") <= key <= ord("9") and key - ord("1") < len(layer_names):
        st["layers"] ^= {layer_names[key - ord("1")]}
    elif key == ord("v"):
        st["view"] = st["view"] + 1 if st["view"] + 1 < n_views else -1
    return st


def preview(ctx, layers, readers, crop, tile, a):
    """cv2 window over the same frame images render writes (mosaic, or one view full tile); writes no file.
    Layer toggles keep each layer's option (racket=bootstrap ...)."""
    names = list(LAYERS)
    st = {"f": min(max(a.frame, 0), ctx.F - 1), "playing": False, "layers": set(layers), "view": -1, "quit": False}
    print(PREVIEW_KEYS)
    print("    layers: " + " ".join(f"{i + 1}={n}" for i, n in enumerate(names[:9])))
    win = ascii_text(f"overlay {ctx.clip} {ctx.variant}")
    try:
        while not st["quit"]:
            lay = {k: layers.get(k) for k in st["layers"]}
            if st["view"] < 0:
                img = _frame_image(ctx, st["f"], lay, readers, "mosaic", crop, tile)
            else:
                img = view_tile(ctx, st["f"], ctx.views[st["view"]], lay, readers)
            cv2.imshow(win, img)
            key = cv2.waitKeyEx(max(1, int(1000 / (ctx.sc["fps"] or 60.0))) if st["playing"] else 0)
            if key == -1:
                if st["playing"]:
                    st["f"] = min(ctx.F - 1, st["f"] + 1)
                    st["playing"] = st["f"] < ctx.F - 1
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:       # window closed with the mouse
                    break
                continue
            st = handle_key(st, key, ctx.F, names, len(ctx.views))
    finally:
        cv2.destroyAllWindows()
    return 0


def _even_width(width):
    """libx264 yuv420p needs even sizes: round the short side up to even; < 2 is an error."""
    if int(width) < 2:
        raise ValueError(f"proxy width must be >= 2, got {width}")
    return (int(width) + 1) // 2 * 2


def _rate_str(fps):
    """60 -> '60', 59.94006 -> '60000/1001': the exact rational for setpts and -r."""
    r = Fraction(fps).limit_denominator(1001)
    return str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"


def proxy_command(src, dst, width, fps):
    """Short side -> width (even), aspect kept (cameras are a mix of portrait and landscape). Frames are re-stamped
    by index (setpts=N/(fps*TB)) before the rate stage, so every timestamp already sits on the output grid and
    `-fps_mode cfr` neither drops nor duplicates: proxy frame k IS source frame k even for a VFR source, and the
    stream is tagged constant-rate (ffmpeg >= 7 rejects -r with passthrough; passthrough alone mis-tags the rate).
    A 10-frame GOP keeps Unity's VideoPlayer seeks fast."""
    w, r = _even_width(width), _rate_str(fps)
    vf = f"setpts=N/({r}*TB),scale='if(gt(iw,ih),-2,{w})':'if(gt(iw,ih),{w},-2)'"
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-fps_mode", "cfr",
            "-r", r, "-g", "10", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", dst]


def make_proxies(clip, width=540, out_root=PROXY_DIR, registry=None, log=print):
    """Every view with a video -> <out_root>/<base clip>/<view>.mp4, verified frame-exact against the source.
    Encodes to <view>.tmp.mp4 and only replaces <view>.mp4 once the frame count matches (a mismatch is kept as
    <view>.bad.mp4 and the command fails), so a bad encode never clobbers a good proxy."""
    reg = registry if registry is not None else config.load_registry()
    if clip not in reg:
        raise KeyError(f"unknown clip {clip!r}; registry has {', '.join(sorted(reg))}")
    base = reg[clip].get("base", clip)
    out_dir = os.path.join(out_root, base)
    check_out_dir(out_dir)                                 # privacy: refuse before anything is probed or encoded
    width = _even_width(width)
    os.makedirs(out_dir, exist_ok=True)
    bad = 0
    for v, src in video_io.resolve_videos(clip, registry=reg, root="").items():
        if not src or not os.path.exists(src):
            log(f"    {v}: no video - skipped")
            continue
        info = video_io.probe(src)
        dst, tmp, bad_path = (os.path.join(out_dir, f"{v}{ext}") for ext in (".mp4", ".tmp.mp4", ".bad.mp4"))
        t0 = time.perf_counter()
        try:
            subprocess.run(proxy_command(src, tmp, width, info["fps"] or 60.0), check=True)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        got = video_io.probe(tmp)["frames"]
        if got != info["frames"]:
            os.replace(tmp, bad_path)
            log(f"    {v}: proxy has {got} frames, source {info['frames']} - NOT frame-exact (kept as {v}.bad.mp4)")
            bad += 1
            continue
        os.replace(tmp, dst)
        if os.path.exists(bad_path):
            os.remove(bad_path)
        log(f"    {v}: {dst} ({got} frames, {time.perf_counter() - t0:.0f} s)")
    return 1 if bad else 0


def _player_points(ctx, f, ci, pid):
    return _uv(ctx, ci, ctx.world[pid][0][f])


def _crop_tile(img, points, tile):
    """Crop around the points with the tile's own aspect, then resize to the tile: no stretch on any camera shape."""
    x0, y0, x1, y1 = crop_box(points, (img.shape[1], img.shape[0]), aspect=tile)
    return cv2.resize(img[y0:y1, x0:x1], tile, interpolation=cv2.INTER_LINEAR)


def _close_all(writers, quiet=False):
    """Close every writer; raise the first failure unless quiet (another exception is already propagating)."""
    first = None
    for w in writers.values():
        try:
            w.close()
        except Exception as e:                                 # noqa: BLE001 - re-raised below
            first = first or e
    if first is not None and not quiet:
        raise first


def _frame_image(ctx, f, layers, readers, mode, crop, tile):
    """views: [tile per view]; mosaic / still: one grid (views + court + stats) with the legend strip under it."""
    tiles = []
    for ci in ctx.views:
        t = view_tile(ctx, f, ci, layers, readers)
        if crop:
            t = _crop_tile(t, _player_points(ctx, f, ci, crop), tile)
        tiles.append(t)
    if mode == "views":
        return tiles
    extra = [court_tile(ctx, f, tile)]
    rows, cols = mosaic_layout(len(tiles))
    if rows * cols > len(tiles) + 1:
        extra.append(stats_tile(ctx, f, tile))
    return compose_mosaic(tiles, extra, (rows, cols), tile)


def tile_size(ctx, scale, tile=None):
    """Mosaic tile (w, h): --tile WxH, else the orientation most selected views have (tie: portrait) at the size of
    the first selected camera with that orientation, x scale."""
    if tile:
        w, _, h = str(tile).lower().partition("x")
        if not (w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0):
            raise ValueError(f"--tile must be WxH (e.g. 960x540), got {tile!r}")
        return int(w), int(h)
    cams = [ctx.sc["cameras"][ci] for ci in ctx.views]
    land = [c for c in cams if c["width"] > c["height"]]
    port = [c for c in cams if c["width"] <= c["height"]]
    c = (land if len(land) > len(port) else port or land)[0]
    return int(round(c["width"] * scale)), int(round(c["height"] * scale))


def render(a, log=print):
    check_out_dir(a.out)                                   # privacy: refuse before anything is loaded or decoded
    if a.stride < 1:
        raise SystemExit(f"--stride must be >= 1, got {a.stride}")
    crop = None
    if a.crop:
        kind, _, crop = a.crop.partition(":")
        if kind != "player" or not crop:
            raise SystemExit(f"--crop must be player:<id>, got {a.crop!r}")
    os.makedirs(a.out, exist_ok=True)
    ctx = OverlayContext(a.clip, a.variant, compare=a.compare, analysis_dir=a.analysis_dir, cache_dir=a.cache_dir,
                         data_dir=a.data_dir, views=a.views, scale=a.scale, source=a.source, log=log)
    if crop and crop not in ctx.world:
        raise SystemExit(f"--crop player {crop!r} not in {a.clip} {a.variant} (players: {', '.join(ctx.pids)})")
    layers = parse_layers(a.layers)
    tile = tile_size(ctx, a.scale, getattr(a, "tile", None))
    readers = ctx.readers()
    for ci in ctx.views:
        c = ctx.sc["cameras"][ci]
        if readers.get(ci) is None:
            log(f"    {c['name']}: no video ({c['video'] or 'not in clips.json'}) - grey tiles")
    try:
        if a.preview:
            return preview(ctx, layers, readers, crop, tile, a)
        if a.mode == "still":
            if not 0 <= a.frame < ctx.F:
                raise SystemExit(f"--frame {a.frame} outside 0..{ctx.F - 1}")
            img = _frame_image(ctx, a.frame, layers, readers, "mosaic", crop, tile)
            p = os.path.join(a.out, f"{a.variant}_still_{a.frame}.png")
            if not cv2.imwrite(p, img):
                raise RuntimeError(f"could not write {p}")
            log(f"[overlay] wrote {p}")
            return 0
        if not a.frames:
            raise SystemExit("--frames a:b is required for video modes (whole clip: --frames 0:)")
        frames = parse_frames(a.frames, ctx.F)[::a.stride]
        if not frames:
            raise SystemExit(f"--frames {a.frames} selects no frame (the clip has {ctx.F})")
        if len(frames) > WHOLE_CLIP_WARN:
            log(f"    WARNING: {len(frames)} frames at full-resolution decode - expect a long run (--source proxy)")
        tag = f"{frames[0]}-{frames[-1] + 1}"
        writers = {}
        t0 = time.perf_counter()
        try:
            for f in frames:
                out = _frame_image(ctx, f, layers, readers, a.mode, crop, tile)
                imgs = out if a.mode == "views" else [out]
                for k, img in enumerate(imgs):
                    key = ctx.names[ctx.views[k]] if a.mode == "views" else "mosaic"
                    if key not in writers:
                        writers[key] = FFmpegWriter(os.path.join(a.out, f"{a.variant}_{key}_{tag}.mp4"),
                                                    ctx.sc["fps"] / a.stride, (img.shape[1], img.shape[0]))
                    writers[key].write(img)
        except BaseException:
            _close_all(writers, quiet=True)                    # keep the original exception
            raise
        _close_all(writers)
        dt = max(time.perf_counter() - t0, 1e-9)
        dec = sum(r.decoded for r in readers.values() if r)
        for w in writers.values():
            log(f"[overlay] wrote {w.path}")
        log(f"[overlay] {len(frames)} frames in {dt:.0f} s ({len(frames) / dt:.1f} fps out, {dec / dt:.1f} decoded fps)")
        return 0
    finally:
        for r in readers.values():
            if r:
                r.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="analysis.overlay_video")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--clip", required=True)
    r.add_argument("--variant", default="pose24")
    r.add_argument("--compare", help="variant of the same clip, or clip:variant")
    r.add_argument("--views", default="all")
    r.add_argument("--layers", default=DEFAULT_LAYERS)
    r.add_argument("--frames")
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--mode", choices=("mosaic", "views", "still"), default="mosaic")
    r.add_argument("--frame", type=int, default=0)
    r.add_argument("--scale", type=float, default=0.25)
    r.add_argument("--tile", help="mosaic tile WxH (default: majority camera orientation x --scale)")
    r.add_argument("--crop", help="player:P1")
    r.add_argument("--preview", action="store_true")
    r.add_argument("--source", choices=("original", "proxy"), default="original")
    r.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    r.add_argument("--cache-dir", default=pv.CACHE_DIR)
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out")
    p = sub.add_parser("proxies")
    p.add_argument("--clip", required=True)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--out", default=PROXY_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "render":
        a.out = a.out or os.path.join("data", "overlays", a.clip)
        return render(a)
    return make_proxies(a.clip, width=a.width, out_root=a.out)


if __name__ == "__main__":
    raise SystemExit(main())
