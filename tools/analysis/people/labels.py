"""Pre-fill + confirm identity labelling (SP3 spec § 4.1; user decision 2026-09-29).

The page (tools/labeler.html, served by serve_calibrator.py at /labeler) shows one camera frame with every kp2d
detection, pre-filled with tracker3d's label; the user accepts (Enter) or corrects (1-4 = P1-P4, N, S = skip).
Labels go to data/gt/labels/<clip>_ids.json (idlabels/1, gitignored) keyed on raw kp2d frames, so they don't depend
on any sync. Each label records whether it confirmed or corrected the proposal (bias check). Frame choice: frames
where tracker3d and another tracker (v2) disagree first, then events (player track start / end, two players < 1 m,
unknown tracks), then every 30th frame; at most 40 frames per clip, each shown for every camera the tracker used."""
from __future__ import annotations

import base64
import json
import os
import threading

import cv2
import numpy as np

from analysis import cvdebug, video_io
from analysis.people import persons as pm
from analysis.people import seeds as sd
from analysis.people.eval_identity import FORMAT


def _player_floor(doc):
    out = {}
    for t in doc["tracks"]:
        if t["label"] == "player" and t["player_id"]:
            fl = np.asarray(t["floor_xz"], float).reshape(-1, 2)
            for k, p in enumerate(fl):
                if p[0] != pm.NONE_XZ:
                    out.setdefault(t["first_frame"] + k, {})[t["player_id"]] = p
    return out


RANK = {"disagree": 0, "track start": 1, "track end": 1, "players close": 1, "unknown": 1, "regular": 2}
QUOTA = (0.5, 0.3)          # share of n_max for disagreements, events; regular frames take the rest
MIN_GAP = 5                 # frames between two picks (regular frames are exempt)
CONF = 0.3                  # a detection is drawn on the page (and must be labelled) when a body keypoint reaches this


def _spread_order(n):
    """Indices 0..n-1 ordered so that every prefix is spread over the range (ends, middle, quarters, ...)."""
    if n <= 0:
        return []
    out, seen, q = [], set(), [(0, n - 1)]
    for i in (0, n - 1):
        if i not in seen:
            seen.add(i)
            out.append(i)
    while q:
        nq = []
        for lo, hi in q:
            if hi - lo < 2:
                continue
            m = (lo + hi) // 2
            if m not in seen:
                seen.add(m)
                out.append(m)
            nq += [(lo, m), (m, hi)]
        q = nq
    return out + [i for i in range(n) if i not in seen]


def frame_list(pdoc, other=None, every=30, n_max=40, near_m=1.0):
    F = int(pdoc["frame_count"])
    reasons = {}

    def add(f, why):
        if 0 <= f < F:
            reasons.setdefault(int(f), []).append(why)
    if other is not None:
        a, b = pm.predictions(pdoc), pm.predictions(other)
        shift = dict(zip(pdoc["views"], pdoc["det_shift"]))
        for key in set(a) | set(b):
            if a.get(key, "N") != b.get(key, "N") and key[0] in shift:
                add(key[1] - shift[key[0]], "disagree")
    for t in pdoc["tracks"]:
        if t["label"] == "player":
            if t["first_frame"] > 0:
                add(t["first_frame"], "track start")
            if t["last_frame"] < F - 1:
                add(t["last_frame"], "track end")
        elif t["label"] == "unknown":
            add((t["first_frame"] + t["last_frame"]) // 2, "unknown")
    for f, ps in _player_floor(pdoc).items():
        q = list(ps.values())
        if any(np.linalg.norm(q[i] - q[j]) < near_m for i in range(len(q)) for j in range(i + 1, len(q))):
            add(f, "players close")
    for f in range(0, F, every):
        add(f, "regular")
    # group by best rank; within a group take frames spread over the clip (a dense run of disagreements at the
    # start must not fill every slot); quotas per group, unused quota passes on, then a back-fill pass
    groups = [sorted(f for f in reasons if min(RANK[r] for r in reasons[f]) == k) for k in range(3)]
    orders = [[g[i] for i in _spread_order(len(g))] for g in groups]
    taken, pos = [], [0, 0, 0]

    def take(k, quota):
        n = 0
        while n < quota and pos[k] < len(orders[k]) and len(taken) < n_max:
            f = orders[k][pos[k]]
            pos[k] += 1
            if "regular" in reasons[f] or all(abs(f - g) >= MIN_GAP for g in taken):
                taken.append(f)
                n += 1
        return n
    carry = 0
    for k, share in enumerate(QUOTA):
        quota = int(share * n_max) + carry
        carry = quota - take(k, quota)
    take(2, n_max - len(taken))
    for k in range(3):
        take(k, n_max - len(taken))
    taken.sort(key=lambda f: (min(RANK[r] for r in reasons[f]), f))
    return [{"f": f, "reasons": sorted(set(reasons[f]))} for f in taken]


def _visible(d):
    """Drawn when any of the 23 body + feet keypoints has score >= CONF; labeler.html box() uses this flag (dets[].visible)."""
    return bool(np.any(np.nan_to_num(d[:23, 2]) >= CONF))


# "U" (accepting an unknown-track proposal as is) is stored but the page never pre-fills or offers it: the user must
# decide P1-P4 / N / S for those detections.
LABELS = ("P1", "P2", "P3", "P4", "N", "S", "U")


def upsert(doc, clip, view, raw, entries, proposals):
    bad = [e.get("label") for e in entries if e.get("label") not in LABELS]
    if bad:
        raise ValueError(f"labels must be one of {', '.join(LABELS)}; got {bad}")
    idx = {(l["view"], l["raw_frame"], l["det"]): i for i, l in enumerate(doc["labels"])}
    n = 0
    for e in entries:
        det, lab = int(e["det"]), str(e["label"])
        prop = proposals.get(det, "")
        row = {"view": view, "raw_frame": int(raw), "det": det, "label": lab, "proposal": prop,
               "source": "manual" if not prop else ("confirmed" if prop == lab else "corrected")}
        key = (view, int(raw), det)
        if key in idx:
            doc["labels"][idx[key]] = row
        else:
            idx[key] = len(doc["labels"])
            doc["labels"].append(row)
        n += 1
    return n


class LabelService:
    def __init__(self, root=".", registry=None, analysis_dir=None, eval_dir=None, labels_dir=None, data_dir=None):
        from analysis.pose24 import config
        self.root = root
        self.reg = registry if registry is not None else config.load_registry()
        self.analysis_dir = analysis_dir or os.path.join(root, config.DEFAULT_ANALYSIS_DIR)
        self.eval_dir = eval_dir or os.path.join(root, "data", "people_eval")
        self.labels_dir = labels_dir or os.path.join(root, "data", "gt", "labels")
        self.data_dir = data_dir or os.path.join(root, "data")
        self._persons, self._raw, self._frames, self._readers = {}, {}, {}, {}
        self._seeds = {}
        self._lock = threading.Lock()                 # video readers: serve_calibrator is threaded, readers are stateful
        self._label_lock = threading.Lock()           # label file read-modify-write

    def _check(self, clip):
        """Only registry clips: the name is joined into read / write paths, so a crafted one (../x) must not pass."""
        if not isinstance(clip, str) or clip not in self.reg:
            raise KeyError(f"clip {clip!r} not in the registry")

    def _ppath(self, clip):
        for p in (pm.path(self.analysis_dir, clip), os.path.join(self.eval_dir, f"{clip}.3d.persons.json")):
            if os.path.exists(p):
                return p
        return None

    def clips(self):
        return [{"clip": c, "persons": os.path.basename(self._ppath(c))} for c in self.reg if self._ppath(c)]

    def persons(self, clip):
        self._check(clip)
        if clip not in self._persons:
            self._persons[clip] = pm.load(self._ppath(clip))
        return self._persons[clip]

    def _raw_frames(self, view):
        if view not in self._raw:
            self._raw[view] = cvdebug.load_raw_kp2d(os.path.join(self.data_dir, "pose2d", f"{view}_kp2d.json"))[0]
        return self._raw[view]

    def _labels_path(self, clip):
        return os.path.join(self.labels_dir, f"{clip}_ids.json")

    def _labels(self, clip):
        p = self._labels_path(clip)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"format": FORMAT, "clip": clip, "labels": []}

    def frames(self, clip):
        self._check(clip)
        if clip not in self._frames:
            doc = self.persons(clip)
            v2 = os.path.join(self.eval_dir, f"{clip}.v2.persons.json")
            other = pm.load(v2) if os.path.exists(v2) else None
            shift = dict(zip(doc["views"], doc["det_shift"]))
            self._frames[clip] = [{"f": x["f"], "view": v, "raw": x["f"] + shift[v], "reasons": x["reasons"]}
                                  for x in frame_list(doc, other) for v in doc["views_used"]]
        have = {}
        for l in self._labels(clip)["labels"]:
            have.setdefault((l["view"], int(l["raw_frame"])), set()).add(int(l["det"]))
        items = [dict(it, done=self._done(it["view"], it["raw"], have.get((it["view"], it["raw"]), set())))
                 for it in self._frames[clip]]
        return {"items": items, "done": sum(it["done"] for it in items)}

    def _done(self, view, raw, labelled):
        """Done = every drawn detection of (view, raw) has a label row (S counts); a frame with none is done."""
        frames = self._raw_frames(view)
        dets = frames[raw] if 0 <= raw < len(frames) else ()
        return all(i in labelled for i, d in enumerate(dets) if _visible(d))

    def frame(self, clip, view, raw):
        self._check(clip)
        raw = int(raw)
        if view not in self.persons(clip)["views"]:
            raise KeyError(f"view {view!r} not in {clip}")
        frames = self._raw_frames(view)
        dets = frames[raw] if 0 <= raw < len(frames) else np.zeros((0, 133, 3), np.float32)
        pred = pm.predictions(self.persons(clip))
        have = {(l["det"]): l["label"] for l in self._labels(clip)["labels"] if l["view"] == view and l["raw_frame"] == raw}
        cal = os.path.join(self.data_dir, "calib", f"{view}_cam.json")
        c = {}
        if os.path.exists(cal):
            with open(cal, "r", encoding="utf-8") as fh:
                c = json.load(fh)
        return {"width": c.get("width"), "height": c.get("height"),
                "dets": [{"det": i, "kps": np.round(np.nan_to_num(d[:23], nan=0.0), 1).tolist(),   # JSON-safe
                          "proposal": pred.get((view, raw, i), "N") if (view, raw, i) in pred else "",
                          "label": have.get(i, ""), "visible": _visible(d)} for i, d in enumerate(dets)]}

    def _get(self, path, idx):
        with self._lock:
            if path not in self._readers:
                self._readers[path] = video_io.VideoReader(path)
            return self._readers[path].get(int(idx))

    def _video(self, clip, view):
        vids = video_io.resolve_videos(clip, registry=self.reg, root=self.root)
        return vids.get(view)

    def image(self, clip, view, raw, max_side=1600):
        self._check(clip)
        path = self._video(clip, view)
        if not path or not os.path.exists(path):
            return None
        img = self._get(path, raw)
        if img is None:
            return None
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def seeds(self, clip, view):
        """Seed crops (who is P1..P4) at the view's first seed time: [{pid, jpeg (base64)}] (cached per clip + view)."""
        self._check(clip)
        if (clip, view) not in self._seeds:
            self._seeds[(clip, view)] = self._seed_crops(clip, view)
        return self._seeds[(clip, view)]

    def _seed_crops(self, clip, view):
        e = self.reg[clip]
        base = self.reg[e.get("base", clip)]
        if not base.get("seeds"):
            return []
        p = os.path.join(self.data_dir, "calib", f"{base['seeds']}_seeds.json")
        if not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8") as fh:
            sv = json.load(fh).get("views", {})
        fps = float(self.persons(clip).get("fps", 60.0))
        out = []
        for v, r, pid, (u, w) in sd.seed_events(sv, [view], fps)[:4]:
            path = self._video(clip, v)
            if not path or not os.path.exists(path):
                return []
            img = self._get(path, r)
            if img is None:
                continue
            h = int(0.12 * img.shape[0])
            y0, x0 = max(0, int(w) - 2 * h), max(0, int(u) - h // 2)
            crop = img[y0:int(w) + h // 4, x0:x0 + h]
            if crop.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                out.append({"pid": pid, "jpeg": base64.b64encode(buf.tobytes()).decode()})
        return out

    def save(self, clip, view, raw, entries):
        self._check(clip)
        if view not in self.persons(clip)["views"]:
            raise KeyError(f"view {view!r} not in {clip}")
        with self._label_lock:
            return self._save(clip, view, raw, entries)

    def _save(self, clip, view, raw, entries):
        doc = self._labels(clip)
        pred = pm.predictions(self.persons(clip))
        props = {d: pred[(view, int(raw), d)] for d in {int(e["det"]) for e in entries} if (view, int(raw), d) in pred}
        n = upsert(doc, clip, view, int(raw), entries, props)
        os.makedirs(self.labels_dir, exist_ok=True)
        p = self._labels_path(clip)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, p)
        return n
