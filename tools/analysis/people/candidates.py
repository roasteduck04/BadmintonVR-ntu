"""Per synced frame: 2D detections -> cross-view 3D person candidates (SP3 spec 2.1).

Detections of two views pair when their floor points (pose_v2.court_pos: ankles, else hips @ 0.9 m, else shoulders
@ 1.4 m; no centroid fallback) are < floor_pair_m apart or their ankle rays pass < ray_pair_m apart, and a 2-view DLT of shoulders/hips/knees/ankles reprojects
< max(tri_px, tri_bbox_frac x bbox height). Consistent pairs merge greedily (lowest error first, one detection per
view) into multi-view candidates with a triangulated floor point and stature. A detection left over whose floor point
is within loose_attach_m of a multi-view candidate lacking that view is attached "loose" (same person, 2D too poor to
pair). The rest become 1-view candidates (grouped across views when their floor points are < merge_1view_m apart), so
players seen by one camera don't vanish. All geometry in the court world (Y-down, floor Y = 0)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis import pose_v2 as pv

TORSO_LEGS = [5, 6, 11, 12, 13, 14, 15, 16]
HEAD = [0, 1, 2, 3, 4]
HEAD_BELOW_TOP = 0.10      # unused since v0.7.5 (review 1 L12): heights use params head_top_m (0.18) everywhere
ANKLE_H = 0.08             # template ankle keypoint height (m); real clips: estimate_ankle_height (T15)


@dataclass
class Candidate:
    frame: int
    dets: dict                          # view -> local detection index in the tracker's (shifted) stream
    floor: np.ndarray                   # (2,) court x, z (m)
    support: int                        # views
    height: float = float("nan")        # m, triangulated head top -> floor
    knee_deg: float = float("nan")      # max knee flexion (deg)
    err_px: float = float("nan")        # mean reprojection of the triangulation
    pix_h: float = float("nan")         # 1-view only: head -> ankle span in pixels
    track_ids: dict = field(default_factory=dict)
    loose: tuple = ()                   # views attached without passing the pairing check
    crowd_dist: float = float("inf")    # floor distance to the nearest other candidate this frame
    tri: bool = False
    ankle_h: float = float("nan")       # triangulated ankle keypoint height above the floor (m)


def floor_point(kps, cam, s_min=pv.W_MIN, ankle_h=ANKLE_H):
    """pose_v2.court_pos, except ankle keypoints are intersected with the plane ankle_h above the floor (they are not
    on it; with phone cameras 2.5-3 m up and 10+ m away, a plane dh off biases the floor spot by dh x range / camera
    height along the ray and breaks cross-view floor pairing). Tracker3D passes the clip's estimate_ankle_height."""
    for (a, b), h in (((15, 16), ankle_h), ((11, 12), pv.HIP_HEIGHT), ((5, 6), pv.SHOULDER_HEIGHT)):
        pts = [kps[j, :2] for j in (a, b) if kps[j, 2] >= s_min]
        if pts:
            u, v = np.mean(pts, axis=0)
            q = pv._plane_point(u, v, *cam, h)
            return None if q is None else np.array(q, float)
    return None


def bbox_height(kps, s_min=pv.W_MIN):
    ys = kps[:17][kps[:17, 2] >= s_min, 1]
    return float(ys.max() - ys.min()) if len(ys) >= 2 else 0.0


def pixel_height(kps, s_min=pv.W_MIN):
    head = kps[HEAD][kps[HEAD, 2] >= s_min, 1]
    ank = kps[[15, 16]][kps[[15, 16], 2] >= s_min, 1]
    return float(ank.max() - head.min()) if len(head) and len(ank) else float("nan")


def _tri2(cams, a, b, ka, kb, idx=TORSO_LEGS, w_min=pv.W_MIN, full=False):
    """Two-view DLT of keypoints `idx` -> median reprojection px over both views (inf when < 3 usable points
    or a point lands behind a camera). full: also the pair's floor (x, z) - triangulated ankles, else hips, else
    None."""
    uv = np.stack([ka[idx, :2], kb[idx, :2]], axis=1).astype(float)
    w = np.stack([ka[idx, 2], kb[idx, 2]], axis=1).astype(float)
    ok = (w >= w_min).all(axis=1)
    if ok.sum() < 3:
        return (np.inf, None) if full else np.inf
    views = [a, b]
    Rt = np.stack([np.hstack([cams[v][1], cams[v][2][:, None]]) for v in views])
    Kinv = np.stack([np.linalg.inv(cams[v][0]) for v in views])
    uvh = np.concatenate([uv[ok], np.ones((int(ok.sum()), 2, 1))], axis=-1)
    xn = np.einsum("vij,nvj->nvi", Kinv, uvh)[..., :2]
    X = pv._dlt(Rt, xn, w[ok])
    pr, depth = pv.project(pv.proj_mats(cams, views), X)
    if not np.isfinite(X).all() or not (depth > 0).all():
        return (np.inf, None) if full else np.inf
    e = float(np.median(np.linalg.norm(pr - uv[ok], axis=-1)))
    if not full:
        return e
    joint = np.asarray(idx)[ok]
    fl = None
    for grp in ((15, 16), (11, 12)):
        rows = X[np.isin(joint, grp)]
        if len(rows):
            fl = rows.mean(axis=0)[[0, 2]]
            break
    return e, fl


def ray_gap(cams, a, b, ka, kb, s_min=pv.W_MIN):
    """Closest distance (m) between the two views' rays through the ankle midpoints (hips when a view has no ankles in
    common): the same person's rays meet whatever the ankle height, while floor points slide along the rays (T15:
    jumps / lunges). inf when no joint pair is visible in both views or the rays are parallel."""
    for grp in ((15, 16), (11, 12)):
        ja = [j for j in grp if ka[j, 2] >= s_min]
        jb = [j for j in grp if kb[j, 2] >= s_min]
        if ja and jb:
            break
    else:
        return np.inf
    c, d = [], []
    for v, k, js in ((a, ka, ja), (b, kb, jb)):
        K, R, t = cams[v]
        u = k[js, :2].mean(axis=0)
        r = R.T @ np.linalg.inv(K) @ np.array([u[0], u[1], 1.0])
        c.append(-R.T @ t)
        d.append(r / np.linalg.norm(r))
    n = np.cross(d[0], d[1])
    nn = float(np.linalg.norm(n))
    return float(abs((c[1] - c[0]) @ n) / nn) if nn > 1e-9 else np.inf


def knee_flexion(X):
    out = []
    for h, k, a in ((11, 13, 15), (12, 14, 16)):
        if np.isfinite(X[[h, k, a]]).all():
            u, w = X[h] - X[k], X[a] - X[k]
            c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
            out.append(180.0 - float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return max(out) if out else float("nan")


def _tid(track_ids, v, i):
    ids = (track_ids or {}).get(v)
    return int(ids[i]) if ids is not None and i < len(ids) and ids[i] is not None else -1


def _mean_floor(C, floors):
    return np.mean([floors[v][i] for v, i in C.items()], axis=0)


def _finite_rows(a):
    return a[np.isfinite(a).all(axis=1)]


def _multi(f, C, dets, cams, floors, track_ids, p):
    vs = sorted(C)
    K = np.stack([dets[v][C[v]] for v in vs], axis=1)[:17].astype(float)      # (17, Vc, 3)
    X, _, err, _ = pv.triangulate_batch({v: cams[v] for v in vs}, vs, K[..., :2], K[..., 2])
    ank, hip, head = _finite_rows(X[[15, 16]]), _finite_rows(X[[11, 12]]), _finite_rows(X[HEAD])
    if len(ank):
        fl = ank.mean(axis=0)[[0, 2]]
    elif len(hip):
        fl = hip.mean(axis=0)[[0, 2]]
    else:
        fl = _mean_floor(C, floors)
    height = float(ank[:, 1].mean() - head[:, 1].min() + p["head_top_m"]) if len(ank) and len(head) else float("nan")
    e = err[np.isfinite(err)]
    return Candidate(frame=f, dets=dict(C), floor=np.asarray(fl, float), support=len(vs), height=height,
                     knee_deg=knee_flexion(X), err_px=float(e.mean()) if len(e) else float("nan"),
                     track_ids={v: _tid(track_ids, v, C[v]) for v in vs}, tri=True,
                     ankle_h=float(-ank[:, 1].mean()) if len(ank) else float("nan"))


def _pairs(dets, cams, views, floors, p):
    """Consistent cross-view pairs (err, (a, i), (b, j), pair floor): floor points < floor_pair_m apart, or (floor points
    < ray_pair_max_floor_m apart and) ankle rays passing < ray_pair_m apart (floor points of a jumping / lunging player
    disagree, T15), then the 2-view DLT gate.
    The pair floor is the pair's triangulated ankles (hips; else the mean floor point)."""
    out = []
    for ia, a in enumerate(views):
        for b in views[ia + 1:]:
            for i, ka in enumerate(dets.get(a, [])):
                fa = floors[a][i]
                if fa is None:
                    continue
                for j, kb in enumerate(dets.get(b, [])):
                    fb = floors[b][j]
                    if fb is None:
                        continue
                    fd = float(np.linalg.norm(fa - fb))
                    if fd >= p["floor_pair_m"] and (fd >= p["ray_pair_max_floor_m"]
                                                    or ray_gap(cams, a, b, ka, kb) >= p["ray_pair_m"]):
                        continue
                    e, fl = _tri2(cams, a, b, ka, kb, full=True)
                    gate = max(p["tri_px"], p["tri_bbox_frac"] * max(bbox_height(ka), bbox_height(kb)))
                    if e < gate:
                        out.append((e, (a, i), (b, j), fl if fl is not None else (fa + fb) / 2))
    return sorted(out, key=lambda x: x[0])


def _cluster(pairs, floors, p):
    """Greedy merge (lowest error first): one detection per view per cluster; a pair joins / merges clusters when its
    triangulated floor is < merge_m from the cluster's (mean of its pairs' floors; per-view floor points slide along
    the rays when the ankles are off the ankle plane)."""
    clusters, where, pos = [], {}, []
    for _, da, db, fl in pairs:
        ca, cb = where.get(da), where.get(db)
        if ca is None and cb is None:
            clusters.append({da[0]: da[1], db[0]: db[1]})
            pos.append([fl])
            where[da] = where[db] = len(clusters) - 1
        elif ca is not None and cb is not None:
            A, B = clusters[ca], clusters[cb]
            if ca == cb or set(A) & set(B) or np.linalg.norm(np.mean(pos[ca], axis=0) - np.mean(pos[cb], axis=0)) >= p["merge_m"]:
                continue
            A.update(B)
            pos[ca] += pos[cb] + [fl]
            for item in B.items():
                where[item] = ca
            clusters[cb], pos[cb] = {}, []
        else:
            c, new = (ca, db) if ca is not None else (cb, da)
            Cl = clusters[c]
            if new[0] in Cl or np.linalg.norm(np.mean(pos[c], axis=0) - fl) >= p["merge_m"]:
                continue
            Cl[new[0]] = new[1]
            pos[c].append(fl)
            where[new] = c
    return [c for c in clusters if c], set(where)


def estimate_ankle_height(cands_by_frame, p, n_views=None) -> float:
    """The clip's ankle keypoint height: median triangulated ankle height of candidates seen by >= ankle_min_support
    views (well-conditioned; all n_views on a smaller rig), clipped to ankle_h_range; params ankle_h_m when fewer than ankle_min_samples. T15: the
    real rigs put ankle keypoints ~0.15 m up (calibration scale + detector), not the template's 0.08 m."""
    need = min(p["ankle_min_support"], n_views) if n_views else p["ankle_min_support"]     # a 2-camera rig: 2 views
    hs = [c.ankle_h for C in cands_by_frame for c in C if c.tri and c.support >= need and np.isfinite(c.ankle_h)]
    if len(hs) < p["ankle_min_samples"]:
        return float(p["ankle_h_m"])
    lo, hi = p["ankle_h_range"]
    return float(np.clip(np.median(hs), lo, hi))


def frame_candidates(f, dets, cams, views, p, track_ids=None, ankle_h=ANKLE_H) -> list:
    views = sorted(views)
    floors = {v: [floor_point(k, cams[v], ankle_h=ankle_h) for k in dets.get(v, [])] for v in views}
    clusters, used = _cluster(_pairs(dets, cams, views, floors, p), floors, p)
    out = [_multi(f, C, dets, cams, floors, track_ids, p) for C in clusters]
    rest = [(v, i) for v in views for i in range(len(dets.get(v, [])))
            if (v, i) not in used and floors[v][i] is not None]
    groups = []
    for v, i in rest:
        near = [c for c in out if v not in c.dets and np.linalg.norm(c.floor - floors[v][i]) < p["loose_attach_m"]]
        if near:
            c = min(near, key=lambda c: np.linalg.norm(c.floor - floors[v][i]))
            c.dets[v] = i
            c.loose = tuple(sorted(c.loose + (v,)))
            c.track_ids[v] = _tid(track_ids, v, i)
            continue
        for g in groups:
            if v not in g and np.linalg.norm(_mean_floor(g, floors) - floors[v][i]) < p["merge_1view_m"]:
                g[v] = i
                break
        else:
            groups.append({v: i})
    for g in groups:
        v0, i0 = next(iter(g.items()))
        out.append(Candidate(frame=f, dets=dict(g), floor=_mean_floor(g, floors), support=len(g),
                             pix_h=pixel_height(dets[v0][i0]) if len(g) == 1 else float("nan"),
                             track_ids={v: _tid(track_ids, v, i) for v, i in g.items()}))
    for a in range(len(out)):
        for b in range(a + 1, len(out)):
            d = float(np.linalg.norm(out[a].floor - out[b].floor))
            out[a].crowd_dist = min(out[a].crowd_dist, d)
            out[b].crowd_dist = min(out[b].crowd_dist, d)
    return out
