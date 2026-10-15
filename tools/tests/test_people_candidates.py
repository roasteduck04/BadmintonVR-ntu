"""Candidates: cross-view pairing, merge, triangulated floor/height, 1-view candidates, loose attach, crowding."""
import numpy as np

import pose24_synth as syn
import people_synth as ps
from analysis import pose_v2 as pv
from analysis import people
from analysis.people import candidates as cd

P = people.load_params()
CAMS = syn.rig(ps.VIEWS)


def _dets(people_, f, views=ps.VIEWS, drop=()):
    """{view: [calibrated (133,3)]} of every person at synced frame f, in `people_` order; drop: {(person, view)}."""
    out = {v: [] for v in views}
    for n, X in people_.items():
        o = syn.project(X[f:f + 1], CAMS, views)
        for v in views:
            if (n, v) in drop:
                continue
            k = o[v][0].astype(np.float32)
            k[:, 2] = pv.norm_score(k[:, 2])
            out[v].append(k)
    return out


def test_two_people_become_two_four_view_candidates():
    people_ = {"p1": ps.standing(5, -1.0, -3.0), "p2": ps.standing(5, 1.5, 3.0)}
    C = cd.frame_candidates(0, _dets(people_, 0), CAMS, list(ps.VIEWS), P)
    assert len(C) == 2 and all(c.support == 4 and c.tri for c in C)
    c1 = min(C, key=lambda c: c.floor[1])
    assert np.allclose(c1.floor, [-1.0, -3.0], atol=0.12)
    assert c1.dets == {v: 0 for v in ps.VIEWS}
    # template stature: head keypoints + 0.10 m, ankles 0.08 m above the floor
    assert 1.4 < c1.height < 2.0 and c1.err_px < 3.0
    assert not np.isinf(c1.crowd_dist) and c1.crowd_dist > 5.0


def test_one_view_person_is_kept_and_close_people_are_crowded():
    people_ = {"p1": ps.standing(5, 0.0, -3.0), "p2": ps.standing(5, 0.35, -3.0), "far": ps.standing(5, -2.0, 4.0)}
    drop = {("far", "a"), ("far", "b"), ("far", "c")}
    C = cd.frame_candidates(0, _dets(people_, 0, drop=drop), CAMS, list(ps.VIEWS), P)
    one = [c for c in C if c.support == 1]
    assert len(one) == 1 and one[0].dets == {"d": 2} and not one[0].tri and np.isfinite(one[0].pix_h)
    close = sorted((c for c in C if c.support == 4), key=lambda c: c.floor[0])
    assert len(close) == 2 and all(c.crowd_dist < P["ambig_m"] for c in close)


def test_detection_that_fails_pairing_is_attached_loose():
    people_ = {"p1": ps.standing(5, 0.0, -3.0)}
    d = _dets(people_, 0)
    d["b"][0] = d["b"][0].copy()
    # shoulders / hips / knees jump +-300 px vertically in a zigzag (a rigid shift is absorbed by the DLT depth, so it
    # would still pair with view d); ankles stay, so the floor point still agrees
    d["b"][0][5:15, 1] += 300.0 * np.array([(-1.0) ** (j // 2) for j in range(5, 15)])
    C = cd.frame_candidates(0, d, CAMS, list(ps.VIEWS), P)
    assert len(C) == 1 and C[0].support == 3 and C[0].loose == ("b",) and C[0].dets["b"] == 0


def test_no_detections_and_track_ids():
    assert cd.frame_candidates(0, {v: [] for v in ps.VIEWS}, CAMS, list(ps.VIEWS), P) == []
    people_ = {"p1": ps.standing(5, 0.0, -3.0)}
    C = cd.frame_candidates(0, _dets(people_, 0), CAMS, list(ps.VIEWS), P, track_ids={v: [7] for v in ps.VIEWS})
    assert C[0].track_ids == {v: 7 for v in ps.VIEWS}


def _raise_ankles(X, dh):
    X = X.copy()
    X[:, [15, 16], 1] += dh           # Y-up world
    return X


def test_ankle_height_is_estimated_and_unbiases_one_view_floor_points():
    # T15 (real clips): ankle keypoints triangulate ~0.15 m above the court, not 0.08 m. A plane at the wrong height
    # moves a 1-view floor point along the ray by dh x range / camera height (~0.5 m at 16 m): pairing, linking and
    # in_lines then break for far players. The clip's own triangulated ankles give the height.
    people_ = {"p1": _raise_ankles(ps.standing(20, 0.0, -3.0), 0.07), "p2": _raise_ankles(ps.standing(20, 1.0, 4.5), 0.07)}
    C = [cd.frame_candidates(f, _dets(people_, f), CAMS, list(ps.VIEWS), P) for f in range(20)]
    assert all(np.isfinite(c.ankle_h) for cc in C for c in cc if c.tri)
    h = cd.estimate_ankle_height(C, P)
    assert abs(h - 0.15) < 0.02
    d = _dets(people_, 0, drop={("p2", v) for v in ("a", "b", "c")})      # p2 seen by camera d only
    def one(**kw):
        return next(c for c in cd.frame_candidates(0, d, CAMS, list(ps.VIEWS), P, **kw) if c.support == 1)
    tri = next(c for c in C[0] if c.floor[1] > 0).floor                      # p2's 4-view floor
    assert np.linalg.norm(one(ankle_h=h).floor - tri) < 0.05
    assert np.linalg.norm(one().floor - tri) > 0.15                           # the fixed 0.08 m plane is biased


def test_ankle_height_falls_back_without_enough_samples():
    assert cd.estimate_ankle_height([[]], P) == P["ankle_h_m"]


def test_a_jumping_player_is_one_candidate():
    # T15 doubles_1side P1 at the net (f1250-1390): airborne / lunging, ankles far above the ankle plane, so each view's
    # floor point slides along its ray (0.7-1.4 m apart across 4 views) while the 2D still triangulates to < 15 px.
    # The floor-distance prefilter / merge split it into four 1-view candidates; the ankle rays still meet.
    X = ps.standing(3, 1.0, 4.5)
    X[:, :, 1] += 0.3                                   # the whole body 0.3 m up (Y-up world)
    C = cd.frame_candidates(0, _dets({"p": X}, 0), CAMS, list(ps.VIEWS), P)
    assert len(C) == 1 and C[0].support == 4 and C[0].tri
    assert np.linalg.norm(C[0].floor - [1.0, 4.5]) < 0.15


def test_ankle_height_on_a_two_camera_rig():
    # T15 singles_05x_2cam: with 2 views no candidate reaches the 3-view support, the estimate fell back to 0.08 m
    two = ["a", "d"]
    people_ = {"p1": _raise_ankles(ps.standing(20, 0.0, -3.0), 0.07), "p2": _raise_ankles(ps.standing(20, 1.0, 4.5), 0.07)}
    C = [cd.frame_candidates(f, _dets(people_, f, views=two), CAMS, two, P) for f in range(20)]
    assert abs(cd.estimate_ankle_height(C, P, n_views=2) - 0.15) < 0.03
