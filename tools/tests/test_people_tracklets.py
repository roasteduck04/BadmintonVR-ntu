"""Tracklets: linking, gaps, splits at ambiguities (with hysteresis) and at the net (inside the court only)."""
import numpy as np

from analysis import people
from analysis.people.candidates import Candidate
from analysis.people import tracklets as tk

P = people.load_params()


def _c(f, x, z, support=3, tid=-1, crowd=np.inf):
    return Candidate(frame=f, dets={"a": 0}, floor=np.array([x, z], float), support=support,
                     track_ids={"a": tid}, crowd_dist=crowd, tri=support > 1)


def test_straight_walk_is_one_tracklet_and_gap_ends_it():
    frames = [[_c(f, 0.02 * f, -3.0)] for f in range(40)]
    T = tk.link(frames, P)
    assert len(T) == 1 and T[0].first == 0 and T[0].last == 39 and len(T[0]) == 40
    frames = [[_c(f, 0.0, -3.0)] if (f < 10 or f > 25) else [] for f in range(40)]   # 15-frame gap > max_gap 10
    assert len(tk.link(frames, P)) == 2
    frames = [[_c(f, 0.0, -3.0)] if (f < 10 or f > 16) else [] for f in range(40)]   # 6-frame gap: coast
    assert len(tk.link(frames, P)) == 1


def test_max_gap_is_the_number_of_missing_frames():
    """"A gap > max_gap frames ends a tracklet": exactly max_gap missing frames still links, max_gap + 1 doesn't."""
    g = P["max_gap"]
    frames = [[_c(f, 0.0, -3.0)] if (f < 10 or f >= 10 + g) else [] for f in range(40)]       # g missing
    assert len(tk.link(frames, P)) == 1
    frames = [[_c(f, 0.0, -3.0)] if (f < 10 or f >= 10 + g + 1) else [] for f in range(40)]   # g + 1 missing
    assert len(tk.link(frames, P)) == 2


def test_two_people_crossing_split_at_the_ambiguity():
    frames = []
    for f in range(60):
        x = -1.5 + 0.05 * f
        d = abs(2 * x)
        crowd = np.hypot(d, 0.35)
        frames.append([_c(f, x, -3.0, crowd=crowd), _c(f, -x, -3.35, crowd=crowd)])
    T = tk.link(frames, P)
    crowded = [t for t in T if t.crowded]
    assert len(crowded) == 2 and all(not (t.first == 0 and t.last == 59) for t in T)
    # every frame of every person is in exactly one tracklet
    assert sum(len(t) for t in T) == 120


def test_net_blocks_links_inside_the_court_only():
    inside = [[_c(f, 0.0, -0.8 + 0.04 * f)] for f in range(40)]            # walks across z = 0 at x = 0
    T = tk.link(inside, P)
    assert len(T) == 2 and all(abs(c.floor[1]) <= P["net_band_m"] or c.floor[1] < 0 for c in T[0].cands)
    outside = [[_c(f, 4.2, -0.8 + 0.04 * f)] for f in range(40)]           # umpire position beside the net post
    assert len(tk.link(outside, P)) == 1


def test_a_player_at_the_net_line_is_not_cut_by_floor_noise():
    # T15 singles_05x: P1 at the net has floor z = -0.02 .. +0.03; the z = 0 rule cut it into 5 fragments whose
    # median z > 0 put them on the far half (lost to P1). Only a candidate beyond the net band on the other side cuts.
    zs = [-3.0 + 0.1 * f for f in range(30)] + [0.03 * (-1) ** f for f in range(60)] + [-0.1 * f for f in range(1, 30)]
    T = tk.link([[_c(f, -2.0, z)] for f, z in enumerate(zs)], P)
    assert len(T) == 1 and T[0].side == -1
    net = [[_c(f, -2.0, 0.02)] for f in range(20)]                          # never off the net line: no side yet
    assert len(tk.link(net, P)) == 1 and tk.link(net, P)[0].side == 0


def test_track_id_breaks_a_tie():
    frames = [[_c(0, 0.0, -3.0, tid=5)]]
    frames.append([_c(1, 0.10, -3.0, tid=9), _c(1, -0.12, -3.0, tid=5)])
    T = tk.link(frames, P)
    t0 = next(t for t in T if t.first == 0)
    assert len(t0) == 2 and t0.cands[1].track_ids["a"] == 5


def test_a_handover_overlap_is_cut_off_the_old_tracklet():
    # T15 doubles_2sides_4angle_2cam: P2 is two candidates (1-view + 2-view, 0.4 m apart) for 3 frames while its old
    # tracklet ends and a new one starts; the 3-frame overlap made the 604-frame new tracklet lose its seed pin and its
    # identity (a player can't be in two tracklets at once). The old tracklet's overlapping tail is split off.
    frames = []
    for f in range(100):
        C = []
        if f <= 50:
            C.append(_c(f, 0.0, -3.0))
        if f >= 48:
            C.append(_c(f, 0.4, -3.0, support=1))
        frames.append(C)
    frames[48][0].crowd_dist = frames[48][1].crowd_dist = 5.0          # not a crowd split (keeps the test on the cut)
    T = tk.cut_handovers(tk.link(frames, P), P)
    old = next(t for t in T if t.first == 0)
    new = next(t for t in T if t.first == 48 and len(t) > 10)
    assert old.last == 47 and new.last == 99
    tail = [t for t in T if t is not old and t is not new]
    assert len(tail) == 1 and tail[0].frames == [48, 49, 50]
    assert len({t.tid for t in T}) == len(T) and sum(len(t) for t in T) == 51 + 52
    # two people far apart overlapping in time are left alone
    far = [[_c(f, 0.0, -3.0)] * (f <= 50) + [_c(f, 2.5, -3.0)] * (f >= 48) for f in range(100)]
    T2 = tk.cut_handovers(tk.link(far, P), P)
    assert sorted(len(t) for t in T2) == [51, 52]
