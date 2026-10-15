"""Appearance per tracklet: colour features from the cache keyed (raw frame, det index), per-view Hellinger."""
import numpy as np

import people_synth as ps
from analysis import people
from analysis.people.candidates import Candidate
from analysis.people.reid import Appearance
from analysis.people.tracklets import Tracklet

P = people.load_params()


def _t(tid, frames, det=0, view="a"):
    t = Tracklet(tid=tid)
    for f in frames:
        t.add(Candidate(frame=f, dets={view: det}, floor=np.zeros(2), support=1))
    return t


def test_colour_distance_per_shared_view_and_shift():
    red, blue = ps.colour(0, 0), ps.colour(6, 6)
    shifts = {"a": 2, "b": 0}
    # stream a is shifted by 2: synced f -> raw f + 2; kp_idx maps local det 0 -> original det 3
    feats = {"a": {(f + 2, 3): red for f in range(0, 20, 2)} | {(f + 2, 4): blue for f in range(0, 20, 2)}}
    kp_idx = {"a": [[3, 4]] * 20, "b": [[0]] * 20}
    app = Appearance("colour", feats, shifts, kp_idx, P)
    ta, tb, tc = _t(0, range(0, 10)), _t(1, range(10, 20)), _t(2, range(0, 10), det=1)
    assert app.dist(ta, tb) < 1e-6 and app.dist(ta, tc) > 0.9
    assert app.dist(ta, _t(3, range(0, 10), view="b")) is None      # no shared view -> unknown
    assert app.gate == P["appearance"]["colour_gate"] and app.weight == P["appearance"]["colour_weight_m"]


def test_osnet_kind_uses_cosine():
    e1, e2 = np.r_[1.0, 0.0, 0.0], np.r_[0.0, 1.0, 0.0]
    feats = {"a": {(f, 0): e1 for f in range(10)} | {(f, 1): e2 for f in range(10)}}
    app = Appearance("osnet", feats, {"a": 0}, {"a": [[0, 1]] * 10}, P)
    assert abs(app.dist(_t(0, range(5)), _t(1, range(5, 10)))) < 1e-9
    assert abs(app.dist(_t(0, range(5)), _t(2, range(5), det=1)) - 1.0) < 1e-9
