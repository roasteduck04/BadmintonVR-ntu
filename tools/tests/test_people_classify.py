"""Classification: hotspots, size plausibility vs the clip's reference height, features -> player / nonplayer."""
import numpy as np
import pytest

import pose24_synth as syn
from analysis import people
from analysis.people import classify as cl
from analysis.people.candidates import Candidate
from analysis.people.reid import Appearance
from analysis.people.tracklets import Tracklet

P = people.load_params()
CAMS = syn.rig(("a", "b", "c", "d"))


def _track(tid, xz, support=3, height=1.5):
    t = Tracklet(tid=tid)
    for f, (x, z) in enumerate(xz):
        t.add(Candidate(frame=f, dets={"a": tid}, floor=np.array([x, z]), support=support, height=height, tri=True))
    return t


def _ctx(T, hs, pins=()):
    app = Appearance("colour", {}, {"a": 0}, {"a": []}, P)
    return {"p": P, "fps": 60.0, "cams": CAMS, "ref_h": 1.5, "hotspots": hs, "app": app,
            "seed_tracklets": [t for t in T if t.tid in pins]}


def test_band_and_size_score():
    assert cl.band(1.0, 0.85, 1.25) == 1.0 and cl.band(0.5, 0.85, 1.25) == 0.0
    c = Candidate(frame=0, dets={"a": 0}, floor=np.zeros(2), support=3, height=1.5, tri=True)
    assert cl.size_score(c, 1.5, CAMS, P) == 1.0
    c.height = 0.95
    assert cl.size_score(c, 1.5, CAMS, P) < 0.5                     # 0.63 x ref standing: implausible
    c.knee_deg = 70.0
    assert cl.size_score(c, 1.5, CAMS, P) == 1.0                    # ... fine in a deep lunge
    one = Candidate(frame=0, dets={"a": 0}, floor=np.array([0.0, -3.0]), support=1)
    one.pix_h = cl.expected_pixel_height(CAMS["a"], one.floor, 1.5)
    assert cl.size_score(one, 1.5, CAMS, P) == 1.0
    one.pix_h *= 0.3                                                 # a child / poster
    assert cl.size_score(one, 1.5, CAMS, P) == 0.0


def test_hotspots_only_outside_the_lines():
    frames = [[Candidate(frame=f, dets={"a": 0}, floor=np.array([-3.5, 7.5]), support=2),
               Candidate(frame=f, dets={"a": 1}, floor=np.array([0.0, -3.0]), support=3)] for f in range(150)]
    hs = cl.hotspots(frames, 60.0, P)
    assert len(hs) == 1 and abs(hs[0]["x"] + 3.5) < 0.01 and hs[0]["frames"] == 150


def test_hotspot_on_a_cell_boundary_is_one_hotspot():
    # (-3.5, 7.5) and z = 0 lie on 0.5 m cell edges: noise splits the person over several cells
    rng = np.random.default_rng(1)
    frames = [[Candidate(frame=f, dets={"a": 0}, floor=np.array([-3.5, 7.5]) + rng.normal(0, 0.02, 2), support=2),
               Candidate(frame=f, dets={"a": 1}, floor=np.array([4.2, 0.0]) + rng.normal(0, 0.02, 2), support=4)]
              for f in range(150)]
    hs = cl.hotspots(frames, 60.0, P)
    assert len(hs) == 2 and all(h["frames"] == 150 for h in hs)
    assert abs(hs[0]["x"] + 3.5) < 0.05 and abs(hs[1]["z"]) < 0.05


def test_player_bystander_and_walker_scores():
    F = 180
    t = np.arange(F) / 60.0
    player = _track(0, np.c_[0.8 * np.sin(2 * np.pi * 0.5 * t), -3.0 + 0.4 * np.cos(2 * np.pi * 0.35 * t)])
    bystander = _track(1, np.tile([-3.5, 7.5], (F, 1)) + np.random.default_rng(0).normal(0, 0.02, (F, 2)), support=2)
    walker = _track(2, np.c_[np.linspace(-5, 5, F), np.full(F, -1.0)])
    short = _track(3, np.tile([0.0, -3.0], (5, 1)))
    T = [player, bystander, walker, short]
    hs = [{"x": -3.5, "z": 7.5, "frames": F}]
    labels, scores, feats = cl.classify(T, {}, _ctx(T, hs))
    assert labels[0] == "player" and labels[1] == "nonplayer"
    assert feats[2]["straight"] > 0.95 and scores[2] < scores[0]
    assert labels[3] == "unknown"                      # 5 frames: too little evidence either way
    labels, scores, _ = cl.classify(T, {1: "P1"}, _ctx(T, hs, pins=(1,)))
    assert labels[1] == "player" and scores[1] == 1.0  # a seed pin overrides the score


def test_reference_height_from_pinned_tracklets():
    T = [_track(0, [(0, -3)] * 20, height=1.40), _track(1, [(0, 3)] * 20, height=1.60), _track(2, [(0, 1)] * 20, height=2.5)]
    assert abs(cl.reference_height(T, {0: "P1", 1: "P2"}, P, log=lambda *a: None) - 1.5) < 1e-9
    assert cl.reference_height(T, {}, P, log=lambda *a: None) == P["ref_height_m"]


def test_one_half_counts_the_net_line_as_the_players_side():
    # T15: a near player at the net line (z jitters -0.03 .. +0.03) is on one half, not half on each
    xz = [(-2.0, -1.5 + 0.05 * f) for f in range(30)] + [(-2.0, 0.03 * (-1) ** f) for f in range(30)]
    t = _track(0, xz)
    assert cl.features(t, _ctx([t], []))["one_half"] == 1.0
    t2 = _track(1, [(0.0, -2.0)] * 20 + [(0.0, 2.0)] * 20)
    assert cl.features(t2, _ctx([t2], []))["one_half"] == 0.5
    t3 = _track(2, [(4.2, 0.02 * (-1) ** f) for f in range(40)])           # only ever on the line (net post): unknown
    assert cl.features(t3, _ctx([t3], []))["one_half"] == 0.5


def test_a_short_fragment_inside_a_hotspot_is_a_nonplayer():
    # T15: bystanders / the net-post person flicker into 2-20 frame fragments; the (n/30)^2 shrink turned the hotspot
    # evidence (clip-level: >= hotspot_min_s of candidates at that spot) into "unknown" too. Only the tracklet's own
    # evidence is shrunk; the hotspot term is not.
    frag = _track(0, np.tile([-3.5, 7.5], (8, 1)), support=1)
    short = _track(1, np.tile([0.0, -3.0], (8, 1)))                          # in the lines, no hotspot: unknown
    T = [frag, short]
    labels, scores, _ = cl.classify(T, {}, _ctx(T, [{"x": -3.5, "z": 7.5, "frames": 300}]))
    assert labels[0] == "nonplayer" and labels[1] == "unknown"


def test_one_view_expected_height_uses_the_3d_height_definition():
    """review 1 L12: a 3D candidate's height = ankle keypoint -> top head keypoint + head_top_m, so a person of the
    reference height has their head keypoints ref_h - head_top_m above their ankle keypoints (ankle_h above the
    floor). The 1-view expectation placed them ref_h - 0.10 m above the floor (~6 cm low, ~4 % ratio bias)."""
    from analysis import pose_v2 as pv
    cam, floor, ref_h, ankle_h, top = CAMS["a"], np.array([0.5, -3.0]), 1.6, 0.14, P["head_top_m"]
    head_y = ankle_h + ref_h - top                                     # elevation (m) of the head keypoints
    uv, _ = pv.project(pv.proj_mats({"v": cam}, ["v"]),
                       np.array([[floor[0], -head_y, floor[1]], [floor[0], -ankle_h, floor[1]]]))
    want = abs(uv[1, 0, 1] - uv[0, 0, 1])
    got = cl.expected_pixel_height(cam, floor, ref_h, ankle_h, head_top_m=top)
    assert got == pytest.approx(want, rel=1e-6)
    c = Candidate(frame=0, dets={"a": 0}, floor=floor, support=1)
    c.pix_h = want
    assert cl.size_score(c, ref_h, CAMS, P, ankle_h) == 1.0
