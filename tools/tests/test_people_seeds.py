"""Seeds: sides, frame_time + additional_seeds events, detection under the click, pins, unusable seeds skipped."""
import numpy as np

from analysis import people
from analysis.people import seeds as sd
from analysis.people.candidates import Candidate
from analysis.people.tracklets import Tracklet
import pose24_synth as syn

CAMS = syn.rig(("a",))


def _det(u0, v0, h=400.0):
    k = np.zeros((133, 3), np.float32)
    k[:23, 0] = u0
    k[:23, 1] = np.linspace(v0 - h, v0, 23)
    k[15:17, 1] = v0
    k[:23, 2] = 0.9
    return k


def test_player_ids_and_sides():
    assert sd.player_ids("singles") == ["P1", "P2"] and sd.player_ids("doubles") == ["P1", "P2", "P3", "P4"]
    K, R, t = CAMS["a"]
    near = syn.project(np.zeros((1, 133, 3)) + [0.0, 0.0, -3.0], CAMS, ["a"])["a"][0, 0, :2]
    far = syn.project(np.zeros((1, 133, 3)) + [0.0, 0.0, 3.0], CAMS, ["a"])["a"][0, 0, :2]
    sv = {"a": {"seeds": {"P1": near.tolist(), "P2": far.tolist()}}}
    logs = []
    assert sd.player_sides(sv, CAMS, "singles", logs.append) == {"P1": "near", "P2": "far"}
    sides = sd.player_sides({}, CAMS, "doubles", logs.append)
    assert sides == {"P1": "near", "P2": "near", "P3": "far", "P4": "far"} and any("no seed" in m for m in logs)


def test_seed_events_include_additional_seeds():
    sv = {"a": {"frame_time": 0.5, "seeds": {"P1": [1, 2]},
                "additional_seeds": [{"frame_time": 1.5, "seeds": {"P1": [3, 4], "P2": [5, 6]}}]},
          "z": {"frame_time": 1.0, "seeds": {"P1": [0, 0]}}}
    ev = sd.seed_events(sv, ["a"], 60.0)
    assert ev == [("a", 30, "P1", (1.0, 2.0)), ("a", 90, "P1", (3.0, 4.0)), ("a", 90, "P2", (5.0, 6.0))]


def test_det_under_pixel():
    dets = [_det(500, 2000), _det(900, 2000)]
    assert sd.det_under_pixel(dets, 905, 1995) == 1 and sd.det_under_pixel(dets, 520, 1960) == 0
    assert sd.det_under_pixel(dets, 3000, 100) is None and sd.det_under_pixel([], 1, 1) is None


def test_det_under_pixel_needs_the_click_at_the_feet():
    # T15: seeds are clicked at the feet. doubles_2sides' 1 s seeds match no detection at their time (feet 80-280 px
    # away, 0.16-0.38 x the person's height; good seeds are <= 0.11 x): a click that is merely inside someone's box
    # (P1 on P2's body behind it) or 150 px from small far feet must not pin that person
    dets = [_det(500, 2000, h=400.0), _det(900, 2000, h=800.0)]
    assert sd.det_under_pixel(dets, 505, 1700) is None          # inside det 0's box, 300 px above its feet
    assert sd.det_under_pixel(dets, 900, 1880) is None          # 120 px above a tall person's feet (0.15 x)
    assert sd.det_under_pixel(dets, 900, 1920) == 1             # 80 px (0.10 x)
    assert sd.det_under_pixel(dets, 500, 1965) == 0             # 35 px: under the 40 px floor for small people
    assert sd.det_under_pixel(dets, 500, 1945) is None          # 55 px on a 400 px person (0.14 x)


def test_pins_skip_unusable_seeds_and_resolve_conflicts():
    kp2d = {"a": [[_det(500, 2000), _det(900, 2000)] for _ in range(40)]}
    t0, t1 = Tracklet(tid=0), Tracklet(tid=1)
    for f in range(40):
        t0.add(Candidate(frame=f, dets={"a": 0}, floor=np.zeros(2), support=1))
        t1.add(Candidate(frame=f, dets={"a": 1}, floor=np.ones(2), support=1))
    logs = []
    ev = [("a", 12, "P1", (505.0, 1990.0)),      # shift 2 -> synced frame 10 -> det 0 -> t0
          ("a", 30, "P2", (905.0, 1990.0)),      # -> t1
          ("a", 99, "P1", (505.0, 1990.0)),      # raw 99 - 2 = 97: outside the 40-frame timeline -> skipped
          ("a", 20, "P2", (2000.0, 100.0))]      # nothing under the click -> skipped
    pins = sd.pin_tracklets(ev, [t0, t1], kp2d, {"a": 2}, 40, log=logs.append)
    assert pins == {0: "P1", 1: "P2"}
    assert sum("skipped" in m for m in logs) == 2
    ev2 = [("a", 12, "P1", (505.0, 1990.0)), ("a", 14, "P2", (505.0, 1990.0))]   # tie on t0 -> not pinned
    assert sd.pin_tracklets(ev2, [t0, t1], kp2d, {"a": 2}, 40, log=logs.append) == {}


def test_skipped_seed_logs_its_distance_and_params_drive_the_limit():
    kp2d = {"a": [[_det(500, 2000)] for _ in range(10)]}
    t0 = Tracklet(tid=0)
    for f in range(10):
        t0.add(Candidate(frame=f, dets={"a": 0}, floor=np.zeros(2), support=1))
    logs = []
    p = people.load_params()
    assert sd.pin_tracklets([("a", 5, "P1", (500.0, 1700.0))], [t0], kp2d, {"a": 0}, 10, log=logs.append, p=p) == {}
    assert any("300 px" in m and "0.75 x" in m for m in logs), logs
    loose = dict(p, seed_feet_frac=0.8)                      # the limit comes from params.json
    assert sd.pin_tracklets([("a", 5, "P1", (500.0, 1700.0))], [t0], kp2d, {"a": 0}, 10, log=logs.append, p=loose) == {0: "P1"}
