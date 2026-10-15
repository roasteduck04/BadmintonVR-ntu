"""SP3 synthetic scenes: files, truth map, seeds (frame_time + additional_seeds), appearance cache."""
import json

import numpy as np

import people_synth as ps
from analysis import appearance
from analysis import people


def test_params_load():
    p = people.load_params()
    assert p["floor_pair_m"] == 0.6 and p["classify"]["player"] == 0.7 and "_comment" not in p


def test_singles_scene_files_and_truth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg, truth, people_ = ps.singles_scene(str(tmp_path), F=120)
    kp = json.load(open(tmp_path / "data" / "pose2d" / "c_kp2d.json"))
    assert len(kp["frames"]) == 120 and all("track_id" in d for fr in kp["frames"] for d in fr)
    # view c is 2 frames late: raw frames 0-1 are empty, raw 2 shows synced frame 0
    assert kp["frames"][0] == [] and len(kp["frames"][2]) >= 4
    # the bystander is hidden in a and c
    assert not any(n == "by" for (v, r, d), n in truth.items() if v in ("a", "c"))
    assert any(n == "by" for (v, r, d), n in truth.items() if v == "b")
    # truth points at the right detection (shuffled order)
    kpa = json.load(open(tmp_path / "data" / "pose2d" / "a_kp2d.json"))
    (_, r, d) = next(k for k, n in truth.items() if n == "p1" and k[0] == "a")
    got = np.array(kpa["frames"][r][d]["keypoints"])
    assert got.shape == (133, 3) and kpa["frames"][r][d]["track_id"] == 1 + sorted(people_).index("p1")
    seeds = json.load(open(tmp_path / "data" / "calib" / "syns_seeds.json"))
    e = seeds["views"]["a"]
    assert e["frame_time"] == 0.5 and e["additional_seeds"][0]["frame_time"] == 1.5 and set(e["seeds"]) == {"P1", "P2"}
    app = appearance.load_clip("syns")
    assert set(app) == {"a", "b", "c", "d"} and all(k[0] % 2 == 0 for k in app["a"])
