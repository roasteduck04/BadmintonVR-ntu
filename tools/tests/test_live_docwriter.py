import json

import numpy as np

import live_synth as ls
from analysis import schema24
from analysis.live import docwriter
from analysis.live.stream_pose24 import StreamPose24


def _outs(n=50):
    cams, players = ls.scene(F=2 * n + 4)
    cfg = ls.cfg(n_steps=n, warmup=20)
    cfg.update({"out_clip": "synth_3cam_live", "timeline_frames": 2 * n + 10, "backend": "cached",
                "pace": "asfast", "skeleton": "smpl-28"})
    cfg["ref_frames"] = [r + 4 for r in cfg["ref_frames"]]          # section starts at frame 4
    s = StreamPose24(cfg, cams)
    outs = []
    for i, r in enumerate(cfg["ref_frames"]):
        outs += s.push(ls.frameset(i, r, cams, players))
    outs += s.flush()
    return cfg, cams, s.tracker.sides, outs


def test_doc_on_timeline_with_interpolated_odd_frames_and_no_nan(tmp_path):
    cfg, cams, sides, outs = _outs()
    outs[30]["players"]["P1"]["J"][:] = np.nan                        # Review Focus 3
    outs[30]["players"]["P1"]["valid"] = False
    d = docwriter.to_doc(cfg, outs, cams, sides)
    assert schema24.validate_analysis24(d) == []
    p1 = next(p for p in d["players"] if p["id"] == "P1")["frames"]
    assert len(p1) == cfg["timeline_frames"]
    assert p1[0]["joints_flat"] is None and p1[3]["joints_flat"] is None      # before the section
    assert p1[4]["joints_flat"] is not None and not p1[4]["interpolated"]     # live frame
    assert p1[5]["joints_flat"] is not None and p1[5]["interpolated"]         # odd frame between two live frames
    assert p1[outs[30]["ref_frame"]]["joints_flat"] is None
    assert p1[outs[30]["ref_frame"] + 1]["joints_flat"] is None               # no interpolation next to a gap
    pv = d["meta"]["pose_variant"]
    assert pv["time_ref_view"] == "a" and [o["view"] for o in pv["offsets_list"]] == list(ls.VIEWS)
    assert d["meta"]["live"]["quality"]["P1"][30] == 3 and len(d["meta"]["live"]["frame_quality"]) == 50
    path = docwriter.write(cfg, outs, cams, sides, str(tmp_path))
    assert path.endswith("synth_3cam_live_pose24.analysis.json")
    json.loads(open(path, encoding="utf-8").read())
