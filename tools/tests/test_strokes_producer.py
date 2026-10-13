import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from analysis import strokes as stk
from analysis import schema, smpl24, synth_analysis as sa
from analysis import racket as rk


def _rest_flat():
    flat = []
    for j in range(smpl24.NUM_SMPL_JOINTS):
        flat.extend([float(smpl24._REST[j, 0]), float(smpl24._REST[j, 1]),
                     float(smpl24._REST[j, 2]), 1.0])
    return flat


def _set_joint(flat, name, y=None, x=None, z=None, conf=None):
    b = stk.JOINT_INDEX[name] * 4
    for off, v in ((0, x), (1, y), (2, z), (3, conf)):
        if v is not None:
            flat[b + off] = v


def test_joint_index_is_smpl24_by_name_not_mediapipe():
    # The F1 bug was hardcoded MediaPipe indices; the contract is SMPL-24.
    assert stk.JOINT_INDEX["right_wrist"] == 21
    assert stk.JOINT_INDEX["left_wrist"] == 20
    assert stk.JOINT_INDEX["head"] == 15
    assert stk.JOINT_INDEX["right_hip"] == 2


def _swing_frames(n=120, fps=30.0, peak=60, width=8, peak_speed=6.0):
    """Right wrist moves along +x, triangular speed to peak_speed at `peak`,
    holding the final x afterwards (a snap-back would be a phantom spike)."""
    xs = []
    x = 0.0
    for i in range(n):
        if peak - width <= i <= peak + width:
            s = peak_speed * (1 - abs(i - peak) / width)
            x += s / fps
        xs.append(x)
    frames = []
    for i in range(n):
        flat = _rest_flat()
        _set_joint(flat, "right_wrist", x=xs[i])
        frames.append({"frame_id": i, "time": i / fps, "joints_flat": flat,
                       "position": [0.0, 4.0], "racket": None, "grip": None,
                       "stroke": None})
    return frames


def test_wrist_speed_zero_when_still():
    frames = [{"frame_id": i, "time": i / 30.0, "joints_flat": _rest_flat(),
               "position": [0.0, 4.0], "racket": None, "grip": None, "stroke": None}
              for i in range(60)]
    sp = stk.wrist_speed(frames, "right", 30.0)
    assert sp.shape == (60,)
    assert np.nanmax(sp) < 0.1


def test_wrist_speed_tracks_moving_wrist():
    frames = _swing_frames(peak_speed=6.0)
    sp = stk.wrist_speed(frames, "right", 30.0)
    assert np.nanmax(sp[52:69]) > 3.0


def test_wrist_speed_nan_on_low_conf():
    frames = _swing_frames()
    frames[40]["joints_flat"][stk.JOINT_INDEX["right_wrist"] * 4 + 3] = 0.1
    sp = stk.wrist_speed(frames, "right", 30.0)
    assert np.isnan(sp[40])


def _multi_swing_frames(peaks, n=240, fps=30.0, width=8, peak_speed=6.0):
    """One continuous +x wrist track with a triangular speed hump at each peak."""
    xs = []
    x = 0.0
    for i in range(n):
        s = 0.0
        for pk in peaks:
            if pk - width <= i <= pk + width:
                s = peak_speed * (1 - abs(i - pk) / width)
        x += s / fps
        xs.append(x)
    frames = []
    for i in range(n):
        flat = _rest_flat()
        _set_joint(flat, "right_wrist", x=xs[i])
        frames.append({"frame_id": i, "time": i / fps, "joints_flat": flat,
                       "position": [0.0, 4.0], "racket": None, "grip": None,
                       "stroke": None})
    return frames


def test_detect_peaks_finds_two():
    frames = _multi_swing_frames([60, 180], n=240)   # two separated swings
    sp = stk.wrist_speed(frames, "right", 30.0)
    peaks = stk.detect_peaks(sp, 30.0)
    assert len(peaks) == 2


def _stroke_doc_frames(wrist_y, post_vy=0.0, peak_speed=6.0, root_z=4.0,
                       n=120, peak=60, fps=30.0, face_normal=None):
    """A right-wrist swing with controlled height and follow-through (SMPL-24)."""
    frames = _swing_frames(n=n, fps=fps, peak=peak, peak_speed=peak_speed)
    kmax = int(0.15 * fps)
    for i, fr in enumerate(frames):
        jf = fr["joints_flat"]
        _set_joint(jf, "head", y=1.6)
        _set_joint(jf, "left_hip", y=1.0)
        _set_joint(jf, "right_hip", y=1.0)
        if i <= peak:
            _set_joint(jf, "right_wrist", y=wrist_y)
        else:
            k = min(i - peak, kmax)
            _set_joint(jf, "right_wrist", y=wrist_y + post_vy * k / fps)
        fr["position"] = [0.0, root_z]
        if face_normal is not None:
            fr["racket"] = {"face_normal": list(face_normal), "which_hand": "right",
                            "confidence": 0.8}
    return frames


def _classify_only_stroke(frames, fps=30.0):
    sp = stk.wrist_speed(frames, "right", fps)
    peaks = stk.detect_peaks(sp, fps)
    assert peaks, "expected at least one detected peak"
    return stk.classify(frames, peaks[0], "right", fps, sp)


def test_overhead_fast_down_is_smash():
    label, conf, f = _classify_only_stroke(
        _stroke_doc_frames(2.0, post_vy=-3.0, peak_speed=7.0))
    assert label == "overhead_smash" and f["wrist_above_head"] and conf >= 0.5


def test_overhead_gentle_is_drop():
    label, _, _ = _classify_only_stroke(
        _stroke_doc_frames(2.0, post_vy=-0.5, peak_speed=3.8))
    assert label == "drop"


def test_overhead_up_follow_is_clear():
    label, _, _ = _classify_only_stroke(
        _stroke_doc_frames(2.0, post_vy=1.0, peak_speed=7.0))
    assert label == "overhead_clear"


def test_low_wrist_upward_is_lift():
    label, _, _ = _classify_only_stroke(
        _stroke_doc_frames(0.6, post_vy=2.0, peak_speed=6.0))
    assert label == "underarm_lift"


def test_near_net_gentle_is_net_shot():
    label, _, _ = _classify_only_stroke(
        _stroke_doc_frames(1.2, post_vy=0.0, peak_speed=3.8, root_z=1.0))
    assert label == "net_shot"


def test_mid_height_horizontal_is_drive():
    label, _, _ = _classify_only_stroke(
        _stroke_doc_frames(1.2, post_vy=0.0, peak_speed=6.0))
    assert label == "drive"


def test_racket_face_forehand_and_backhand():
    fore = _stroke_doc_frames(2.0, peak_speed=6.0, face_normal=[1.0, 0.0, 0.0])
    back = _stroke_doc_frames(2.0, peak_speed=6.0, face_normal=[-1.0, 0.0, 0.0])
    pf = stk.detect_peaks(stk.wrist_speed(fore, "right", 30.0), 30.0)[0]
    pb = stk.detect_peaks(stk.wrist_speed(back, "right", 30.0), 30.0)[0]
    assert stk.racket_face(fore, pf, "right") == "forehand"
    assert stk.racket_face(back, pb, "right") == "backhand"


def test_racket_face_none_without_racket():
    frames = _stroke_doc_frames(2.0, peak_speed=6.0)   # racket None
    p = stk.detect_peaks(stk.wrist_speed(frames, "right", 30.0), 30.0)[0]
    assert stk.racket_face(frames, p, "right") is None


def _doc_with_frames(frames_by_side):
    n = len(next(iter(frames_by_side.values())))
    doc = sa.build_synth_doc(frames=n)
    for p in doc["players"]:
        if p["side"] in frames_by_side:
            p["frames"] = frames_by_side[p["side"]]
    return doc


def test_hand_from_racket_majority():
    frames = _stroke_doc_frames(2.0, peak_speed=6.0, face_normal=[1.0, 0.0, 0.0])
    assert stk.hand_from_racket(frames) == "right"
    plain = _stroke_doc_frames(2.0, peak_speed=6.0)
    assert stk.hand_from_racket(plain) is None


def test_fill_strokes_keeps_doc_valid_and_labels_peaks():
    frames = _stroke_doc_frames(2.0, post_vy=-3.0, peak_speed=7.0)
    doc = _doc_with_frames({"near": frames, "far": _stroke_doc_frames(2.0, peak_speed=7.0)})
    stk.fill_player_strokes(doc, min_peak_speed=3.0)
    assert schema.validate_analysis(doc) == []
    labelled = [pf for pf in doc["players"][0]["frames"] if pf["stroke"] is not None]
    assert len(labelled) >= 1
    assert labelled[0]["stroke"]["label"] in stk.STROKE_LABELS
    assert 0.0 < labelled[0]["stroke"]["confidence"] <= 0.9


def test_fill_strokes_clears_prior_cosmetic_labels():
    frames = _stroke_doc_frames(2.0, peak_speed=1.0)   # too slow -> no peak
    doc = _doc_with_frames({"near": frames, "far": frames})
    doc["players"][0]["frames"][3]["stroke"] = {"label": "smash", "confidence": 0.9}
    stk.fill_player_strokes(doc, min_peak_speed=3.0)
    assert all(pf["stroke"] is None for pf in doc["players"][0]["frames"])


def test_fill_strokes_provenance():
    frames = _stroke_doc_frames(2.0, post_vy=-3.0, peak_speed=7.0)
    doc = _doc_with_frames({"near": frames, "far": frames})
    stk.fill_player_strokes(doc)
    assert doc["meta"]["producers"]["stroke"] == "heuristic_v2"


def test_fill_strokes_adds_face_when_racket_present():
    frames = _stroke_doc_frames(2.0, post_vy=-3.0, peak_speed=7.0,
                                face_normal=[1.0, 0.0, 0.0])
    doc = _doc_with_frames({"near": frames, "far": frames})
    stk.fill_player_strokes(doc)
    labelled = [pf for pf in doc["players"][0]["frames"] if pf["stroke"] is not None]
    assert labelled and labelled[0]["stroke"]["face"] == "forehand"


def test_cli_writes_valid_analysis_with_strokes(tmp_path):
    out = tmp_path / "strokes.analysis.json"
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "analysis.strokes",
         "--out", str(out), "--frames", "120", "--min-peak-speed", "2.0"],
        cwd=str(repo), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_analysis(doc) == []
    assert doc["meta"]["producers"]["stroke"] == "heuristic_v2"
    total = sum(1 for p in doc["players"] for pf in p["frames"]
                if pf["stroke"] is not None)
    assert total >= 1              # gentle synth motion still yields strokes at 2.0 m/s
