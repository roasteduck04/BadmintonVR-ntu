"""Strokes v2 producer — per-player stroke labels on the analysis.json v3 contract.

Contract-native successor to tools/label_moves.py, which hardcoded MediaPipe-33 indices
(the F1 bug). Here every joint is resolved by NAME from analysis.skeleton.JOINT_NAMES,
so it reads the right joints on the SMPL-24 body-local pose. Stroke moments are wrist-speed
peaks (pose is body-local, so wrist speed is already body-relative); each is classified with
explainable rules and enriched with forehand/backhand from the racket face-normal.

Writes per-frame `stroke: {label, confidence, face?}` on contact peaks. See analysis/schema.py.
"""
from __future__ import annotations

import numpy as np

from analysis import skeleton

JOINT_INDEX = {name: i for i, name in enumerate(skeleton.JOINT_NAMES)}

STROKE_LABELS = ("overhead_smash", "overhead_clear", "drop",
                 "underarm_lift", "net_shot", "drive")

TH_DROP_SPEED = 4.5
TH_SMASH_VY = -1.5
TH_NET_Z = 2.0
TH_NET_SPEED = 5.0
TH_LIFT_VY = 1.0
POST_WINDOW_S = 0.15


def joint_xyz(joints_flat, name):
    b = JOINT_INDEX[name] * 4
    return np.array(joints_flat[b:b + 3], dtype=np.float64)


def joint_conf(joints_flat, name):
    return joints_flat[JOINT_INDEX[name] * 4 + 3]


def wrist_speed(frames, hand, fps, conf_cutoff=0.3):
    """Body-relative wrist speed m/s per frame; NaN where conf < cutoff or pose missing."""
    wname = "right_wrist" if hand == "right" else "left_wrist"
    n = len(frames)
    pos = np.full((n, 3), np.nan)
    for i, fr in enumerate(frames):
        jf = fr.get("joints_flat")
        if jf is not None and joint_conf(jf, wname) >= conf_cutoff:
            pos[i] = joint_xyz(jf, wname)
    speed = np.full(n, np.nan)
    if n > 1:
        speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
        speed[0] = speed[1]
    else:
        speed[0] = 0.0
    k = 5
    sm = np.copy(speed)
    for i in range(n):
        if not np.isnan(speed[i]):
            sm[i] = np.nanmean(speed[max(0, i - k // 2):i + k // 2 + 1])
    return sm


def detect_peaks(speed, fps, min_peak_speed=3.0, min_gap_s=0.5):
    """Local maxima above min_peak_speed, at least min_gap_s apart. Ascending indices."""
    n = len(speed)
    gap = max(1, int(min_gap_s * fps))
    candidates = [i for i in range(1, n - 1)
                  if not np.isnan(speed[i]) and speed[i] >= min_peak_speed
                  and speed[i] >= np.nanmax(speed[max(0, i - gap):i + gap + 1]) - 1e-9]
    peaks = []
    for c in candidates:
        if not peaks or c - peaks[-1] >= gap:
            peaks.append(c)
        elif speed[c] > speed[peaks[-1]]:
            peaks[-1] = c
    return peaks


CONF_MIN = 0.1      # joints below this are dropouts (xyz zeroed to the floor point): never compared


def _mean_y(jf, names):
    """Mean height of the named joints that are confident, or None."""
    ys = [joint_xyz(jf, n)[1] for n in names if joint_conf(jf, n) >= CONF_MIN]
    return float(np.mean(ys)) if ys else None


def stroke_features(frames, peak, hand, fps, speed):
    """Explainable features at a wrist-speed peak. Null frames after the peak and unconfident head / hips are
    skipped (review 2 M4): a missing comparison reads as False."""
    wname = "right_wrist" if hand == "right" else "left_wrist"
    jf = frames[peak]["joints_flat"]
    wy = joint_xyz(jf, wname)[1]
    head_y = _mean_y(jf, ("head",))
    hip_y = _mean_y(jf, ("left_hip", "right_hip"))
    k = max(1, int(POST_WINDOW_S * fps))
    post_vy = 0.0
    for hi in range(min(len(frames) - 1, peak + k), peak, -1):     # last usable frame in the window
        jh = frames[hi].get("joints_flat")
        if jh is not None and joint_conf(jh, wname) >= CONF_MIN:
            post_vy = (joint_xyz(jh, wname)[1] - wy) * fps / (hi - peak)
            break
    pos = frames[peak].get("position")
    root_z = abs(pos[1]) if pos and len(pos) == 2 else 99.0
    lo, hi2 = max(0, peak - k), min(len(frames), peak + k + 1)
    return {"peak_speed": float(np.nanmax(speed[lo:hi2])),
            "wrist_above_head": bool(head_y is not None and wy > head_y),
            "wrist_below_hip": bool(hip_y is not None and wy < hip_y),
            "post_vy": float(post_vy), "root_z": float(root_z)}


def racket_face(frames, peak, hand):
    """Forehand/backhand from the racket face-normal, or None if no racket.

    Heuristic: a face-normal pointing to the player's own hand side is a forehand.
    Body-local +x is the model's left; `which_hand` disambiguates the sign.
    """
    rk = frames[peak].get("racket")
    if not rk:
        return None
    fn = rk.get("face_normal")
    if not fn or len(fn) != 3:
        return None
    is_right = hand == "right"
    return "forehand" if (fn[0] >= 0.0) == is_right else "backhand"


def _confidence(margin_frac, second_agrees):
    conf = 0.5
    if margin_frac >= 0.5:
        conf += 0.2
    if second_agrees:
        conf += 0.1
    return min(conf, 0.9)


def classify(frames, peak, hand, fps, speed):
    f = stroke_features(frames, peak, hand, fps, speed)
    ps, vy = f["peak_speed"], f["post_vy"]
    if f["wrist_above_head"]:
        if ps < TH_DROP_SPEED:
            return "drop", _confidence((TH_DROP_SPEED - ps) / TH_DROP_SPEED, vy > TH_SMASH_VY), f
        if vy < TH_SMASH_VY:
            return "overhead_smash", _confidence((TH_SMASH_VY - vy) / abs(TH_SMASH_VY), ps > 6.0), f
        return "overhead_clear", _confidence((vy - TH_SMASH_VY) / abs(TH_SMASH_VY), ps >= TH_DROP_SPEED), f
    if f["root_z"] <= TH_NET_Z and ps < TH_NET_SPEED:
        return "net_shot", _confidence((TH_NET_Z - f["root_z"]) / TH_NET_Z, ps < TH_DROP_SPEED), f
    if f["wrist_below_hip"] or vy > TH_LIFT_VY:
        return "underarm_lift", _confidence(max(vy - TH_LIFT_VY, 0.0) / TH_LIFT_VY,
                                            f["wrist_below_hip"]), f
    return "drive", _confidence(0.0, not f["wrist_above_head"]), f


def hand_from_racket(frames):
    """Majority racket.which_hand across frames, or None if no racket carries a hand."""
    counts = {"right": 0, "left": 0}
    for fr in frames:
        rk = fr.get("racket")
        if rk and rk.get("which_hand") in counts:
            counts[rk["which_hand"]] += 1
    if counts["right"] == 0 and counts["left"] == 0:
        return None
    return "right" if counts["right"] >= counts["left"] else "left"


def fill_player_strokes(doc, min_peak_speed=3.0, hand=None):
    """Label each player's strokes: detect wrist-speed peaks, classify, write per-frame
    stroke on the peaks (None elsewhere). Hand comes from the racket when not forced.
    Tags meta.producers.stroke.
    """
    fps = float(doc["meta"]["fps"])
    for player in doc["players"]:
        frames = player["frames"]
        h = hand or hand_from_racket(frames) or "right"
        speed = wrist_speed(frames, h, fps)
        peaks = detect_peaks(speed, fps, min_peak_speed)
        for pf in frames:
            pf["stroke"] = None
        for p in peaks:
            label, conf, _feats = classify(frames, p, h, fps, speed)
            stroke = {"label": label, "confidence": round(float(conf), 2)}
            face = racket_face(frames, p, h)
            if face:
                stroke["face"] = face
            frames[p]["stroke"] = stroke
    doc.setdefault("meta", {}).setdefault("producers", {})["stroke"] = "heuristic_v2"
    return doc


def main(argv=None):
    import argparse
    import json
    import os

    from analysis import schema, synth_analysis, racket

    ap = argparse.ArgumentParser(
        description="Strokes v2 producer: label per-player strokes on analysis.json "
                    "(foundation demo builds a synthetic clip + racket first).")
    ap.add_argument("--out", required=True, help="output analysis.json")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-peak-speed", type=float, default=2.0,
                    help="wrist-speed peak gate (synth ~2.8 m/s; real footage ~3.0)")
    args = ap.parse_args(argv)

    doc = synth_analysis.build_synth_doc(frames=args.frames, fps=args.fps)
    racket.fill_player_rackets(doc, racket.synthetic_racket_tracks(doc),
                               "synthetic_racket_bodylocal")
    fill_player_strokes(doc, min_peak_speed=args.min_peak_speed)

    errs = schema.validate_analysis(doc)
    if errs:
        raise SystemExit("analysis doc failed validation:\n  " + "\n  ".join(errs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    total = sum(1 for p in doc["players"] for pf in p["frames"] if pf["stroke"])
    print(f"wrote {args.out}: {args.frames} frames, {total} strokes labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
