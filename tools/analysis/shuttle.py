"""Shuttle producer — parabolic trajectory between rally contacts, the fourth Phase 2 producer.

The shuttle track is a sequence of parabolic arcs between contacts. A contact is a
stroke-labelled frame at the hitting player's court position, so this producer consumes
the strokes and position slices of the contract. Each arc interpolates court XZ linearly
and models height as a parabola peaking mid-flight.

Contract slice: fills `shuttle.frames[].position` = [x, y, z] (or null). See analysis/schema.py.

Foundation-first: the real input (TrackNetV3 2D shuttle detections lifted to court 3D) is the
replacement seam. The synthetic fallback derives contacts from the contract itself — the same
physics synth_analysis uses, now owned by a real producer.
"""
from __future__ import annotations

SHUTTLE_PEAK_M = 2.6


def parabola_point(src_xz, dst_xz, u, peak_h):
    """Court [x, y, z] at fraction u in [0,1] of an arc from src to dst.

    XZ interpolate linearly; height is a parabola: y = 4*peak_h*u*(1-u) (apex peak_h at
    u=0.5, zero at both contacts). Never negative for u in [0,1].
    """
    x = src_xz[0] + (dst_xz[0] - src_xz[0]) * u
    z = src_xz[1] + (dst_xz[1] - src_xz[1]) * u
    y = 4.0 * peak_h * u * (1.0 - u)
    return [round(float(x), 4), round(float(y), 4), round(float(z), 4)]


def build_shuttle_track(contacts, frame_count, peak_h=SHUTTLE_PEAK_M):
    """Stitch parabolic arcs between consecutive contacts into a per-frame track.

    contacts: list of (frame:int, [x,z]), ascending, unique frames. Returns
    (positions, confs), each length frame_count. Frames outside the contact span, or with
    fewer than two contacts, are None (no evidence -> no invented position).
    """
    positions = [None] * frame_count
    confs = [None] * frame_count
    pts = [(int(f), xz) for f, xz in contacts if 0 <= int(f) < frame_count]
    for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
        if f1 <= f0:
            continue
        for f in range(f0, f1 + 1):
            u = (f - f0) / (f1 - f0)
            positions[f] = parabola_point(p0, p1, u, peak_h)
            confs[f] = 0.8
    return positions, confs


def contacts_from_doc(doc):
    """One contact per stroke-labelled frame at the hitter's court position.

    Reads the strokes + position slices. Ties (two players hitting the same frame) keep the
    first player. Returns [(frame, [x,z])], ascending and unique by frame.
    """
    by_frame = {}
    for player in doc["players"]:
        for pf in player["frames"]:
            if pf.get("stroke") is None:
                continue
            pos = pf.get("position")
            if not pos or len(pos) != 2:
                continue
            f = int(pf["frame_id"])
            by_frame.setdefault(f, [float(pos[0]), float(pos[1])])
    return [(f, by_frame[f]) for f in sorted(by_frame)]


def fill_shuttle(doc, positions, confs, producer_name):
    """Write a per-frame shuttle track into the doc. positions/confs length = frame_count."""
    fps = float(doc["meta"]["fps"])
    frames = []
    for i, (pos, conf) in enumerate(zip(positions, confs)):
        frames.append({"frame_id": i, "time": round(i / fps, 4),
                       "position": pos, "confidence": conf})
    doc["shuttle"] = {"frames": frames}
    doc.setdefault("meta", {}).setdefault("producers", {})["shuttle"] = producer_name
    return doc


def synthetic_shuttle_track(doc, peak_h=SHUTTLE_PEAK_M):
    """Derive the shuttle track from the doc's own contacts (stroke + position slices).
    REPLACEMENT SEAM — swap for TrackNetV3 2D detections lifted to court 3D when footage exists.
    """
    contacts = contacts_from_doc(doc)
    return build_shuttle_track(contacts, len(doc["players"][0]["frames"]), peak_h)


def main(argv=None):
    import argparse
    import json
    import os

    from analysis import schema, synth_analysis, racket, strokes

    ap = argparse.ArgumentParser(
        description="Shuttle producer: fill analysis.json with a parabolic shuttle track "
                    "from rally contacts (foundation demo chains racket + strokes first).")
    ap.add_argument("--out", required=True, help="output analysis.json")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-peak-speed", type=float, default=2.0)
    args = ap.parse_args(argv)

    doc = synth_analysis.build_synth_doc(frames=args.frames, fps=args.fps)
    racket.fill_player_rackets(doc, racket.synthetic_racket_tracks(doc),
                               "synthetic_racket_bodylocal")
    strokes.fill_player_strokes(doc, min_peak_speed=args.min_peak_speed)
    positions, confs = synthetic_shuttle_track(doc)
    fill_shuttle(doc, positions, confs, "parabola_from_contacts")

    errs = schema.validate_analysis(doc)
    if errs:
        raise SystemExit("analysis doc failed validation:\n  " + "\n  ".join(errs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    tracked = sum(1 for p in positions if p is not None)
    print(f"wrote {args.out}: {args.frames} frames, {tracked} shuttle frames tracked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
