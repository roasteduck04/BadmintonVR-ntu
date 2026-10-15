"""Automated tracking quality evaluation.

Reads one or two analysis.json files and computes per-player and aggregate
metrics. Designed for before/after comparison of pipeline changes.

Usage:
    PYTHONPATH=tools python -m analysis.eval_tracking analysis.json
    PYTHONPATH=tools python -m analysis.eval_tracking new.json --compare old.json
    PYTHONPATH=tools python -m analysis.eval_tracking analysis.json --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

# --- Parameters (spec §Parameters) ---
JUMP_THRESHOLD_1M = 1.0
JUMP_THRESHOLD_2M = 2.0
CLOSE_PLAYER_DIST = 0.5
OVERLAP_DIST = 0.3


def _pos_dist(a, b):
    """Euclidean distance between two [x, z] court positions."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _player_metrics(player):
    """Compute per-player metrics from a player dict."""
    frames = player["frames"]
    n = len(frames)
    side = player.get("side")
    pid = player.get("id", "?")

    pose_ok = 0
    pos_ok = 0
    gaps = []
    cur_gap = 0
    jumps_1m = 0
    jumps_2m = 0
    jump_max = 0.0
    prev_pos = None
    positions = []
    wrong_side = 0

    for f in frames:
        jf = f.get("joints_flat")
        pos = f.get("position")

        if jf is not None:
            pose_ok += 1
            if cur_gap > 0:
                gaps.append(cur_gap)
                cur_gap = 0
        else:
            cur_gap += 1

        if pos is not None:
            pos_ok += 1
            positions.append(pos)
            if prev_pos is not None:
                d = _pos_dist(pos, prev_pos)
                if d > JUMP_THRESHOLD_1M:
                    jumps_1m += 1
                if d > JUMP_THRESHOLD_2M:
                    jumps_2m += 1
                if d > jump_max:
                    jump_max = d
            prev_pos = pos

            # Wrong side: near = negative Z, far = positive Z (tracker: side "near" if z < 0)
            if side == "near" and pos[1] > 0:
                wrong_side += 1
            elif side == "far" and pos[1] < 0:
                wrong_side += 1
        else:
            prev_pos = None     # jumps are frame-to-frame: never measured across a gap (review 2 L15)

    if cur_gap > 0:
        gaps.append(cur_gap)

    z_vals = [p[1] for p in positions] if positions else []

    return {
        "pose_coverage": round(100.0 * pose_ok / max(n, 1), 1),
        "position_coverage": round(100.0 * pos_ok / max(n, 1), 1),
        "gap_count": len(gaps),
        "gap_max": max(gaps) if gaps else 0,
        "gap_mean": round(sum(gaps) / len(gaps), 1) if gaps else 0.0,
        "jump_count_1m": jumps_1m,
        "jump_count_2m": jumps_2m,
        "jump_max": round(jump_max, 3),
        "z_range": [round(min(z_vals), 3), round(max(z_vals), 3)] if z_vals else None,
        "z_mean": round(sum(z_vals) / len(z_vals), 3) if z_vals else None,
        "wrong_side_frames": wrong_side,
        "close_teammate_frames": 0,  # filled in aggregate pass
    }


def compute_metrics(doc):
    """Compute per-player and aggregate metrics from an analysis.json dict.

    Returns {pid: {metric: value}, "_aggregate": {metric: value}}.
    """
    players = doc.get("players", [])
    result = {}

    for p in players:
        pid = p.get("id", "?")
        result[pid] = _player_metrics(p)

    # Aggregate: close teammates and player overlap
    overlap_frames = 0
    frame_count = doc.get("meta", {}).get("frame_count", 0)

    # Build per-frame position lookup
    positions_by_frame = {}
    for p in players:
        pid = p["id"]
        for i, f in enumerate(p["frames"]):
            pos = f.get("position")
            if pos is not None:
                positions_by_frame.setdefault(i, []).append(
                    (pid, p.get("side"), pos))

    # Close teammate: same-side players within CLOSE_PLAYER_DIST
    teammate_counts = {p["id"]: 0 for p in players}
    for frame_idx, entries in positions_by_frame.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pid_a, side_a, pos_a = entries[i]
                pid_b, side_b, pos_b = entries[j]
                d = _pos_dist(pos_a, pos_b)
                if d < OVERLAP_DIST:
                    overlap_frames += 1
                if side_a == side_b and d < CLOSE_PLAYER_DIST:
                    teammate_counts[pid_a] = teammate_counts.get(pid_a, 0) + 1
                    teammate_counts[pid_b] = teammate_counts.get(pid_b, 0) + 1

    for pid in result:
        result[pid]["close_teammate_frames"] = teammate_counts.get(pid, 0)

    reproj = doc.get("meta", {}).get("reprojection_px_median")
    result["_aggregate"] = {
        "reprojection_median": reproj,
        "total_swap_suspect": sum(
            result[p["id"]]["wrong_side_frames"] for p in players),
        "player_overlap_frames": overlap_frames,
    }

    return result


def compare_metrics(metrics_new, metrics_old):
    """Compare two sets of metrics. Returns list of comparison rows."""
    rows = []
    # Per-player metrics
    scalar_keys = [
        "pose_coverage", "position_coverage", "gap_count", "gap_max",
        "gap_mean", "jump_count_1m", "jump_count_2m", "jump_max",
        "wrong_side_frames", "close_teammate_frames",
    ]
    all_pids = sorted(set(list(metrics_new.keys()) + list(metrics_old.keys()))
                      - {"_aggregate"})
    for pid in all_pids:
        mn = metrics_new.get(pid, {})
        mo = metrics_old.get(pid, {})
        for key in scalar_keys:
            vn = mn.get(key, 0)
            vo = mo.get(key, 0)
            if isinstance(vn, (int, float)) and isinstance(vo, (int, float)):
                rows.append({
                    "player": pid, "metric": key,
                    "new": vn, "old": vo,
                    "delta": round(vn - vo, 3),
                })

    # Aggregate metrics
    agg_keys = ["total_swap_suspect", "player_overlap_frames"]
    an = metrics_new.get("_aggregate", {})
    ao = metrics_old.get("_aggregate", {})
    for key in agg_keys:
        vn = an.get(key, 0)
        vo = ao.get(key, 0)
        if isinstance(vn, (int, float)) and isinstance(vo, (int, float)):
            rows.append({
                "player": "_aggregate", "metric": key,
                "new": vn, "old": vo,
                "delta": round(vn - vo, 3),
            })

    return rows


def _format_report(metrics, clip_name=""):
    """Format metrics as a human-readable table."""
    lines = []
    if clip_name:
        lines.append(f"=== {clip_name} ===\n")
    for pid in sorted(k for k in metrics if k != "_aggregate"):
        pm = metrics[pid]
        lines.append(f"  {pid}:")
        lines.append(f"    pose_coverage:       {pm['pose_coverage']}%")
        lines.append(f"    position_coverage:   {pm['position_coverage']}%")
        lines.append(f"    gaps:                {pm['gap_count']} (max {pm['gap_max']}, mean {pm['gap_mean']})")
        lines.append(f"    jumps (>1m/>2m):     {pm['jump_count_1m']} / {pm['jump_count_2m']} (max {pm['jump_max']}m)")
        lines.append(f"    wrong_side_frames:   {pm['wrong_side_frames']}")
        lines.append(f"    close_teammate:      {pm['close_teammate_frames']}")
        if pm.get("z_range"):
            lines.append(f"    z_range:             {pm['z_range']}")
        lines.append("")
    agg = metrics.get("_aggregate", {})
    lines.append("  Aggregate:")
    lines.append(f"    reprojection_median: {agg.get('reprojection_median', 'N/A')}")
    lines.append(f"    total_swap_suspect:  {agg.get('total_swap_suspect', 0)}")
    lines.append(f"    player_overlap:      {agg.get('player_overlap_frames', 0)}")
    return "\n".join(lines)


def _format_comparison(rows, clip_name=""):
    """Format comparison as a table with deltas."""
    lines = []
    if clip_name:
        lines.append(f"=== {clip_name}: NEW vs OLD ===\n")
    lines.append(f"{'':24s} {'NEW':>10s} {'OLD':>10s} {'delta':>10s}")
    lines.append("-" * 56)
    for r in rows:
        label = f"{r['player']} {r['metric']}"
        vn = r["new"]
        vo = r["old"]
        d = r["delta"]
        suffix = "%" if "coverage" in r["metric"] else ""
        suffix_m = "m" if "jump_max" in r["metric"] else ""
        lines.append(f"{label:24s} {vn:>10}{suffix} {vo:>10}{suffix} {d:>+10.1f}{suffix_m}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Evaluate tracking quality from analysis.json")
    ap.add_argument("input", help="Path to analysis.json")
    ap.add_argument("--compare", default=None,
                    help="Path to second analysis.json for comparison")
    ap.add_argument("--json", default=None, metavar="OUT",
                    help="Write metrics as JSON to this file")
    args = ap.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        doc = json.load(f)
    metrics = compute_metrics(doc)

    clip_name = doc.get("clip_id", "")

    metrics_old = None
    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            doc_old = json.load(f)
        metrics_old = compute_metrics(doc_old)
        rows = compare_metrics(metrics, metrics_old)
        print(_format_comparison(rows, clip_name))
    else:
        print(_format_report(metrics, clip_name))

    if args.json:
        out = {"metrics": metrics}
        if args.compare:
            out["comparison"] = compare_metrics(metrics, metrics_old)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
