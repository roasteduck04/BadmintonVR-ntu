"""Round-6 experiment (test branch only): run the live racket / shuttle 3D (rs_stream.RS3D) again, on the laptop,
without the 2D models - to tune / time the 3D part in seconds.

    python tools/experiments/rs_replay.py run <run dir> --out <dir> [--set shuttle3d=tri ...]
        the stream's own recorded 2D (rs_inputs.jsonl.gz, written by every run) with other 3D settings
    python tools/experiments/rs_replay.py offline --out <dir> [--clip singles_05x] [--views back_051,...]
            [--frames 300:900] [--racket2d data/racket2d] [--set ...]
        the offline stored 2D instead (racket: data/racket2d = RacketVision detector + its model, or
        data/racket2d_bootstrap = the offline best; shuttle: data/shuttle with the per-view scale of the offline
        tracker; hands from the offline pose24 doc), cameras and offsets of the offline builds.
Each writes racket_shuttle.jsonl + rs_summary.json + a config stub, so rs_report.py scores it like a run."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import rs_report  # noqa: E402
import rs_stream  # noqa: E402


def _sets(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        import yaml
        out[k] = yaml.safe_load(v)
    return out


def _stub(out_dir, clip, views, fps, warmup=0):
    import yaml
    with open(os.path.join(out_dir, "config.resolved.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"clip": clip, "views": list(views), "fps": fps, "warmup_steps": warmup}, fh)


def cmd_run(a):
    import shutil
    import yaml
    inp = os.path.join(a.run, "rs_inputs.jsonl.gz")
    rs_stream.replay(inp, a.out, _sets(a.set))
    for f in ("config.resolved.yaml", "report.json"):
        if os.path.exists(os.path.join(a.run, f)):
            shutil.copy(os.path.join(a.run, f), a.out)
    print(json.dumps(rs_report.score_run(a.out), indent=1, default=str)[:4000])


def cmd_offline(a):
    from analysis import racket_rigidfit as rf
    clip, views = a.clip, a.views.split(",")
    doc = rs_report._json(rs_report.find_ref(f"{clip}_pose24.analysis.json"))
    tr = rs_report._json(rs_report.find_ref(f"{clip}.shuttle_tracker.json"))
    off = {v: float(tr["sync"]["fractional"].get(v, 0.0)) * a.offset_sign for v in views}
    scale = {v: float(tr["per_view_scale"].get(v, 1.0)) for v in views}
    cams = {c["name"]: [c["K"], c["R"], c["t"]] for c in doc["meta"]["cameras"] if c["name"] in views}
    lo, hi = (int(x) for x in a.frames.split(":"))
    refs = list(range(lo, hi))
    steps = {i: {v: [[int(round(r + off[v]))], 0.0] for v in views} for i, r in enumerate(refs)}
    names = doc["skeleton"]["joint_names"]
    jid = [names.index(n) for n in ("left_wrist", "right_wrist", "left_hand", "right_hand", "pelvis")]
    st = rs_stream.settings({"exp_racket": "det", "exp_shuttle": True})
    st.update(_sets(a.set))
    R3 = rs_stream.RS3D(a.out if os.makedirs(a.out, exist_ok=True) is None else a.out, st,
                        {v: rf.Camera(*c) for v, c in cams.items()}, steps, off, float(doc["meta"].get("fps", 60.0)))
    r2d = {v: json.load(open(os.path.join(REPO, a.racket2d, f"{v}_racket2d.json")))["frames"] for v in views}
    s2d = {v: json.load(open(os.path.join(REPO, "data", "shuttle", f"{v}_shuttle.json")))["frames"] for v in views}
    t0 = time.perf_counter()
    for i, r in enumerate(refs):
        for v in views:
            f = steps[i][v][0][0]
            for k in (f + 8,):                         # 2D arrives ahead of the frame, as in a stream
                if 0 <= k < len(r2d[v]):
                    R3._add_2d(v, [{"f": k, "racket": [{"kp": c} for c in (r2d[v][k] or [])]}])
                if 0 <= k < len(s2d[v]):
                    p = s2d[v][k]
                    R3._add_2d(v, [{"f": k, "sh": [[k, None if p is None else p[0] * scale[v],
                                                    None if p is None else p[1] * scale[v], 1.0]]}])
            if i == 0:
                for k in range(f - 10, f + 8):
                    if 0 <= k < len(r2d[v]):
                        R3._add_2d(v, [{"f": k, "racket": [{"kp": c} for c in (r2d[v][k] or [])]}])
                    if 0 <= k < len(s2d[v]):
                        p = s2d[v][k]
                        R3._add_2d(v, [{"f": k, "sh": [[k, None if p is None else p[0] * scale[v],
                                                        None if p is None else p[1] * scale[v], 1.0]]}])
        players = {}
        for p in doc["players"]:
            fr = p["frames"][r] if r < len(p["frames"]) else None
            if not fr or fr.get("joints_flat") is None or fr.get("position") is None:
                players[p["id"]] = {"valid": False, "quality": 0, "hands": [[np.nan] * 3] * 4, "hc": [0.0] * 4,
                                    "pelvis": [np.nan] * 3}
                continue
            jf, ps = fr["joints_flat"], fr["position"]
            W = [[jf[j * 4] + ps[0], jf[j * 4 + 1], jf[j * 4 + 2] + ps[1]] for j in jid]
            C = [jf[j * 4 + 3] if jf[j * 4 + 3] is not None else 0.0 for j in jid]
            players[p["id"]] = {"valid": True, "quality": 0, "hands": W[:4], "hc": C[:4], "pelvis": W[4]}
        R3.pending.append((0.0, (i, r, None, None, players)))
        R3.process(force=False, now=0.0)
    R3.process(force=True)
    R3.finish()
    dt = time.perf_counter() - t0
    _stub(a.out, clip, views, 60)
    print(f"{len(refs)} frames in {dt:.1f} s ({1000 * dt / len(refs):.1f} ms/frame incl. reading)")
    rep = rs_report.score_run(a.out)
    keep = {k: rep.get(k) for k in ("racket", "shuttle", "racket_ms", "shuttle_ms", "hand_votes")}
    print(json.dumps(keep, indent=1, default=str)[:5000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", default=[])
    o = sub.add_parser("offline")
    o.add_argument("--out", required=True)
    o.add_argument("--clip", default="singles_05x")
    o.add_argument("--views", default="back_051,backright_051,frontleft_051")
    o.add_argument("--frames", default="300:900")
    o.add_argument("--racket2d", default=os.path.join("data", "racket2d"))
    o.add_argument("--offset-sign", type=float, default=1.0)
    o.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    (cmd_run if a.cmd == "run" else cmd_offline)(a)


if __name__ == "__main__":
    main()
