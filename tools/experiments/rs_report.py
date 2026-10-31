"""Round-6 experiment (test branch only): score the racket / shuttle of live runs (rs_stream.py) against the offline
builds of the same clip, next to the pose numbers of each run.

    python tools/experiments/rs_report.py <run dir | campaign folder | campaign zip> [...] [--out <folder>]

References (found in Assets/StreamingAssets/analysis, data/analysis or --ref-dir; the offline sidecars of singles_05x
are in the archive folder the analysis files were moved to):
- racket: <clip>_pose24.racket_bootstrap.json (5 cameras at 4K, fine-tuned model on guide boxes, rigid fit, smoothed),
  body-local -> world with the positions of <clip>_pose24.analysis.json. Scored per player: frames fitted (of the
  frames the offline racket has), grip / head error (cm), shaft angle (deg).
- shuttle: <clip>.shuttle_tracker.json (5 cameras, physics arcs over improved triangulation; video frames of the
  reference camera). Scored: frames with a live position, of the frames the offline one has; position error (cm) on
  the frames both have, all and offline 'observed' only; the best whole-frame time shift (a sync check).
Frames are matched by the live ref_frame = the offline frame index (both on the reference camera's video frames)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REF_DIRS = [os.path.join(REPO, "Assets", "StreamingAssets", "analysis"), os.path.join(REPO, "data", "analysis"),
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "archive")]


def _json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def find_ref(name, extra=()):
    for d in list(extra) + REF_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _yaml(p):
    import yaml
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _st(x, scale=1.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] * scale
    return {"median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2),
            "n": int(len(x))} if len(x) else None


def _angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.sum(a * b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ------------------------------------------------------------------------------------------------ references
def racket_reference(clip, extra=()):
    sc_p = find_ref(f"{clip}_pose24.racket_bootstrap.json", extra)
    doc_p = find_ref(f"{clip}_pose24.analysis.json", extra)
    if not sc_p or not doc_p:
        return None
    sc, doc = _json(sc_p), _json(doc_p)
    pos = {p["id"]: [f.get("position") if f else None for f in p["frames"]] for p in doc["players"]}
    out = {}
    for p in sc["players"]:
        n = len(p["frames"])
        g, h, ax = (np.full((n, 3), np.nan) for _ in range(3))
        for i, f in enumerate(p["frames"]):
            ps = pos.get(p["id"], [None] * n)[i] if i < len(pos.get(p["id"], [])) else None
            if not f or ps is None or f.get("status") not in ("detected", "partial"):
                continue
            off = np.array([ps[0], 0.0, ps[1]])
            g[i], h[i], ax[i] = np.asarray(f["grip"]) + off, np.asarray(f["head"]) + off, f["axis"]
        out[p["id"]] = {"grip": g, "head": h, "axis": ax, "hand": p.get("which_hand")}
    return {"path": sc_p, "doc": doc_p, "players": out}


def shuttle_reference(clip, extra=()):
    p = find_ref(f"{clip}.shuttle_tracker.json", extra)
    if not p:
        return None
    d = _json(p)
    n = len(d["frames"])
    X = np.full((n, 3), np.nan)
    src = [None] * n
    for i, f in enumerate(d["frames"]):
        if f and f.get("position") is not None:
            X[i] = f["position"]
            src[i] = f.get("source")
    return {"path": p, "X": X, "source": src}


# ------------------------------------------------------------------------------------------------ one run
def score_run(run_dir, ref_dirs=()):
    p = os.path.join(run_dir, "racket_shuttle.jsonl")
    if not os.path.exists(p):
        return None
    lines = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    summ = _json(os.path.join(run_dir, "rs_summary.json")) if os.path.exists(os.path.join(run_dir, "rs_summary.json")) \
        else {}
    cfg = _yaml(os.path.join(run_dir, "config.resolved.yaml"))
    rep = _json(os.path.join(run_dir, "report.json")) if os.path.exists(os.path.join(run_dir, "report.json")) else {}
    clip = cfg.get("clip")
    warm = int(cfg.get("warmup_steps") or 0)
    out = {"run": os.path.basename(run_dir.rstrip("/\\")), "clip": clip, "views": cfg.get("views"),
           "fps": cfg.get("fps"), "frames": len(lines), "settings": summ.get("settings"), "ep": summ.get("ep"),
           "providers": summ.get("providers"), "worker": summ.get("worker"), "racket_ms": summ.get("racket_ms"),
           "shuttle_ms": summ.get("shuttle_ms"), "lag_ms": summ.get("lag_ms"),
           "after_pose_ms": summ.get("after_pose_ms"), "hand_votes": summ.get("hand_votes"),
           "racket2d_frames_with_cands": summ.get("racket2d_frames_with_cands"),
           "shuttle2d_found": summ.get("shuttle2d_found"), "errors": summ.get("errors")}
    t = rep.get("timing") or {}
    out["pose"] = {"fps": t.get("achieved_output_fps"), "x_realtime": t.get("realtime_factor"),
                   "keeps_up": t.get("keeps_up"), "keeps_up_2d": t.get("keeps_up_2d"),
                   "delay_p50_ms": ((t.get("stages") or {}).get("e2e") or {}).get("p50"),
                   "e2e_ms_start_end": t.get("e2e_ms_start_end")}
    lines = [ln for ln in lines if ln["step"] >= warm]
    out["frames_after_warmup"] = len(lines)

    rr = racket_reference(clip, ref_dirs) if lines and any("racket" in ln for ln in lines) else None
    if rr:
        res = {}
        live_ids = sorted({pid for ln in lines for pid in (ln.get("racket") or {})})
        for lid in live_ids:
            rows = [(ln["ref_frame"], ln["racket"].get(lid)) for ln in lines]
            # the offline player of this live player: nearest median grip
            best = None
            for oid, o in rr["players"].items():
                d = [np.linalg.norm(np.asarray(r["grip"]) - o["grip"][f]) for f, r in rows
                     if r and f < len(o["grip"]) and np.isfinite(o["grip"][f]).all()]
                if d and (best is None or np.median(d) < best[1]):
                    best = (oid, float(np.median(d)))
            if best is None:
                res[lid] = {"fitted": sum(1 for _, r in rows if r), "frames": len(rows)}
                continue
            o = rr["players"][best[0]]
            has = [(f, r) for f, r in rows if f < len(o["grip"]) and np.isfinite(o["grip"][f]).all()]
            both = [(f, r) for f, r in has if r]
            eg = [np.linalg.norm(np.asarray(r["grip"]) - o["grip"][f]) for f, r in both]
            eh = [np.linalg.norm(np.asarray(r["head"]) - o["head"][f]) for f, r in both]
            ea = [float(_angle(r["axis"], o["axis"][f])) for f, r in both]
            shifts = {}
            for sft in range(-2, 3):
                d = [np.linalg.norm(np.asarray(r["head"]) - o["head"][f + sft]) for f, r in rows
                     if r and 0 <= f + sft < len(o["head"]) and np.isfinite(o["head"][f + sft]).all()]
                shifts[sft] = round(float(np.median(d)) * 100, 1) if d else None
            res[lid] = {"offline": best[0], "frames": len(rows), "offline_frames": len(has), "fitted": len(both),
                        "head_cm_by_shift": shifts,
                        "coverage": round(len(both) / len(has), 3) if has else None,
                        "extra": sum(1 for f, r in rows if r) - len(both),
                        "grip_cm": _st(eg, 100), "head_cm": _st(eh, 100), "shaft_deg": _st(ea),
                        "views": _st([r["views"] for _, r in both]), "hand": next((r["hand"] for _, r in both), None),
                        "offline_hand": o.get("hand")}
        out["racket"] = {"reference": os.path.basename(rr["path"]), "players": res}
    sr = shuttle_reference(clip, ref_dirs) if lines and any("shuttle" in ln for ln in lines) else None
    if sr:
        X = sr["X"]
        live = {ln["ref_frame"]: ln.get("shuttle") for ln in lines}
        fr = sorted(live)
        has = [f for f in fr if f < len(X) and np.isfinite(X[f]).all()]
        got = [f for f in fr if live[f]]
        both = [f for f in has if live[f]]
        e = [np.linalg.norm(np.asarray(live[f]["pos"]) - X[f]) for f in both]
        eo = [np.linalg.norm(np.asarray(live[f]["pos"]) - X[f]) for f in both if sr["source"][f] == "observed"]
        shifts = {}
        for s in range(-3, 4):
            d = [np.linalg.norm(np.asarray(live[f]["pos"]) - X[f + s]) for f in got
                 if 0 <= f + s < len(X) and np.isfinite(X[f + s]).all()]
            shifts[s] = round(float(np.median(d)) * 100, 1) if d else None
        modes = {}
        for f in got:
            modes[live[f].get("mode")] = modes.get(live[f].get("mode"), 0) + 1
        out["shuttle"] = {"reference": os.path.basename(sr["path"]), "frames": len(fr), "offline_frames": len(has),
                          "live_frames": len(got), "both": len(both),
                          "coverage": round(len(both) / len(has), 3) if has else None,
                          "live_only": len(got) - len(both),
                          "pos_cm": _st(e, 100), "pos_observed_cm": _st(eo, 100),
                          "err_px": _st([live[f]["err_px"] for f in got]),
                          "speed_kmh_p90": _st([live[f].get("speed_kmh") for f in got
                                                if live[f].get("speed_kmh") is not None]),
                          "median_cm_by_shift": shifts, "modes": modes}
        ok = {k: v for k, v in shifts.items() if v is not None}
        if ok:
            b = min(ok, key=ok.get)
            eb = [np.linalg.norm(np.asarray(live[f]["pos"]) - X[f + b]) for f in got
                  if 0 <= f + b < len(X) and np.isfinite(X[f + b]).all()]
            out["shuttle"]["best_shift"] = b
            out["shuttle"]["pos_cm_best_shift"] = _st(eb, 100)
    return out


# ------------------------------------------------------------------------------------------------ tables
def _f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def _g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def table(rows) -> str:
    L = ["| run | racket 2D (ep) | shuttle 2D (ep) | 2D extra ms/step p50 (det / racket / TrackNet) | pose fps | "
         "keeps up | racket fitted % (P1 / P2) | grip cm | head cm | shaft ° | shuttle % | shuttle cm (observed) | "
         "shuttle cm at best shift (frames) | 3D ms racket / shuttle | lag after pose ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r.get("settings") or {}
        w = r.get("worker") or {}
        rk = _g(r, "racket", "players") or {}
        cov = " / ".join(_f(100 * (v.get("coverage") or 0), 0) for _, v in sorted(rk.items())) or "–"
        gr = " / ".join(_f(_g(v, "grip_cm", "median")) for _, v in sorted(rk.items())) or "–"
        hd = " / ".join(_f(_g(v, "head_cm", "median")) for _, v in sorted(rk.items())) or "–"
        sh = " / ".join(_f(_g(v, "shaft_deg", "median")) for _, v in sorted(rk.items())) or "–"
        su = r.get("shuttle") or {}
        ep = r.get("ep") or {}
        rk2d = f"{s.get('racket')}{'/' + str(s.get('det_every')) if s.get('racket') == 'det' and s.get('det_every', 1) > 1 else ''}" \
            f" ({ep.get('racket', '?')})" if s.get("racket") else "off"
        sh2d = f"stride {s.get('stride')} ({ep.get('shuttle', '?')})" if s.get("shuttle") else "off"
        L.append(f"| {r['run']} | {rk2d} | {sh2d} | {_f(_g(w, 'all_ms', 'median'))} ({_f(_g(w, 'det_ms', 'median'))} / "
                 f"{_f(_g(w, 'pose_ms', 'median'))} / {_f(_g(w, 'tn_ms', 'median'))}) | {_f(_g(r, 'pose', 'fps'))} | "
                 f"{ {True: '✓', False: '✗'}.get(_g(r, 'pose', 'keeps_up'), '–') } | {cov} | {gr} | {hd} | {sh} | "
                 f"{_f(100 * su['coverage'], 0) if su.get('coverage') is not None else '–'} | "
                 f"{_f(_g(su, 'pos_cm', 'median'))} ({_f(_g(su, 'pos_observed_cm', 'median'))}) | "
                 f"{_f(_g(su, 'pos_cm_best_shift', 'median'))} ({su.get('best_shift', '–')}) | "
                 f"{_f(_g(r, 'racket_ms', 'median'))} / {_f(_g(r, 'shuttle_ms', 'median'))} | "
                 f"{_f(_g(r, 'after_pose_ms', 'median'), 0)} |")
    return "\n".join(L)


def run_dirs(src):
    """Run folders under src (a run, a campaign folder or zip)."""
    if src.endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="rs_report_")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    if os.path.exists(os.path.join(src, "racket_shuttle.jsonl")):
        return [src]
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(src, "**", "racket_shuttle.jsonl"), recursive=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", default=None, help="folder for rs_report.md / .json (default: the first src folder)")
    ap.add_argument("--ref-dir", action="append", default=[])
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles (cp932) cannot print the table's dashes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for s in a.src:
        for d in run_dirs(s):
            r = score_run(d, a.ref_dir)
            if r:
                rows.append(r)
    if not rows:
        sys.exit("no racket_shuttle.jsonl found")
    out = a.out or (a.src[0] if os.path.isdir(a.src[0]) else os.path.dirname(os.path.abspath(a.src[0])))
    os.makedirs(out, exist_ok=True)
    md = table(rows)
    with open(os.path.join(out, "rs_report.md"), "w", encoding="utf-8") as fh:
        fh.write("# Racket + shuttle (round 6)\n\n" + md + "\n")
    with open(os.path.join(out, "rs_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, default=str)
    print(md)
    print("->", os.path.join(out, "rs_report.md"))


if __name__ == "__main__":
    main()
