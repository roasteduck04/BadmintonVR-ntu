"""Report for the camera-subset study (spec § 6): long CSV (score.ROW_KEYS columns) write / read, pivot by camera
count (median over rows, best / worst subset), a 2×2 error-vs-n_cams figure and a markdown table. No pandas.

CSV: None / NaN / ±inf → empty cell; read back empty → None (test_view → "", the deployment marker), "True" /
"False" → bool (player_seen, matched, is_reference, …), n_cams / ref_cams / heldout_n → int, other numeric cells → float; text columns (clip, subset, player, the JSON
columns used_frac / excluded, …) stay str. Pivot groups by (clip_type, ref_cams, mode, method, n_cams): a 4-camera
and a 5-camera reference (e.g. doubles_1side vs doubles_2sides) are never pooled. A subset's rank = median of its rows (players × test views ×
clips in the group); lower is better except `coverage`. Method labels (fit, mono-len-oracle, mono-len-oracle+motionbert-lite,
...) are separate rows; method_name gives their short display name (pivot "name" column, figure legend). Unseen rows (player_seen False) are left out of every error
metric and count in `coverage` with their row value (0.0 when missing); `n_unseen` counts them in every metric."""
from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

from analysis.pose24.camsubsets import score

PIVOT_METRICS = ["heldout_px_median", "w_mpjpe_cm_median", "pa_mpjpe_cm_median", "pelvis_xz_cm_median",
                 "knee_mae_deg", "elbow_mae_deg", "trunk_fwd_mae_deg", "coverage"]
HIGHER_BETTER = {"coverage"}
_STR = {"clip", "clip_type", "test_view", "subset", "method", "mode", "player", "used_frac", "excluded"}
_INT = {"n_cams", "ref_cams", "heldout_n"}
_LABEL = {"heldout_px_median": "held-out 2D error, median (px)", "w_mpjpe_cm_median": "W-MPJPE vs reference, median (cm)",
          "pa_mpjpe_cm_median": "PA-MPJPE vs reference, median (cm)", "pelvis_xz_cm_median": "pelvis XZ vs reference, median (cm)",
          "knee_mae_deg": "knee flexion MAE (deg)", "elbow_mae_deg": "elbow flexion MAE (deg)",
          "trunk_fwd_mae_deg": "trunk forward lean MAE (deg)", "coverage": "coverage (fraction)"}


_LENGTH_NAME = {"oracle": "rig lengths", "prior": "template lengths"}
_PRIOR_NAME = {"motionbert": "MotionBERT", "wham": "WHAM", "gvhmr": "GVHMR"}


def method_name(method):
    """Short display name of a method label: "fit" -> "multi-cam fit", "mono-len-oracle" -> "1 cam (rig lengths)",
    "mono-len-oracle+motionbert-lite" -> "1 cam + MotionBERT-lite (rig lengths)"; other labels unchanged."""
    if not isinstance(method, str):
        return method
    if method == "fit":
        return "multi-cam fit"
    if not method.startswith("mono-len-"):
        return method
    src, _, prior = method[len("mono-len-"):].partition("+")
    ln = _LENGTH_NAME.get(src, f"{src} lengths")
    if not prior:
        return f"1 cam ({ln})"
    head, sep, tail = prior.partition("-")
    return f"1 cam + {_PRIOR_NAME.get(head, head)}{sep}{tail} ({ln})"


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v)) if math.isfinite(v) else ""
    if isinstance(v, np.integer):
        return str(int(v))
    return str(v)


def write_csv(rows, path: str) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(score.ROW_KEYS)
        for r in rows:
            w.writerow([_cell(r.get(k)) for k in score.ROW_KEYS])
    return path


def _parse(k: str, s: str):
    if s == "":
        return "" if k == "test_view" else None
    if s in ("True", "False"):
        return s == "True"
    if k in _STR:
        return s
    if k in _INT:
        try:
            return int(s)
        except ValueError:
            return int(float(s))
    try:
        v = float(s)
    except ValueError:
        return s
    return v if math.isfinite(v) else None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _parse(k, v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _median(vals):
    return float(np.median(vals)) if vals else None


def pivot(rows, metrics=PIVOT_METRICS, by=("clip_type", "ref_cams", "mode", "method", "n_cams"),
          held_out_only=True) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if bool(r.get("test_view")) == bool(held_out_only):
            groups[tuple(r.get(k) for k in by)].append(r)
    out = []
    for g in sorted(groups, key=lambda t: tuple((x is None, x if _finite(x) else 0, str(x)) for x in t)):
        rs = groups[g]
        n_unseen = sum(1 for r in rs if r.get("player_seen") is False)
        for m in metrics:
            per = defaultdict(list)
            for r in rs:
                unseen = r.get("player_seen") is False
                v = r.get(m)
                if m in HIGHER_BETTER:
                    if unseen and not _finite(v):
                        v = 0.0
                elif unseen:
                    continue
                if _finite(v):
                    per[r.get("subset")].append(float(v))
            allv = [v for vs in per.values() for v in vs]
            ranked = sorted(((float(np.median(vs)), str(s)) for s, vs in per.items()),
                            key=lambda t: ((-t[0] if m in HIGHER_BETTER else t[0]), t[1]))
            row = dict(zip(by, g))
            row.update(metric=m, median=_median(allv), best_subset=ranked[0][1] if ranked else None,
                       best=ranked[0][0] if ranked else None, worst_subset=ranked[-1][1] if ranked else None,
                       worst=ranked[-1][0] if ranked else None, n_subsets=len(ranked), n_rows=len(rs),
                       n_unseen=n_unseen)
            out.append(row)
    return out


def figure_series(pv) -> list:
    """[((clip_type, ref_cams, mode, method), legend label)] of the pivot rows, one line per reference size."""
    keys = sorted({(p.get("clip_type"), p.get("ref_cams"), p.get("mode"), p.get("method")) for p in pv},
                  key=lambda t: tuple(map(str, t)))
    multi = len({(k[2], k[3]) for k in keys}) > 1
    return [(k, f"{k[0]} ref {k[1]} cams" + (f" {k[2]}/{method_name(k[3])}" if multi else "")) for k in keys]


def figure(rows, path: str, metrics=("heldout_px_median", "w_mpjpe_cm_median", "pelvis_xz_cm_median", "knee_mae_deg"),
           held_out_only=True) -> str:
    """2×2 panels of `metrics` vs n_cams: per (clip_type, ref_cams[, mode/method]) the median line, a best–worst
    subset band and the best pair (n_cams 2) marked. A panel with no finite data says "no data"; the figure is
    closed after saving."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv = pivot(rows, metrics=list(metrics), held_out_only=held_out_only)
    series = figure_series(pv)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    try:
        for ax, m in zip(axes.ravel(), metrics):
            ax.set_title(_LABEL.get(m, m), fontsize=10)
            drawn = False
            for i, (s, lab) in enumerate(series):
                pts = sorted((p for p in pv if p["metric"] == m
                              and (p.get("clip_type"), p.get("ref_cams"), p.get("mode"), p.get("method")) == s
                              and p["median"] is not None and p["n_cams"] is not None), key=lambda p: p["n_cams"])
                if not pts:
                    continue
                drawn = True
                col = f"C{i % 10}"
                x = [p["n_cams"] for p in pts]
                ax.plot(x, [p["median"] for p in pts], "-o", color=col, label=lab, ms=4)
                ax.fill_between(x, [p["best"] for p in pts], [p["worst"] for p in pts], color=col, alpha=0.18, lw=0)
                for p in pts:
                    if p["n_cams"] == 2 and p["best"] is not None:
                        ax.plot([2], [p["best"]], marker="*", ms=11, color=col, ls="none")
                        ax.annotate(p["best_subset"], (2, p["best"]), textcoords="offset points", xytext=(5, -10),
                                    fontsize=7, color=col)
            if drawn:
                ax.set_xlabel("cameras in subset")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
                ax.xaxis.get_major_locator().set_params(integer=True)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(("held-out" if held_out_only else "deployment") + " rows: median line, best–worst subset band, "
                     "★ best pair", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120, format="png")
    finally:
        plt.close(fig)
    return path


def _fmt(v) -> str:
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def markdown_table(pivot_rows) -> str:
    cols = ["clip_type", "ref_cams", "mode", "method", "name", "n_cams", "metric", "median", "best", "best_subset",
            "worst", "worst_subset", "n_rows", "n_unseen"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(_fmt(method_name(p.get("method")) if c == "name" else p.get(c)) for c in cols) + " |"
              for p in pivot_rows]
    return "\n".join(lines) + "\n"


RIG_METRICS = ("w_mpjpe_cm_median", "pa_mpjpe_cm_median", "pelvis_xz_cm_median", "knee_mae_deg", "elbow_mae_deg")
_LENGTH_ORDER = ("oracle", "prior")
_PRIOR_ORDER = ("", "motionbert-lite", "motionbert-full", "wham")


def mono_methods(rows) -> list:
    """Every 1-camera method label in `rows` ("mono-len-<src>[+<prior>]"), ordered by length source (oracle, prior,
    others) then prior (none, MotionBERT lite / full, WHAM, others alphabetically)."""
    def key(m):
        src, _, prior = m[len("mono-len-"):].partition("+")
        return (_LENGTH_ORDER.index(src) if src in _LENGTH_ORDER else len(_LENGTH_ORDER), src,
                _PRIOR_ORDER.index(prior) if prior in _PRIOR_ORDER else len(_PRIOR_ORDER), prior)
    return sorted({r.get("method") for r in rows if isinstance(r.get("method"), str)
                   and r["method"].startswith("mono-len-")}, key=key)


def vs_full_rig_table(rows, methods=None, metrics=RIG_METRICS, ref_cams=5, n_cams=1, mode="oracle") -> list[dict]:
    """Medians "against the full rig" (the report's v0.7.4 tables): rows with ref_cams == `ref_cams`, no
    held-out test view (deployment rows, test_view ""), not the reference itself, `mode`, `n_cams`, seen players only;
    one median per (clip_type, method) over clips x players x views. `methods` None = every 1-camera method in `rows`
    (`mono_methods`). Returns [{clip_type, method, name, n_rows, <metric>: median or None}] in clip_type then
    `methods` order; (clip_type, method) pairs without rows are left out."""
    methods = mono_methods(rows) if methods is None else methods
    groups = defaultdict(list)
    for r in rows:
        if (r.get("ref_cams") == ref_cams and not r.get("test_view") and r.get("is_reference") is not True
                and r.get("mode") == mode and r.get("n_cams") == n_cams and r.get("player_seen") is not False
                and r.get("method") in methods):
            groups[(r.get("clip_type"), r.get("method"))].append(r)
    out = []
    for ct in sorted({k[0] for k in groups}, key=str):
        for m in methods:
            rs = groups.get((ct, m))
            if not rs:
                continue
            row = dict(clip_type=ct, method=m, name=method_name(m), n_rows=len(rs))
            for k in metrics:
                row[k] = _median([float(r[k]) for r in rs if _finite(r.get(k))])
            out.append(row)
    return out


def vs_full_rig_markdown(table, metrics=RIG_METRICS) -> str:
    cols = ["clip_type", "name", "n_rows", *metrics]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join("-" if t.get(c) is None else (f"{t[c]:.1f}" if isinstance(t[c], float) else str(t[c]))
                                for c in cols) + " |" for t in table]
    return "\n".join(lines) + "\n"
