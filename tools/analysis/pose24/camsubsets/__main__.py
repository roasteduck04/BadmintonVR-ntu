"""python -m analysis.pose24.camsubsets build | score | report | export | cost — the SP5 camera-subset study (v0.7.4).

    PYTHONPATH=tools python -m analysis.pose24.camsubsets build --clip singles_05x --sizes 1,2,3,4 --methods fit,mono \
            --mode oracle --dry-run                       # job list + runtime estimate, runs nothing
    PYTHONPATH=tools python -m analysis.pose24.camsubsets build --all5 --mode e2e --e2e-subsets pairs
    PYTHONPATH=tools python -m analysis.pose24.camsubsets build --all5 --sizes 1 --methods mono \
            --mono-prior motionbert-lite,motionbert-full  # + mono-len-<src>+<prior> rows (data/smpl_mono/<clip>/)
    PYTHONPATH=tools python -m analysis.pose24.camsubsets score --all            # -> <out-dir>/<clip>/rows.csv
    PYTHONPATH=tools python -m analysis.pose24.camsubsets report --all --figure data/pose_eval/camsubsets.png
    PYTHONPATH=tools python -m analysis.pose24.camsubsets export --clip singles_05x --subset backleft_051,frontright_051
    PYTHONPATH=tools python -m analysis.pose24.camsubsets cost --all --gpu-timing data/pose2d   # -> camsubsets_cost.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import build as pb
from analysis.pose24 import config, track
from analysis.pose24.__main__ import _each
from analysis.pose24.camsubsets import e2e, oracle, report, score
from analysis.pose24.camsubsets import subsets as cs

ALL5 = ("singles_1x", "singles_05x", "doubles_2sides")
DEFAULT_REPORT = os.path.join("data", "pose_eval", "camsubsets.csv")
DEFAULT_COST = os.path.join("data", "pose_eval", "camsubsets_cost.csv")
DEFAULT_FIT_S = 180.0                                      # per fit job when the full doc has no runtime_s
OVERHEAD = 1.5
MONO_OPTS = {"fit": None, "mono-len-oracle": {"lengths": "oracle"}, "mono-len-prior": {"lengths": "prior"}}
MODES = ("oracle", "e2e")
_PRIOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _mono_lengths(s):
    """--mono-lengths: comma list of mono.LENGTH_SOURCES (checked at argument time)."""
    from analysis.pose24 import mono
    out = _csv(s)
    bad = [x for x in out if x not in mono.LENGTH_SOURCES]
    if not out or bad:
        raise argparse.ArgumentTypeError(f"{s!r}: comma list of {', '.join(mono.LENGTH_SOURCES)}")
    return out


def _mono_priors(s):
    """--mono-prior: comma list of prior method names (file <prior-dir>/<clip>/<view>_<name>.npz; no path parts)."""
    out = _csv(s)
    bad = [x for x in out if not _PRIOR_NAME.match(x) or ".." in x]
    if not out or bad:
        raise argparse.ArgumentTypeError(f"{s!r}: comma list of prior method names (e.g. motionbert-lite)")
    return out


def parse_mono_label(label):
    """"mono-len-<src>[+<prior>]" -> (length source, prior method or None)."""
    if not label.startswith("mono-len-"):
        raise ValueError(f"not a mono label: {label!r}")
    src, _, prior = label[len("mono-len-"):].partition("+")
    return src, prior or None


def prior_file(prior_dir, clip, view, prior):
    """The learned prior of `view`: <prior_dir>/<clip>/<view>_<prior>.npz (the producers' layout; camsubsets clips
    are full registry clips, so clip == the cache's timeline_of)."""
    if prior_dir is None:
        from analysis.pose24.priors import motionbert
        prior_dir = motionbert.DEFAULT_OUT_DIR
    return os.path.join(prior_dir, clip, f"{view}_{prior}.npz")


def mono_opts_for(label, clip=None, views=(), prior_dir=None):
    """run_fits mono_opts of a job label: MONO_OPTS for "fit" / the no-prior labels (unchanged: stored results stay
    fresh); a prior label adds "prior_paths" {view: prior file}."""
    if label in MONO_OPTS:
        return MONO_OPTS[label]
    src, prior = parse_mono_label(label)
    o = {"lengths": src}
    if prior:
        o["prior_paths"] = {v: prior_file(prior_dir, clip, v, prior) for v in views}
    return o


def _clips(a, reg):
    if a.all:
        return [c for c in config.all_clips(reg) if "base" not in reg[c]]
    clips = (list(ALL5) if getattr(a, "all5", False) else []) + (a.clip or [])
    if not clips:
        sys.exit("give --clip, --all5 or --all")
    unknown = [c for c in clips if c not in reg or "base" in reg[c]]
    if unknown:
        sys.exit(f"not full registry clips: {', '.join(unknown)}")
    return list(dict.fromkeys(clips))


def _cache_meta(path):
    """Views / excluded / fps / build_id of a pose24 debug cache without loading its arrays."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no pose24 cache at {path}: run `python -m analysis.pose24 build` for the clip first")
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def build_jobs(rows, methods, mono_lengths, sizes, mono_priors=()) -> dict:
    """{method label: [(label, subset)]} (R5): "fit" for every distinct fit, a mono label per length for 1-view rows,
    and per length x learned prior "mono-len-<length>+<prior>" (v0.7.5)."""
    jobs = {}
    if "fit" in methods:
        jobs["fit"] = [("fit", s) for s in cs.distinct_fits(rows)]
    if "mono" in methods and 1 in sizes:
        for ln in mono_lengths:
            for pr in [None] + list(mono_priors or ()):
                lab = f"mono-len-{ln}" + (f"+{pr}" if pr else "")
                jobs[lab] = [(lab, (v,)) for v in cs.mono_views(rows)]
    return jobs


def e2e_subsets(spec, kept) -> list:
    if spec == "pairs":
        return [tuple(c) for c in combinations(sorted(kept), 2)]
    out = [cs.parse_key(k) for k in _csv(spec)]
    bad = sorted({v for s in out for v in s} - set(kept))
    if bad:
        raise ValueError(f"--e2e-subsets views {', '.join(bad)} are not kept views ({', '.join(sorted(kept))})")
    return out


def _full_doc(analysis_dir, clip):
    p = pb.output_path(analysis_dir, clip)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _runtime_s(doc, step):
    v = ((((doc or {}).get("meta") or {}).get("pose24") or {}).get("runtime_s") or {}).get(step)
    return (float(v), "the full build") if v else (DEFAULT_FIT_S, "default")


def _cmd_build(a):
    reg = config.load_registry(a.registry)
    clips = _clips(a, reg)
    sizes = tuple(int(s) for s in _csv(a.sizes))
    methods = _csv(a.methods or ("fit,mono" if a.mode == "oracle" else "fit"))
    unknown = sorted(set(methods) - {"fit", "mono"})
    if unknown:
        sys.exit(f"unknown --methods {', '.join(unknown)} (fit, mono)")
    lengths = a.mono_lengths
    tests = None if a.test_views == "all" else _csv(a.test_views)
    workers = a.workers or (6 if a.mode == "oracle" else 3)
    fit_opts = {"passes": a.fit_passes} if a.fit_passes else None

    def run(c):
        cpath = track.cache_path(c, a.cache_dir)
        meta = _cache_meta(cpath)
        excl = meta.get("excluded") or {}
        kept = sorted(v for v in meta["views"] if v not in excl)
        rows = cs.study_rows(meta["views"], sizes, tests, excluded=excl)
        n_prior = n_present = 0
        per_prior = {}                                   # prior name -> [jobs, files present, an expected path]
        if a.mode == "oracle":
            jobs = build_jobs(rows, methods, lengths, sizes, a.mono_prior)
            n_fit = len(jobs.get("fit", []))
            n_mono = sum(len(j) for k, j in jobs.items() if k != "fit")
            opts = {lab: mono_opts_for(lab, c, [s[0] for _, s in js], a.prior_dir) for lab, js in jobs.items()}
            for lab, js in jobs.items():
                pp = (opts[lab] or {}).get("prior_paths")
                if pp is not None:
                    have = sum(1 for _, s in js if os.path.exists(pp[s[0]]))
                    n_prior += len(js)
                    n_present += have
                    st = per_prior.setdefault(parse_mono_label(lab)[1], [0, 0, pp[js[0][1][0]] if js else None])
                    st[0] += len(js)
                    st[1] += have
        else:
            if "mono" in methods:
                print(f"[camsubsets] {c}: e2e runs method fit only (mono ignored)")
            subs = e2e_subsets(a.e2e_subsets, kept)
            n_fit, n_mono = len(subs), 0
        if a.dry_run:
            step = "fit" if a.mode == "oracle" else "total"
            sec, src = _runtime_s(_full_doc(a.analysis_dir, c), step)
            mins = sec * (n_fit + n_mono) * OVERHEAD / workers / 60
            print(f"[camsubsets] {c} ({a.mode}): kept views {', '.join(kept)} ({len(kept)}; excluded: "
                  f"{', '.join(sorted(excl)) or 'none'})")
            what = f"{n_fit} distinct fits" if a.mode == "oracle" else f"{n_fit} e2e builds"
            print(f"    {len(rows)} rows; {what}; {n_mono} mono jobs"
                  + (f" ({n_prior} with a learned prior ({n_present} prior file present))" if n_prior else ""))
            print(f"    estimate: {n_fit + n_mono} jobs x {sec:.0f} s ({src} {step}) x {OVERHEAD} / {workers} workers "
                  f"= {mins:.1f} min")
            return None
        none = [(pr, st) for pr, st in per_prior.items() if st[0] and not st[1]]
        if none:                                         # a typo in --mono-prior / --prior-dir: don't skip silently
            raise RuntimeError("; ".join(
                f"--mono-prior {pr}: no prior file for any of {st[0]} jobs of {c} (expected e.g. {st[2]}); "
                f"check the prior name and --prior-dir" for pr, st in none))
        if a.mode == "oracle":
            done = 0
            for lab, js in jobs.items():
                done += len(oracle.run_fits(c, js, cpath, out_dir=a.out_dir, mode="oracle", workers=workers,
                                            fit_opts=fit_opts, mono_opts=opts[lab]))
        else:
            done = len(e2e.run_e2e(c, subs, cpath, out_dir=a.out_dir, workers=workers, registry=reg,
                                   keep_builds=a.keep_builds))
        n_run = n_fit + n_mono - (n_prior - n_present)           # prior jobs without a file: skipped, not failed
        if done < n_run:
            raise RuntimeError(f"{n_run - done} of {n_run} jobs failed (tracebacks above)")
    return _each(clips, run, True, tag="camsubsets")


def _stored(out_dir, clip, mode):
    """{method: {subset key: result path}} of the npz results under <out_dir>/<clip>/<mode>/<method>/."""
    root = os.path.join(out_dir, clip, mode)
    out = {}
    if not os.path.isdir(root):
        return out
    for m in sorted(os.listdir(root)):
        d = os.path.join(root, m)
        if m.startswith("_") or not os.path.isdir(d):
            continue
        ks = {f[:-4]: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".npz") and ".tmp" not in f}
        if ks:
            out[m] = ks
    return out


def expected_for(cache, clip, mode, method, key, prior_dir=None):
    """The freshness key `build` would give this stored result (oracle.expected_meta / e2e.expected_meta); None for
    a method label this CLI doesn't produce (then only build_id is checked)."""
    if mode == "e2e":
        return e2e.expected_meta(cache)
    views = cs.parse_key(key)
    try:
        mo = oracle.job_mono_opts(mono_opts_for(method, clip, views, prior_dir), views)
        return oracle.expected_meta(cache, method, None, mo)
    except ValueError:
        return None


def stale_reason(meta, want):
    """Why a stored result's meta doesn't match the key `want` (None: fresh). The build's fit_opts (--fit-passes) is
    a choice, not staleness; a learned prior whose file is gone now can't be checked (its sha1 is skipped)."""
    if meta.get("build_id") != want.get("build_id"):
        return f"build {meta.get('build_id')} != cache {want.get('build_id')}"
    if meta.get("method") != want.get("method"):
        return f"method {meta.get('method')} != {want.get('method')}"
    got, exp = dict(meta.get("fit_settings") or {}), dict(want.get("fit_settings") or {})
    exp["fit_opts"] = got.get("fit_opts", exp.get("fit_opts"))
    if "prior" in exp and (exp["prior"] or {}).get("sha1") is None:
        exp["prior"] = got.get("prior")
    diff = sorted(k for k in set(got) | set(exp) if got.get(k) != exp.get(k))
    return f"{', '.join(diff)} changed since it was fitted (rerun build)" if diff else None


def _load(paths, build_id, log, expect=None):
    """expect(key) -> the freshness key the result must match (review 1 L8), or None: build_id only."""
    out = {}
    for k, p in paths.items():
        r = oracle.load_result(p, None)
        if r is None:
            log(f"    {p}: unreadable, skipped")
            continue
        want = expect(k) if expect is not None else None
        why = stale_reason(r["meta"], want) if want is not None else (
            None if r["meta"].get("build_id") == build_id
            else f"build {r['meta'].get('build_id')} != cache {build_id}")
        if why:
            log(f"    {p}: stale ({why}), skipped")
        else:
            out[k] = r
    return out


def score_clip(clip, reg, cache_dir, out_dir, analysis_dir, modes, sizes=None, log=print, prior_dir=None) -> list:
    from analysis.pose24 import doc as docmod
    cache = docmod.load_debug_cache(track.cache_path(clip, cache_dir))
    bid = cache.get("build_id")

    def expect(mode, method):
        return lambda k: expected_for(cache, clip, mode, method, k, prior_dir)
    stored = {m: _stored(out_dir, clip, m) for m in modes}
    refs = _load(_stored(out_dir, clip, "oracle").get("fit", {}), bid, log,         # references: always oracle fits
                 expect("oracle", "fit"))
    if sizes is None:
        sizes = sorted({len(cs.parse_key(k)) for st in stored.values() for ks in st.values() for k in ks})
    rows = cs.study_rows(cache["views"], sizes, None, excluded=cache.get("excluded") or {})
    doc = _full_doc(analysis_dir, clip)
    rallies = (doc or {}).get("rallies")
    if doc is None:
        log(f"    {clip}: no full pose24 doc in {analysis_dir}: swing windows over the whole clip")
    kw = dict(full_cache=cache, fps=float(cache["fps"]), clip_type=reg[clip]["mode"], rallies=rallies, log=log)
    out = []
    for mode in modes:
        for method, paths in stored[mode].items():
            sub = refs if (mode, method) == ("oracle", "fit") else _load(paths, bid, log, expect(mode, method))
            rs = [r for r in rows if cs.key(r["subset"]) in sub]
            em = {k: r["meta"].get("e2e") or {} for k, r in sub.items()} if mode == "e2e" else None
            if (mode, method) == ("oracle", "fit"):
                out += score.score_rows(clip, rs, sub, method=method, mode=mode, **kw)
                continue
            groups = defaultdict(list)                     # subsets from this store, references from oracle/fit
            for r in rs:
                groups[cs.key(r["reference"])].append(r)
            for kref, g in groups.items():
                same = [r for r in g if cs.key(r["subset"]) == kref]
                if same:
                    log(f"    {mode}/{method} {kref}: subset equals its reference, {len(same)} row(s) not scored")
                g = [r for r in g if cs.key(r["subset"]) != kref]
                res = {cs.key(r["subset"]): sub[cs.key(r["subset"])] for r in g}
                if kref in refs:
                    res[kref] = refs[kref]
                out += score.score_rows(clip, g, res, method=method, mode=mode, e2e_meta=em, **kw)
            log(f"    {clip} {mode}/{method}: {len(sub)} results, {len(rs)} rows")
    return out


def _cmd_score(a):
    reg = config.load_registry(a.registry)
    clips = _clips(a, reg)
    modes = MODES if a.mode == "both" else (a.mode,)
    sizes = [int(s) for s in _csv(a.sizes)] or None

    def run(c):
        if not os.path.isdir(os.path.join(a.out_dir, c)):
            return f"no results under {os.path.join(a.out_dir, c)}"
        rows = score_clip(c, reg, a.cache_dir, a.out_dir, a.analysis_dir, modes, sizes, prior_dir=a.prior_dir)
        p = report.write_csv(rows, os.path.join(a.out_dir, c, "rows.csv"))
        print(f"[camsubsets] {c}: {len(rows)} rows -> {p}")
    return _each(clips, run, True, tag="camsubsets")


def _cmd_report(a):
    reg = config.load_registry(a.registry)
    clips = _clips(a, reg)
    rows = []
    for c in clips:
        p = os.path.join(a.out_dir, c, "rows.csv")
        if os.path.exists(p):
            rows += report.read_csv(p)
        else:
            print(f"[camsubsets] {c}: no {p} (run score first), skipped")
    if not rows:
        print("[camsubsets] no rows to report")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    report.write_csv(rows, a.out)                         # every file first: a console print can't lose them
    md = report.markdown_table(report.pivot(rows))
    rig = report.vs_full_rig_table(rows)                   # 1-camera methods vs the full 5-camera rig (ref_cams 5)
    if rig:
        md += "\n1 camera vs the full 5-camera rig (ref_cams 5, deployment rows, seen players):\n\n"
        md += report.vs_full_rig_markdown(rig)
    if a.pivot_md:
        with open(a.pivot_md, "w", encoding="utf-8") as fh:
            fh.write(md)
    fig = report.figure(rows, a.figure) if a.figure else None
    print(md)
    print(f"[camsubsets] {len(rows)} rows -> {a.out}")
    if a.pivot_md:
        print(f"[camsubsets] pivot -> {a.pivot_md}")
    if fig:
        print(f"[camsubsets] figure -> {fig}")
    return 0


def _cmd_export(a):
    from analysis.pose24.camsubsets import export
    subset = [v for part in _csv(a.subset) for v in part.split("+") if v]
    mono_opts = None
    if a.method.startswith("mono-len-"):          # the study's labels: lengths (+ learned prior) -> mono_opts (L9)
        src, prior = parse_mono_label(a.method)
        mono_opts = {"lengths": src}
        if prior:
            if len(subset) != 1:
                sys.exit(f"--method {a.method} takes exactly 1 view")
            p = prior_file(a.prior_dir, a.clip, subset[0], prior)
            if not os.path.exists(p):
                sys.exit(f"--method {a.method}: no prior file {p} (check the prior name and --prior-dir)")
            mono_opts["prior_path"] = p
    export.export(a.clip, subset, registry_path=a.registry, method=a.method, mono_opts=mono_opts, name=a.name,
                  analysis_dir=a.analysis_dir, cache_dir=a.cache_dir)
    return 0


def _stored_metas(out_dir, clip, build_id, log):
    """{mode: {method: {key: result meta}}} of the store (meta only, no arrays); stale results (another
    build_id than the cache's) are skipped."""
    out = {}
    for mode in MODES:
        for method, paths in _stored(out_dir, clip, mode).items():
            for k, p in paths.items():
                try:
                    with np.load(p) as z:
                        m = json.loads(str(z["meta"]))
                except (OSError, ValueError, KeyError, EOFError):
                    log(f"    {p}: unreadable, skipped")
                    continue
                if build_id is not None and m.get("build_id") != build_id:
                    log(f"    {p}: stale (build {m.get('build_id')} != cache {build_id}), skipped")
                    continue
                out.setdefault(mode, {}).setdefault(method, {})[k] = m
    return out


def _cmd_cost(a):
    from analysis.pose24.camsubsets import cost
    reg = config.load_registry(a.registry)
    clips = _clips(a, reg)
    rows = []

    def run(c):
        cpath = track.cache_path(c, a.cache_dir)
        bid = _cache_meta(cpath).get("build_id") if os.path.exists(cpath) else None
        doc = _full_doc(a.analysis_dir, c)
        rt = ((((doc or {}).get("meta") or {}).get("pose24") or {}).get("runtime_s"))
        timing = cost.load_timing(a.gpu_timing, c)
        if a.gpu_timing and timing is None:
            print(f"[camsubsets] {c}: no GPU timing for this clip in {a.gpu_timing} (gpu_2d_s empty)")
        rs = cost.cost_rows(c, reg, rt, _stored_metas(a.out_dir, c, bid, print), gpu_timing=timing,
                            data_dir=a.data_dir, setup_min_per_cam=a.setup_min_per_cam)
        rows.extend(rs)
        print(f"[camsubsets] {c}: {len(rs)} cost rows")
    rc = _each(clips, run, True, tag="camsubsets")
    if rows:
        print(cost.table(rows))
        print(f"[camsubsets] {len(rows)} rows -> {cost.write_csv(rows, a.out)}")
    return rc


def _common(p):
    p.add_argument("--out-dir", default=oracle.DEFAULT_OUT, help="subset result store")
    p.add_argument("--analysis-dir", default=config.DEFAULT_ANALYSIS_DIR)
    p.add_argument("--cache-dir", default=pv.CACHE_DIR, help="pose24 debug caches")
    p.add_argument("--registry", default=config.REGISTRY_PATH, help="clips.json")


def _clip_args(p, all5=True):
    p.add_argument("--clip", action="append")
    if all5:
        p.add_argument("--all5", action="store_true", help=f"the true 5-camera clips: {', '.join(ALL5)}")
    p.add_argument("--all", action="store_true", help="every full registry clip")


def _safe_console():
    """Unencodable characters (en dash, ·, °, ≥ on a cp932 console) print as "?" instead of raising; the
    console encoding is kept (forcing UTF-8 would garble a cp932 console). Files are always written UTF-8."""
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            try:
                st.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv=None):
    _safe_console()
    ap = argparse.ArgumentParser(prog="python -m analysis.pose24.camsubsets", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fit (oracle) or build (e2e) every camera subset -> <out-dir>/<clip>/<mode>/")
    _clip_args(b)
    b.add_argument("--sizes", default="1,2,3,4", help="subset sizes (1 = mono)")
    b.add_argument("--methods", default=None,
                   help="fit, mono (one label per --mono-lengths value); default fit,mono (oracle) / fit (e2e)")
    b.add_argument("--mono-lengths", type=_mono_lengths, default="oracle,prior", help="oracle, prior")
    b.add_argument("--mono-prior", type=_mono_priors, default=[],
                   help="comma list of learned prior methods (e.g. motionbert-lite,motionbert-full): + jobs "
                        "mono-len-<length>+<prior> per view with <prior-dir>/<clip>/<view>_<prior>.npz (a missing "
                        "file: job skipped); default none")
    b.add_argument("--prior-dir", default=None, help="learned prior root (default data/smpl_mono)")
    b.add_argument("--mode", choices=MODES, default="oracle")
    b.add_argument("--test-views", default="all", help="all | comma-separated held-out views")
    b.add_argument("--e2e-subsets", default="pairs", help="pairs | comma list of keys a+b,c+d (mode e2e)")
    b.add_argument("--workers", type=int, default=None, help="default 6 (oracle) / 3 (e2e)")
    b.add_argument("--fit-passes", type=int, default=None, help="fit_opts passes (default: the SP1 fit's)")
    b.add_argument("--keep-builds", action="store_true", help="e2e: keep the scratch builds")
    b.add_argument("--dry-run", action="store_true", help="print jobs + runtime estimate, run nothing")
    _common(b)
    s = sub.add_parser("score", help="stored results -> <out-dir>/<clip>/rows.csv")
    _clip_args(s)
    s.add_argument("--mode", choices=MODES + ("both",), default="both")
    s.add_argument("--sizes", default=None, help="default: the subset sizes found in the store")
    s.add_argument("--prior-dir", default=None, help="learned prior root (default data/smpl_mono): prior results "
                                                     "are fresh only for the prior file they were fitted with")
    _common(s)
    r = sub.add_parser("report", help="combined CSV + pivot table (+ figure)")
    _clip_args(r)
    r.add_argument("--out", default=DEFAULT_REPORT)
    r.add_argument("--figure", default=None, help="png path")
    r.add_argument("--pivot-md", default=None, help="write the pivot markdown table here")
    _common(r)
    e = sub.add_parser("export", help="registry entry + real pose24 build of one subset")
    e.add_argument("--clip", required=True)
    e.add_argument("--subset", required=True, help="v1,v2 (or v1+v2)")
    e.add_argument("--method", default="fit", help="fit | mono-len-<oracle|prior>[+<prior>] (the report's labels)")
    e.add_argument("--prior-dir", default=None, help="learned prior root (default data/smpl_mono)")
    e.add_argument("--name", default=None)
    _common(e)
    k = sub.add_parser("cost", help="cost table per stored subset (capture MB, CPU-s, GPU-s) -> CSV")
    _clip_args(k)
    k.add_argument("--gpu-timing", default=None, help="<clip>_gpu_timing.json or a directory of them "
                                                      "(tools/cost/pose2d_gpu_timed.py)")
    k.add_argument("--out", default=DEFAULT_COST)
    k.add_argument("--data-dir", default=".", help="root the clips.json video paths are relative to (the repo root)")
    k.add_argument("--setup-min-per-cam", type=float, default=None, help="per-camera setup minutes (default: "
                                                                          "not measured)")
    _common(k)
    a = ap.parse_args(argv)
    return {"build": _cmd_build, "score": _cmd_score, "report": _cmd_report, "export": _cmd_export,
            "cost": _cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
