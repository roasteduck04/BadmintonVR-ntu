"""camsubsets CLI: dry-run counts on the real caches (skipped without data), synthetic build -> score -> report."""
import json
import os

import pytest

import camsubsets_synth as cs_syn
from analysis.pose24.camsubsets import __main__ as cli
from analysis.pose24.camsubsets import e2e, oracle
from analysis.pose24.camsubsets import report
from analysis.pose24.camsubsets import subsets as cs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "data", "pose_cache")
ANALYSIS_DIR = os.path.join(ROOT, "Assets", "StreamingAssets", "analysis")


def _no_runs(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not fit or build")
    monkeypatch.setattr(oracle, "run_fits", boom)
    monkeypatch.setattr(e2e, "run_e2e", boom)
    monkeypatch.setattr(oracle, "fit_subset", boom)


def _real_dry_run(monkeypatch, capsys, clip):
    if not os.path.exists(os.path.join(CACHE_DIR, f"{clip}_pose24.npz")):
        pytest.skip(f"no local pose24 cache for {clip}")
    _no_runs(monkeypatch)
    rc = cli.main(["build", "--clip", clip, "--sizes", "2,3,4", "--mode", "oracle", "--dry-run",
                   "--cache-dir", CACHE_DIR, "--analysis-dir", ANALYSIS_DIR])
    assert rc == 0
    return capsys.readouterr().out


def test_dry_run_counts_fits(monkeypatch, capsys):
    out = _real_dry_run(monkeypatch, capsys, "singles_05x")
    assert "26 distinct fits" in out and "min" in out


def test_dry_run_excludes_frontright_21(monkeypatch, capsys):
    out = _real_dry_run(monkeypatch, capsys, "doubles_1side")
    assert "11 distinct fits" in out
    kept = next(l for l in out.splitlines() if "kept views" in l)
    assert "frontright_21" not in kept.split("excluded")[0]


def _registry(tmp_path, clips=("synfull",)):
    path = tmp_path / "clips.json"
    path.write_text(json.dumps({"clips": {c: {"mode": "singles", "seeds": c, "views": cs_syn.VIEWS} for c in clips}}),
                    encoding="utf-8")
    return str(path)


def test_jobs_expand_mono_labels_and_fit_subsets():
    rows = cs.study_rows(cs_syn.VIEWS, (1, 2))
    jobs = cli.build_jobs(rows, ["fit", "mono"], ["oracle", "prior"], (1, 2))
    assert set(jobs) == {"fit", "mono-len-oracle", "mono-len-prior"}
    assert [s for _, s in jobs["fit"]] == cs.distinct_fits(rows)
    assert jobs["mono-len-prior"] == [("mono-len-prior", (v,)) for v in cs_syn.VIEWS]
    assert set(cli.build_jobs(rows, ["fit", "mono"], ["oracle"], (2,))) == {"fit"}          # size 1 not asked
    assert cli.MONO_OPTS["mono-len-oracle"] == {"lengths": "oracle"} and cli.MONO_OPTS["fit"] is None


def test_e2e_subsets_pairs_and_keys():
    assert cli.e2e_subsets("pairs", ["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert cli.e2e_subsets("c+a,b+d", ["a", "b", "c", "d"]) == [("a", "c"), ("b", "d")]
    with pytest.raises(ValueError, match="zz"):
        cli.e2e_subsets("a+zz", ["a", "b"])


def test_dry_run_synthetic_mono_and_failure_isolation(tmp_path, monkeypatch, capsys):
    cs_syn.full_cache(tmp_path, F=10)                                      # synfull only; "missing" has no cache
    reg = _registry(tmp_path, ("missing", "synfull"))
    _no_runs(monkeypatch)
    rc = cli.main(["build", "--clip", "missing", "--clip", "synfull", "--sizes", "1,2", "--methods", "fit,mono",
                   "--dry-run", "--registry", reg, "--cache-dir", str(tmp_path), "--analysis-dir", str(tmp_path),
                   "--workers", "2"])
    cap = capsys.readouterr()
    assert rc == 1 and "missing" in cap.err and "FAILED" in cap.err
    fits = len(cs.distinct_fits(cs.study_rows(cs_syn.VIEWS, (1, 2))))
    assert f"{fits} distinct fits" in cap.out and "8 mono jobs" in cap.out
    # (fits + 8 mono) x 180 s default x 1.5 / 2 workers
    assert f"{(fits + 8) * 180 * 1.5 / 2 / 60:.1f} min" in cap.out


def test_build_score_report_synthetic(tmp_path, capsys):
    cache_path, cache, _ = cs_syn.full_cache(tmp_path, F=30)
    reg = _registry(tmp_path)
    out_dir, adir = str(tmp_path / "subsets"), str(tmp_path / "an")
    common = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir", adir]
    assert cli.main(["build", "--clip", "synfull", "--sizes", "3", "--methods", "fit", "--mode", "oracle",
                     "--test-views", "a", "--workers", "1", "--fit-passes", "1"] + common) == 0
    rows = cs.study_rows(cache["views"], (3,), ["a"], excluded=cache["excluded"])
    for s in cs.distinct_fits(rows):
        assert os.path.exists(oracle.result_path(out_dir, "synfull", "oracle", "fit", s))
    assert cli.main(["score", "--clip", "synfull", "--mode", "oracle"] + common) == 0
    per_clip = os.path.join(out_dir, "synfull", "rows.csv")
    got = report.read_csv(per_clip)
    assert got and {r["method"] for r in got} == {"fit"} and {r["mode"] for r in got} == {"oracle"}
    assert {r["clip_type"] for r in got} == {"singles"} and any(r["test_view"] == "a" for r in got)
    assert any(r["test_view"] == "" for r in got)                          # deployment rows
    combined, md = str(tmp_path / "all.csv"), str(tmp_path / "pivot.md")
    capsys.readouterr()
    assert cli.main(["report", "--clip", "synfull", "--out", combined, "--pivot-md", md] + common) == 0
    assert len(report.read_csv(combined)) == len(got)
    assert "| clip_type |" in capsys.readouterr().out and open(md, encoding="utf-8").read().startswith("| clip_type")


def test_score_e2e_uses_oracle_references_and_e2e_meta(tmp_path):
    cache_path, cache, _ = cs_syn.full_cache(tmp_path, F=30)
    reg = _registry(tmp_path)
    out_dir = str(tmp_path / "subsets")
    common = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir", str(tmp_path)]
    assert cli.main(["build", "--clip", "synfull", "--sizes", "3", "--methods", "fit", "--test-views", "a",
                     "--workers", "1", "--fit-passes", "1"] + common) == 0
    res = oracle.fit_subset(cache, ["a", "c"], fit_opts={"passes": 1})
    em = {"shift": 2, "spread": 0.25, "match": {"P1": "P1", "P2": "P2"}, "identity": {}}
    oracle.save_result(oracle.result_path(out_dir, "synfull", "e2e", "fit", ("a", "c")), res,
                       dict(oracle.expected_meta(cache, "fit", None, None), e2e=em))
    assert cli.main(["score", "--clip", "synfull"] + common) == 0
    got = report.read_csv(os.path.join(out_dir, "synfull", "rows.csv"))
    e = [r for r in got if r["mode"] == "e2e"]
    assert e and {r["subset"] for r in e} == {"a+c"} and all(r["sync_shift_f"] == 2.0 for r in e)
    assert {r["test_view"] for r in e} == {"b", "d", ""}                  # a+c held out against b, d + deployment
    assert any(r["mode"] == "oracle" for r in got)


@pytest.mark.parametrize("bad", ["oracle,wham", "", "bogus"])
def test_build_rejects_unknown_mono_lengths_at_parse_time(bad, monkeypatch, capsys):
    _no_runs(monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli.main(["build", "--clip", "synfull", "--mono-lengths", bad, "--dry-run"])
    assert e.value.code == 2 and "--mono-lengths" in capsys.readouterr().err


def test_report_cp932_console_writes_files(tmp_path, monkeypatch):
    import io
    import sys
    from analysis.pose24.camsubsets import score
    reg = _registry(tmp_path)
    out_dir = tmp_path / "subsets"
    (out_dir / "synfull").mkdir(parents=True)
    rows = []
    for s, n, v in (("a", 1, None), ("a+b", 2, 8.0), ("a+c", 2, 12.0)):      # None -> "–" in the pivot md
        r = dict.fromkeys(score.ROW_KEYS)
        r.update(clip="synfull", clip_type="singles", mode="oracle", method="fit", player="P1", player_seen=True,
                 test_view="t", matched=True, subset=s, n_cams=n, ref_cams=3, heldout_px_median=v, coverage=1.0)
        rows.append(r)
    report.write_csv(rows, str(out_dir / "synfull" / "rows.csv"))
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    monkeypatch.setattr(sys, "stdout", console)
    md, fig = tmp_path / "pivot.md", tmp_path / "fig.png"
    rc = cli.main(["report", "--clip", "synfull", "--out", str(tmp_path / "all.csv"), "--pivot-md", str(md),
                   "--figure", str(fig), "--registry", reg, "--out-dir", str(out_dir), "--cache-dir", str(tmp_path),
                   "--analysis-dir", str(tmp_path)])
    assert rc == 0 and md.exists() and fig.exists()
    assert "\u2013" in md.read_bytes().decode("utf-8")
    console.flush()
    assert b"?" in console.buffer.getvalue()                                  # en dash replaced on the console


def test_report_appends_vs_full_rig_table(tmp_path):
    from analysis.pose24.camsubsets import score
    reg = _registry(tmp_path)
    out_dir = tmp_path / "subsets"
    (out_dir / "synfull").mkdir(parents=True)
    rows = []
    for m, v in (("mono-len-oracle", 18.0), ("mono-len-oracle+motionbert-full", 16.0)):
        r = dict.fromkeys(score.ROW_KEYS)
        r.update(clip="synfull", clip_type="singles", mode="oracle", method=m, player="P1", player_seen=True,
                 test_view="", is_reference=False, matched=True, subset="b", n_cams=1, ref_cams=5,
                 w_mpjpe_cm_median=v, coverage=1.0)
        rows.append(r)
    report.write_csv(rows, str(out_dir / "synfull" / "rows.csv"))
    md = tmp_path / "pivot.md"
    assert cli.main(["report", "--clip", "synfull", "--out", str(tmp_path / "all.csv"), "--pivot-md", str(md),
                     "--registry", reg, "--out-dir", str(out_dir), "--cache-dir", str(tmp_path),
                     "--analysis-dir", str(tmp_path)]) == 0
    text = md.read_text(encoding="utf-8")
    assert "vs the full 5-camera rig" in text and "| singles | 1 cam + MotionBERT-full (rig lengths) | 1 | 16.0 |" in text


def test_score_skips_results_made_under_other_settings(tmp_path, capsys, monkeypatch):
    """review 1 L8: score checked only build_id, so results fitted under another regressor / fit / prior were scored
    silently. The build's own --fit-passes choice is not staleness."""
    from analysis.pose24 import regress
    cache_path, cache, _ = cs_syn.full_cache(tmp_path, F=30)
    reg = _registry(tmp_path)
    out_dir = str(tmp_path / "subsets")
    common = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir", str(tmp_path)]
    assert cli.main(["build", "--clip", "synfull", "--sizes", "3", "--methods", "fit", "--mode", "oracle",
                     "--test-views", "a", "--workers", "1", "--fit-passes", "1"] + common) == 0
    capsys.readouterr()
    assert cli.main(["score", "--clip", "synfull", "--mode", "oracle"] + common) == 0
    out = capsys.readouterr().out
    assert "stale" not in out and report.read_csv(os.path.join(out_dir, "synfull", "rows.csv"))
    monkeypatch.setitem(regress.CONSTANTS, "hjc_medial", 0.71)
    assert cli.main(["score", "--clip", "synfull", "--mode", "oracle"] + common) == 0
    out = capsys.readouterr().out
    assert "stale" in out and "regressor" in out
    assert not report.read_csv(os.path.join(out_dir, "synfull", "rows.csv"))


def test_export_cli_passes_the_prior_and_lengths_of_a_mono_label(tmp_path, monkeypatch):
    """review 1 L9: `export --method mono-len-oracle+<prior>` (report's best method) raised; the CLI now turns the
    label into mono_opts (lengths + the view's prior file)."""
    from analysis.pose24.camsubsets import export
    seen = []
    monkeypatch.setattr(export, "export", lambda clip, subset, **k: seen.append((clip, subset, k)) or "doc")
    pdir = tmp_path / "priors"
    (pdir / "synfull").mkdir(parents=True)
    pf = pdir / "synfull" / "c_wham.npz"
    pf.write_bytes(b"x")
    common = ["--registry", _registry(tmp_path), "--cache-dir", str(tmp_path), "--analysis-dir", str(tmp_path)]
    assert cli.main(["export", "--clip", "synfull", "--subset", "c", "--method", "mono-len-oracle+wham",
                     "--prior-dir", str(pdir)] + common) == 0
    clip, subset, k = seen[-1]
    assert (clip, subset, k["method"]) == ("synfull", ["c"], "mono-len-oracle+wham")
    assert k["mono_opts"] == {"lengths": "oracle", "prior_path": str(pf)}
    assert cli.main(["export", "--clip", "synfull", "--subset", "c", "--method", "mono-len-prior"] + common) == 0
    assert seen[-1][2]["mono_opts"] == {"lengths": "prior"}
    assert cli.main(["export", "--clip", "synfull", "--subset", "a,b"] + common) == 0
    assert seen[-1][2]["mono_opts"] is None and seen[-1][2]["method"] == "fit"
    with pytest.raises(SystemExit, match="no prior file"):
        cli.main(["export", "--clip", "synfull", "--subset", "d", "--method", "mono-len-oracle+wham",
                  "--prior-dir", str(pdir)] + common)
