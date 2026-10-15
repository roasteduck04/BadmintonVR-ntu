"""camsubsets with a learned mono depth prior (v0.7.5 Task 4): --mono-prior jobs, prior freshness, worker loading,
missing-file skips and the new method rows in score / report."""
import os

import numpy as np
import pytest

import camsubsets_synth as cs_syn
from analysis.pose24 import doc as docmod
from analysis.pose24 import kpset, mono
from analysis.pose24.camsubsets import __main__ as cli
from analysis.pose24.camsubsets import oracle, report
from analysis.pose24.camsubsets import subsets as cs
from analysis.pose24.priors import common

PRIOR = "motionbert-lite"


def _registry(tmp_path):
    import json
    path = tmp_path / "clips.json"
    path.write_text(json.dumps({"clips": {"synfull": {"mode": "singles", "seeds": "synfull", "views": cs_syn.VIEWS}}}),
                    encoding="utf-8")
    return str(path)


def _write_prior(prior_dir, cache, players, view, method=PRIOR, build_id=cs_syn.BUILD_ID, pids=("P1", "P2"),
                 noise=0.0):
    """A perfect (truth-derived) prior of `view` at <prior_dir>/synfull/<view>_<method>.npz."""
    F = int(cache["frame_count"])
    R = np.asarray(cache["cams"][view][1], float)
    per = {}
    for pid in pids:
        X = kpset.from133(players[pid]) * np.array([1.0, -1.0, 1.0])
        rel = (X - X[:, [11, 12]].mean(1, keepdims=True)) @ R.T
        rel = rel + noise * np.random.default_rng(0).standard_normal(rel.shape)
        per[pid] = {"kp": np.arange(35), "rel_cam": rel, "conf": np.ones(F)}
    meta = {"method": method, "clip": "synfull", "view": view, "timeline_of": "synfull", "fps": float(cache["fps"]),
            "frame_count": F, "build_id": build_id, "norm": "crop"}
    if build_id is None:
        meta.pop("build_id")
        path = os.path.join(prior_dir, "synfull", f"{view}_{method}.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {"meta": np.array(__import__("json").dumps(meta))}
        for pid, d in per.items():
            arrays.update({f"{pid}_kp": d["kp"], f"{pid}_rel_cam": d["rel_cam"], f"{pid}_conf": d["conf"]})
        np.savez(path, **arrays)
        return path
    path = os.path.join(prior_dir, "synfull", f"{view}_{method}.npz")
    common.write_prior(path, meta, per)
    return path


# --- labels / jobs ---

def test_jobs_add_prior_labels_and_keep_the_no_prior_ones():
    rows = cs.study_rows(cs_syn.VIEWS, (1, 2))
    jobs = cli.build_jobs(rows, ["fit", "mono"], ["oracle", "prior"], (1, 2),
                          mono_priors=["motionbert-lite", "motionbert-full"])
    assert set(jobs) == {"fit", "mono-len-oracle", "mono-len-prior", "mono-len-oracle+motionbert-lite",
                         "mono-len-oracle+motionbert-full", "mono-len-prior+motionbert-lite",
                         "mono-len-prior+motionbert-full"}
    lab = "mono-len-oracle+motionbert-lite"
    assert jobs[lab] == [(lab, (v,)) for v in cs_syn.VIEWS]
    assert set(cli.build_jobs(rows, ["fit", "mono"], ["oracle", "prior"], (1, 2))) == {
        "fit", "mono-len-oracle", "mono-len-prior"}                       # default: no prior labels


def test_label_parsing_and_mono_opts():
    assert cli.parse_mono_label("mono-len-oracle") == ("oracle", None)
    assert cli.parse_mono_label("mono-len-prior+motionbert-full") == ("prior", "motionbert-full")
    assert cli.mono_opts_for("mono-len-oracle") == {"lengths": "oracle"}             # unchanged (stored results)
    assert cli.mono_opts_for("mono-len-prior") == {"lengths": "prior"}
    assert cli.mono_opts_for("fit") is None
    o = cli.mono_opts_for("mono-len-oracle+motionbert-lite", clip="c1", views=["a", "b"], prior_dir="pd")
    assert o == {"lengths": "oracle", "prior_paths": {"a": os.path.join("pd", "c1", "a_motionbert-lite.npz"),
                                                      "b": os.path.join("pd", "c1", "b_motionbert-lite.npz")}}


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", "x+y"])
def test_mono_prior_names_are_checked_at_parse_time(bad, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["build", "--clip", "synfull", "--mono-prior", bad, "--dry-run"])
    assert e.value.code == 2 and "--mono-prior" in capsys.readouterr().err


# --- freshness key ---

def test_no_prior_freshness_key_has_no_prior_entry(tmp_path):
    from analysis.pose24 import confidence, fit, regress
    _, cache, _ = cs_syn.full_cache(tmp_path, F=10)
    want = {"build_id": cs_syn.BUILD_ID, "method": "mono-len-oracle",
            "fit_settings": docmod.json_safe({"fit": fit.settings(), "fit_opts": {}, "mono_opts": {"lengths": "oracle"},
                                              "regressor": regress.CONSTANTS, "confidence": confidence.settings(),
                                              "skeleton": "smpl-28", "rev": oracle.RESULT_REV,
                                              "mono": mono.settings()})}
    assert oracle.expected_meta(cache, "mono-len-oracle", None, {"lengths": "oracle"}) == want
    assert "prior" not in want["fit_settings"]


def test_prior_freshness_key_has_the_file_sha1_not_its_path(tmp_path):
    _, cache, players = cs_syn.full_cache(tmp_path, F=10)
    p = _write_prior(str(tmp_path / "pr"), cache, players, "c")
    lab = "mono-len-oracle+" + PRIOR
    m = oracle.expected_meta(cache, lab, None, {"lengths": "oracle", "prior_path": p})
    st = m["fit_settings"]
    assert st["mono_opts"] == {"lengths": "oracle"} and st["prior"]["sha1"] == oracle.file_sha1(p)
    assert len(st["prior"]["sha1"]) == 40 and p not in str(m)
    import shutil
    p2 = str(tmp_path / "elsewhere.npz")
    shutil.copy(p, p2)
    assert oracle.expected_meta(cache, lab, None, {"lengths": "oracle", "prior_path": p2}) == m   # same bytes
    _write_prior(str(tmp_path / "pr"), cache, players, "c", noise=0.01)                       # new content
    assert oracle.expected_meta(cache, lab, None, {"lengths": "oracle", "prior_path": p}) != m


def test_job_mono_opts_picks_the_view_path():
    o = {"lengths": "oracle", "prior_paths": {"a": "pa", "b": "pb"}}
    assert oracle.job_mono_opts(o, ("b",)) == {"lengths": "oracle", "prior_path": "pb"}
    assert oracle.job_mono_opts({"lengths": "prior"}, ("a",)) == {"lengths": "prior"}
    assert oracle.job_mono_opts(None, ("a",)) is None


# --- fit_subset with a prior ---

def test_fit_subset_uses_the_prior_and_refuses_another_build(tmp_path, monkeypatch):
    _, cache, players = cs_syn.full_cache(tmp_path, F=40)
    p = _write_prior(str(tmp_path / "pr"), cache, players, "c")
    lab = "mono-len-oracle+" + PRIOR
    seen = []
    real = mono.fit_mono

    def spy(*a, **k):
        seen.append(k.get("prior"))
        return real(*a, **k)
    monkeypatch.setattr(mono, "fit_mono", spy)
    res = oracle.fit_subset(cache, ["c"], method=lab, mono_opts={"lengths": "oracle", "prior_path": p, "passes": 1})
    monkeypatch.undo()
    assert res["method"] == lab and len(seen) == 2 and all(s is not None for s in seen)
    assert {s["meta"]["method"] for s in seen} == {PRIOR}
    assert np.isfinite(res["players"]["P1"]["J"]).any()
    bad = _write_prior(str(tmp_path / "bad"), cache, players, "c", build_id="otherbuild")
    with pytest.raises(ValueError, match="build"):
        oracle.fit_subset(cache, ["c"], method=lab, mono_opts={"lengths": "oracle", "prior_path": bad, "passes": 1})
    wrongview = _write_prior(str(tmp_path / "wv"), cache, players, "d")
    with pytest.raises(ValueError, match="view"):
        oracle.fit_subset(cache, ["c"], method=lab, mono_opts={"lengths": "oracle", "prior_path": wrongview,
                                                               "passes": 1})


def test_fit_subset_prior_without_build_id_warns_and_missing_pid_fits_without(tmp_path):
    _, cache, players = cs_syn.full_cache(tmp_path, F=30)
    p = _write_prior(str(tmp_path / "old"), cache, players, "c", build_id=None, pids=("P1",))
    with pytest.warns(UserWarning, match="build_id"):
        res = oracle.fit_subset(cache, ["c"], method="mono-len-oracle+" + PRIOR,
                                mono_opts={"lengths": "oracle", "prior_path": p, "passes": 1})
    assert set(res["players"]) == {"P1", "P2"}                     # P2 not in the prior: fitted without one


# --- run_fits / store / CLI ---

def test_run_fits_prior_jobs_in_spawn_workers_and_freshness(tmp_path):
    path, cache, players = cs_syn.full_cache(tmp_path, F=20)
    pd = str(tmp_path / "pr")
    pa, pc = _write_prior(pd, cache, players, "a"), _write_prior(pd, cache, players, "c")
    lab = "mono-len-oracle+" + PRIOR
    opts = {"lengths": "oracle", "passes": 1, "prior_paths": {"a": pa, "c": pc}}
    out = str(tmp_path / "s")
    got = oracle.run_fits("synfull", [(lab, ("a",)), (lab, ("c",))], path, out, workers=2, mono_opts=opts,
                          log=lambda *a: None)
    assert set(got) == {"a", "c"}
    rp = oracle.result_path(out, "synfull", "oracle", lab, ("c",))
    assert got["c"] == rp and os.path.basename(os.path.dirname(rp)) == lab and os.path.exists(rp)
    m = oracle.load_result(rp, None)["meta"]
    assert m["fit_settings"]["prior"]["sha1"] == oracle.file_sha1(pc) and "prior_paths" not in str(m)
    logs = []
    oracle.run_fits("synfull", [(lab, ("a",)), (lab, ("c",))], path, out, workers=2, mono_opts=opts, log=logs.append)
    assert "2 fresh, 0 to fit" in logs[0]
    _write_prior(pd, cache, players, "c", noise=0.01)              # c's prior changed -> only c is stale
    logs = []
    oracle.run_fits("synfull", [(lab, ("a",)), (lab, ("c",))], path, out, workers=1, mono_opts=opts, log=logs.append)
    assert "1 fresh, 1 to fit" in logs[0]
    assert cli._stored(out, "synfull", "oracle").keys() == {lab}


def test_cli_build_skips_missing_priors_and_dry_run_counts(tmp_path, monkeypatch, capsys):
    path, cache, players = cs_syn.full_cache(tmp_path, F=20)
    reg = _registry(tmp_path)
    pd = str(tmp_path / "pr")
    _write_prior(pd, cache, players, "c")
    out_dir = str(tmp_path / "subsets")
    common_args = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir",
                   str(tmp_path), "--prior-dir", pd]
    assert cli.main(["build", "--clip", "synfull", "--sizes", "1", "--methods", "mono", "--mono-lengths", "oracle",
                     "--mono-prior", PRIOR, "--dry-run", "--workers", "1"] + common_args) == 0
    o = capsys.readouterr().out
    assert "8 mono jobs" in o and "4 with a learned prior (1 prior file present)" in o
    assert cli.main(["build", "--clip", "synfull", "--sizes", "1", "--methods", "mono", "--mono-lengths", "oracle",
                     "--mono-prior", PRIOR, "--workers", "1", "--fit-passes", "1"] + common_args) == 0
    o = capsys.readouterr().out
    lab = "mono-len-oracle+" + PRIOR
    for v in ("a", "b", "d"):
        assert f"{lab} {v}: skipped: no prior" in o
    st = cli._stored(out_dir, "synfull", "oracle")
    assert set(st) == {"mono-len-oracle", lab} and set(st[lab]) == {"c"} and set(st["mono-len-oracle"]) == set("abcd")


def test_score_report_treat_prior_label_as_its_own_method(tmp_path, capsys):
    path, cache, players = cs_syn.full_cache(tmp_path, F=30)
    reg = _registry(tmp_path)
    pd = str(tmp_path / "pr")
    for v in cs_syn.VIEWS:
        _write_prior(pd, cache, players, v)
    out_dir = str(tmp_path / "subsets")
    common_args = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir",
                   str(tmp_path)]
    assert cli.main(["build", "--clip", "synfull", "--sizes", "1,3", "--methods", "fit,mono", "--mono-lengths",
                     "oracle", "--mono-prior", PRIOR, "--test-views", "a", "--workers", "1", "--fit-passes", "1",
                     "--prior-dir", pd] + common_args) == 0
    assert cli.main(["score", "--clip", "synfull", "--mode", "oracle"] + common_args) == 0
    rows = report.read_csv(os.path.join(out_dir, "synfull", "rows.csv"))
    lab = "mono-len-oracle+" + PRIOR
    assert {r["method"] for r in rows} == {"fit", "mono-len-oracle", lab}
    pv = report.pivot(rows, metrics=["heldout_px_median"])
    assert {p["method"] for p in pv if p["n_cams"] == 1} == {"mono-len-oracle", lab}
    md = report.markdown_table(pv)
    assert "1 cam + MotionBERT-lite (rig lengths)" in md and "1 cam (rig lengths)" in md
    labs = [lab_ for _, lab_ in report.figure_series(pv)]
    assert any("1 cam + MotionBERT-lite (rig lengths)" in s for s in labs)
    fig = report.figure(rows, str(tmp_path / "f.png"))
    assert open(fig, "rb").read(4) == b"\x89PNG"


def test_method_display_names():
    assert report.method_name("fit") == "multi-cam fit"
    assert report.method_name("mono-len-oracle") == "1 cam (rig lengths)"
    assert report.method_name("mono-len-prior") == "1 cam (template lengths)"
    assert report.method_name("mono-len-oracle+motionbert-lite") == "1 cam + MotionBERT-lite (rig lengths)"
    assert report.method_name("mono-len-prior+motionbert-full") == "1 cam + MotionBERT-full (template lengths)"
    assert report.method_name("mono-len-oracle+wham") == "1 cam + WHAM (rig lengths)"
    assert report.method_name("whatever") == "whatever"
    assert report.method_name(None) is None


def test_cli_build_fails_when_a_named_prior_has_no_file_at_all(tmp_path, capsys):
    path, cache, players = cs_syn.full_cache(tmp_path, F=20)
    reg = _registry(tmp_path)
    pd = str(tmp_path / "pr")
    _write_prior(pd, cache, players, "c")                        # motionbert-lite exists for one view only
    out_dir = str(tmp_path / "subsets")
    common_args = ["--registry", reg, "--cache-dir", str(tmp_path), "--out-dir", out_dir, "--analysis-dir",
                   str(tmp_path), "--prior-dir", pd, "--sizes", "1", "--methods", "mono", "--mono-lengths", "oracle",
                   "--workers", "1", "--fit-passes", "1"]
    typo = "motionbert-ful"
    assert cli.main(["build", "--clip", "synfull", "--mono-prior", typo, "--dry-run"] + common_args) == 0   # info only
    capsys.readouterr()
    assert cli.main(["build", "--clip", "synfull", "--mono-prior", f"{PRIOR},{typo}"] + common_args) != 0
    err = capsys.readouterr().err
    assert f"--mono-prior {typo}: no prior file" in err and os.path.join(pd, "synfull", f"a_{typo}.npz") in err
    assert f"--mono-prior {PRIOR}:" not in err                    # a partly present prior still skips per view
    assert not cli._stored(out_dir, "synfull", "oracle")          # nothing was fitted
