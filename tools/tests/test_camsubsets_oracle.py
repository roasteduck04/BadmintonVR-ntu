"""camsubsets oracle: slicing, subset fit, store, pool."""
import os

import numpy as np
import pytest

import camsubsets_synth as cs_syn
from analysis.pose24.camsubsets import oracle


def test_slice_bundle_restricts_views_and_carries_excluded(tmp_path):
    _, cache, _ = cs_syn.full_cache(tmp_path)
    cache["excluded"] = {"e": 927.0}
    s = oracle.slice_bundle(cache, ["c", "a"])
    assert s["views"] == ["a", "c"] and set(s["cams"]) == {"a", "c"} and s["excluded"] == {"e": 927.0}
    assert cache["views"] == ["a", "b", "c", "d"]                   # original untouched
    with pytest.raises(ValueError):
        oracle.slice_bundle(cache, ["a", "zz"])


def test_fit_subset_pair_matches_truth_and_flags_unseen_player(tmp_path):
    _, cache, players = cs_syn.full_cache(tmp_path)
    res = oracle.fit_subset(cache, ["a", "b"], fit_opts={"passes": 1})
    p1 = res["players"]["P1"]
    assert p1["coverage"] > 0.95
    err = np.nanmedian(np.linalg.norm(p1["X"][:, 11] - players["P1"][:, 11], axis=-1))
    assert err < 0.05                                                 # hip within 5 cm
    assert res["players"]["P2"]["coverage"] == 0.0                    # a+b never see P2
    with pytest.raises(ValueError):
        oracle.fit_subset(cache, ["a"], method="fit")


def test_fit_subset_slices_lr_swapped_by_view(tmp_path):
    _, cache, _ = cs_syn.full_cache(tmp_path, F=30)
    assert cache["extra"]["P1"]["lr_swapped"].shape == (4, 30)       # (V, F) like real caches
    lr = oracle.fit_subset(cache, ["c", "a"], fit_opts={"passes": 1})["players"]["P1"]["lr_swapped"]
    assert lr.shape == (30, 2) and lr.dtype == bool
    assert not lr[:, 0].any()                                         # view a
    np.testing.assert_array_equal(np.nonzero(lr[:, 1])[0], np.arange(5, 10))   # view c, frames 5-9


def _blas_threads():
    from threadpoolctl import threadpool_info
    return [d["num_threads"] for d in threadpool_info() if d.get("user_api") == "blas"]


def test_pool_workers_use_one_blas_thread(tmp_path):
    pytest.importorskip("threadpoolctl")
    path, _, _ = cs_syn.full_cache(tmp_path, F=10)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, initializer=oracle._init_worker, initargs=(path,)) as ex:
        info = ex.submit(_blas_threads).result()
    assert info and all(n == 1 for n in info)


def test_run_fits_restores_parent_thread_env(tmp_path, monkeypatch):
    path, _, _ = cs_syn.full_cache(tmp_path, F=10)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")                          # sentinel
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    got = oracle.run_fits("synfull", [("fit", ("a", "c"))], path, str(tmp_path / "s"), workers=2,
                          fit_opts={"passes": 1}, log=lambda *a: None)
    assert set(got) == {"a+c"}
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ and "MKL_NUM_THREADS" not in os.environ


def test_fit_subset_method_labels(tmp_path):
    _, cache, _ = cs_syn.full_cache(tmp_path, F=10)
    with pytest.raises(ValueError):
        oracle.fit_subset(cache, ["a", "b"], method="mono-len-oracle")   # mono takes exactly 1 view
    with pytest.raises(ValueError):
        oracle.fit_subset(cache, ["a", "b"], method="bogus")
    with pytest.raises(ValueError):
        oracle.fit_subset(cache, ["a"], method="bogus")


def test_store_skips_fresh_and_recomputes_stale(tmp_path):
    path, cache, _ = cs_syn.full_cache(tmp_path)
    out = str(tmp_path / "subsets")
    jobs = [("fit", ("a", "c")), ("fit", ("b", "d"))]
    got = oracle.run_fits("synfull", jobs, path, out, workers=1, fit_opts={"passes": 1}, log=lambda *a: None)
    assert set(got) == {"a+c", "b+d"}
    assert got["a+c"] == oracle.result_path(out, "synfull", "oracle", "fit", ("c", "a"))
    m0 = os.path.getmtime(got["a+c"])
    oracle.run_fits("synfull", jobs, path, out, workers=1, fit_opts={"passes": 1}, log=lambda *a: None)
    assert os.path.getmtime(got["a+c"]) == m0                         # fresh -> skipped
    exp = oracle.expected_meta(cache, "fit", {"passes": 2}, None)
    assert oracle.load_result(got["a+c"], exp) is None                # other fit settings -> stale
    r = oracle.load_result(got["a+c"], oracle.expected_meta(cache, "fit", {"passes": 1}, None))
    assert r is not None and r["method"] == "fit" and r["views"] == ["a", "c"]
    assert r["players"]["P1"]["J"].dtype == np.float32 and r["players"]["P1"]["inlier"].dtype == bool
    assert not any(f.endswith(".tmp.npz") for f in os.listdir(os.path.dirname(got["a+c"])))


def test_partial_file_is_never_read(tmp_path):
    p = str(tmp_path / "x.npz")
    with open(p + ".tmp.npz", "wb") as fh:
        fh.write(b"garbage")                                          # killed mid-write
    assert oracle.load_result(p, {"build_id": "x", "fit_settings": {}, "method": "fit"}) is None
    with open(p, "wb") as fh:
        fh.write(b"PK\x03\x04truncated")                              # corrupt final file
    assert oracle.load_result(p, None) is None


def test_pool_matches_serial(tmp_path):
    path, _, _ = cs_syn.full_cache(tmp_path)
    kw = dict(fit_opts={"passes": 1}, log=lambda *a: None)
    a = oracle.run_fits("synfull", [("fit", ("a", "c"))], path, str(tmp_path / "s1"), workers=1, **kw)
    b = oracle.run_fits("synfull", [("fit", ("a", "c"))], path, str(tmp_path / "s2"), workers=2, **kw)
    ra = oracle.load_result(a["a+c"], None)
    rb = oracle.load_result(b["a+c"], None)
    np.testing.assert_allclose(ra["players"]["P1"]["J"], rb["players"]["P1"]["J"], atol=1e-5)


def test_fit_subset_mono_one_view(tmp_path):
    _, cache, players = cs_syn.full_cache(tmp_path)
    assert set(cache["players_lengths"]["P1"]) == {"edges", "soft", "stature"}
    res = oracle.fit_subset(cache, ["c"], method="mono", mono_opts={"lengths": "oracle"})
    p1 = res["players"]["P1"]
    assert p1["coverage"] > 0.9 and np.isfinite(p1["J"][np.isfinite(p1["J"]).all(-1).all(-1)]).all()
    assert np.isfinite(p1["J"]).all(-1).all(-1).mean() > 0.9
    assert p1["limb_lengths_m"] == cache["players_lengths"]["P1"]["edges"]      # oracle lengths from the cache
    err = np.nanmedian(np.linalg.norm((p1["X"][:, 11] - players["P1"][:, 11])[:, [0, 2]], axis=-1))
    assert err < 0.15


def _key_settings(**extra):
    from analysis.pose24 import confidence, fit, regress
    from analysis.pose24 import doc as docmod
    return docmod.json_safe(dict({"fit": fit.settings(), "fit_opts": {}, "mono_opts": {},
                                  "regressor": regress.CONSTANTS, "confidence": confidence.settings(),
                                  "skeleton": "smpl-28", "rev": oracle.RESULT_REV}, **extra))


def test_fit_freshness_key_and_mono_adds_mono_settings(tmp_path):
    """M4: only mono methods get mono.settings(). Review 1 M3: the key also holds the regressor, the confidence
    model, the skeleton and the result code revision (every stored J / C / sigma depends on them)."""
    import json
    from analysis.pose24 import doc as docmod
    from analysis.pose24 import mono
    _, cache, _ = cs_syn.full_cache(tmp_path, F=10)
    want = {"build_id": cs_syn.BUILD_ID, "method": "fit", "fit_settings": _key_settings()}
    got = oracle.expected_meta(cache, "fit", None, None)
    assert got == want and json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
    m = oracle.expected_meta(cache, "mono-len-oracle", None, {"lengths": "oracle"})
    assert m["fit_settings"]["mono"] == docmod.json_safe(mono.settings())
    assert m["fit_settings"]["fit"] == want["fit_settings"]["fit"]


def test_freshness_key_follows_regressor_confidence_and_skeleton(tmp_path, monkeypatch):
    """review 1 M3: tuning regress.CONSTANTS or the confidence model must make stored results stale."""
    from analysis.pose24 import confidence, regress
    _, cache, _ = cs_syn.full_cache(tmp_path, F=10)
    base = {m: oracle.expected_meta(cache, m, None, mo) for m, mo in (("fit", None),
                                                                     ("mono-len-oracle", {"lengths": "oracle"}))}
    for m, mo in (("fit", None), ("mono-len-oracle", {"lengths": "oracle"})):
        assert oracle.expected_meta(cache, m, None, mo, skeleton="smpl-24") != base[m]
    monkeypatch.setitem(regress.CONSTANTS, "hjc_medial", 0.71)
    assert all(oracle.expected_meta(cache, m, None, mo) != base[m]
               for m, mo in (("fit", None), ("mono-len-oracle", {"lengths": "oracle"})))
    monkeypatch.undo()
    monkeypatch.setattr(confidence, "CONF_SCALE_MM", 41.0)
    assert oracle.expected_meta(cache, "fit", None, None) != base["fit"]
    monkeypatch.undo()
    assert oracle.expected_meta(cache, "fit", None, None) == base["fit"]


def test_never_seen_knee_is_covered_by_the_pelvis(tmp_path):
    """R16: a subset that never measures one knee keeps its pelvis: coverage stays, full_body 0, row scored."""
    from analysis.pose24.camsubsets import score
    _, cache, _ = cs_syn.full_cache(tmp_path, F=40)
    for v in cache["views"]:
        cache["obs"]["P1"][v][:, 13, 2] = 0.0                          # left_knee never seen by any view
    res = {k: oracle.fit_subset(cache, list(k.split("+")), fit_opts={"passes": 1}) for k in ("a+c", "a+b+c")}
    p1 = res["a+c"]["players"]["P1"]
    assert p1["coverage"] > 0.9 and p1["full_body"] == 0.0 and 0 < p1["measured"] <= p1["coverage"]
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    r = next(r for r in score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
             if r["player"] == "P1")
    assert r["player_seen"] is True and r["pelvis_xz_cm_median"] is not None and r["full_body"] == 0.0


def test_results_record_cpu_seconds_workers_and_no_mono_stature(tmp_path):
    path, cache, _ = cs_syn.full_cache(tmp_path, F=20)
    res = oracle.fit_subset(cache, ["a", "c"], fit_opts={"passes": 1})
    assert res["cpu_s"] > 0 and res["players"]["P1"]["stature_m"] is not None
    mono = oracle.fit_subset(cache, ["c"], method="mono-len-oracle", mono_opts={"lengths": "oracle"})
    assert all(p["stature_m"] is None for p in mono["players"].values())    # mono lengths are an input (M3)
    got = oracle.run_fits("synfull", [("fit", ("a", "c"))], path, str(tmp_path / "s"), workers=2,
                          fit_opts={"passes": 1}, log=lambda *a: None)
    m = oracle.load_result(got["a+c"], None)["meta"]
    assert m["workers"] == 2 and m["cpu_s"] > 0 and "full_body" in m["players"]["P1"]


def test_oracle_j_is_nan_where_the_doc_would_write_conf_zero(tmp_path, monkeypatch):
    """review 1 L3: e2e reads J from the doc (NaN where the 5-decimal conf is 0); the oracle must drop the same
    joints, else identical fits give different coverage / W-MPJPE."""
    from analysis.pose24 import confidence
    _, cache, _ = cs_syn.full_cache(tmp_path, F=30)
    sig = oracle.fit_subset(cache, ["a", "b"], fit_opts={"passes": 1})["players"]["P1"]["sigma"]
    # scale chosen so the joints above the median sigma get a conf that rounds to 0 (exp(-12.3) < 5e-6)
    monkeypatch.setattr(confidence, "CONF_SCALE_MM", float(np.nanmedian(sig)) / 12.3)
    res = oracle.fit_subset(cache, ["a", "b"], fit_opts={"passes": 1})
    p1 = res["players"]["P1"]
    zero = np.round(p1["C"], 5) == 0
    assert zero.any() and (~zero).any()
    assert np.isnan(p1["J"][zero]).all()
    assert np.isfinite(p1["J"][~zero]).all()
