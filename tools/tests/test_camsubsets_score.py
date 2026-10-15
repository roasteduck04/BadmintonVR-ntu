"""camsubsets scoring: long rows from stored subset results."""
import json

import numpy as np

import camsubsets_synth as cs_syn
from analysis.pose24.camsubsets import oracle, score
from analysis.pose24.camsubsets import subsets as cs


def _results(tmp_path, cache_path):
    rows = cs.study_rows(cs_syn.VIEWS, sizes=(2, 3))
    fits = cs.distinct_fits(rows)
    got = oracle.run_fits("synfull", [("fit", f) for f in fits], cache_path, str(tmp_path / "s"), workers=1,
                          fit_opts={"passes": 1}, log=lambda *a: None)
    return rows, {k: oracle.load_result(p, None) for k, p in got.items()}


def test_rows_have_all_columns_and_heldout_scored_on_t(tmp_path):
    path, cache, _ = cs_syn.full_cache(tmp_path)
    rows, res = _results(tmp_path, path)
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
    assert out and all(list(r) == score.ROW_KEYS for r in out)
    r = next(r for r in out if r["test_view"] == "d" and r["subset"] == "a+c" and r["player"] == "P1")
    assert r["heldout_px_median"] < 15 and r["w_mpjpe_cm_median"] < 10 and r["n_cams"] == 2
    assert score.ROW_KEYS.index("heldout_n") == score.ROW_KEYS.index("heldout_px_p90") + 1
    assert isinstance(r["heldout_n"], int) and r["heldout_n"] >= 200
    dep = [r for r in out if r["test_view"] == "" and r["player"] == "P1"]
    assert dep and all(r["heldout_px_median"] is None for r in dep)   # deployment rows: 3D vs full only


def test_unseen_player_row_is_flagged_not_zero(tmp_path):
    path, cache, _ = cs_syn.full_cache(tmp_path)
    rows, res = _results(tmp_path, path)
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
    r = next(r for r in out if r["subset"] == "a+b" and r["player"] == "P2")
    assert r["player_seen"] is False and r["coverage"] == 0.0
    assert r["w_mpjpe_cm_median"] is None and r["heldout_px_median"] is None and r["knee_mae_deg"] is None


def _small(tmp_path, F=40):
    path, cache, _ = cs_syn.full_cache(tmp_path, F=F)
    got = oracle.run_fits("synfull", [("fit", ("a", "c")), ("fit", ("a", "b", "c"))], path, str(tmp_path / "s"),
                          workers=1, fit_opts={"passes": 1}, log=lambda *a: None)
    return cache, {k: oracle.load_result(p, None) for k, p in got.items()}


def test_one_view_subset_row_has_no_ray_angles(tmp_path):
    cache, res = _small(tmp_path)
    two = res["a+c"]
    mono = dict(two, views=["c"], method="mono-len-oracle",
                players={pid: dict(p, inlier=p["inlier"][..., 1:], lr_swapped=p["lr_swapped"][:, 1:])
                         for pid, p in two["players"].items()})
    res["c"] = mono
    rows = [{"test_view": "d", "subset": ("c",), "reference": ("a", "b", "c"), "n_cams": 1}]
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "mono-len-oracle", "oracle")
    r = next(r for r in out if r["player"] == "P1")
    assert r["player_seen"] is True and r["n_cams"] == 1
    assert r["ray_angle_min_deg"] is None and r["ray_angle_max_deg"] is None
    assert json.loads(r["used_frac"]).keys() == {"c"}
    r2 = score.score_rows("synfull", [dict(rows[0], subset=("a", "c"), n_cams=2)], res, cache, 60.0, "singles",
                          "fit", "oracle")[0]
    assert 0 < r2["ray_angle_min_deg"] == r2["ray_angle_max_deg"] < 180


def test_e2e_meta_match_identity_and_shift_carried(tmp_path):
    cache, res = _small(tmp_path)
    sub = res["a+c"]
    res["a+c"] = dict(sub, players={"P9": sub["players"]["P1"], "P2": sub["players"]["P2"]})
    meta = {"a+c": {"shift": 2, "spread": 0.5, "match": {"P9": "P1", "P2": "P2"},
                    "identity": {"P1": {"wrong_frames": 3, "switches": 1}}}}
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "e2e", e2e_meta=meta,
                           log=lambda *a: None)
    r = next(r for r in out if r["player"] == "P1")
    assert r["matched"] is True and r["player_seen"] is True and r["w_mpjpe_cm_median"] < 10
    assert r["wrong_person_frames"] == 3 and r["id_switches"] == 1
    assert r["sync_shift_f"] == 2 and r["sync_spread_f"] == 0.5
    assert "wrong_frames" not in r and list(r) == score.ROW_KEYS


def test_missing_result_row_skipped_with_log(tmp_path):
    cache, res = _small(tmp_path, F=30)
    logs = []
    rows = [{"test_view": "d", "subset": ("b", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    assert score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle", log=logs.append) == []
    assert logs and "b+c" in logs[0]


_METRIC_COLS = [c for c in score.ROW_KEYS if c.endswith(("_median", "_p90", "_deg")) or c in (
    "lr_swap_frames", "jerk", "limb_cv", "skate", "mean_sigma_mm", "stature_m", "used_frac", "bbox_h_test_px",
    "measured", "heldout_n")]


def _all_unmatched(out):
    assert out and all(r["matched"] is False and r["player_seen"] is False and r["coverage"] == 0.0 for r in out)
    assert all(r[c] is None for r in out for c in _METRIC_COLS)


def test_e2e_empty_match_means_unmatched_not_identity(tmp_path):
    cache, res = _small(tmp_path, F=30)
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "e2e",
                           e2e_meta={"a+c": {"shift": 0, "spread": 0.0, "match": {}, "identity": {}}},
                           log=lambda *a: None)
    _all_unmatched(out)


def test_e2e_key_missing_from_meta_is_unmatched_and_logged(tmp_path):
    cache, res = _small(tmp_path, F=30)
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    logs = []
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "e2e", e2e_meta={}, log=logs.append)
    _all_unmatched(out)
    assert any("a+c" in m and "unmatched" in m for m in logs)


def _knee_never_seen(res, pid="P1"):
    """A stored result whose subset never measured one knee (regress NaNs it for the whole clip), with the old
    all-joints stored scalars (coverage / measured 0) that scoring must not trust (R16)."""
    from analysis import skeleton24 as sk
    p = dict(res["players"][pid])
    J = p["J"].copy()
    J[:, list(sk.SMPL28_NAMES).index("left_knee")] = np.nan
    p.update(J=J, coverage=0.0, measured=0.0)
    return dict(res, players=dict(res["players"], **{pid: p}))


def test_extrapolated_pelvis_is_not_seen(tmp_path):
    """R19: a finite pelvis with (almost) no frame of >= 2 inlier views (one-view extrapolation) is unseen."""
    cache, res = _small(tmp_path)
    sub = res["a+c"]
    p = dict(sub["players"]["P1"])
    inl = p["inlier"].copy()
    inl[..., 1] = False                                                 # only view a ever inlier
    p["inlier"] = inl
    res["a+c"] = dict(sub, players=dict(sub["players"], P1=p))
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    r = next(r for r in score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
             if r["player"] == "P1")
    assert r["player_seen"] is False and r["coverage"] > 0.9 and r["measured"] == 0.0
    assert r["pelvis_xz_cm_median"] is None and r["w_mpjpe_cm_median"] is None


def test_never_seen_knee_keeps_player_seen_with_pelvis_metrics(tmp_path):
    cache, res = _small(tmp_path)
    res["a+c"] = _knee_never_seen(res["a+c"])
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    r = next(r for r in score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
             if r["player"] == "P1")
    assert r["player_seen"] is True and r["coverage"] > 0.9 and r["full_body"] == 0.0
    assert r["pelvis_xz_cm_median"] is not None and r["pelvis_xz_cm_median"] < 10
    assert 0 < r["measured"] <= r["coverage"] and r["w_mpjpe_cm_median"] is not None
    assert score.ROW_KEYS.index("full_body") == score.ROW_KEYS.index("measured") + 1


def test_reference_row_is_flagged_and_not_self_scored(tmp_path):
    path, cache, _ = cs_syn.full_cache(tmp_path)
    rows, res = _results(tmp_path, path)
    out = score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle")
    assert score.ROW_KEYS.index("is_reference") == score.ROW_KEYS.index("ref_cams") + 1
    r = next(r for r in out if r["test_view"] == "d" and r["subset"] == "a+b+c" and r["player"] == "P1")
    assert r["is_reference"] is True and r["player_seen"] is True
    vs = [c for c in score.ROW_KEYS if c.startswith(("w_mpjpe", "root_mpjpe", "pa_mpjpe", "pelvis_"))
          or c.endswith("_mae_deg")] + ["lr_swap_frames"]
    assert all(r[c] is None for c in vs)
    assert r["heldout_px_median"] is not None and r["coverage"] > 0.9 and r["jerk"] is not None
    other = next(r for r in out if r["test_view"] == "d" and r["subset"] == "a+c" and r["player"] == "P1")
    assert other["is_reference"] is False and other["w_mpjpe_cm_median"] is not None


def test_reference_not_seeing_the_player_makes_the_row_unseen(tmp_path):
    cache, res = _small(tmp_path)
    ref = res["a+b+c"]
    p = dict(ref["players"]["P1"], J=np.full_like(ref["players"]["P1"]["J"], np.nan))
    res["a+b+c"] = dict(ref, players=dict(ref["players"], P1=p))
    rows = [{"test_view": "d", "subset": ("a", "c"), "reference": ("a", "b", "c"), "n_cams": 2}]
    logs = []
    r = next(r for r in score.score_rows("synfull", rows, res, cache, 60.0, "singles", "fit", "oracle",
                                         log=logs.append) if r["player"] == "P1")
    assert r["player_seen"] is False and r["matched"] is True and r["coverage"] > 0.9
    assert r["w_mpjpe_cm_median"] is None and r["heldout_px_median"] is None
    assert any("P1" in m and "reference" in m for m in logs)


def test_mono_row_has_no_stature_and_one_view_measured(tmp_path):
    cache, res = _small(tmp_path)
    two = res["a+c"]
    res["c"] = dict(two, views=["c"], method="mono-len-oracle",
                    players={pid: dict(p, inlier=p["inlier"][..., 1:], lr_swapped=p["lr_swapped"][:, 1:],
                                       stature_m=1.7) for pid, p in two["players"].items()})
    rows = [{"test_view": "d", "subset": ("c",), "reference": ("a", "b", "c"), "n_cams": 1}]
    r = next(r for r in score.score_rows("synfull", rows, res, cache, 60.0, "singles", "mono-len-oracle", "oracle")
             if r["player"] == "P1")
    assert r["stature_m"] is None and r["measured"] > 0.5                 # measured: >= 1 inlier view (mono)
