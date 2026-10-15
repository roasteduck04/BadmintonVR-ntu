"""camsubsets subset enumeration and held-out bookkeeping."""
from math import comb

import pytest

from analysis.pose24.camsubsets import subsets as cs

V5 = ["back_051", "backleft_051", "backright_051", "frontleft_051", "frontright_051"]


def test_counts_per_test_view_and_clip():
    rows = [r for r in cs.study_rows(V5) if r["test_view"] is not None]
    per_t = {t: [r for r in rows if r["test_view"] == t] for t in V5}
    assert all(len(v) == 15 for v in per_t.values())                  # 4 + 6 + 4 + 1
    assert len(rows) == 75
    rows4 = [r for r in cs.study_rows(V5[:4]) if r["test_view"] is not None]
    assert len(rows4) == 4 * (comb(3, 1) + comb(3, 2) + comb(3, 3))  # 7 per t


def test_test_view_never_in_candidate_or_reference():
    for r in cs.study_rows(V5):
        if r["test_view"] is not None:
            assert r["test_view"] not in r["subset"] and r["test_view"] not in r["reference"]
            assert set(r["subset"]) <= set(r["reference"])


def test_deployment_rows_reference_is_full():
    dep = [r for r in cs.study_rows(V5) if r["test_view"] is None]
    assert all(r["reference"] == tuple(V5) for r in dep)
    assert len(dep) == comb(5, 1) + comb(5, 2) + comb(5, 3) + comb(5, 4)


def test_distinct_fits_are_25_plus_full():
    fits = cs.distinct_fits(cs.study_rows(V5))
    assert len(fits) == 26 and tuple(V5) in fits and all(len(f) >= 2 for f in fits)
    assert cs.mono_views(cs.study_rows(V5)) == V5


def test_excluded_views_never_enter_rows():
    v = ["back_21", "backleft_21", "backright_21", "frontleft_21", "frontright_21"]
    rows = cs.study_rows(v, excluded=["frontright_21"])
    assert all("frontright_21" not in r["subset"] + r["reference"] and r["test_view"] != "frontright_21"
               for r in rows)
    assert len(cs.distinct_fits(rows)) == 11                          # C(4,2)+C(4,3)+full


def test_sizes_filter_and_keys():
    rows = cs.study_rows(V5, sizes=(2,))
    assert {r["n_cams"] for r in rows if r["subset"] != r["reference"]} == {2}
    assert cs.key(("frontright_051", "backleft_051")) == "backleft_051+frontright_051"
    assert cs.parse_key("backleft_051+frontright_051") == ("backleft_051", "frontright_051")
    assert [cs.abbr(x) for x in V5] == ["b", "bl", "br", "fl", "fr"]
    with pytest.raises(ValueError):
        cs.abbr("side_cam")
