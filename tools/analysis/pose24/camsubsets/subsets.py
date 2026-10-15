"""Camera subsets for the SP5 study (spec § 3): held-out rows per test view, deployment rows, distinct fits."""
from __future__ import annotations

from itertools import combinations

_ABBR = (("backleft", "bl"), ("backright", "br"), ("frontleft", "fl"), ("frontright", "fr"), ("back", "b"))


def key(subset) -> str:
    return "+".join(sorted(subset))


def parse_key(s: str) -> tuple:
    return tuple(sorted(s.split("+")))


def abbr(view: str) -> str:
    stem = view.rsplit("_", 1)[0]
    for name, a in _ABBR:
        if stem == name:
            return a
    raise ValueError(f"no abbreviation for view {view!r}")


def _subsets(views, sizes, max_size):
    return [tuple(c) for k in sorted(set(sizes)) if 1 <= k <= max_size for c in combinations(sorted(views), k)]


def study_rows(views, sizes=(1, 2, 3, 4), test_views=None, excluded=()) -> list:
    kept = sorted(v for v in views if v not in set(excluded))
    tests = kept if test_views is None else [t for t in sorted(test_views) if t in kept]
    rows = []
    for t in tests:
        ref = tuple(v for v in kept if v != t)
        for s in _subsets(ref, sizes, len(ref)):
            rows.append({"test_view": t, "subset": s, "reference": ref, "n_cams": len(s)})
        if len(ref) not in sizes:          # the reference is always a candidate (the LOVO row)
            rows.append({"test_view": t, "subset": ref, "reference": ref, "n_cams": len(ref)})
    for s in _subsets(kept, sizes, len(kept) - 1):
        rows.append({"test_view": None, "subset": s, "reference": tuple(kept), "n_cams": len(s)})
    return rows


def distinct_fits(rows, min_views=2) -> list:
    out = {r["subset"] for r in rows if len(r["subset"]) >= min_views} | {r["reference"] for r in rows}
    return sorted(out, key=lambda s: (len(s), s))


def mono_views(rows) -> list:
    return sorted({r["subset"][0] for r in rows if len(r["subset"]) == 1})
