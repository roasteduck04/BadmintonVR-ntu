"""Fixed-lag L/R swap (spec § 6.3): the offline detector (lrswap.detect_swaps: per-frame consensus cost + Viterbi)
run on the UNSWAPPED observations of [history + window]; only the `commit` frames' decisions are used. Committed
frames never change afterwards. `span_obs` covers [history + window]; `commit` is a slice RELATIVE to the start of
`span_obs` (not the clip), so a span starting at absolute frame 10 with slice(20, 80) commits absolute frames 30..89."""
from __future__ import annotations

from analysis.pose24 import lrswap


def window_swaps(cams, views, span_obs, commit) -> dict:
    b = {"views": list(views), "cams": {v: cams[v] for v in views}, "obs": span_obs}
    return {pid: lrswap.detect_swaps(b, pid)[:, commit] for pid in span_obs}
