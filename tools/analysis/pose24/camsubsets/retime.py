"""Re-time an end-to-end subset build (own sync) onto the full build's timeline (spec § 8; SP1 F5, SP2 F7)."""
from __future__ import annotations

import numpy as np


def frame_shift(offsets_ref, offsets_sub):
    d = [float(offsets_ref[v]) - float(offsets_sub[v]) for v in offsets_ref if v in offsets_sub]
    if not d:
        return None, None
    return float(np.median(d)), float(max(d) - min(d))


def resample(arr, shift, n_out):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return np.full((int(n_out),) + a.shape[1:], np.nan)
    t = np.arange(n_out) + float(shift)
    i0 = np.floor(t).astype(int)
    fr = (t - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    ok = (t >= 0) & (t <= len(a) - 1)
    i0c = np.clip(i0, 0, len(a) - 1)
    i1c = np.clip(i0 + 1, 0, len(a) - 1)
    out = (1.0 - fr) * a[i0c] + fr * np.where(fr > 0, a[i1c], 0.0)
    exact = (fr == 0).reshape(-1)
    out[exact] = a[i0c[exact]]
    out[~ok] = np.nan
    return out
