"""Fallback path (spec § 6.5): causal per-keypoint EMA of the raw robust triangulation (Y-down court world).
A keypoint missing for > max_hold steps is dropped; a new observation restarts it (no smoothing across gaps)."""
from __future__ import annotations

import numpy as np


class EmaSmoother:
    def __init__(self, alpha, max_hold):
        self.alpha, self.max_hold = float(alpha), int(max_hold)
        self.x = None
        self.miss = None

    def push(self, X):
        X = np.asarray(X, float)
        if self.x is None:
            self.x = np.full(X.shape, np.nan)
            self.miss = np.zeros(X.shape[0], int)
        seen = np.isfinite(X).all(-1)
        have = np.isfinite(self.x).all(-1)
        upd = seen & have
        self.x[upd] = self.alpha * X[upd] + (1 - self.alpha) * self.x[upd]
        new = seen & ~have
        self.x[new] = X[new]
        self.miss[seen] = 0
        self.miss[~seen] += 1
        self.x[self.miss > self.max_hold] = np.nan
        return self.x.copy()
