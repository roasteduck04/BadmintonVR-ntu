"""ByteTrack integration for multi-person pose tracking.

Provides bboxes_from_keypoints() to derive bounding boxes from RTMPose
keypoints, and wraps BYTETracker for frame-by-frame tracking.

If the `bytetracker` pip package is available, it is used directly.
Otherwise a small vendored IOU+Hungarian tracker (MIT-style, ported from
the ByteTrack reference implementation's ideas) is used as a fallback so
this module has no hard external dependency beyond numpy/scipy.
"""
from __future__ import annotations

import numpy as np

# RTMPose wholebody scores are raw SimCC values (0-10, median ~4.3), not probabilities, so gates such as
# score_min=0.1 / track_thresh=0.5 passed everything. They are mapped with the same logistic as
# analysis.pose_v2.norm_score (2.5 -> 0.3, 3.0 -> 0.7; kept in sync by test_byte_tracker) before any gate.
# This module stays standalone (GPU boxes run it without tools/analysis), hence the copied constants.
SCORE_MID = 2.75
SCORE_WIDTH = 0.30


def calibrated_scores(scores, raw=None):
    """Per-keypoint scores as probabilities. raw=None: raw SimCC if any score > 1, else already probabilities."""
    s = np.asarray(scores, dtype=np.float64)
    if raw is None:
        raw = bool(s.size) and float(np.nanmax(s)) > 1.0
    if not raw:
        return np.asarray(scores)
    return 1.0 / (1.0 + np.exp(-(s - SCORE_MID) / SCORE_WIDTH))


def bboxes_from_keypoints(keypoints, scores, pad_ratio=0.1, score_min=0.1, raw=None):
    """Derive bounding boxes from keypoint arrays.

    Args:
        keypoints: (N, K, 3) array of [x, y, score] per keypoint (score
            channel in keypoints is ignored — `scores` is authoritative)
        scores: (N, K) array of per-keypoint scores (raw RTMPose SimCC or probabilities, see `raw`)
        pad_ratio: fractional padding on each side
        score_min: minimum keypoint probability to include
        raw: True = raw SimCC scores, False = probabilities, None = detect (any score > 1 -> raw)

    Returns:
        (N, 5) array of [x1, y1, x2, y2, mean keypoint probability]
    """
    n = keypoints.shape[0]
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)
    scores = calibrated_scores(scores, raw)

    bboxes = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        mask = scores[i] >= score_min
        if not mask.any():
            mask = np.ones(scores.shape[1], dtype=bool)
        pts = keypoints[i, mask, :2]
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        w, h = x2 - x1, y2 - y1
        x1 -= w * pad_ratio
        y1 -= h * pad_ratio
        x2 += w * pad_ratio
        y2 += h * pad_ratio
        bboxes[i] = [x1, y1, x2, y2, float(scores[i, mask].mean())]
    return bboxes


def _iou(a, b):
    """Compute IOU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ByteTrack wrapper — try pip package first, fall back to vendored.
# NOTE: the pip branch is untested (package not installed in any environment used so far; its update() /
# t.tlbr / t.track_id API is unverified). All produced kp2d files so far came from the vendored fallback.
try:
    from bytetracker import BYTETracker as _BYTETracker

    class ByteTracker:
        """Thin wrapper around the `bytetracker` pip package that returns
        (track_id, detection_index) pairs for a given frame's bboxes."""

        def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
            self._tracker = _BYTETracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )

        def update(self, bboxes):
            """Update tracker with (N, 5) bboxes [x1,y1,x2,y2,score].
            Returns list of (track_id, det_index) pairs."""
            if len(bboxes) == 0:
                return []
            tracks = self._tracker.update(bboxes)
            # Map tracks back to detection indices by bbox overlap
            return self._match_tracks_to_dets(tracks, bboxes)

        def _match_tracks_to_dets(self, tracks, bboxes):
            pairs = []
            for t in tracks:
                tlbr = t.tlbr
                best_idx, best_iou = -1, 0.0
                for j in range(len(bboxes)):
                    iou = _iou(tlbr, bboxes[j, :4])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_idx >= 0:
                    pairs.append((int(t.track_id), best_idx))
            return pairs

except ImportError:
    # Vendored fallback — minimal IOU/Hungarian tracker as ByteTrack
    # fallback. Install `bytetracker` (pip) for the full Kalman-filter
    # based implementation.
    class ByteTracker:
        """Minimal IOU-based tracker as a ByteTrack fallback.

        Matches existing tracks to new detections via Hungarian assignment
        on IOU cost, ages out tracks that go unmatched for more than
        `track_buffer` frames, and spawns new tracks for high-score
        unmatched detections.
        """

        def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
            self.track_thresh = track_thresh
            self.track_buffer = track_buffer
            self.match_thresh = match_thresh
            self._tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'lost': int}
            self._next_id = 1

        def update(self, bboxes):
            if len(bboxes) == 0:
                for tid in list(self._tracks):
                    self._tracks[tid]['lost'] += 1
                    if self._tracks[tid]['lost'] > self.track_buffer:
                        del self._tracks[tid]
                return []

            from scipy.optimize import linear_sum_assignment

            # Build IOU cost matrix between existing tracks and detections
            track_ids = list(self._tracks.keys())
            if track_ids:
                cost = np.zeros((len(track_ids), len(bboxes)), dtype=np.float32)
                for i, tid in enumerate(track_ids):
                    for j in range(len(bboxes)):
                        cost[i, j] = 1.0 - _iou(
                            self._tracks[tid]['bbox'], bboxes[j, :4])

                row_ind, col_ind = linear_sum_assignment(cost)
                matched_dets = set()
                matched_tracks = set()
                pairs = []
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < (1.0 - self.match_thresh + 0.5):
                        tid = track_ids[r]
                        self._tracks[tid]['bbox'] = bboxes[c, :4].copy()
                        self._tracks[tid]['lost'] = 0
                        matched_dets.add(c)
                        matched_tracks.add(tid)
                        pairs.append((tid, c))

                # Remove stale tracks
                for tid in track_ids:
                    if tid not in matched_tracks:
                        self._tracks[tid]['lost'] += 1
                        if self._tracks[tid]['lost'] > self.track_buffer:
                            del self._tracks[tid]
            else:
                matched_dets = set()
                pairs = []

            # Create new tracks for unmatched detections
            for j in range(len(bboxes)):
                if j not in matched_dets and bboxes[j, 4] >= self.track_thresh:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = {
                        'bbox': bboxes[j, :4].copy(), 'lost': 0}
                    pairs.append((tid, j))

            return pairs
