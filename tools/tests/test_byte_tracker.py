import numpy as np


def test_bboxes_from_keypoints_basic():
    """Derive [x1, y1, x2, y2, score] bboxes from wholebody keypoints."""
    from byte_tracker import bboxes_from_keypoints

    # 2 detections, 5 keypoints each (simplified)
    keypoints = np.array([
        [[100, 200, 0.9], [150, 180, 0.8], [120, 300, 0.7],
         [90, 250, 0.6], [160, 280, 0.5]],
        [[400, 100, 0.8], [450, 80, 0.7], [420, 200, 0.9],
         [390, 150, 0.6], [460, 180, 0.5]],
    ], dtype=np.float32)
    scores = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.8, 0.7, 0.9, 0.6, 0.5],
    ], dtype=np.float32)

    bboxes = bboxes_from_keypoints(keypoints, scores, pad_ratio=0.1)

    assert bboxes.shape == (2, 5)  # [x1, y1, x2, y2, score]
    # First detection: x range [90, 160], y range [180, 300], padded 10%
    assert bboxes[0, 0] < 90   # x1 padded left
    assert bboxes[0, 2] > 160  # x2 padded right
    assert bboxes[0, 1] < 180  # y1 padded up
    assert bboxes[0, 3] > 300  # y2 padded down
    assert 0.0 < bboxes[0, 4] <= 1.0  # mean score


def test_bboxes_from_keypoints_filters_low_score():
    """Keypoints with score < 0.1 are excluded from bbox computation."""
    from byte_tracker import bboxes_from_keypoints

    keypoints = np.array([
        [[100, 200, 0.9], [500, 500, 0.05], [120, 250, 0.8]],
    ], dtype=np.float32)
    scores = np.array([[0.9, 0.05, 0.8]], dtype=np.float32)

    bboxes = bboxes_from_keypoints(keypoints, scores, pad_ratio=0.0)

    # [500, 500] should be excluded (score < 0.1)
    assert bboxes[0, 2] < 200  # x2 should NOT extend to 500


def test_bboxes_from_keypoints_empty():
    """No detections returns empty array."""
    from byte_tracker import bboxes_from_keypoints

    keypoints = np.zeros((0, 5, 3), dtype=np.float32)
    scores = np.zeros((0, 5), dtype=np.float32)
    bboxes = bboxes_from_keypoints(keypoints, scores)
    assert bboxes.shape == (0, 5)


def test_byte_tracker_assigns_consistent_ids():
    """Same detections across frames keep the same track IDs."""
    from byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=5, match_thresh=0.3)

    # Frame 1: two persons
    bboxes1 = np.array([
        [100, 100, 200, 300, 0.9],
        [400, 100, 500, 300, 0.8],
    ], dtype=np.float32)
    pairs1 = tracker.update(bboxes1)
    assert len(pairs1) == 2
    id_left = [tid for tid, idx in pairs1 if idx == 0][0]
    id_right = [tid for tid, idx in pairs1 if idx == 1][0]
    assert id_left != id_right

    # Frame 2: same positions, slight shift
    bboxes2 = np.array([
        [105, 102, 205, 302, 0.9],
        [402, 98, 502, 298, 0.8],
    ], dtype=np.float32)
    pairs2 = tracker.update(bboxes2)
    id_left2 = [tid for tid, idx in pairs2 if idx == 0][0]
    id_right2 = [tid for tid, idx in pairs2 if idx == 1][0]
    assert id_left2 == id_left
    assert id_right2 == id_right


def test_byte_tracker_reassociates_after_gap():
    """Track survives a few frames of missing detection."""
    from byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=10, match_thresh=0.3)

    bbox = np.array([[100, 100, 200, 300, 0.9]], dtype=np.float32)
    pairs = tracker.update(bbox)
    original_id = pairs[0][0]

    # 5 frames with no detections
    for _ in range(5):
        tracker.update(np.zeros((0, 5), dtype=np.float32))

    # Same bbox returns — should get same ID (within buffer)
    pairs2 = tracker.update(bbox)
    assert len(pairs2) == 1
    assert pairs2[0][0] == original_id


def test_raw_simcc_scores_are_calibrated_before_gating():
    """RTMPose scores are raw SimCC (0-10, median ~4.3): gates of 0.1 / 0.5 were no-ops (review 4 L13)."""
    from byte_tracker import bboxes_from_keypoints, calibrated_scores
    from analysis.pose_v2 import norm_score
    raw = np.array([[4.3, 1.0, 5.0]], dtype=np.float32)
    assert np.allclose(calibrated_scores(raw), norm_score(raw))                 # same logistic as pose_v2
    keypoints = np.array([[[100, 200, 0], [500, 500, 0], [120, 250, 0]]], dtype=np.float32)
    b = bboxes_from_keypoints(keypoints, raw, pad_ratio=0.0)
    assert b[0, 2] < 200                                    # raw 1.0 (p ~ 0.003) no longer stretches the box
    assert 0.9 < b[0, 4] <= 1.0                             # box score is a probability, comparable to track_thresh
    weak = bboxes_from_keypoints(keypoints, np.array([[2.5, 2.4, 2.6]], dtype=np.float32), pad_ratio=0.0)
    assert weak[0, 4] < 0.5                                 # a weak detection no longer starts a track at 0.5


def test_probability_scores_pass_through_unchanged():
    from byte_tracker import calibrated_scores
    p = np.array([[0.9, 0.05, 0.8]], dtype=np.float32)
    assert np.array_equal(calibrated_scores(p), p)
    assert np.array_equal(calibrated_scores(p, raw=False), p)
    assert calibrated_scores(p, raw=True).max() < 0.01
