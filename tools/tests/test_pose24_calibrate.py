"""pose24 calibrate-scores."""
import numpy as np
import pytest

from analysis.pose24 import calibrate


def test_fit_logistic_recovers_known_parameters():
    rng = np.random.default_rng(0)
    raw = rng.uniform(1.0, 6.0, 20000)
    y = rng.uniform(size=raw.size) < 1 / (1 + np.exp(-(raw - 3.2) / 0.4))
    mid, width = calibrate.fit_logistic(raw, y)
    assert mid == pytest.approx(3.2, abs=0.05) and width == pytest.approx(0.4, abs=0.04)


def test_reliability_table_bins():
    p = np.linspace(0.01, 0.99, 1000)
    rows = calibrate.reliability_table(p, np.random.default_rng(1).uniform(size=1000) < p)
    assert len(rows) == 10 and all(abs(r["predicted"] - r["observed"]) < 0.2 for r in rows)


def test_frames_with_one_or_two_confident_body_keypoints_give_no_samples():
    """The bbox height needs a body: 1-2 confident body keypoints (bbox ~0 px) must not label samples
    (eval's heldout_errors also requires bbox > 10 px)."""
    import pose24_synth as syn
    from analysis.pose24 import scores
    views = ("a", "b", "c")
    b = syn.bundle({"P1": syn.motion(12, z0=-3.0)}, syn.rig(views), views, noise_px=0.5)
    params = scores.load_params(path=None)
    assert len(calibrate.heldout_samples(b, params, step=1)[0]) > 0
    for v in views:
        D = b["obs"]["P1"][v]
        D[:6, 1:17, 2] = 0.0                 # frames 0-5: only the nose
        D[6:, [0] + list(range(3, 17)), 2] = 0.0   # frames 6-11: only the two eyes (same height)
    raw, y, g = calibrate.heldout_samples(b, params, step=1)
    assert len(raw) == 0


def test_clips_built_with_different_score_params_are_refused(monkeypatch, tmp_path):
    """reliability_before is computed with one set of params: mixing builds made with different ones is an error."""
    from analysis.pose24 import scores
    from analysis.pose24 import doc as docmod
    a = scores.load_params(path=None)
    b = {"groups": dict(a["groups"], body={"mid": 3.1, "width": 0.3}), "source": "x.json"}
    caches = {"c1": {"score_params": a}, "c2": {"score_params": b}}
    monkeypatch.setattr(docmod, "load_debug_cache", lambda p: caches["c1" if "c1" in p else "c2"])
    monkeypatch.setattr(calibrate, "heldout_samples", lambda *a, **k: (np.ones(3), np.ones(3, bool), np.zeros(3, int)))
    with pytest.raises(ValueError, match="score"):
        calibrate.calibrate_scores(["c1", "c2"], str(tmp_path), str(tmp_path / "o.json"), log=lambda *a: None)
