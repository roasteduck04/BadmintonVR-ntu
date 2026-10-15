"""camsubsets array metrics on constructed poses."""
import numpy as np
import pytest

import camsubsets_synth as cs_syn
import pose24_synth as syn
from analysis import skeleton24 as sk
from analysis.pose24 import eval as ev
from analysis.pose24.camsubsets import metrics as m

N = sk.SMPL28_NAMES


def _walk(F=30):
    J = sk.REST28[None].repeat(F, 0).copy()
    J[:, :, 0] += np.linspace(0, 1, F)[:, None]
    return J


def test_vs_ref_zero_and_angles():
    J = _walk()
    r = m.vs_ref(J, J.copy(), N)
    assert r["w_mpjpe_cm"]["median"] < 1e-9
    assert all(v["mae"] < 1e-6 for v in r["angle_mae_deg"].values())


def test_lr_swap_frames_detects_relabelled_frames():
    J = _walk(20)
    S = J.copy()
    for a in range(len(N)):
        if N[a].startswith("left_"):
            b = N.index("right_" + N[a][5:])
            S[5:9, [a, b]] = S[5:9, [b, a]]
    assert m.lr_swap_frames(S, J, N) == 4
    assert m.lr_swap_frames(J, J, N) == 0


def test_ray_angles_opposite_corners_near_180():
    cams = syn.rig(["a", "d"])                 # a (0,-3,-11) vs d (5,-2.5,11): roughly opposite ends
    pel = np.tile([[0.0, -0.9, 0.0]], (10, 1))
    ang = m.ray_angles_deg(cams, ["a", "d"], pel)
    assert ang["a+d"] > 140


def test_used_frac_counts_body_inliers():
    inl = np.zeros((10, 35, 2), bool)
    inl[:4, 5, 0] = True
    inl[:, 10, 1] = True                        # keypoint 10 = right wrist (body)
    assert m.used_frac(inl, ["a", "b"]) == {"a": 0.4, "b": 1.0}


def test_swing_windows_mark_fast_wrist_frames():
    J = sk.REST28[None].repeat(120, 0).copy()
    rw = N.index("right_wrist")
    J[60:64, rw, 1] += np.array([0.0, 0.3, 0.6, 0.9])            # a fast swing at f60-63
    w = m.swing_windows(J, 60.0, N, pad=5)
    assert w[60] and w[56] and not w[10] and not w[100]


def test_swing_windows_degenerate_inputs_return_false():
    J = np.full((50, len(N), 3), np.nan)
    assert not m.swing_windows(J, 60.0, N).any()
    still = sk.REST28[None].repeat(50, 0).copy()
    assert not m.swing_windows(still, 60.0, N).any()                 # zero speed: nothing above the percentile
    w = m.swing_windows(still, 60.0, N, rallies=[{"start": 40, "end": 500}, {"start": 900, "end": 950}])
    assert w.shape == (50,) and not w.any()


def test_compare_joints_mask_shorter_or_longer_than_arrays():
    J = _walk(30)
    r = ev.compare_joints(J, J[:25], N, mask=np.ones(20, bool))
    assert r["w_mpjpe_cm"]["n"] == 20
    r = ev.compare_joints(J[:20], J, N, mask=np.ones(30, bool))
    assert r["w_mpjpe_cm"]["n"] == 20


def test_vs_ref_swing_block():
    J = _walk(30)
    sw = np.zeros(30, bool)
    sw[10:15] = True
    r = m.vs_ref(J, J.copy(), N, swing=sw)
    assert all(v["mae"] < 1e-6 for v in r["swing"].values())
    assert "swing" not in m.vs_ref(J, J.copy(), N, swing=np.zeros(30, bool))


def test_identity_proxy_counts_wrong_frames_and_switches():
    ref = {"P1": np.tile([[0.0, 0, 0]], (20, 1)), "P2": np.tile([[3.0, 0, 0]], (20, 1))}
    sub = ref["P1"].copy()
    sub[5:8] = ref["P2"][5:8]
    sub[12:14] = ref["P2"][12:14]
    r = m.identity_proxy(sub, ref, "P1")
    assert r == {"wrong_frames": 5, "switches": 2}


def test_quality_keys_and_constant_limbs():
    J = _walk(40)
    C = np.ones(J.shape[:2])
    r = m.quality(J, C, np.full(J.shape[:2], 12.0), 60.0, N)
    assert r["limb_cv"] < 1e-9 and r["mean_sigma_mm"] == pytest.approx(12.0)
    assert {"jerk", "bone_cv"} <= set(r)


def test_heldout_px_near_zero_for_truth(tmp_path):
    from analysis.pose24 import kpset
    _, cache, players = cs_syn.full_cache(tmp_path)
    X35 = kpset.from133(players["P1"])
    r = m.heldout_px(X35, cache, "P1", "d")
    assert r["px"]["median"] < 3.0
    assert m.heldout_px(X35, cache, "P1", "zz") is None


def _det_masked(cache, pid, view, frames):
    c = dict(cache, extra={q: dict(e) for q, e in cache["extra"].items()})
    d = np.array(c["extra"][pid]["det_index"])
    d[cache["views"].index(view), frames] = -1
    c["extra"][pid]["det_index"] = d
    return c


def test_heldout_px_only_on_frames_with_a_reference_detection(tmp_path):
    from analysis.pose24 import kpset
    _, cache, players = cs_syn.full_cache(tmp_path)
    X35 = kpset.from133(players["P1"])
    F = cache["frame_count"]
    full = m.heldout_px(X35, cache, "P1", "d")
    assert full["px"]["n"] >= m.HELDOUT_MIN_SAMPLES
    none = _det_masked(cache, "P1", "d", slice(None))              # no detection in d on any frame
    assert m.heldout_px(X35, none, "P1", "d") is None and m.bbox_height_px(none, "P1", "d") is None
    half = _det_masked(cache, "P1", "d", slice(0, F // 2))
    h = m.heldout_px(X35, half, "P1", "d")
    assert h["px"]["n"] == full["px"]["n"] // 2
    r, bbox = ev.heldout_errors(X35[:, ev.LIMB_FIT], cache, "P1", "d")   # the kept half only
    assert h["px"]["median"] == pytest.approx(ev._row([(r[F // 2:], bbox[F // 2:])])["px"]["median"])
    assert m.bbox_height_px(half, "P1", "d") == pytest.approx(float(np.nanmedian(bbox[F // 2:])))
    few = _det_masked(cache, "P1", "d", slice(10, None))          # 10 frames x 12 keypoints = 120 < 200
    assert m.heldout_px(X35, few, "P1", "d") is None
    nodet = dict(cache, extra={q: {k: v for k, v in e.items() if k != "det_index"} for q, e in cache["extra"].items()})
    assert m.heldout_px(X35, nodet, "P1", "d")["px"]["n"] == full["px"]["n"]     # no det_index: old behaviour


def test_bbox_height_px_matches_heldout_bbox(tmp_path):
    _, cache, _ = cs_syn.full_cache(tmp_path)
    h = m.bbox_height_px(cache, "P1", "d")
    _, bbox = ev.heldout_errors(np.zeros((cache["frame_count"], len(ev.LIMB_FIT), 3)), cache, "P1", "d")
    assert h is not None and h > 10 and h == pytest.approx(float(np.nanmedian(bbox)))
    assert m.bbox_height_px(cache, "P2", "a") is None                  # P2 unseen by a
