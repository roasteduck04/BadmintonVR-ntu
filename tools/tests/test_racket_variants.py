import numpy as np
import pytest

from analysis import racket_rigidfit as rf
from analysis import racket_variants as rv
from tests.test_racket_rigidfit import _look_at_camera


def test_hand_joint_indices_prefers_hand_then_mcp_then_wrist():
    assert rv.hand_joint_indices(["left_wrist", "right_wrist", "left_hand", "right_hand"]) == (2, 3)
    assert rv.hand_joint_indices(["left_middle1", "x", "right_middle1", "left_wrist", "right_wrist"]) == (0, 2)
    assert rv.hand_joint_indices(["left_wrist", "right_wrist"]) == (0, 1)
    with pytest.raises(ValueError):
        rv.hand_joint_indices(["pelvis"])


def test_player_world_hands_falls_back_to_the_wrist_per_frame():
    """Review 2 M1: a 58-joint build_real doc has the middle1 name but never a confident middle1; hands must come
    from the wrist then (they were all NaN -> NaN ppm -> an empty sidecar)."""
    from analysis import skeleton
    names = list(skeleton.JOINT_NAMES)
    N = {nm: i for i, nm in enumerate(names)}

    def frame(mcp_conf):
        a = np.zeros((58, 4))
        a[N["right_wrist"]] = [0.3, 1.0, 0.1, 0.9]
        a[N["right_middle1"]] = [0.3, 1.0, 0.2, mcp_conf]
        a[N["left_wrist"]] = [-0.3, 1.0, 0.1, 0.9]
        return {"joints_flat": a.ravel().tolist(), "position": [1.0, 2.0]}
    doc = {"skeleton": {"joint_names": names}, "players": [{"id": "P1", "frames": [frame(0.0), frame(0.9)]}]}
    hands, pos = rv.player_world_hands(doc)["P1"]
    assert np.allclose(hands[0, 1], [1.3, 1.0, 2.1]) and np.allclose(hands[0, 0], [0.7, 1.0, 2.1])
    assert np.allclose(hands[1, 1], [1.3, 1.0, 2.2])                     # confident MCP preferred
    assert np.allclose(pos[1], [1.0, 2.0])


def test_load_pose2d_compact_calibrates_raw_simcc_scores(tmp_path):
    """Review 2 L2: pose2d scores are raw SimCC (0-10); the 0.3 gates (MCP vs wrist, person distance, raw-hand
    triangulation) only gate on calibrated probabilities."""
    import json
    from analysis import pose_v2 as pv

    def person(mcp_raw):
        k = [[100.0 + i, 200.0 + i, 4.0] for i in range(133)]
        k[100] = [500.0, 500.0, mcp_raw]                 # left middle MCP
        return {"keypoints": k}
    path = tmp_path / "v_kp2d.json"
    path.write_text(json.dumps({"keypoint_format": "coco_wholebody", "frames": [[person(2.0), person(3.5)]]}))
    body, hands = rv.load_pose2d_compact(str(path), [5, 6])
    assert np.allclose(body[0][:, :, 2], pv.norm_score(4.0))
    assert np.allclose(hands[0][0, 0, :2], [109.0, 209.0])        # MCP raw 2.0 (p 0.07): the wrist
    assert np.allclose(hands[0][1, 0, :2], [500.0, 500.0])        # MCP raw 3.5 (p 0.92): the MCP
    assert hands[0][1, 0, 2] == pytest.approx(pv.norm_score(3.5))


_SYNC_NAMES = ["pelvis", "left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_knee", "right_knee", "left_ankle", "right_ankle"]
_SYNC_REST = {"pelvis": [0, 0.95, 0], "left_hip": [0.1, 0.92, 0], "right_hip": [-0.1, 0.92, 0],
              "left_shoulder": [0.18, 1.42, 0], "right_shoulder": [-0.18, 1.42, 0], "left_elbow": [0.25, 1.15, 0],
              "right_elbow": [-0.25, 1.15, 0], "left_wrist": [0.28, 0.9, 0.05], "right_wrist": [-0.28, 0.9, 0.05],
              "left_knee": [0.1, 0.5, 0.02], "right_knee": [-0.1, 0.5, 0.02], "left_ankle": [0.1, 0.08, 0],
              "right_ankle": [-0.1, 0.08, 0]}


def _sync_body(t):
    """World joints (len(_SYNC_NAMES), 3) of a player running about and swinging both arms at time t."""
    p = np.array([1.5 * np.sin(0.05 * t), 0.0, -2.0 + 1.0 * np.sin(0.033 * t)])
    out = []
    for nm in _SYNC_NAMES:
        x = np.array(_SYNC_REST[nm], float) + p
        if "wrist" in nm or "elbow" in nm:
            x += [0.0, 0.25 * np.sin(0.21 * t), 0.3 * np.cos(0.17 * t)]
        out.append(x)
    return np.array(out)


def test_sidecar_sync_is_on_the_reference_video_timeline_not_the_lagged_body():
    """Review 2 L14: the per-view shift that best fits the (EMA / Kalman lagged) base body absorbs the lag, so a
    'video' sidecar was really on the body timeline. The fractional estimator referenced to the doc's sync view
    recovers the true offsets."""
    lag, true, n = 2, {"a": 0, "b": 3, "c": -2}, 240
    cams = {"a": _look_at_camera([0.0, 2.5, -12.0], [0.0, 1.0, -2.0]),
            "b": _look_at_camera([-7.0, 2.5, -4.0], [0.0, 1.0, -2.0]),
            "c": _look_at_camera([7.0, 2.5, -4.0], [0.0, 1.0, -2.0])}
    frames = []
    for f in range(n):
        W = _sync_body(f - lag)                         # the base body trails the video by `lag` frames
        flat = np.zeros((len(_SYNC_NAMES), 4))
        flat[:, :3] = W - [W[0, 0], 0.0, W[0, 2]]
        flat[:, 3] = 0.9
        frames.append({"joints_flat": flat.ravel().tolist(), "position": [float(W[0, 0]), float(W[0, 2])]})
    doc = {"meta": {"frame_count": n, "cameras": [{"name": v} for v in "cba"]},
           "skeleton": {"joint_names": _SYNC_NAMES}, "players": [{"id": "P1", "frames": frames}]}
    body_idx, body3d = rv.world_body_joints(doc)
    assert 11 in body_idx and 12 in body_idx                  # hips: pose_v2.SYNC_JOINTS needs them
    sel = [_SYNC_NAMES.index(rv._COCO_BODY[c]) for c in body_idx]
    body_proj = {v: {"P1": rv._project_joints(cams[v], body3d["P1"])[0]} for v in cams}
    body_by_view = {}
    for v, cam in cams.items():
        raw = []
        for g in range(n + 10):
            uv, _ = rv._project_joints(cam, _sync_body(g - true[v])[sel])
            raw.append(np.concatenate([uv, np.full((len(sel), 1), 0.9)], axis=1)[None])
        body_by_view[v] = raw
    coarse = rv.estimate_offsets_pose2d(body_by_view, body_proj, n, log=lambda *a: None)
    assert coarse == {v: true[v] - lag for v in true}        # the old, lag-biased sidecar shifts
    frac = rv.refine_offsets_to_ref(doc, cams, body_by_view, body_proj, coarse, "a", log=lambda *a: None)
    assert frac == pytest.approx({v: float(o) for v, o in true.items()}, abs=0.15)
    shifts, sync = rv.reference_shifts(coarse, frac, "a")
    assert shifts == true and sync["reference"] == "a" and sync["body_lag_frames"] == pytest.approx(lag, abs=0.15)
    shifts, sync = rv.reference_shifts(coarse, {}, "a")      # reference unusable: body shifts, said so
    assert shifts == coarse and sync["reference"] is None


def test_sidecar_path():
    assert rv.sidecar_path("a/b/singles_1x.analysis.json", "rigid") == "a/b/singles_1x.racket_rigid.json"


def _track(n=3):
    R = rf.rotation_from_shaft([0.2, 0.9, 0.1], roll=0.4)
    return {"grip": np.tile([1.0, 1.1, 2.0], (n, 1)), "R": np.tile(R, (n, 1, 1)),
            "valid": np.array([True, False, True][:n]), "status": np.array(["detected", "missing", "partial"][:n]),
            "confidence": np.full(n, 0.5), "views": np.array([3, 0, 1][:n]), "err": np.array([0.05, np.nan, 0.08][:n])}


def test_track_to_frames_is_body_local_and_rigid():
    tpl = rf.RacketTemplate()
    pos = np.array([[0.5, 1.5], [0.5, 1.5], [np.nan, np.nan]])
    frames = rv.track_to_frames(_track(), tpl, pos, "right")
    assert frames[1] is None and frames[2] is None
    r = frames[0]
    assert np.allclose(r["grip"], [0.5, 1.1, 0.5])
    axis, head, grip, side = (np.array(r[k]) for k in ("axis", "head", "grip", "side"))
    assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(head - grip), np.linalg.norm(tpl.head_center()), atol=1e-4)
    assert np.isclose(np.linalg.norm(side - head), tpl.head_width / 2, atol=1e-4)
    assert abs(float(np.array(r["face_normal"]) @ axis)) < 1e-4
    assert r["which_hand"] == "right" and r["status"] == "detected" and r["views"] == 3


def test_validate_sidecar_catches_mismatches():
    sc = {"format": rv.SIDECAR_FORMAT, "frame_count": 2,
          "players": [{"id": "P1", "frames": [None, {"grip": [0, 0, 0], "head": [0, 1, 0], "side": [0.1, 1, 0],
                                                    "axis": [0, 1, 0], "face_normal": [0, 0, 1],
                                                    "which_hand": "right"}]}]}
    doc = {"players": [{"id": "P1"}], "meta": {"frame_count": 2}}
    assert rv.validate_sidecar(sc, doc) == []
    doc_bad = {"players": [{"id": "P2"}], "meta": {"frame_count": 3}}
    assert len(rv.validate_sidecar(sc, doc_bad)) == 2
    sc["players"][0]["frames"][1]["which_hand"] = "both"
    assert any("which_hand" in e for e in rv.validate_sidecar(sc))


def test_pose2d_offset_recovers_known_shift():
    cam = _look_at_camera([0.0, 1.6, -9.0], [0.0, 1.2, -2.0])
    n, J, true_shift = 200, 6, 3
    t = np.arange(n + 20)
    base = np.stack([np.stack([0.4 * np.sin(0.15 * t + j), 0.8 + 0.2 * j + 0.1 * np.cos(0.2 * t), -2.0 + 0.3 * np.sin(0.1 * t)], axis=1)
                     for j in range(J)], axis=1)
    body3d = base[:n]
    proj_uv, _ = rv._project_joints(cam, body3d)
    raw = []
    for g in range(n + 20):
        f = g - true_shift
        if 0 <= f < n:
            uv, _ = rv._project_joints(cam, base[f][None])
            raw.append(np.concatenate([uv[0], np.full((J, 1), 0.9)], axis=1)[None])
        else:
            raw.append(np.zeros((0, J, 3)))
    off = rv.estimate_offsets_pose2d({"cam": raw}, {"cam": {"P1": proj_uv}}, n, log=lambda *a: None)
    assert off == {"cam": true_shift}


def test_pose2d_offset_skips_view_that_never_matches():
    cam = _look_at_camera([0.0, 1.6, -9.0], [0.0, 1.2, -2.0])
    n, J = 100, 6
    body3d = np.tile(np.array([[0.0, 1.0 + 0.1 * j, -2.0] for j in range(J)]), (n, 1, 1))
    proj_uv, _ = rv._project_joints(cam, body3d)
    far = [np.concatenate([proj_uv[f] + 900.0, np.full((J, 1), 0.9)], axis=1)[None] for f in range(n)]
    health = {}
    off = rv.estimate_offsets_pose2d({"cam": far}, {"cam": {"P1": proj_uv}}, n, log=lambda *a: None, health=health)
    assert off == {} and not np.isfinite(health["cam"])


def test_assign_candidates_by_nearest_hand_with_alternatives():
    n = 1
    def cand(x, y):
        return np.array([[x, y - 300, 0.9], [x, y - 150, 0.9], [x, y, 0.9], [x - 30, y - 220, 0.9], [x + 30, y - 220, 0.9]])
    r2d = {"v": [np.stack([cand(100, 500), cand(900, 500), cand(130, 520)])]}
    hand2d = {"v": {"P1": np.array([[[np.nan, np.nan, 0], [105, 505, 0.9]]]),
                    "P2": np.array([[[895, 498, 0.9], [np.nan, np.nan, 0]]])}}
    ppm = {"v": {"P1": np.array([400.0]), "P2": np.array([400.0])}}
    frames, votes = rv.assign_candidates(r2d, hand2d, ppm, {"v": 0}, ["P1", "P2"], n)
    p1 = frames["P1"][0]["v"]
    assert np.allclose(p1[0][rf.HANDLE, :2], [100, 500])
    assert len(p1) == 2 and np.allclose(p1[1][rf.HANDLE, :2], [130, 520])
    assert np.allclose(frames["P2"][0]["v"][0][rf.HANDLE, :2], [900, 500])
    assert votes["P1"] == [0, 1] and votes["P2"] == [1, 0]


def test_relag_track_trails_a_step():
    n = 20
    tr = {"grip": np.zeros((n, 3)), "R": np.tile(np.eye(3), (n, 1, 1)), "valid": np.ones(n, bool)}
    tr["grip"][10:, 0] = 1.0
    out = rv.relag_track(tr, alpha=0.25)
    assert out["grip"][9, 0] == 0.0
    assert np.isclose(out["grip"][10, 0], 0.25)
    assert 0.9 < out["grip"][19, 0] < 1.0


def test_estimate_view_sync_without_pose2d_returns_empty():
    cam = _look_at_camera([0.0, 1.6, -9.0], [0.0, 1.2, -2.0])
    doc = {"meta": {"frame_count": 3}, "skeleton": {"joint_names": ["left_shoulder", "right_shoulder"]},
           "players": [{"id": "P1", "frames": [{"joints_flat": None, "position": None}] * 3}]}
    offsets, health, body_proj, body_by_view, hands_by_view = rv.estimate_view_sync(
        doc, {"cam": cam}, None, log=lambda *a: None)
    assert offsets == {} and health == {} and body_by_view == {} and hands_by_view == {}
    assert body_proj["cam"]["P1"].shape == (3, 2, 2)
