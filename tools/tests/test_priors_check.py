"""v0.7.5 monoprior: priors check (prior vs the full rig's camera-frame root-relative 3D), on a synthetic body seen by
a known camera."""
import json

import numpy as np
import pytest

from analysis.pose24 import kpset, mono
from analysis.pose24.priors import check, common
from analysis.pose24.priors import __main__ as pcli

KP = np.array([k for _, k in common.H36M_TO_KP])
# a standing body in the Y-down world (m): +Y down, hips at y = -0.9
REST = {0: (0.0, -1.6, 0.05), 5: (0.18, -1.4, 0.0), 6: (-0.18, -1.4, 0.0), 7: (0.25, -1.15, 0.05),
        8: (-0.25, -1.15, 0.05), 9: (0.28, -0.9, 0.15), 10: (-0.28, -0.9, 0.15), 11: (0.1, -0.9, 0.0),
        12: (-0.1, -0.9, 0.0), 13: (0.1, -0.48, 0.03), 14: (-0.1, -0.48, 0.03), 15: (0.1, -0.08, 0.0),
        16: (-0.1, -0.08, 0.0)}


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _camera():
    """A camera 8 m behind the court looking along +Z, tilted down 15 degrees (det R = +1)."""
    a = np.deg2rad(15)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    R = Rx
    C = np.array([0.5, -3.0, -8.0])
    return np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]]), R, -R @ C


def _body(F=300, seed=0):
    """(F,35,3) Y-down: the rest body turning about its vertical axis and walking, limbs swinging."""
    rng = np.random.default_rng(seed)
    X = np.full((F, 35, 3), np.nan)
    base = np.array([REST[k] for k in KP])
    for f in range(F):
        B = base.copy()
        sw = 0.3 * np.sin(f / 9.0)
        B[KP == 9, 2] += sw
        B[KP == 10, 2] -= sw
        B[KP == 15, 2] -= 0.5 * sw
        B[KP == 16, 2] += 0.5 * sw
        X[f, KP] = B @ _rot_y(f / 25.0).T + np.array([np.sin(f / 50.0), 0, 0.01 * f]) + rng.normal(0, 0.003, B.shape)
    return X


def _truth_prior(X, cam, unit=0.37):
    """The exact camera-frame root-relative 3D in arbitrary units (as a learned prior would give)."""
    K, R, t = cam
    Xc = X[:, KP] @ R.T + t
    root = 0.5 * (Xc[:, list(KP).index(11)] + Xc[:, list(KP).index(12)])
    return {"kp": KP.copy(), "rel_cam": (Xc - root[:, None]) * unit, "conf": np.full(len(X), 0.9)}


def _lengths(X):
    leg = np.linalg.norm(X[:, 11] - X[:, 13], axis=-1) + np.linalg.norm(X[:, 13] - X[:, 15], axis=-1)
    th = float(np.median(np.linalg.norm(X[:, 11] - X[:, 13], axis=-1)))
    return {"edges": {"thigh": th, "shank": float(np.median(leg)) - th}}


def test_rig_rel_cam_matches_projection_convention():
    cam = _camera()
    X = _body(10)
    rel = check.rig_rel_cam(X, cam, KP)
    K, R, t = cam
    Xc = X[:, KP] @ R.T + t
    uv = Xc[..., :2] / Xc[..., 2:]
    # the camera-frame points project with K to the pixels pose_v2's projection gives
    from analysis import pose_v2 as pv
    uv_pv, depth = pv.project(pv.proj_mats({"v": cam}, ["v"]), X[:, KP])
    assert np.allclose(uv * 1000 + [960, 540], uv_pv[..., 0, :], atol=1e-6)
    assert np.allclose(depth[..., 0], Xc[..., 2])
    hips = rel[:, [list(KP).index(11), list(KP).index(12)]].mean(1)
    assert np.allclose(hips, 0, atol=1e-12)


def test_truth_prior_gives_r_one_and_zero_dz():
    cam, X = _camera(), _body()
    res = check.compare(_truth_prior(X, cam), X, cam, _lengths(X))
    assert res["scale"] == pytest.approx(1 / 0.37, rel=0.02)
    for name in ("left_knee", "right_ankle", "left_wrist", "right_elbow"):
        j = res["joints"][name]
        assert j["r_z"] > 0.99 and j["r_x"] > 0.99 and j["r_y"] > 0.95
        assert j["med_abs_dz_cm"] < 1.0
        assert j["n"] == len(X)
    assert res["groups"]["legs_arms"]["median_r_z"] > 0.99
    assert check.verdict(res) == "PASS"


def test_z_flipped_prior_gives_r_minus_one():
    cam, X = _camera(), _body()
    p = _truth_prior(X, cam)
    p["rel_cam"][..., 2] *= -1
    res = check.compare(p, X, cam, _lengths(X))
    for name in ("left_knee", "right_ankle", "left_wrist", "right_elbow"):
        assert res["joints"][name]["r_z"] < -0.99
        assert res["joints"][name]["r_x"] > 0.99
    assert res["groups"]["legs_arms"]["median_r_z"] < -0.99
    assert check.verdict(res) == "FLIP"


def test_low_conf_and_missing_rig_frames_are_left_out():
    cam, X = _camera(), _body()
    p = _truth_prior(X, cam)
    p["conf"][:50] = mono.PRIOR_CONF_MIN - 0.01
    X[50:60, 13] = np.nan                                    # left knee missing in the rig
    res = check.compare(p, X, cam, _lengths(X))
    assert res["joints"]["right_knee"]["n"] == len(X) - 50
    assert res["joints"]["left_knee"]["n"] == len(X) - 60


def test_noise_prior_is_blocked():
    cam, X = _camera(), _body()
    p = _truth_prior(X, cam)
    p["rel_cam"] = np.random.default_rng(3).normal(size=p["rel_cam"].shape)
    res = check.compare(p, X, cam, _lengths(X))
    assert check.verdict(res) == "BLOCKED"


def test_run_writes_json_and_table(tmp_path, capsys):
    cam, X = _camera(), _body()
    p = _truth_prior(X, cam)
    clip, view, method = "c", "v1", "motionbert-lite"
    meta = {"method": method, "clip": clip, "view": view, "timeline_of": clip, "fps": 60.0,
            "frame_count": len(X), "build_id": "b1", "norm": "crop"}
    prior_path = tmp_path / clip / f"{view}_{method}.npz"
    common.write_prior(prior_path, meta, {"P1": p})
    cache = {"views": [view], "cams": {view: cam}, "frame_count": len(X), "build_id": "b1",
             "extra": {"P1": {"X": (X * [1, -1, 1]).astype(np.float32)}},
             "players_lengths": {"P1": dict(_lengths(X), soft={}, stature={})}}
    res = check.run(clip, view, method, cache=cache, prior_dir=tmp_path)
    out = tmp_path / clip / f"{view}_{method}.check.json"
    data = json.loads(out.read_text())
    assert data["verdict"] == "PASS" and data["players"]["P1"]["joints"]["left_knee"]["r_z"] > 0.99
    assert res["path"] == str(out)
    table = check.format_table(res)
    table.encode("ascii")
    assert "left_knee" in table and "PASS" in table


def test_cli_check_subcommand_parses(monkeypatch):
    seen = {}
    monkeypatch.setattr(check, "run", lambda clip, view, method, **kw: seen.update(c=clip, v=view, m=method) or
                        {"verdict": "PASS", "path": "x", "players": {}, "pooled": {"joints": {}, "groups": {}}})
    monkeypatch.setattr(check, "format_table", lambda r: "table")
    monkeypatch.setattr(pcli, "_load_cache", lambda clip: {})
    pcli.main(["check", "--clip", "c", "--view", "v", "--method", "motionbert-lite-image"])
    assert seen == {"c": "c", "v": "v", "m": "motionbert-lite-image"}
