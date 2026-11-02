# tools/tests/test_live_mono.py
"""v0.8.7 one camera: config, body model sources, per-step points, a synthetic 1-camera stream, the server."""
import json

import numpy as np
import pytest

import live_synth as ls
import pose24_synth as syn
from analysis import skeleton24 as sk
from analysis.live import mono_live, outputs
from analysis.live.stream_pose24 import StreamPose24
from analysis.pose24 import kpset
from test_live_config import load

PELVIS = sk.SMPL28_NAMES.index("pelvis")


# ------------------------------------------------------------------------------------------------ config
def test_one_view_turns_on_the_one_camera_mode():
    c = load(overrides=["views=[back_051]"])
    assert c["views"] == ["back_051"] and c["mono"]["on"]
    assert c["warmup"]["min_support_views"] == 1
    assert c["offsets"] == {"back_051": 0.0} and c["offsets_source"] == "one camera"
    assert c["out_clip"] == "singles_05x_1cam_live30"


def test_three_views_keep_the_mono_mode_off():
    c = load()
    assert not c["mono"]["on"] and c["warmup"]["min_support_views"] == 2


def test_yaml_off_and_bad_heights():
    assert load(overrides=["views=[back_051]", "mono.body_models=off"])["mono"]["body_models"] == "off"
    with pytest.raises(ValueError, match="1.0-2.5 m"):
        load(overrides=["views=[back_051]", "mono.heights_m.P1=17.5"])
    with pytest.raises(ValueError, match="height_scale"):
        load(overrides=["views=[back_051]", "mono.height_scale=3"])
    with pytest.raises(ValueError, match="auto or off"):
        load(overrides=["views=[back_051]", "mono.body_models=always"])


# ------------------------------------------------------------------------------------------------ body model
def _cfg(tmp_path, **mono):
    return {"runs_dir": str(tmp_path), "mode": "singles",
            "mono": dict({"body_models": "auto", "heights_m": {}, "height_scale": 0.88, "stature_m": 1.50}, **mono)}


def _cam():
    return syn.rig(("a",))["a"]


def _stature(m):
    return m.lengths["stature"]["measured_m"]


def test_template_height_and_scaled_real_height(tmp_path):
    m, src = mono_live.body_model(_cfg(tmp_path), _cam(), "P1")
    assert _stature(m) == pytest.approx(1.50) and src == "template 1.50 m"
    m, src = mono_live.body_model(_cfg(tmp_path, heights_m={"P1": 1.80}), _cam(), "P1")
    assert _stature(m) == pytest.approx(1.80 * 0.88) and src.startswith("height 1.80 m x 0.88")
    m2, _ = mono_live.body_model(_cfg(tmp_path, heights_m={"P1": 1.80}), _cam(), "P2")
    assert _stature(m2) == pytest.approx(1.50)
    assert m.edges[1].max() > m2.edges[1].max()              # longer limbs for the taller player
    assert m.sgn == kpset.world_sign({"a": _cam()})


def _write_warm(tmp_path, views=("a", "b", "c"), mode="singles", stature=1.62):
    L = mono_live.mono.lengths_for("prior", None, "P1", stature_m=stature)
    (tmp_path / "engine").mkdir(exist_ok=True)
    ws = {"models": {"P1": {"lengths": L, "rest": [], "sgn": 1.0}, "P2": None}, "views": list(views), "fps": 30.0}
    if mode:
        ws["mode"] = mode
    (tmp_path / "engine" / "warm_state.json").write_text(json.dumps(ws), encoding="utf-8")


def test_saved_multi_camera_models_come_first(tmp_path):
    _write_warm(tmp_path)
    cfg = _cfg(tmp_path, heights_m={"P1": 1.90})
    saved, path = mono_live.load_saved(cfg, ["P1", "P2"])
    assert set(saved) == {"P1"} and path.endswith("warm_state.json")
    m, src = mono_live.body_model(cfg, _cam(), "P1", saved)
    assert _stature(m) == pytest.approx(1.62, abs=1e-6) and src == "the last multi-camera warm-up"
    _, src2 = mono_live.body_model(cfg, _cam(), "P2", saved)          # no saved model: its height / template
    assert src2 == "template 1.50 m"


@pytest.mark.parametrize("kw, cfg_kw", [({"views": ("a",)}, {}), ({"mode": "doubles"}, {}), ({}, {"body_models": "off"}),
                                        ({"mode": None}, {})])
def test_saved_models_are_ignored_when_they_do_not_fit(tmp_path, kw, cfg_kw):
    _write_warm(tmp_path, **kw)
    saved, _ = mono_live.load_saved(_cfg(tmp_path, **cfg_kw), ["P1", "P2"])
    if kw.get("mode", "x") is None:          # an older file without a mode: accepted only with the same players
        assert saved == {} or set(saved) <= {"P1", "P2"}
    else:
        assert saved == {}


def test_no_warm_state_file(tmp_path):
    assert mono_live.load_saved(_cfg(tmp_path), ["P1", "P2"]) == ({}, None)


# ------------------------------------------------------------------------------------------------ per-step points
def test_step_points_reproject_exactly_and_stand_on_the_floor(tmp_path):
    cam = _cam()
    X = syn.motion(4, 60.0, x0=0.5, z0=-3.0)[2:3]                     # (1, 133, 3) one frame
    uv = syn.project(X, {"a": cam}, ["a"], 0.0, np.random.default_rng(0))["a"]
    u35, w35 = kpset.obs35(uv[:, :, None, :2].astype(float), uv[:, :, None, 2].astype(float))
    m, _ = mono_live.body_model(_cfg(tmp_path), cam, "P1")
    P = mono_live.step_points(u35[:, :, 0], w35[:, :, 0], cam, m, 0.08)
    assert np.isfinite(P).all()
    K, R, t = (np.asarray(a, float) for a in cam)
    pr = (P @ R.T + t) @ K.T
    pr = pr[:, :2] / pr[:, 2:]
    seen = w35[0, :, 0] >= 0.3
    assert np.abs(pr[seen] - u35[0, seen, 0]).max() < 1e-6
    assert P[kpset.CONTACT, 1].max() > -0.3                          # feet near the floor (Y-down world)
    none = mono_live.step_points(u35[:, :, 0], np.zeros_like(w35[:, :, 0]), cam, m, 0.08)
    assert np.isnan(none).all()


# ------------------------------------------------------------------------------------------------ stream
def _run_mono(tmp_path, n=80, mono=None):
    cams, players = ls.scene(F=2 * n + 4)
    cfg = ls.cfg(n_steps=n)
    cfg.update(views=["a"], time_ref="a", offsets={"a": 0.0}, offsets_source="one camera", runs_dir=str(tmp_path))
    cfg["warmup"]["min_support_views"] = 1
    cfg["mono"] = dict(cfg["mono"], on=True, **(mono or {}))
    events = []
    s = StreamPose24(cfg, {"a": cams["a"]}, events=events.append)
    outs = []
    for i, r in enumerate(cfg["ref_frames"]):
        fs = ls.frameset(i, r, cams, players)
        fs["views"] = {"a": fs["views"]["a"]}
        outs += s.push(fs)
    outs += s.flush()
    s.close()
    return cfg, players, outs, events


def _pelvis_xz_err(players, outs, pid):
    err = []
    for o in outs[30:]:
        J = o["players"][pid]["J"]
        tr = ls.truth(players, pid, o["ref_frame"])[[11, 12]].mean(axis=0)
        err.append(np.linalg.norm(J[PELVIS][[0, 2]] - tr[[0, 2]]))
    return float(np.median(err))


def test_one_camera_stream_outputs_every_step_and_finds_both_players(tmp_path):
    cfg, players, outs, events = _run_mono(tmp_path)
    assert [o["step"] for o in outs] == list(range(80))
    assert all(o["quality"] == outputs.WARMUP for o in outs[:30])
    assert all(o["quality"] == outputs.FITTED for o in outs[30:])
    assert all(o["players"][p]["valid"] for o in outs for p in ("P1", "P2"))
    src = next(e for e in events if e["ev"] == "mono_models")["source"]
    assert src == {"P1": "template 1.50 m", "P2": "template 1.50 m"}
    for pid in ("P1", "P2"):
        assert _pelvis_xz_err(players, outs, pid) < 0.35            # 1 camera: court position to a few dm


def test_one_camera_stream_exports_a_warm_state_with_its_mode(tmp_path):
    cams, players = ls.scene(F=40)
    cfg = ls.cfg(n_steps=10, warmup=4)
    cfg.update(views=["a"], time_ref="a", offsets={"a": 0.0}, runs_dir=str(tmp_path))
    cfg["mono"] = dict(cfg["mono"], on=True)
    s = StreamPose24(cfg, {"a": cams["a"]})
    ws = s.export_warm_state()
    s.close()
    assert ws["mode"] == "singles" and ws["views"] == ["a"] and set(ws["models"]) == {"P1", "P2"}


# ------------------------------------------------------------------------------------------------ server
def test_server_accepts_one_angle_and_one_camera(tmp_path):
    from fastapi import HTTPException
    from pipeline_server import server as srv
    s = srv.Server(unity_project=tmp_path, runs_dir=tmp_path / "runs")
    out = s.set_input({"kind": "video", "clip": "singles_05x", "views": ["back_051"]})
    assert out["views"] == ["back_051"]
    out = s.set_input({"kind": "camera", "clip": "singles_05x",
                       "cameras": [{"name": "cam1", "url": "http://x:8090/a.mjpg", "use": True}]})
    assert [c["name"] for c in out["cameras"] if c["use"]] == ["cam1"]
    with pytest.raises(HTTPException):
        s.set_input({"kind": "video", "clip": "singles_05x", "views": []})
