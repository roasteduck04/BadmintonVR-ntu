"""pose24 clip registry + config resolution (no hardcoded 5-view fallback, review L2)."""
import os

import pytest

from analysis.pose24 import config

REG = {"clipA": {"mode": "singles", "seeds": "clipA", "time_ref": "a", "views": ["b", "a", "c"],
                 "videos": {"a": "va.mov"}},
       "clipA_2cam": {"base": "clipA", "views": ["c", "b"], "timeline_of": "clipA"}}


def _res(tmp_path, clip="clipA", **kw):
    return config.resolve(clip, registry=REG, data_dir=str(tmp_path), analysis_dir=str(tmp_path), **kw)


def test_full_clip_resolution(tmp_path):
    cfg = _res(tmp_path)
    assert cfg["views"] == ["a", "b", "c"] and cfg["out_clip"] == "clipA" and cfg["time_ref"] == "a"
    assert cfg["timeline_of"] is None and cfg["base_doc_path"] is None and cfg["mode"] == "singles"
    assert cfg["cam_paths"]["b"] == os.path.join(str(tmp_path), "calib", "b_cam.json")
    assert cfg["kp2d_paths"]["c"] == os.path.join(str(tmp_path), "pose2d", "c_kp2d.json")
    assert cfg["seeds_path"].endswith("clipA_seeds.json") and cfg["videos"]["a"] == "va.mov"


def test_derived_entry_uses_base_and_timeline(tmp_path):
    cfg = _res(tmp_path, "clipA_2cam")
    assert cfg["clip"] == "clipA" and cfg["out_clip"] == "clipA_2cam" and cfg["views"] == ["b", "c"]
    assert cfg["timeline_of"] == "clipA" and cfg["time_ref"] == "b" and cfg["all_views"] == ["a", "b", "c"]


def test_views_override_rules(tmp_path):
    with pytest.raises(ValueError, match="out-clip"):
        _res(tmp_path, views=["a", "b"])
    with pytest.raises(ValueError, match="unknown"):
        _res(tmp_path, views=["a", "zz"], out_clip="x")
    with pytest.raises(ValueError, match=">= 2"):
        _res(tmp_path, views=["a"], out_clip="x")
    cfg = _res(tmp_path, views=["c", "a"], out_clip="clipA_ac", timeline_of="clipA")
    assert cfg["views"] == ["a", "c"] and cfg["timeline_of"] == "clipA" and cfg["out_clip"] == "clipA_ac"


def test_base_doc_found_for_out_clip(tmp_path):
    (tmp_path / "clipA.analysis.json").write_text("{}")
    assert _res(tmp_path)["base_doc_path"] == os.path.join(str(tmp_path), "clipA.analysis.json")
    assert _res(tmp_path, "clipA_2cam")["base_doc_path"] is None


def test_unknown_clip():
    with pytest.raises(KeyError):
        config.resolve("nope", registry=REG)


def test_registry_file_lists_every_clip():
    reg = config.load_registry()
    assert {"singles_1x", "singles_05x", "doubles_1side", "doubles_2sides", "doubles_2sides_4angle",
            "singles_05x_2cam", "doubles_2sides_4angle_2cam"} <= set(reg)
    assert reg["singles_05x_2cam"]["views"] == ["backleft_051", "frontright_051"]
    order = config.all_clips(reg)
    assert order.index("singles_05x") < order.index("singles_05x_2cam")
    for e in reg.values():
        if "base" not in e:
            assert len(e["views"]) >= 2 and e["time_ref"] == sorted(e["views"])[0]
            assert set(e["videos"]) == set(e["views"])


def test_derived_entry_with_other_views_needs_its_own_out_clip(tmp_path):
    """`build --clip clipA_2cam --views a,c` must not overwrite the registry's clipA_2cam build."""
    with pytest.raises(ValueError, match="out-clip"):
        _res(tmp_path, "clipA_2cam", views=["a", "c"])
    assert _res(tmp_path, "clipA_2cam", views=["b", "c"])["out_clip"] == "clipA_2cam"      # same views: fine
    assert _res(tmp_path, "clipA_2cam", views=["a", "c"], out_clip="clipA_ac")["views"] == ["a", "c"]


def test_bad_base_in_the_registry_is_named():
    reg = dict(REG, broken={"base": "missing", "views": ["a", "b"]})
    with pytest.raises(KeyError, match="broken.*missing"):
        config.resolve("broken", registry=reg)


def test_multiview_clips_skips_mono_entries():
    reg = dict(REG, clipA_1cam_a={"base": "clipA", "views": ["a"], "mono": True})
    assert config.multiview_clips(reg) == ["clipA", "clipA_2cam"]
    assert config.all_clips(reg) == ["clipA", "clipA_2cam", "clipA_1cam_a"]


def test_every_multiview_clip_of_the_real_registry_resolves(tmp_path):
    """review 1 M2: the real clips.json holds a mono entry; every --all multi-view consumer must skip it."""
    reg = config.load_registry()
    assert any(e.get("mono") for e in reg.values())
    for c in config.multiview_clips(reg):
        config.resolve(c, registry=reg, data_dir=str(tmp_path), analysis_dir=str(tmp_path))
