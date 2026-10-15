import json
import os
import re
import shutil

import pytest

import pose24_synth as syn
from analysis.pose24 import build as pb
from analysis.pose24 import config
from analysis.pose24 import doc as docmod
from analysis.pose24 import track
from analysis.pose24.camsubsets import export as ex


EXPORTED = re.compile(r".+_\d+cam_\w+$")          # <clip>_<N>cam_<abbr...>: written by export, not hand-made


def test_export_name_reuses_registry_entry_and_abbreviates():
    live = config.load_registry()
    reg = {k: v for k, v in live.items() if not EXPORTED.match(k)}   # independent of later exports
    assert "singles_05x_2cam" in reg and "doubles_2sides_4angle_2cam" in reg
    assert ex.export_name("singles_05x", ("frontright_051", "backleft_051"), reg) == "singles_05x_2cam"
    assert ex.export_name("singles_05x", ("back_051", "frontleft_051"), reg) == "singles_05x_2cam_b_fl"
    assert ex.export_name("singles_05x", ("back_051",), reg) == "singles_05x_1cam_b"
    if live.get("singles_05x_1cam_b", {}).get("mono"):                  # the live mono export of back_051
        assert ex.export_name("singles_05x", ("back_051",), live, mono=True, lengths="oracle") == "singles_05x_1cam_b"
        assert ex.export_name("singles_05x", ("back_051",), live) == "singles_05x_1cam_b_2"


def test_add_registry_entry_keeps_file_valid_and_is_idempotent(tmp_path):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    before = p.read_text(encoding="utf-8")
    assert ex.add_registry_entry(str(p), "singles_05x_2cam_b_fl", "singles_05x", ["back_051", "frontleft_051"],
                                 "singles_05x")
    reg = config.load_registry(str(p))
    assert reg["singles_05x_2cam_b_fl"] == {"base": "singles_05x", "views": ["back_051", "frontleft_051"],
                                            "timeline_of": "singles_05x"}
    assert p.read_text(encoding="utf-8").startswith(before.rsplit("\n  }", 1)[0])     # old lines untouched
    assert not ex.add_registry_entry(str(p), "singles_05x_2cam_b_fl", "singles_05x",
                                     ["back_051", "frontleft_051"], "singles_05x")
    with pytest.raises(ValueError):
        ex.add_registry_entry(str(p), "singles_05x_2cam_b_fl", "singles_05x", ["back_051"], "singles_05x")


def test_add_registry_entry_changes_only_the_previous_last_line(tmp_path):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    old = p.read_text(encoding="utf-8").split("\n")
    ex.add_registry_entry(str(p), "x_2cam", "singles_05x", ["back_051", "frontleft_051"], "singles_05x")
    new = p.read_text(encoding="utf-8").split("\n")
    k = max(i for i, s in enumerate(old) if s == "  }") - 1              # last line of the last entry
    assert new[:k] == old[:k] and new[k] == old[k] + "," and new[k + 2:] == old[k + 1:]
    assert new[k + 1] == ('    "x_2cam": {"base": "singles_05x", "views": ["back_051", "frontleft_051"], '
                          '"timeline_of": "singles_05x"}')


@pytest.mark.parametrize("text", ['{"clips": {"a": {"views": ["x"]}}}\n',
                                  '{\n  "clips": {\n    "a": {"views": ["x"]}\n  },\n  "other": 1\n}\n'])
def test_add_registry_entry_refuses_unknown_layout_and_writes_nothing(tmp_path, text):
    p = tmp_path / "clips.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        ex.add_registry_entry(str(p), "b", "a", ["x"], "a")
    assert p.read_text(encoding="utf-8") == text


def test_add_registry_entry_into_empty_clips(tmp_path):
    p = tmp_path / "clips.json"
    p.write_text('{\n  "clips": {\n  }\n}\n', encoding="utf-8")
    assert ex.add_registry_entry(str(p), "b", "a", ["x", "y"], "a")
    assert config.load_registry(str(p)) == {"b": {"base": "a", "views": ["x", "y"], "timeline_of": "a"}}


@pytest.mark.parametrize("subset", [("back_051", "frontleft_051"), ("back_11",)])
def test_export_mono_needs_exactly_one_known_view_before_writing(tmp_path, subset):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        ex.export("singles_05x", subset, registry_path=str(p), method="mono-len-prior", log=lambda *a: None)
    with pytest.raises(ValueError):
        ex.export("singles_05x", ("back_051",), registry_path=str(p), method="mono-len-wham", log=lambda *a: None)
    assert p.read_text(encoding="utf-8") == before


def test_export_name_matches_on_the_mono_flag_and_lengths():
    b, fl = "back_051", "frontleft_051"
    reg = {"c": {"views": [b, fl]}, "c_1cam_b": {"base": "c", "views": [b], "timeline_of": "c"},
           "m": {"base": "c", "views": [fl], "timeline_of": "c", "mono": True, "mono_lengths": "oracle"}}
    assert ex.export_name("c", [b], reg) == "c_1cam_b"                         # the fit entry
    assert ex.export_name("c", [b], reg, mono=True) == "c_1cam_b_2"            # never the fit entry's name
    assert ex.export_name("c", [fl], reg, mono=True) == "m"
    assert ex.export_name("c", [fl], reg, mono=True, lengths="prior") == "c_1cam_fl_prior"
    assert ex.export_name("c", [fl], reg) == "c_1cam_fl"                       # a fit never reuses a mono entry


def test_export_name_matches_on_the_mono_prior():
    b = "back_051"
    reg = {"c": {"views": [b]}, "c_1cam_b": {"base": "c", "views": [b], "timeline_of": "c", "mono": True,
                                           "mono_lengths": "oracle"},
           "c_1cam_b_2": {"base": "c", "views": [b], "timeline_of": "c", "mono": True, "mono_lengths": "oracle",
                          "mono_prior": "p.npz"}}
    assert ex.export_name("c", [b], reg, mono=True) == "c_1cam_b"
    assert ex.export_name("c", [b], reg, mono=True, prior="p.npz") == "c_1cam_b_2"
    assert ex.export_name("c", [b], reg, mono=True, prior="q.npz") == "c_1cam_b_3"      # never a prior-free entry
    del reg["c_1cam_b_2"]
    assert ex.export_name("c", [b], reg, mono=True, prior="p.npz") == "c_1cam_b_2"


def test_export_mono_records_the_prior_and_never_reuses_a_prior_free_entry(tmp_path, monkeypatch):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    calls = []
    monkeypatch.setattr(pb, "build", lambda c, **k: calls.append((c, k)) or "doc.json")
    ex.export("singles_05x", ("back_051",), registry_path=str(p), method="mono-len-oracle", log=lambda *a: None)
    ex.export("singles_05x", ("back_051",), registry_path=str(p), method="mono-len-oracle",
              mono_opts={"prior_path": "pri/wham.npz"}, log=lambda *a: None)
    reg = config.load_registry(str(p))
    assert "mono_prior" not in reg["singles_05x_1cam_b"]
    assert reg["singles_05x_1cam_b_2"]["mono_prior"] == "pri/wham.npz"
    assert [k["out_clip"] for _, k in calls] == ["singles_05x_1cam_b", "singles_05x_1cam_b_2"]
    assert calls[1][1]["mono_opts"] == {"prior_path": "pri/wham.npz", "lengths": "oracle"}
    with pytest.raises(ValueError, match="registry entry"):
        ex.export("singles_05x", ("back_051",), registry_path=str(p), method="mono-len-oracle",
                  name="singles_05x_1cam_b", mono_opts={"prior_path": "pri/wham.npz"}, log=lambda *a: None)


def test_add_registry_entry_is_atomic(tmp_path, monkeypatch):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    before = p.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ex.os, "replace", boom)
    with pytest.raises(OSError):
        ex.add_registry_entry(str(p), "x_2cam", "singles_05x", ["back_051", "frontleft_051"], "singles_05x")
    assert p.read_text(encoding="utf-8") == before and os.listdir(tmp_path) == ["clips.json"]
    monkeypatch.undo()
    assert ex.add_registry_entry(str(p), "x_1cam", "singles_05x", ["back_051"], "singles_05x",
                                 extra={"mono": True, "mono_lengths": "prior"})
    assert config.load_registry(str(p))["x_1cam"] == {"base": "singles_05x", "views": ["back_051"],
                                                      "timeline_of": "singles_05x", "mono": True,
                                                      "mono_lengths": "prior"}
    assert os.listdir(tmp_path) == ["clips.json"]


def test_export_rejects_unknown_views_before_writing(tmp_path):
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        ex.export("singles_05x", ("back_051", "back_11"), registry_path=str(p), log=lambda *a: None)
    assert p.read_text(encoding="utf-8") == before


def test_export_fit_is_a_real_pose24_build_on_the_full_timeline(tmp_path):
    views = ["a", "b", "c", "d"]
    players = {"P1": syn.motion(120, z0=-3.0), "P2": syn.motion(120, x0=0.5, z0=4.0)}
    entry = syn.write_clip(str(tmp_path), "synclip", players, syn.rig(views), views, shifts={"b": 2})
    regp = tmp_path / "clips.json"
    regp.write_text(json.dumps({"clips": {"synclip": entry}}, indent=2) + "\n", encoding="utf-8")
    kw = dict(analysis_dir=str(tmp_path / "analysis"), cache_dir=str(tmp_path / "cache"),
              data_dir=str(tmp_path / "data"), fit_opts={"passes": 1})
    os.makedirs(kw["analysis_dir"])
    pb.build("synclip", registry=config.load_registry(str(regp)), log=lambda *a: None, **kw)
    logs = []
    doc_path = ex.export("synclip", ("c", "a"), registry_path=str(regp), name="synclip_2cam_ac",
                         log=lambda *a: logs.append(" ".join(map(str, a))), **kw)
    assert config.load_registry(str(regp))["synclip_2cam_ac"] == {"base": "synclip", "views": ["a", "c"],
                                                                  "timeline_of": "synclip"}
    assert doc_path == pb.output_path(kw["analysis_dir"], "synclip_2cam_ac") and os.path.exists(doc_path)
    cpath = track.cache_path("synclip_2cam_ac", kw["cache_dir"])
    assert os.path.exists(cpath)
    with open(doc_path, encoding="utf-8") as fh:
        d = json.load(fh)
    assert docmod.load_debug_cache(cpath)["build_id"] == d["meta"]["pose24"]["build_id"]
    assert d["meta"]["pose_variant"]["timeline_of"] == "synclip"
    assert any("cvdebug build --clip synclip_2cam_ac --variant pose24" in s for s in logs)
    assert any("--compare synclip:pose24 --frames 300:600" in s for s in logs)


@pytest.mark.parametrize("subset,method", [(("back_051", "frontleft_051"), "fit"), (("back_051",), "mono-len-oracle")])
def test_export_rejects_views_excluded_in_the_full_cache(tmp_path, monkeypatch, subset, method):
    import numpy as np
    p = tmp_path / "clips.json"
    shutil.copy(config.REGISTRY_PATH, p)
    before = p.read_text(encoding="utf-8")
    cdir = tmp_path / "cache"
    cdir.mkdir()
    np.savez(track.cache_path("singles_05x", str(cdir)), meta=json.dumps({"excluded": {"back_051": "view health"}}))
    monkeypatch.setattr(pb, "build", lambda c, **k: pytest.fail("built an excluded view"))
    with pytest.raises(ValueError, match="excluded"):
        ex.export("singles_05x", subset, registry_path=str(p), method=method, cache_dir=str(cdir),
                  log=lambda *a: None)
    assert p.read_text(encoding="utf-8") == before
    monkeypatch.setattr(pb, "build", lambda c, **k: "doc.json")               # a kept view still exports
    assert ex.export("singles_05x", ("frontleft_051",) if method != "fit" else ("frontleft_051", "frontright_051"),
                     registry_path=str(p), method=method, cache_dir=str(cdir), log=lambda *a: None) == "doc.json"
