import json
import os

import numpy as np
import pytest

import pose24_synth as syn
from analysis.pose24 import build as pb
from analysis.pose24 import doc as docmod
from analysis.pose24 import track
from analysis.pose24.camsubsets import e2e


def test_match_by_pelvis_handles_permutation_and_missing():
    t = np.linspace(0, 1, 50)[:, None]
    ref = {"P1": np.c_[t, 0 * t, 0 * t], "P2": np.c_[t, 0 * t, 3 + 0 * t], "P3": np.c_[-t, 0 * t, 6 + 0 * t]}
    sub = {"P1": ref["P2"] + 0.05, "P2": ref["P1"] - 0.05}            # permuted, P3 missing
    assert e2e.match_by_pelvis(sub, ref) == {"P1": "P2", "P2": "P1"}
    far = {"P9": ref["P1"] + np.array([5.0, 0, 0])}
    assert e2e.match_by_pelvis(far, ref) == {}


def test_shift_without_shared_views_raises():
    with pytest.raises(ValueError, match="no shared view"):
        e2e._shift({"offsets": {"a": 0.0}}, {"offsets": {"b": 1.0}})
    assert e2e._shift({"offsets": {"a": 2.0, "b": 0.0}}, {"offsets": {"a": 1.0, "b": -1.0}}) == (1.0, 0.0)


def test_scratch_dirs_and_out_clip(tmp_path):
    a, c = e2e.scratch_dirs(str(tmp_path), "clipx")
    assert a == os.path.join(str(tmp_path), "clipx", "e2e", "_builds", "analysis")
    assert c == os.path.join(str(tmp_path), "clipx", "e2e", "_builds", "cache")
    assert e2e.out_clip("clipx", ("frontleft_41", "back_41")) == "clipx_e2e_b_fl"
    assert e2e.out_clip("synclip", ("c", "b")) == "synclip_e2e_b_c"            # no abbreviation: the view name


def test_doc_joints_blanks_conf0_placeholders():
    doc = {"skeleton": {"joint_names": ["pelvis", "left_hand"]},
           "players": [{"id": "P1", "frames": [
               {"position": [1.0, 2.0], "joints_flat": [0, 0.9, 0, 0.8, 0.1, 1.0, 0.2, 0.0]},   # hand conf 0
               {"position": None, "joints_flat": None}]}]}
    J, C = e2e.doc_joints(doc, "P1", ["left_hand", "pelvis", "nose"])
    assert np.isnan(J[0, 0]).all() and np.allclose(J[0, 1], [1.0, 0.9, 2.0])        # C == 0: placeholder -> NaN (R16)
    assert C[0, 0] == 0.0 and C[0, 1] == 0.8
    assert np.isnan(J[0, 2]).all() and np.isnan(C[0, 2]) and np.isnan(J[1]).all()


def test_failed_build_cleans_its_scratch_files(tmp_path, monkeypatch):
    out = str(tmp_path)
    adir, cdir = e2e.scratch_dirs(out, "clipx")

    def boom(clip, subset, out_dir, **kw):
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(adir, exist_ok=True)
        open(track.cache_path(e2e.out_clip(clip, subset), cdir), "w").close()
        open(track.cache_path(e2e.out_clip(clip, ("a", "b", "c")), cdir), "w").close()   # another subset's: kept
        raise RuntimeError("build failed")
    monkeypatch.setattr(e2e, "build_one", boom)
    with pytest.raises(RuntimeError):
        e2e._build_and_convert("clipx", ("a", "b"), out, None, None, "data", full_cache={"build_id": "x"})
    assert os.listdir(cdir) == [os.path.basename(track.cache_path("clipx_e2e_a_b_c", cdir))]


def test_e2e_pair_build_is_retimed_onto_the_full_timeline(tmp_path):
    """Synthetic 4-view clip with view shifts; full build, then an e2e pair (own sync) -> re-timed result
    agrees with the full build's P1 within a few cm."""
    views = ["a", "b", "c", "d"]
    cams = syn.rig(views)
    players = {"P1": syn.motion(150, z0=-3.0), "P2": syn.motion(150, x0=0.5, z0=4.0)}
    entry = syn.write_clip(str(tmp_path), "synclip", players, cams, views, shifts={"b": 2, "c": -1})
    reg = {"synclip": entry}
    adir = str(tmp_path / "analysis")
    cdir = str(tmp_path / "cache")
    os.makedirs(adir)
    pb.build("synclip", registry=reg, data_dir=str(tmp_path / "data"), analysis_dir=adir, cache_dir=cdir,
             log=lambda *a: None)
    full = docmod.load_debug_cache(track.cache_path("synclip", cdir))
    out = str(tmp_path / "subsets")
    got = e2e.run_e2e("synclip", [("b", "c")], track.cache_path("synclip", cdir), out, workers=1, registry=reg,
                      data_dir=str(tmp_path / "data"), log=lambda *a: None)
    from analysis.pose24.camsubsets import oracle
    res = oracle.load_result(got["b+c"], None)
    meta = res["meta"]
    # sub-frame sync: the full build puts c at -0.6 (true -1), the pair at c - b = -3.4 (true -3) -> spread 0.6
    assert meta["e2e"]["spread"] < 1.0
    assert meta["build_id"] == full["build_id"] and meta["e2e_build_id"] != full["build_id"]
    assert meta["workers"] == 1 and 0 < meta["cpu_s"]                    # process time (R18)
    rt = meta["e2e_runtime_s"]                                             # the e2e doc's runtime_s (cost table)
    assert rt["track"] >= 0 and rt["fit"] > 0 and rt["total"] >= rt["fit"] and "e2e_runtime_s" not in meta["e2e"]
    assert sorted(meta["e2e"]["match"].values()) == ["P1", "P2"]
    assert set(meta["e2e"]["identity"]) == {"P1", "P2"}
    pid = next(s for s, q in meta["e2e"]["match"].items() if q == "P1")
    p = res["players"][pid]
    n = full["frame_count"]
    assert p["J"].shape[0] == n and p["lr_swapped"].shape == (n, 2) and p["inlier"].dtype == bool
    X, Xr = p["X"], full["extra"]["P1"]["X"]
    d = np.nanmedian(np.linalg.norm(X[:, 11] - Xr[:, 11], axis=-1))
    assert d < 0.05
    assert not os.path.exists(os.path.join(out, "synclip", "e2e", "_builds"))       # scratch cleaned
    # fresh on a re-run (nothing rebuilt), stale when the full build changes
    again = e2e.run_e2e("synclip", [("b", "c")], track.cache_path("synclip", cdir), out, workers=1, registry=reg,
                        data_dir=str(tmp_path / "data"), log=lambda *a: None)
    assert again == got
    assert oracle.load_result(got["b+c"], e2e.expected_meta(dict(full, build_id="other"))) is None
    # R16: a hand joint with conf 0 (a filled placeholder) is NaN, and coverage (pelvis-based) is unchanged
    base_cov = p["coverage"]
    doc_path = e2e.build_one("synclip", ("b", "c"), out, registry=reg, data_dir=str(tmp_path / "data"),
                             log=lambda *a: None)
    cpath = track.cache_path(e2e.out_clip("synclip", ("b", "c")), e2e.scratch_dirs(out, "synclip")[1])
    with open(doc_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    hand = doc["skeleton"]["joint_names"].index("left_hand")
    q = next(q for q in doc["players"] if q["id"] == pid)
    for fr in q["frames"]:
        if fr.get("joints_flat"):
            fr["joints_flat"][4 * hand + 3] = 0.0
    with open(doc_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    res2, _ = e2e.to_result(doc_path, cpath, full)
    p2 = res2["players"][pid]
    assert p2["coverage"] == pytest.approx(base_cov) and base_cov > 0.9
    assert 0 < p2["measured"] <= p2["coverage"] and p2["full_body"] == 0.0 < p["full_body"]
    assert np.nanmax(p2["C"][:, hand]) == 0.0 and not np.isfinite(p2["J"][:, hand]).any()


def test_cleanup_error_after_a_saved_result_is_logged_not_failed(tmp_path, monkeypatch):
    out = str(tmp_path)
    res = {"players": {}, "views": ["a", "b"], "method": "fit", "runtime_s": 0.0}
    monkeypatch.setattr(e2e, "build_one", lambda *a, **k: "doc.json")
    monkeypatch.setattr(e2e, "to_result", lambda *a, **k: (dict(res), {"build_id": "e", "runtime_s": {}}))

    def boom(*a):
        raise PermissionError("file in use")
    monkeypatch.setattr(e2e, "_cleanup", boom)
    logs = []
    full = {"build_id": "x"}
    sec = e2e._build_and_convert("clipx", ("a", "b"), out, None, None, "data", full_cache=full, workers=3,
                                 log=logs.append)
    from analysis.pose24.camsubsets import oracle
    got = oracle.load_result(oracle.result_path(out, "clipx", "e2e", "fit", ("a", "b")), e2e.expected_meta(full))
    assert sec >= 0 and got is not None and got["meta"]["workers"] == 3 and got["meta"]["cpu_s"] >= 0
    assert any("cleanup" in m and "file in use" in m for m in logs)
