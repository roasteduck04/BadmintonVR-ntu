"""pose24 end to end on a synthetic 4-camera singles clip (+ a 2-camera subset on its timeline)."""
import json
import os

import numpy as np
import pytest

import pose24_synth as syn
from analysis import schema24
from analysis.pose24 import build as pb
from analysis.pose24 import doc as docmod
from analysis.pose24 import track

F = 150


def _write_base(adir, players):
    """A base doc + rigid racket sidecar whose player ids are swapped relative to the tracker's."""
    def pos(X, f):
        h = 0.5 * (X[f, 11] + X[f, 12])
        return [float(h[0]), float(h[2])]
    racket = {"grip": [0.3, 1.0, 0.1], "head": [0.3, 1.6, 0.1], "side": [0.4, 1.6, 0.1], "which_hand": "right",
              "confidence": 0.9, "status": "detected"}
    bps = []
    for bid, pid in (("P1", "P2"), ("P2", "P1")):
        frames = [{"frame_id": f, "position": pos(players[pid], f), "racket": dict(racket), "grip": None,
                   "stroke": {"label": "smash", "confidence": 0.8} if f == 5 else None} for f in range(F)]
        bps.append({"id": bid, "side": "far" if bid == "P1" else "near", "frames": frames})
    base = {"schema_version": "4.0", "clip_id": "synclip", "meta": {"fps": 60.0, "frame_count": F,
            "producers": {"racket": "rigid_fit", "shuttle": "tracknet"}}, "players": bps,
            "shuttle": {"frames": [{"frame_id": 0, "position": [0.0, float("nan"), 0.0]}]},
            "rallies": [{"start": 0, "end": F - 1}]}
    with open(os.path.join(adir, "synclip.analysis.json"), "w") as fh:
        json.dump(base, fh)                                   # NaN on purpose, like old build_real output
    sc = {"format": "racket_variant/1", "variant": "rigid", "clip_id": "synclip", "time_base": "analysis",
          "frame_count": F, "players": [{"id": p["id"], "which_hand": "right",
                                          "frames": [dict(fr["racket"]) for fr in p["frames"]]} for p in bps]}
    with open(os.path.join(adir, "synclip.racket_rigid.json"), "w") as fh:
        json.dump(sc, fh)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("p24")
    old = os.getcwd()
    os.chdir(root)
    try:
        views = ("a", "b", "c", "d")
        players = {"P1": syn.motion(F, x0=-0.5, z0=-3.0, side_bend=15.0), "P2": syn.motion(F, x0=0.5, z0=3.0)}
        entry = syn.write_clip(str(root), "synclip", players, syn.rig(views), views, shifts={"c": 2})
        reg = {"synclip": entry, "synclip_2cam": {"base": "synclip", "views": ["b", "c"], "timeline_of": "synclip"}}
        adir = str(root / "analysis")
        os.makedirs(adir)
        _write_base(adir, players)
        kw = dict(analysis_dir=adir, cache_dir=str(root / "cache"), data_dir=str(root / "data"), registry=reg,
                  fit_opts={"passes": 1}, log=lambda *a: None)
        full = pb.build("synclip", **kw)
        sub = pb.build("synclip_2cam", **kw)
        yield {"root": root, "full": full, "sub": sub, "players": players, "adir": adir, "kw": kw}
    finally:
        os.chdir(old)


def _load(path):
    return json.loads(open(path, encoding="utf-8").read(), parse_constant=lambda c: pytest.fail(f"{c} in JSON"))


def test_doc_validates_labelled_and_has_no_nan(built):
    d = _load(built["full"])
    assert schema24.validate_analysis24(d) == []
    assert d["skeleton"]["type"] == "smpl-28" and d["meta"]["label"] == "SMPL-28 · triangulated · 4 cams (a, b, c, d)"
    assert [c["name"] for c in d["meta"]["cameras"]] == ["a", "b", "c", "d"]
    pvm = d["meta"]["pose_variant"]
    assert pvm["name"] == "pose24" and pvm["view_health_checked"] is True and pvm["time_ref_view"] == "a"
    assert [o["view"] for o in pvm["offsets_list"]] == ["a", "b", "c", "d"]
    assert pvm["offsets_list"][2]["offset"] == pytest.approx(2.0, abs=0.35)
    assert set(d["meta"]["pose24"]["runtime_s"]) >= {"track", "lrswap", "fit", "total"}
    assert "head_heading" in d["players"][0]["frames"][60]["facing"]


def test_facing_heading_points_the_way_the_player_faces(built):
    """The synthetic players have real (mirror) handedness; P1 faces +z, P2 faces -z (review 1 M1)."""
    d = _load(built["full"])
    for pid, want in (("P1", 0.0), ("P2", np.pi)):
        p = next(p for p in d["players"] if p["id"] == pid)
        assert d["meta"]["pose24"]["players"][pid]["forward_sign"] == -1.0
        fr = [f["facing"] for f in p["frames"][10:140] if f["facing"]]
        assert len(fr) > 100
        for key in ("heading", "shoulder_heading", "head_heading"):
            dev = np.abs(np.remainder(np.array([f[key] for f in fr]) - want + np.pi, 2 * np.pi) - np.pi)
            assert np.median(dev) < 0.35, (pid, key, np.median(dev))


def test_pose_matches_truth(built):
    d = _load(built["full"])
    p = next(p for p in d["players"] if p["id"] == "P1")
    X = built["players"]["P1"]
    names = d["skeleton"]["joint_names"]
    errs = []
    for f in range(10, 140):
        fr = p["frames"][f]
        a = np.array(fr["joints_flat"]).reshape(-1, 4)
        w = a[:, :3] + [fr["position"][0], 0.0, fr["position"][1]]
        errs.append(np.linalg.norm(w[names.index("right_wrist")] - X[f, 10]))
    assert np.median(errs) < 0.03
    conf = np.array([np.array(fr["joints_flat"]).reshape(-1, 4)[:, 3] for fr in p["frames"][10:140]])
    assert np.median(conf[:, names.index("left_knee")]) > 0.5


def test_racket_carried_by_position_not_id(built):
    d = _load(built["full"])
    p1 = next(p for p in d["players"] if p["id"] == "P1")          # near player = base "P2"
    X = built["players"]["P1"]
    fr = p1["frames"][50]
    hip = 0.5 * (X[50, 11] + X[50, 12])
    assert fr["racket"]["grip"][0] + fr["position"][0] == pytest.approx(0.3 + hip[0], abs=1e-3)
    assert p1["frames"][5]["stroke"]["label"] == "smash"
    assert d["shuttle"]["frames"][0]["position"][1] is None
    sc = _load(os.path.join(built["adir"], "synclip_pose24.racket_rigid.json"))
    assert [q["id"] for q in sc["players"]] == [q["id"] for q in d["players"]] and sc["frame_count"] == F
    g = sc["players"][[q["id"] for q in d["players"]].index("P1")]["frames"][50]["grip"]
    assert g[0] + fr["position"][0] == pytest.approx(0.3 + hip[0], abs=1e-3)


def test_debug_cache_keeps_cv_metadata(built):
    c = docmod.load_debug_cache(track.cache_path("synclip", built["kw"]["cache_dir"]))
    x = c["extra"]["P1"]
    assert x["inlier_mask"].shape == (F, 35, 4) and x["resid_px"].dtype == np.float16
    assert x["sigma_mm"].shape == (F, 28) and x["filled"].shape == (F, 28) and x["X"].shape == (F, 35, 3)
    assert x["det_index"].shape == (4, F) and x["lr_swapped"].shape == (4, F) and x["contact"].shape == (F, 6)
    assert c["format"] == "pose24_cache/1" and c["time_ref_view"] == "a" and "d" in c["offsets_all"]


def test_filled_joints_flagged(built):
    c = docmod.load_debug_cache(track.cache_path("synclip", built["kw"]["cache_dir"]))
    assert c["extra"]["P1"]["filled"][20:130].mean() < 0.1       # 4 clean views: almost nothing unseen


def test_subset_build_on_full_timeline(built):
    s = _load(built["sub"])
    assert schema24.validate_analysis24(s) == []
    pvm = s["meta"]["pose_variant"]
    assert s["meta"]["frame_count"] == F and pvm["time_ref_view"] == "a" and pvm["timeline_of"] == "synclip"
    assert pvm["view_health_checked"] is False
    assert s["meta"]["label"] == "SMPL-28 · triangulated · 2 cams (b, c)"
    assert all(fr["racket"] is None for p in s["players"] for fr in p["frames"])
    assert not os.path.exists(os.path.join(built["adir"], "synclip_2cam_pose24.racket_rigid.json"))


def test_base_doc_on_another_timeline_is_skipped_with_a_log_line(built, tmp_path):
    """doubles_2sides_4angle_2cam: its own base doc is on the old 2-view sync timeline; with --timeline-of it
    is not carried, and the build says so."""
    adir = str(tmp_path)
    with open(os.path.join(adir, "synclip_2cam.analysis.json"), "w") as fh:
        json.dump({"schema_version": "4.0", "meta": {"frame_count": F}, "players": []}, fh)
    lines = []
    out = pb.build("synclip_2cam", **dict(built["kw"], analysis_dir=adir, log=lines.append))
    assert any("synclip_2cam.analysis.json" in ln and "timeline" in ln and "not carried" in ln for ln in lines)
    assert _load(out)["meta"]["pose_variant"]["base"] is None


def test_base_doc_synced_to_another_reference_view_is_not_carried(built, tmp_path):
    """Review 1 L13: the base doc's sync reference (sorted-first base camera, here "b") differs from this build's
    time_ref_view ("a"): frames are on different timelines, so nothing is copied by frame index."""
    adir = str(tmp_path / "analysis")
    os.makedirs(adir)
    _write_base(adir, built["players"])
    path = os.path.join(adir, "synclip.analysis.json")
    with open(path) as fh:
        base = json.load(fh)
    base["meta"]["cameras"] = [{"name": v} for v in ("c", "b", "d")]
    with open(path, "w") as fh:
        json.dump(base, fh)
    lines = []
    out = pb.build("synclip", **dict(built["kw"], analysis_dir=adir, cache_dir=str(tmp_path / "cache"),
                                     log=lines.append))
    d = _load(out)
    assert d["meta"]["pose_variant"]["base"] is None and d["meta"]["pose_variant"]["time_ref_view"] == "a"
    assert all(fr.get("racket") is None for p in d["players"] for fr in p["frames"])
    assert not os.path.exists(os.path.join(adir, "synclip_pose24.racket_rigid.json"))
    assert any("synced to b" in ln and "not carried" in ln for ln in lines)


def test_rebuild_without_base_removes_stale_sidecars(built):
    """A subset build carries no racket: a sidecar left by an earlier build (e.g. one that had a base doc)
    must not be served next to the new doc."""
    stale = os.path.join(built["adir"], "synclip_2cam_pose24.racket_bootstrap.json")
    with open(stale, "w") as fh:
        fh.write("{}")
    pb.build("synclip_2cam", **built["kw"])
    assert not os.path.exists(stale)


def test_doc_written_last(built, monkeypatch, tmp_path):
    """A failing debug-cache write leaves no new doc behind (a failed build never writes a doc without its cache;
    the doc / cache pair is checked by meta.pose24.build_id)."""
    kw = dict(built["kw"], analysis_dir=str(tmp_path))

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(docmod, "save_debug_cache", boom)
    with pytest.raises(OSError):
        pb.build("synclip_2cam", **kw)
    assert not os.path.exists(pb.output_path(str(tmp_path), "synclip_2cam"))
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def _fake_registry(monkeypatch, clips):
    from analysis.pose24 import config
    monkeypatch.setattr(config, "load_registry", lambda *a, **k: {c: {"mode": "singles", "views": ["a", "b"]}
                                                                   for c in clips})


def test_cli_build_all_isolates_a_failing_clip(monkeypatch, capsys):
    from analysis.pose24 import __main__ as cli
    _fake_registry(monkeypatch, ["c1", "c2", "c3"])
    calls = []

    def fake_build(c, **kw):
        calls.append(c)
        if c == "c2":
            raise RuntimeError("no seeds for c2")
        return c
    monkeypatch.setattr(pb, "build", fake_build)
    assert cli.main(["build", "--all"]) == 1
    assert calls == ["c1", "c2", "c3"]
    cap = capsys.readouterr()
    assert "no seeds for c2" in cap.err and "2/3 ok" in cap.out and "failed: c2" in cap.out


def test_cli_eval_all_skips_unbuilt_and_isolates_failures(monkeypatch, capsys, tmp_path):
    from analysis.pose24 import __main__ as cli
    from analysis.pose24 import eval as ev
    monkeypatch.chdir(tmp_path)
    _fake_registry(monkeypatch, ["c1", "c2", "c3"])
    adir = str(tmp_path / "analysis")
    for c in ("c1", "c3"):                      # c2 was never built
        os.makedirs(adir, exist_ok=True)
        open(pb.output_path(adir, c), "w").close()
        os.makedirs(os.path.dirname(track.cache_path(c)), exist_ok=True)
        open(track.cache_path(c), "w").close()
    calls = []

    def fake_eval(c, **kw):
        calls.append(c)
        if c == "c1":
            raise ValueError("bad cache c1")
        return {}
    monkeypatch.setattr(ev, "evaluate", fake_eval)
    assert cli.main(["eval", "--all", "--analysis-dir", adir, "--no-posefit"]) == 1
    assert calls == ["c1", "c3"]
    cap = capsys.readouterr()
    assert "c2: skipped" in cap.out and "bad cache c1" in cap.err and "failed: c1" in cap.out


def test_cli_list_shows_registry_entries_and_outputs(built, capsys, monkeypatch):
    from analysis.pose24 import __main__ as cli
    from analysis.pose24 import config
    monkeypatch.setattr(config, "load_registry", lambda *a, **k: built["kw"]["registry"])
    assert cli.main(["list", "--analysis-dir", built["adir"]]) == 0
    rows = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines()}
    assert set(rows) == {"synclip", "synclip_2cam"}
    assert " doc " in rows["synclip"] and "4 views" in rows["synclip"]
    assert "subset of synclip" in rows["synclip_2cam"] and "2 views" in rows["synclip_2cam"]


def test_eval_subset_against_reference(built):
    from analysis.pose24 import eval as ev
    kw = built["kw"]
    res = ev.evaluate("synclip_2cam", reference="synclip", posefit=False, analysis_dir=kw["analysis_dir"],
                      cache_dir=kw["cache_dir"], out_dir=str(built["root"] / "eval"), log=lambda *a: None)
    assert set(res["heldout"]) == {"a", "d"} and res["heldout"]["a"]["px"]["median"] < 15
    vr = res["vs_reference"]
    assert vr["w_mpjpe_cm"]["median"] < 5 and vr["pelvis_xz_cm"]["median"] < 3
    assert "left_knee_flexion" in vr["angle_mae_deg"]
    assert res["reliability"]["source"] == "heldout errors vs build sigma"
    assert (built["root"] / "eval" / "synclip_2cam.json").exists()


def test_eval_lovo_on_full_build(built):
    from analysis.pose24 import eval as ev
    kw = built["kw"]
    res = ev.evaluate("synclip", lovo=True, posefit=False, analysis_dir=kw["analysis_dir"], cache_dir=kw["cache_dir"],
                      out_dir=str(built["root"] / "eval"), fit_opts={"passes": 1}, log=lambda *a: None)
    assert set(res["lovo"]) == {"a", "b", "c", "d"} and all(r["px"]["median"] < 15 for r in res["lovo"].values())
    assert res["players"]["P1"]["pose24"]["coverage"] > 0.95
    assert res["reliability"]["source"] == "lovo refit errors vs full-build sigma"   # sigma included view v


def test_doc_and_cache_share_a_build_id(built):
    d = _load(built["full"])
    c = docmod.load_debug_cache(track.cache_path("synclip", built["kw"]["cache_dir"]))
    assert len(d["meta"]["pose24"]["build_id"]) == 32 and c["build_id"] == d["meta"]["pose24"]["build_id"]
    s = _load(built["sub"])
    assert s["meta"]["pose24"]["build_id"] != d["meta"]["pose24"]["build_id"]


def test_every_view_failing_view_health_is_loud(built, tmp_path, monkeypatch):
    """A sync reference with a bad calibration makes every view disagree, so nothing is dropped (ledger T5):
    the build must say so and mark the doc."""
    from analysis import pose_v2 as pv
    monkeypatch.setattr(pv, "view_health", lambda cams, views, obs: {v: 99.0 for v in views})
    lines = []
    kw = dict(built["kw"], analysis_dir=str(tmp_path / "a"), cache_dir=str(tmp_path / "c"), log=lines.append)
    os.makedirs(kw["analysis_dir"])
    d = _load(pb.build("synclip", **kw))
    assert any("WARNING" in ln and "every view" in ln for ln in lines)
    assert d["meta"]["pose_variant"]["view_health_suspect"] is True and d["meta"]["pose_variant"]["excluded_views"] == {}
    assert _load(built["full"])["meta"]["pose_variant"]["view_health_suspect"] is False


def test_debug_cache_refuses_ids_that_collide_with_its_keys(tmp_path):
    with pytest.raises(ValueError, match="_"):
        docmod.save_debug_cache(str(tmp_path / "x.npz"), {"obs": {"P_1": {}}}, {}, {})


def _swap_ids(pid):
    return {"P1": "P2", "P2": "P1"}.get(pid, pid)


def test_eval_posefit_row_matched_by_position_not_id(built, tmp_path):
    """The posefit doc + its pose_v2 cache use their own player ids (here swapped): the posefit row must land on
    the pose24 player it describes (review L3), not on the one with the same id."""
    from analysis.pose24 import eval as ev
    kw = built["kw"]
    adir, cdir = str(tmp_path / "analysis"), str(tmp_path / "cache")
    os.makedirs(adir), os.makedirs(cdir)
    d = _load(built["full"])
    for p in d["players"]:
        p["id"] = _swap_ids(p["id"])
    with open(os.path.join(adir, "synclip_posefit.analysis.json"), "w") as fh:
        json.dump(d, fh)
    with open(pb.output_path(adir, "synclip"), "w") as fh:
        fh.write(open(built["full"], encoding="utf-8").read())
    src = track.cache_path("synclip", kw["cache_dir"])
    with np.load(src) as z:
        arrs = {k: z[k] for k in z.files}

    def rename(k):
        pid, _, rest = k.partition("_")
        return f"{_swap_ids(pid)}_{rest}" if rest and pid in ("P1", "P2") else k
    np.savez(os.path.join(cdir, "synclip.npz"), **{rename(k): v for k, v in arrs.items()})   # pose_v2-style, swapped
    np.savez(track.cache_path("synclip", cdir), **arrs)
    res = ev.evaluate("synclip", posefit=True, analysis_dir=adir, cache_dir=cdir, out_dir=str(tmp_path / "eval"),
                      log=lambda *a: None)
    for pid in ("P1", "P2"):
        row, ref = res["players"][pid]["posefit"], res["players"][pid]["pose24"]
        assert row["reproj_px"] == pytest.approx(ref["reproj_px"], rel=1e-6)
        assert row["coverage"] == pytest.approx(ref["coverage"])


def test_calibrate_scores_keeps_defaults_when_scores_carry_no_information(built, tmp_path):
    from analysis.pose24 import calibrate
    b = docmod.load_debug_cache(track.cache_path("synclip", built["kw"]["cache_dir"]))
    raw, y, g = calibrate.heldout_samples(b, b["score_params"])
    assert len(raw) >= 500 and len(raw) == len(y) == len(g) and set(np.unique(g)) >= {0}
    out = tmp_path / "cal.json"
    doc = calibrate.calibrate_scores(["synclip"], built["kw"]["cache_dir"], str(out), log=lambda *a: None)
    assert out.exists() and doc["groups"]["body"]["kept"]              # synthetic raw scores are all 4.5


def test_player_seen_from_one_end_keeps_facing_and_nonzero_confidence(tmp_path, monkeypatch):
    """Real singles far player seen only by the back cameras: hip sigma ~150 mm (conf ~0.02 < facing's 0.1).
    Three cameras along one baseline: the far player's honest confidences are small but not placeholders,
    and facing still comes out (pose24 passes a 1e-3 facing threshold)."""
    from analysis import multiview as mv
    monkeypatch.chdir(tmp_path)
    views = ("a", "b", "c")
    K = mv.intrinsics_from_fov(syn.W, syn.H, 70.0)
    cams = {v: (K,) + syn.look_at([x, -3.0, -11.0]) for v, x in zip(views, (-3.5, 0.0, 3.5))}
    n = 90
    players = {"P1": syn.motion(n, x0=-0.5, z0=-3.0), "P2": syn.motion(n, x0=0.5, z0=5.0)}
    entry = syn.write_clip(str(tmp_path), "oneend", players, cams, views)
    adir = str(tmp_path / "analysis")
    os.makedirs(adir)
    out = pb.build("oneend", analysis_dir=adir, cache_dir=str(tmp_path / "cache"), data_dir=str(tmp_path / "data"),
                   registry={"oneend": entry}, fit_opts={"passes": 1}, log=lambda *a: None)
    d = _load(out)
    far = next(p for p in d["players"] if p["id"] == "P2")
    posed = [fr for fr in far["frames"] if fr["joints_flat"] is not None]
    assert len(posed) > 0.9 * n
    conf = np.array([np.array(fr["joints_flat"]).reshape(-1, 4)[:, 3] for fr in posed])
    assert (conf > 0).all() and np.median(conf[:, 1:3]) < 0.1          # honest, small, not placeholders
    assert sum(fr["facing"] is not None for fr in far["frames"]) > 0.8 * n


def test_cli_rejects_all_with_timeline_of(monkeypatch):
    from analysis.pose24 import __main__ as cli
    _fake_registry(monkeypatch, ["c1"])
    with pytest.raises(SystemExit, match="timeline-of"):
        cli.main(["build", "--all", "--timeline-of", "c1"])


def test_cli_warns_when_lovo_is_ignored(monkeypatch, capsys):
    from analysis.pose24 import __main__ as cli
    from analysis.pose24 import eval as ev
    _fake_registry(monkeypatch, ["c1", "c2"])
    seen = {}
    monkeypatch.setattr(ev, "evaluate", lambda c, **kw: seen.update(kw) or {})
    assert cli.main(["eval", "--clip", "c2", "--reference", "c1", "--lovo", "--no-posefit"]) == 0
    assert seen["lovo"] is False and "--lovo" in capsys.readouterr().out


def test_cli_calibrate_clips_are_stripped_and_not_empty(monkeypatch):
    from analysis.pose24 import __main__ as cli
    from analysis.pose24 import calibrate
    seen = []
    monkeypatch.setattr(calibrate, "calibrate_scores", lambda clips, out_path=None, **k: seen.append(clips))
    assert cli.main(["calibrate-scores", "--clips", " a , b,,", "--out", "x.json"]) == 0
    assert seen == [["a", "b"]]
    with pytest.raises(SystemExit):
        cli.main(["calibrate-scores", "--clips", " , "])


def test_calibrate_reads_raw_scores_from_kp2d_not_by_inverting(built, tmp_path):
    """review 1 L5: inverting the cached float32 probabilities saturates every raw score above ~6.9. The held-out
    samples take the detector's own raw score from kp2d via det_index (here rewritten to 9.0)."""
    from analysis.pose24 import calibrate, scores
    b = docmod.load_debug_cache(track.cache_path("synclip", built["kw"]["cache_dir"]))
    assert scores.to_raw(scores.calibrate(np.full((1, 133), 9.0), b["score_params"]).astype(np.float32),
                         b["score_params"])[0, 0] < 7.0                      # the old path: saturated
    src = os.path.join(built["kw"]["data_dir"], "pose2d")
    dst = tmp_path / "data" / "pose2d"
    dst.mkdir(parents=True)
    for fn in os.listdir(src):
        d = json.load(open(os.path.join(src, fn), encoding="utf-8"))
        for fr in d["frames"]:
            for det in fr:
                for kp in det["keypoints"]:
                    kp[2] = 9.0
        (dst / fn).write_text(json.dumps(d), encoding="utf-8")
    raw = calibrate.kp2d_raw_scores(b, str(tmp_path / "data"))
    assert set(raw) == set(b["obs"])
    r = raw["P1"]
    assert r.shape == (len(b["obs"]["P1"][b["views"][0]]), 133, len(b["views"]))
    assert np.nanmin(r) == 9.0 and np.isfinite(r).mean() > 0.9
    got, y, g = calibrate.heldout_samples(b, b["score_params"], raw=raw)
    assert len(got) >= 500 and np.all(got == 9.0)
    assert calibrate.kp2d_raw_scores(b, str(tmp_path / "nowhere")) is None       # no kp2d: fall back to inversion
