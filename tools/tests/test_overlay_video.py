import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

import cvdebug_synth as cs
from analysis import cvdebug as cd
from analysis import overlay_video as ov
from analysis import video_io

S = 0.25
HAS_FFMPEG = shutil.which("ffmpeg") is not None
TILE = (int(3840 * S), int(2160 * S))            # (h, w) of a 2160x3840 view at scale 0.25


@pytest.fixture(scope="module")
def vb(tmp_path_factory):
    """Synthetic 4-view clip; only view c (2 frames late) gets a full-size numbered MJPG video: the overlay checks it
    against the kp2d header (size + frame count). a, b, d have no video (-> grey tiles). Writing all four full-size
    videos would take ~60 s."""
    root = tmp_path_factory.mktemp("ovl")
    b = cs.built_clip(root, F=150)
    vid = os.path.join(str(root), "data", "video", "c.avi")
    cs.write_numbered_video(vid, b["F"], 2160, 3840)
    b["reg"]["syn"]["videos"]["c"] = vid
    court = {"corner_nl": {"px": [0.0, 0.0], "court_xz": [-3.05, -6.7]}}
    for v in ("a", "b", "c", "d"):               # one clicked court point per view (calib layer)
        cp = os.path.join(b["data_dir"], "calib", f"{v}_cam.json")
        with open(cp, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        c["court_points"] = court
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(c, fh)
    return b


def _ctx(b, **kw):
    return ov.OverlayContext("syn", "pose24", analysis_dir=b["adir"], cache_dir=b["cache_dir"],
                             data_dir=b["data_dir"], registry=b["reg"], log=lambda *a: None, **kw)


def _black():
    return np.zeros(TILE + (3,), np.uint8)


def test_parse_layers_and_frames():
    assert ov.parse_layers("dets,kp,racket=rigid") == {"dets": None, "kp": None, "racket": "rigid"}
    with pytest.raises(ValueError, match="unknown layer"):
        ov.parse_layers("dets,bogus")
    assert ov.parse_layers(ov.DEFAULT_LAYERS).keys() <= set(ov.LAYERS)
    assert ov.parse_frames("5:8", 100) == [5, 6, 7] and ov.parse_frames("95:200", 100) == list(range(95, 100))
    assert ov.parse_frames(":3", 100) == [0, 1, 2]


def test_context_loads_sidecar_cameras_and_world(vb):
    ctx = _ctx(vb, views="b,c")
    assert ctx.names == ["a", "b", "c", "d"] and ctx.views == [1, 2]
    assert set(ctx.world) == {"P1", "P2"} and ctx.F == ctx.doc["meta"]["frame_count"] == vb["F"] - 2
    assert len(ctx.parents) == len(ctx.sc["joint_names"]) and ctx.parents[0] == -1
    assert len(ctx.court_pts["c"]) == 1


def test_render_view_draws_the_projected_skeleton(vb):
    ctx = _ctx(vb, scale=S)
    ci = ctx.names.index("b")
    img = ov.render_view(ctx, 10, ci, _black(), ov.parse_layers("skel"))
    J, _ = ctx.world["P1"]
    uv = cd.project(ctx.cams[ci], J[10, :1])[0] * S
    x, y = int(round(uv[0])), int(round(uv[1]))
    assert img[max(0, y - 3):y + 4, max(0, x - 3):x + 4].any()          # pelvis dot drawn where it projects
    full = ov.render_view(ctx, 10, ci, _black(), ov.parse_layers(ov.DEFAULT_LAYERS))
    assert full.sum() > img.sum()


def test_every_layer_draws_something_without_crashing(vb):
    ctx = _ctx(vb)
    ci = ctx.names.index("b")
    for name in ("dets", "kp", "skel", "resid", "bbox", "court", "text", "aux"):
        img = ov.render_view(ctx, 20, ci, _black(), ov.parse_layers(name))
        assert img.any(), name
    # missing racket / shuttle sidecars: nothing drawn, no crash
    assert not ov.render_view(ctx, 20, ci, _black(), ov.parse_layers("racket=rigid,shuttle")).any()


def test_calib_layer_marks_the_clicked_point_and_its_residual(vb):
    ctx = _ctx(vb)
    ci = ctx.names.index("b")
    img = ov.render_view(ctx, 5, ci, _black(), ov.parse_layers("calib"))
    assert img[:6, :6].any()                     # the clicked point (px 0,0) is marked


def test_compare_variant_adds_a_white_skeleton(vb):
    ctx = _ctx(vb, compare="pose24")             # the doc against itself: white skeleton on top
    ci = ctx.names.index("b")
    img = ov.render_view(ctx, 10, ci, _black(), ov.parse_layers("skel"))
    assert (img == 255).all(axis=2).any() and ctx.cmp_label == "syn pose24"


def test_view_tile_reads_the_synced_raw_frame(vb):
    ctx = _ctx(vb)
    ci = ctx.names.index("c")
    assert ctx.sc["cameras"][ci]["offset"] == pytest.approx(2.0, abs=0.49)   # sync recovered c's +2
    idx, rem = ov.raw_index(ctx, 10, ci)
    assert idx == 12 and abs(rem) <= 0.5
    readers = ctx.readers()
    assert readers[ctx.names.index("a")] is None            # no video for a
    frame = readers[ci].get(12)
    assert frame.shape[:2] == TILE and cs.read_number(frame) == 12
    tile = ov.view_tile(ctx, 10, ci, {}, readers=readers)   # no layers: the raw frame itself
    assert cs.read_number(tile) == 12


def test_frame_outside_video_gives_grey_tile(vb):
    ctx = _ctx(vb)
    ci = ctx.names.index("c")
    n = ctx.headers["c"]["frame_count"]
    ctx.sc["cameras"][ci]["offset"] = 5.0        # raw = f + 5: the doc's last frame lies past the video's end
    f = ctx.F - 1
    assert ov.raw_index(ctx, f, ci)[0] >= n
    tile = ov.view_tile(ctx, f, ci, ov.parse_layers("text"), readers=ctx.readers())
    assert tile.shape[:2] == TILE and tile.mean() > 20       # grey + text, not a crash
    no_video = ov.view_tile(ctx, 5, ctx.names.index("a"), ov.parse_layers("text"), readers=ctx.readers())
    assert no_video.shape[:2] == TILE and no_video.mean() > 20


def test_video_size_mismatch_is_an_error_naming_the_view(vb, tmp_path):
    ctx = _ctx(vb, views="d")
    small = cs.write_numbered_video(str(tmp_path / "d_small.avi"), vb["F"], 216, 384)
    ctx.sc["cameras"][ctx.names.index("d")]["video"] = small
    with pytest.raises(video_io.VideoError, match="d: decoded frames are 216x384"):
        ctx.readers()


def test_excluded_view_banner_drawn(vb):
    ctx = _ctx(vb)
    ci = ctx.names.index("d")
    ctx.sc["cameras"][ci]["excluded"], ctx.sc["cameras"][ci]["health_px"] = True, 927.0
    img = ov.render_view(ctx, 5, ci, _black(), ov.parse_layers("text"))
    assert (img[2:10, :, 2] > 200).mean() > 0.5          # red banner across the top


def test_stale_sidecar_is_rebuilt_in_memory(vb):
    path, sc = cd.build("syn", "pose24", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                        data_dir=vb["data_dir"], registry=vb["reg"], log=lambda *a: None)
    try:
        ctx = _ctx(vb)
        assert ctx.sc == sc                           # fresh sidecar on disk: used as is
        stale = dict(sc, source_build_id="old")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stale, fh)
        logs = []
        ctx = ov.OverlayContext("syn", "pose24", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                                data_dir=vb["data_dir"], registry=vb["reg"], log=logs.append)
        assert ctx.sc["source_build_id"] == sc["source_build_id"] and any("stale" in s for s in logs)
        with open(path, "r", encoding="utf-8") as fh:
            assert json.load(fh)["source_build_id"] == "old"   # rebuilt in memory only: the file is not touched
        old_format = json.loads(json.dumps(sc))              # same build, written before cameras had det_shift
        for c in old_format["cameras"]:
            del c["det_shift"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(old_format, fh)
        logs = []
        ctx = ov.OverlayContext("syn", "pose24", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                                data_dir=vb["data_dir"], registry=vb["reg"], log=logs.append)
        assert "det_shift" in ctx.sc["cameras"][0] and any("older" in s for s in logs)
    finally:
        os.remove(path)


def test_detection_layers_read_the_trackers_raw_frame(vb):
    """offset exactly +0.5: the video frame is f + 1 (floor(f + o + 0.5)) but pose24 det_index points into raw kp2d
    frame f + det_shift = f + 0 (the tracker's np.round). Detection order differs between the two raw frames."""
    ctx = _ctx(vb)
    ci, f = ctx.names.index("b"), 20
    cam = ctx.sc["cameras"][ci]
    cam["offset"], cam["det_shift"] = 0.5, 0
    raw = ctx.raw["b"]
    raw[f + 1] = raw[f][::-1].copy()                          # same people, swapped order in the next raw frame
    raw[f + 1][..., :2] += 300.0                              # ... and elsewhere, so a wrong read is visible
    assert ov.raw_index(ctx, f, ci)[0] == f                   # equidistant raw frames: follow the tracker (f + 0)
    cam["det_shift"] = 1                                      # the tracker's other choice: the video follows it
    assert ov.raw_index(ctx, f, ci) == (f + 1, -0.5) and ov.det_frame(ctx, f, ci) == f + 1
    assert not any("2D from raw" in s for s in ov._text_lines(ctx, f, ci))
    cam["offset"] = 2.5                                       # det_shift 1 is not one of f+2 / f+3: floor(f+o+.5)
    assert ov.raw_index(ctx, f, ci)[0] == f + 3 and any(f"2D from raw {f + 1}" in s for s in ov._text_lines(ctx, f, ci))
    cam["offset"], cam["det_shift"] = 0.5, 0
    dets, assigned = ov._detections(ctx, f, ci)
    assert np.array_equal(dets, raw[f])
    di = ctx.splayer["P1"]["det_index"][ci * ctx.F + f]
    assert di >= 0 and assigned[di] == 0
    kp = dets[di]
    img = ov.render_view(ctx, f, ci, _black(), ov.parse_layers("kp"))
    ok = kp[:17, 2] >= 0.3
    x, y = (kp[:17][ok, :2].mean(axis=0) * S).astype(int)
    assert img[y - 40:y + 40, x - 40:x + 40].any()           # P1's keypoints drawn from raw frame f
    assert not any("2D from raw" in s for s in ov._text_lines(ctx, f, ci))    # video and 2D: the same raw frame
    assert not any("2D from raw" in s for s in ov._text_lines(ctx, f, ctx.names.index("a")))


def _fake_ctx(offset, det_shift):
    from types import SimpleNamespace
    return SimpleNamespace(sc={"cameras": [{"offset": offset, "det_shift": det_shift}]})


@pytest.mark.parametrize("offset, det_shift, want", [
    (0.5, 0, 7), (0.5, 1, 8), (0.3, 0, 7), (0.3, 1, 7), (0.7, 0, 8), (-0.5, -1, 6), (-0.5, 0, 7), (0.5, 3, 8),
    (1.5, 1, 8), (1.5, 2, 9)])
def test_raw_index_on_an_exact_half_follows_the_trackers_frame(offset, det_shift, want):
    idx, rem = ov.raw_index(_fake_ctx(offset, det_shift), 7, 0)
    assert idx == want and idx + rem == pytest.approx(7 + offset)
    assert video_io.view_frame(7, offset, det_shift=det_shift)[0] == want
    assert video_io.view_frame(7, 0.3) == video_io.view_frame(7, 0.3, det_shift=None)


def test_unknown_view_name_lists_the_cameras(vb):
    with pytest.raises(ValueError, match=r"unknown view 'zz'.*a, b, c, d"):
        _ctx(vb, views="b,zz")


def test_compare_on_another_timeline_warns(vb):
    logs = []
    ov.OverlayContext("syn", "pose24", compare="pose24", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                      data_dir=vb["data_dir"], registry=vb["reg"], log=logs.append)
    assert not any("WARNING" in s for s in logs)                # same doc: same timeline
    d = json.load(open(cd.doc_path(vb["adir"], "syn", "pose24"), encoding="utf-8"))
    d["meta"]["pose_variant"]["offsets"]["c"] += 0.5
    p = cd.doc_path(vb["adir"], "syn", "cmpx")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    try:
        logs = []
        ov.OverlayContext("syn", "pose24", compare="cmpx", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                          data_dir=vb["data_dir"], registry=vb["reg"], log=logs.append)
        warn = [s for s in logs if "WARNING" in s]
        assert len(warn) == 1 and "syn pose24" in warn[0] and "syn cmpx" in warn[0] and "c" in warn[0]
        d["meta"]["pose_variant"]["offsets"]["c"] -= 0.5
        d["meta"]["pose_variant"]["time_ref_view"] = "zz"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        logs = []
        ov.OverlayContext("syn", "pose24", compare="cmpx", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"],
                          data_dir=vb["data_dir"], registry=vb["reg"], log=logs.append)
        assert any("WARNING" in s and "zz" in s for s in logs)
    finally:
        os.remove(p)


def test_missing_proxy_falls_back_to_the_checked_original(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "PROXY_DIR", str(tmp_path / "no_proxies"))
    checked = []
    real = video_io.check_video
    monkeypatch.setattr(ov.video_io, "check_video", lambda v, p, h, log=print: checked.append(v) or real(v, p, h, log))
    logs = []
    ctx = ov.OverlayContext("syn", "pose24", analysis_dir=vb["adir"], cache_dir=vb["cache_dir"], data_dir=vb["data_dir"],
                            registry=vb["reg"], views="c", source="proxy", log=logs.append)
    ci = ctx.names.index("c")
    r = ctx.readers()[ci]
    assert r is not None and r.path == ctx.sc["cameras"][ci]["video"] and r.scale == ctx.scale
    assert checked == ["c"] and any("proxy" in s and "original" in s for s in logs)
    r.close()


def _tile_ctx(sizes, views=None):
    from types import SimpleNamespace
    return SimpleNamespace(sc={"cameras": [{"name": f"v{i}", "width": w, "height": h} for i, (w, h) in enumerate(sizes)]},
                           views=list(range(len(sizes))) if views is None else views)


def test_mosaic_tile_follows_the_majority_orientation():
    P, L = (2160, 3840), (3840, 2160)
    assert ov.tile_size(_tile_ctx([P, L, L, (1920, 1080)]), 0.25) == (960, 540)     # landscape majority, portrait first
    assert ov.tile_size(_tile_ctx([L, P, P]), 0.25) == (540, 960)
    assert ov.tile_size(_tile_ctx([L, P]), 0.25) == (540, 960)                      # tie -> portrait
    assert ov.tile_size(_tile_ctx([P, (1080, 1920), (1920, 1080), (1920, 1080), (1920, 1080)]), 0.5) == (960, 540)
    assert ov.tile_size(_tile_ctx([P, L, L], views=[0]), 0.25) == (540, 960)        # only the selected views count
    assert ov.tile_size(_tile_ctx([P, L, L]), 0.25, "800x450") == (800, 450)        # --tile override
    with pytest.raises(ValueError, match="--tile"):
        ov.tile_size(_tile_ctx([P]), 0.25, "800")


def test_render_tile_override_sets_the_mosaic_size(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    rc = ov.main(_render_args(vb, "--clip", "syn", "--mode", "still", "--frame", "3", "--views", "a,b",
                              "--tile", "200x120", "--out", str(tmp_path)))
    img = cv2.imread(str(tmp_path / "pose24_still_3.png"))
    assert rc == 0 and img.shape[1] == 2 * 200 and img.shape[0] == 2 * 120 + 28


def test_aux_layer_fills_contact_flagged_feet(vb):
    ctx = _ctx(vb)
    ci, f = ctx.names.index("b"), 30
    sp = ctx.splayer["P1"]
    sp["contact"][f * 6:(f + 1) * 6] = [0] * 6
    hollow = ov.render_view(ctx, f, ci, _black(), ov.parse_layers("aux"))
    sp["contact"][f * 6:(f + 1) * 6] = [1] * 6
    filled = ov.render_view(ctx, f, ci, _black(), ov.parse_layers("aux"))
    assert (filled > 0).any(axis=2).sum() > (hollow > 0).any(axis=2).sum()


def test_text_is_ascii_even_when_the_doc_label_is_not(vb):
    ctx = _ctx(vb)
    ctx.doc["meta"]["label"] = "SMPL-28 · triangulated — 4 cams"
    lines = ov._text_lines(ctx, 5, 0)
    assert all(s.isascii() for s in lines) and "SMPL-28 | triangulated - 4 cams" in lines[0]


def test_court_and_stats_tiles(vb):
    ctx = _ctx(vb)
    court = ov.court_tile(ctx, 5, (540, 960))
    assert court.shape == (960, 540, 3)
    assert (court == np.array(ov.draw.player_bgr(0), np.uint8)).all(axis=2).any()   # P1's track / dot
    stats = ov.stats_tile(ctx, 5, (540, 960))
    assert stats.shape == (960, 540, 3) and stats.any()
    shuttle = {"frames": [{"position": [0.0, 2.0, -1.0]}] * ctx.F}
    ctx._shuttle["tracker"] = shuttle
    img = ov.render_view(ctx, 5, ctx.names.index("b"), _black(), ov.parse_layers("shuttle"))
    assert (img == np.array(ov.draw.SHUTTLE_BGR, np.uint8)).all(axis=2).any()
    blank = ov.blank_tile((540, 960), "x")
    assert blank.shape == (960, 540, 3) and blank.mean() > 20


# --- Task 8: modes, crop, ffmpeg writer, output guard, render CLI ---


def _render_args(vb, *extra):
    return ["render", *extra, "--analysis-dir", vb["adir"], "--cache-dir", vb["cache_dir"],
            "--data-dir", vb["data_dir"]]


@pytest.mark.parametrize("n,layout", [(5, (2, 3)), (4, (2, 3)), (2, (2, 2)), (3, (2, 2))])
def test_mosaic_layout(n, layout):
    assert ov.mosaic_layout(n) == layout


def test_compose_mosaic_places_tiles():
    t = [np.full((40, 20, 3), 50 * (i + 1), np.uint8) for i in range(5)]
    img = ov.compose_mosaic(t, [np.full((40, 20, 3), 255, np.uint8)], (2, 3), (20, 40))
    assert img.shape[1] == 60 and img.shape[0] >= 80
    assert (img[0:40, 20:40] == 100).all() and (img[40:80, 40:60] == 255).all()
    assert img.shape[0] == 80 + 28                            # legend strip under the grid


def test_crop_box_keeps_aspect_and_stays_inside():
    x0, y0, x1, y1 = ov.crop_box(np.array([[100.0, 200.0], [140.0, 400.0]]), (540, 960))
    assert 0 <= x0 < x1 <= 540 and 0 <= y0 < y1 <= 960
    assert (x1 - x0) / (y1 - y0) == pytest.approx(9 / 16, rel=0.05)
    assert ov.crop_box(np.full((3, 2), np.nan), (540, 960)) == (0, 0, 540, 960)   # player not visible: whole tile


def test_crop_box_follows_a_landscape_aspect():
    x0, y0, x1, y1 = ov.crop_box(np.array([[400.0, 200.0], [440.0, 300.0]]), (960, 540), aspect=(960, 540))
    assert 0 <= x0 < x1 <= 960 and 0 <= y0 < y1 <= 540
    assert (x1 - x0) / (y1 - y0) == pytest.approx(16 / 9, rel=0.05)


def test_crop_tile_is_not_stretched_on_a_landscape_view():
    img = np.zeros((540, 960, 3), np.uint8)
    cv2.circle(img, (420, 250), 20, (255, 255, 255), -1)        # a round blob stays round after crop + resize
    pts = np.array([[400.0, 200.0], [440.0, 300.0]])
    t = ov._crop_tile(img, pts, (960, 540))
    assert t.shape == (540, 960, 3)
    ys, xs = np.nonzero(t[..., 0] > 127)
    assert (xs.max() - xs.min()) / (ys.max() - ys.min()) == pytest.approx(1.0, rel=0.1)


def test_compose_mosaic_letterboxes_a_tile_of_another_aspect():
    land = np.full((20, 40, 3), 255, np.uint8)                   # 2:1 landscape into a 1:2 portrait slot
    img = ov.compose_mosaic([land], [], (1, 1), (20, 40))
    slot = img[:40, :20]
    rows = np.nonzero((slot == 255).all(axis=(1, 2)))[0]
    assert len(rows) == 10 and rows.min() == 15                  # 20x10 band, centred: aspect kept
    assert (slot[:15] == 30).all() and (slot[25:] == 30).all()


def test_close_all_closes_every_writer_and_raises_the_first_error():
    closed = []

    class W:
        def __init__(self, k, fail):
            self.k, self.fail = k, fail

        def close(self):
            closed.append(self.k)
            if self.fail:
                raise RuntimeError(f"close {self.k}")
    ws = {k: W(k, k in ("b", "c")) for k in "abcd"}
    with pytest.raises(RuntimeError, match="close b"):
        ov._close_all(ws)
    assert closed == list("abcd")
    closed.clear()
    ov._close_all(ws, quiet=True)                                # an exception is already in flight: swallow
    assert closed == list("abcd")


def test_out_dir_guard(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for bad in ("Assets/StreamingAssets/overlays", "docs/img", "Assets"):
        with pytest.raises(ValueError, match="not allowed"):
            ov.check_out_dir(os.path.join(repo, bad))
    ov.check_out_dir(str(tmp_path / "data" / "overlays" / "x"))
    ov.check_out_dir(os.path.join(repo, "AssetsX"))           # prefix only, not under Assets/


def test_render_refuses_a_bad_out_dir_before_loading_anything(vb, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("OverlayContext built before the out-dir check")
    monkeypatch.setattr(ov, "OverlayContext", boom)
    bad = os.path.join(ov.REPO, "Assets", "StreamingAssets", "overlays")
    with pytest.raises(ValueError, match="not allowed"):
        ov.main(_render_args(vb, "--clip", "syn", "--mode", "still", "--out", bad))
    assert not os.path.exists(bad)


def test_ffmpeg_writer_errors_clearly_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(ov.shutil, "which", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found on PATH"):
        ov.FFmpegWriter(str(tmp_path / "x.mp4"), 30, (20, 20))


@pytest.mark.slow
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_ffmpeg_writer_pads_odd_size_and_reports_failure(tmp_path):
    p = str(tmp_path / "odd.mp4")
    w = ov.FFmpegWriter(p, 30, (41, 23))
    assert w.size == (42, 24)
    for i in range(4):
        w.write(np.full((23, 41, 3), 40 * i, np.uint8))
    w.close()
    info = video_io.probe(p)
    assert info["frames"] == 4 and (info["width"], info["height"]) == (42, 24)
    bad = ov.FFmpegWriter(str(tmp_path / "no_such_dir" / "x.mp4"), 30, (20, 20))
    with pytest.raises(RuntimeError, match="ffmpeg failed"):   # from write (broken pipe) or from close
        bad.write(np.zeros((20, 20, 3), np.uint8))
        bad.close()


def test_still_mode_end_to_end(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    out = tmp_path / "overlays"
    rc = ov.main(_render_args(vb, "--clip", "syn", "--variant", "pose24", "--mode", "still", "--frame", "10",
                              "--out", str(out)))
    assert rc == 0
    img = cv2.imread(str(out / "pose24_still_10.png"))
    assert img is not None and img.shape[1] == 3 * 540 and img.shape[0] >= 2 * 960


@pytest.mark.slow
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_mosaic_video_frame_count(vb, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    out = tmp_path / "overlays"
    rc = ov.main(_render_args(vb, "--clip", "syn_2cam", "--variant", "pose24", "--compare", "syn:pose24",
                              "--mode", "mosaic", "--frames", "5:17", "--scale", "0.25", "--out", str(out)))
    assert rc == 0
    info = video_io.probe(str(out / "pose24_mosaic_5-17.mp4"))
    assert info["frames"] == 12 and (info["width"], info["height"]) == (3 * 540, 2 * 960 + 28)   # b,c + held-out a,d
    log = capsys.readouterr().out
    assert "[overlay] 12 frames in" in log and "fps out" in log and "decoded fps" in log


@pytest.mark.slow
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_views_mode_writes_one_video_per_view(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    rc = ov.main(_render_args(vb, "--clip", "syn", "--mode", "views", "--views", "a,b", "--frames", "3:9",
                              "--stride", "2", "--out", str(tmp_path)))
    assert rc == 0
    for v in ("a", "b"):
        assert video_io.probe(str(tmp_path / f"pose24_{v}_3-8.mp4"))["frames"] == 3


def test_video_mode_requires_frames(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    with pytest.raises(SystemExit, match="--frames"):
        ov.main(_render_args(vb, "--clip", "syn", "--mode", "mosaic", "--views", "a", "--out", str(tmp_path)))


@pytest.mark.parametrize("stride", ["0", "-2"])
def test_stride_below_one_is_rejected(vb, tmp_path, stride):
    with pytest.raises(SystemExit, match="--stride"):
        ov.main(_render_args(vb, "--clip", "syn", "--frames", "0:5", "--stride", stride, "--out", str(tmp_path)))


def test_still_write_failure_is_an_error(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    monkeypatch.setattr(ov.cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="could not write"):
        ov.main(_render_args(vb, "--clip", "syn", "--mode", "still", "--frame", "3", "--views", "a",
                             "--out", str(tmp_path)))


def test_crop_player_mode_still(vb, tmp_path, monkeypatch):
    monkeypatch.setattr(ov.config, "load_registry", lambda *a, **k: vb["reg"])
    rc = ov.main(_render_args(vb, "--clip", "syn", "--variant", "pose24", "--mode", "still", "--frame", "10",
                              "--crop", "player:P1", "--views", "a,b", "--out", str(tmp_path)))
    assert rc == 0 and os.path.exists(tmp_path / "pose24_still_10.png")


def test_handle_key_steps_toggles_and_cycles():
    st = {"f": 5, "playing": False, "layers": {"skel", "kp"}, "view": -1, "quit": False}
    names = ["dets", "kp", "skel"]
    st = ov.handle_key(st, ord("d"), 10, names, 3)
    assert st["f"] == 6
    st = ov.handle_key(st, 2424832, 10, names, 3)             # left arrow
    assert st["f"] == 5
    st = ov.handle_key(st, ord("2"), 10, names, 3)
    assert "kp" not in st["layers"]
    st = ov.handle_key(st, ord("v"), 10, names, 3)
    assert st["view"] == 0
    for _ in range(3):
        st = ov.handle_key(st, ord("v"), 10, names, 3)
    assert st["view"] == -1
    st = ov.handle_key(st, ord(" "), 10, names, 3)
    assert st["playing"]
    assert ov.handle_key(dict(st, f=9), ord("d"), 10, names, 3)["f"] == 9     # clamps at the end
    assert ov.handle_key(dict(st, f=0), ord("a"), 10, names, 3)["f"] == 0     # clamps at the start
    assert ov.handle_key(st, ord("q"), 10, names, 3)["quit"]
    assert ov.handle_key(st, 27, 10, names, 3)["quit"]


def test_handle_key_does_not_mutate_state_and_ignores_unknown_keys():
    st = {"f": 2, "playing": False, "layers": {"kp"}, "view": -1, "quit": False}
    out = ov.handle_key(st, ord("1"), 10, ["dets", "kp"], 2)
    assert out["layers"] == {"kp", "dets"} and st["layers"] == {"kp"}
    assert ov.handle_key(st, ord("9"), 10, ["dets", "kp"], 2) == st       # no 9th layer
    assert ov.handle_key(st, ord("x"), 10, ["dets", "kp"], 2) == st


def test_proxy_command_is_cfr_short_gop_short_side_scaled():
    cmd = ov.proxy_command("in.mov", "out.mp4", 540, 60.0)
    s = " ".join(cmd)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "setpts=N/(60*TB),scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)'"   # re-stamped by index
    assert "-fps_mode cfr" in s                     # safe after setpts: stamps are on the grid (VFR test below)
    assert "-r 60" in s and "-g 10" in s and "libx264" in s and "-an" in s
    assert cmd[-1] == "out.mp4" and cmd[cmd.index("-i") + 1] == "in.mov"
    ntsc = ov.proxy_command("in.mov", "out.mp4", 540, 60000 / 1001)
    assert ntsc[ntsc.index("-vf") + 1].startswith("setpts=N/(60000/1001*TB),") and "60000/1001" in ntsc


@pytest.mark.parametrize("width, want", [(540, 540), (541, 542), (107, 108), (2, 2)])
def test_proxy_width_is_even(width, want):
    assert f"-2,{want})" in ov.proxy_command("a", "b", width, 60.0)[ov.proxy_command("a", "b", width, 60.0).index("-vf") + 1]


@pytest.mark.parametrize("width", [1, 0, -4])
def test_proxy_width_below_two_is_rejected(width, tmp_path):
    with pytest.raises(ValueError, match="width"):
        ov.proxy_command("a", "b", width, 60.0)
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": None}}}
    with pytest.raises(ValueError, match="width"):
        ov.make_proxies("k", width=width, out_root=str(tmp_path / "p"), registry=reg, log=lambda *a: None)
    assert not (tmp_path / "p").exists()


def test_make_proxies_refuses_assets_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(ov.video_io, "probe", lambda *a, **k: pytest.fail("probed before the privacy check"))
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": str(tmp_path / "a.mp4")}}}
    with pytest.raises(ValueError, match="not allowed"):
        ov.make_proxies("k", out_root=os.path.join(ov.REPO, "Assets", "proxy_x"), registry=reg,
                        log=lambda *a: None)
    assert not os.path.exists(os.path.join(ov.REPO, "Assets", "proxy_x"))


def test_make_proxies_frame_mismatch_is_kept_as_bad_and_fails(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": str(src)}}}

    out = tmp_path / "proxy" / "k"
    out.mkdir(parents=True)
    (out / "a.mp4").write_bytes(b"old good proxy")

    def fake_run(cmd, check):
        with open(cmd[-1], "wb") as fh:
            fh.write(b"y")
    monkeypatch.setattr(ov.subprocess, "run", fake_run)
    monkeypatch.setattr(ov.video_io, "probe", lambda p: {"frames": 45 if p == str(src) else 44, "fps": 60.0})
    msgs = []
    rc = ov.make_proxies("k", out_root=str(tmp_path / "proxy"), registry=reg, log=msgs.append)
    assert rc == 1
    assert (out / "a.bad.mp4").read_bytes() == b"y" and not (out / "a.tmp.mp4").exists()
    assert (out / "a.mp4").read_bytes() == b"old good proxy"              # never clobbered by a bad encode
    assert any("NOT frame-exact" in m for m in msgs)


def test_make_proxies_ffmpeg_failure_removes_tmp_and_reraises(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": str(src)}}}

    def fake_run(cmd, check):
        with open(cmd[-1], "wb") as fh:
            fh.write(b"partial")
        raise ov.subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(ov.subprocess, "run", fake_run)
    monkeypatch.setattr(ov.video_io, "probe", lambda p: {"frames": 45, "fps": 60.0})
    with pytest.raises(ov.subprocess.CalledProcessError):
        ov.make_proxies("k", out_root=str(tmp_path / "proxy"), registry=reg, log=lambda *a: None)
    assert os.listdir(tmp_path / "proxy" / "k") == []


def test_make_proxies_success_replaces_and_clears_stale_bad(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": str(src)}}}
    out = tmp_path / "proxy" / "k"
    out.mkdir(parents=True)
    (out / "a.bad.mp4").write_bytes(b"stale")
    (out / "a.mp4").write_bytes(b"old")

    def fake_run(cmd, check):
        assert cmd[-1].endswith("a.tmp.mp4")
        with open(cmd[-1], "wb") as fh:
            fh.write(b"new")
    monkeypatch.setattr(ov.subprocess, "run", fake_run)
    monkeypatch.setattr(ov.video_io, "probe", lambda p: {"frames": 45, "fps": 60.0})
    assert ov.make_proxies("k", out_root=str(tmp_path / "proxy"), registry=reg, log=lambda *a: None) == 0
    assert sorted(os.listdir(out)) == ["a.mp4"] and (out / "a.mp4").read_bytes() == b"new"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("w, h, want", [(216, 384, (108, 192)), (384, 216, (192, 108))])   # portrait, landscape
def test_make_proxies_is_frame_exact(tmp_path, w, h, want):
    src = cs.write_numbered_video(str(tmp_path / "v" / "a.mp4"), 45, w, h, codec="h264")
    reg = {"k": {"mode": "singles", "views": ["a", "b"], "videos": {"a": src, "b": None}}}
    rc = ov.make_proxies("k", width=108, out_root=str(tmp_path / "proxy"), registry=reg, log=lambda *a: None)
    dst = tmp_path / "proxy" / "k" / "a.mp4"
    assert rc == 0 and dst.exists() and sorted(os.listdir(dst.parent)) == ["a.mp4"]
    info = video_io.probe(str(dst))
    assert info["frames"] == 45 and (info["width"], info["height"]) == want         # short side -> width
    with video_io.VideoReader(str(dst)) as r:
        assert cs.read_number(r.get(31)) == 31


def test_preview_keys_says_which_layers_digits_reach():
    assert "first nine" in ov.PREVIEW_KEYS and "--layers" in ov.PREVIEW_KEYS
    assert ov.PREVIEW_KEYS.isascii()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_make_proxies_is_index_exact_for_vfr_source(tmp_path):
    """A 2-frame gap at frame 10 then frames 15-19 squeezed into ~1 frame: plain `-fps_mode cfr -r 60` gives 30
    frames with 9, 10 duplicated and 18, 19 dropped (count check passes, frames wrong - checked with ffmpeg 7.1);
    re-stamping by index keeps proxy frame k == source frame k."""
    num = cs.write_numbered_video(str(tmp_path / "v" / "n.mp4"), 30, 216, 384, codec="h264")
    src = str(tmp_path / "v" / "vfr.mp4")
    pts = "if(lt(N,10),N,if(lt(N,15),N+2,if(lt(N,20),17+0.15*(N-15),N)))"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", num, "-vf",
                    f"setpts='({pts})/(60*TB)'", "-fps_mode", "passthrough", "-c:v", "libx264",
                    "-crf", "10", "-pix_fmt", "yuv420p", src], check=True)
    assert video_io.probe(src)["vfr"]
    reg = {"k": {"mode": "singles", "views": ["a"], "videos": {"a": src}}}
    assert ov.make_proxies("k", width=108, out_root=str(tmp_path / "proxy"), registry=reg, log=lambda *a: None) == 0
    dst = str(tmp_path / "proxy" / "k" / "a.mp4")
    info = video_io.probe(dst)
    assert info["frames"] == 30 and not info["vfr"]
    with video_io.VideoReader(dst) as r:
        assert [cs.read_number(r.get(k)) for k in range(30)] == list(range(30))


def test_compare_matches_players_by_position_not_id(vb):
    """review 1 L7: a compare doc whose labels are permuted must still draw its skeleton on the same person."""
    d = json.load(open(cd.doc_path(vb["adir"], "syn", "pose24"), encoding="utf-8"))
    for p in d["players"]:
        p["id"] = {"P1": "P2", "P2": "P1"}[p["id"]]
    p = cd.doc_path(vb["adir"], "syn", "cmpswap")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    try:
        ctx = _ctx(vb, compare="cmpswap")
        for pid in ("P1", "P2"):
            a, b = ctx.world[pid][0], ctx.cmp_world[pid][0]
            ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
            assert ok.any() and np.allclose(a[ok], b[ok])
    finally:
        os.remove(p)


def test_parents_fallback_follows_the_skeleton_type_and_joint_count():
    """v0.7.5 follow-up: a 58-joint doc without skeleton.parents got the SMPL-28 parents (truncated to 28 entries,
    so the hands / face drew nothing or the wrong bones). The fallback follows skeleton.type (smplx-58 =
    schema.SKELETON_TYPE) or, with no type, the joint count."""
    from analysis import schema
    from analysis import skeleton24 as sk
    names58 = list(schema.JOINT_NAMES)
    for skel in ({"type": schema.SKELETON_TYPE, "joint_names": names58}, {"joint_names": names58}):
        assert ov.OverlayContext._parents({"skeleton": skel}) == list(schema.PARENTS)
    for kind in ("smpl-24", "smpl-28"):
        names, parents, _ = sk.TYPES[kind]
        assert ov.OverlayContext._parents({"skeleton": {"joint_names": list(names)}}) == list(parents)
        assert ov.OverlayContext._parents({"skeleton": {"type": kind, "joint_names": list(names)}}) == list(parents)
    given = list(range(-1, 27))
    assert ov.OverlayContext._parents({"skeleton": {"type": "smpl-28", "joint_names": ["x"] * 28,
                                                    "parents": given}}) == given
    with pytest.raises(ValueError, match="parents"):
        ov.OverlayContext._parents({"skeleton": {"joint_names": ["x"] * 30}})
