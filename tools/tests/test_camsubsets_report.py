"""camsubsets report: long CSV round trip, pivot by camera count, figure, markdown table."""
import csv
import json

from analysis.pose24.camsubsets import report, score


def _row(**kw):
    r = dict.fromkeys(score.ROW_KEYS)
    r.update(clip="c", clip_type="singles", mode="oracle", method="fit", player="P1", player_seen=True,
             test_view="t", matched=True)
    r.update(kw)
    return r


def test_csv_roundtrip(tmp_path):
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=8.5, heldout_n=720, coverage=0.99),
            _row(subset="a+c", n_cams=2, heldout_px_median=None, player_seen=False, coverage=0.0)]
    p = tmp_path / "x.csv"
    report.write_csv(rows, str(p))
    back = report.read_csv(str(p))
    assert type(back[0]["heldout_n"]) is int and back[0]["heldout_n"] == 720 and back[1]["heldout_n"] is None
    assert back[0]["heldout_px_median"] == 8.5 and back[1]["heldout_px_median"] is None
    assert back[1]["player_seen"] is False and back[0]["n_cams"] == 2
    with open(p) as fh:
        assert next(csv.reader(fh)) == score.ROW_KEYS


def test_csv_types_restored_and_strings_kept(tmp_path):
    rows = [_row(subset="1+2", n_cams=2, ref_cams=5, test_view="", player="1", used_frac=json.dumps({"a": 0.5}),
                 excluded=json.dumps([]), view_health_carried=True, lr_swap_frames=3)]
    p = tmp_path / "x.csv"
    report.write_csv(rows, str(p))
    b = report.read_csv(str(p))[0]
    assert b["subset"] == "1+2" and b["player"] == "1" and b["test_view"] == ""   # deployment contract kept
    assert isinstance(b["n_cams"], int) and isinstance(b["ref_cams"], int) and b["ref_cams"] == 5
    assert b["used_frac"] == '{"a": 0.5}' and b["excluded"] == "[]"
    assert b["view_health_carried"] is True and b["matched"] is True and b["lr_swap_frames"] == 3.0
    assert list(b) == score.ROW_KEYS


def test_csv_nonfinite_written_empty(tmp_path):
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=float("nan"), w_mpjpe_cm_median=float("inf"),
                 pa_mpjpe_cm_median=-float("inf"), coverage=1.0)]
    p = tmp_path / "x.csv"
    report.write_csv(rows, str(p))
    text = p.read_text()
    assert "nan" not in text.lower() and "inf" not in text.lower()
    b = report.read_csv(str(p))[0]
    assert b["heldout_px_median"] is None and b["w_mpjpe_cm_median"] is None and b["pa_mpjpe_cm_median"] is None
    assert b["coverage"] == 1.0


def test_pivot_skips_unseen_rows():
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=8.0, coverage=1.0),
            _row(subset="a+c", n_cams=2, heldout_px_median=12.0, coverage=1.0),
            _row(subset="b+c", n_cams=2, heldout_px_median=None, player_seen=False, coverage=0.0),
            _row(subset="a+b+c", n_cams=3, heldout_px_median=6.0, coverage=1.0)]
    pv = {(p["n_cams"], p["metric"]): p for p in report.pivot(rows)}
    two = pv[(2, "heldout_px_median")]
    assert two["median"] == 10.0 and two["best_subset"] == "a+b" and two["worst_subset"] == "a+c"
    assert two["n_unseen"] == 1
    cov = pv[(2, "coverage")]
    assert cov["median"] == 1.0 and cov["worst_subset"] == "b+c" and cov["worst"] == 0.0


def test_pivot_ranks_subsets_by_their_median_and_splits_deployment():
    rows = [_row(subset="a+b", n_cams=2, test_view=t, player=p, heldout_px_median=v, coverage=1.0)
            for t, p, v in (("c", "P1", 1.0), ("c", "P2", 20.0), ("d", "P1", 21.0))]   # median 20
    rows += [_row(subset="a+c", n_cams=2, test_view=t, heldout_px_median=v, coverage=1.0)
             for t, v in (("b", 10.0), ("d", 11.0))]                                     # median 10.5
    rows += [_row(subset="a+b", n_cams=2, test_view="", heldout_px_median=None, w_mpjpe_cm_median=3.0,
                  coverage=1.0)]
    pv = {(p["n_cams"], p["metric"]): p for p in report.pivot(rows)}
    h = pv[(2, "heldout_px_median")]
    assert h["best_subset"] == "a+c" and h["best"] == 10.5 and h["worst_subset"] == "a+b" and h["worst"] == 20.0
    assert h["median"] == 11.0 and h["n_rows"] == 5 and h["n_unseen"] == 0
    assert pv[(2, "w_mpjpe_cm_median")]["median"] is None                              # held-out rows have none
    dep = {(p["n_cams"], p["metric"]): p for p in report.pivot(rows, held_out_only=False)}
    assert dep[(2, "w_mpjpe_cm_median")]["median"] == 3.0 and dep[(2, "w_mpjpe_cm_median")]["n_rows"] == 1
    assert dep[(2, "heldout_px_median")]["best_subset"] is None


def test_pivot_groups_by_clip_type_mode_method():
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=5.0, coverage=1.0),
            _row(subset="a+b", n_cams=2, heldout_px_median=9.0, coverage=1.0, mode="e2e"),
            _row(subset="a+b", n_cams=2, heldout_px_median=7.0, coverage=1.0, clip_type="doubles")]
    pv = report.pivot(rows, metrics=["heldout_px_median"])
    got = {(p["clip_type"], p["mode"], p["method"], p["n_cams"]): p["median"] for p in pv}
    assert got == {("singles", "oracle", "fit", 2): 5.0, ("singles", "e2e", "fit", 2): 9.0,
                   ("doubles", "oracle", "fit", 2): 7.0}


def test_figure_writes_png(tmp_path):
    rows = [_row(subset=s, n_cams=n, heldout_px_median=v, w_mpjpe_cm_median=v / 2, pelvis_xz_cm_median=v / 3,
                 knee_mae_deg=v / 2, coverage=1.0)
            for s, n, v in (("a", 1, 30.0), ("a+b", 2, 10.0), ("a+c", 2, 12.0), ("a+b+c", 3, 7.0))]
    p = report.figure(rows, str(tmp_path / "f.png"))
    assert open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_figure_missing_data_does_not_raise(tmp_path):
    import matplotlib.pyplot as plt
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=8.0, coverage=1.0),
            _row(subset="a+c", n_cams=2, player_seen=False, coverage=0.0)]
    n0 = len(plt.get_fignums())
    p = report.figure(rows, str(tmp_path / "g.png"))
    assert open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n" and len(plt.get_fignums()) == n0
    p2 = report.figure([], str(tmp_path / "h.png"))
    assert open(p2, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_markdown_table():
    rows = [_row(subset="a+b", n_cams=2, heldout_px_median=8.0, coverage=1.0),
            _row(subset="a+c", n_cams=2, heldout_px_median=12.0, coverage=1.0)]
    md = report.markdown_table(report.pivot(rows, metrics=["heldout_px_median", "knee_mae_deg"]))
    lines = md.strip().splitlines()
    assert lines[0].startswith("|") and set(lines[1].replace("|", "").strip()) <= set("-: ")
    assert len(lines) == 4 and "a+b" in lines[2] and "10.00" in lines[2] and "a+c" in lines[2]
    assert lines[3].count("–") >= 3                                                 # knee: no data


def test_pivot_keeps_4_and_5_camera_references_apart():
    rows = [_row(subset="a+b+c", n_cams=3, ref_cams=3, heldout_px_median=2.0, coverage=1.0, is_reference=True),
            _row(subset="a+b+c", n_cams=3, ref_cams=4, heldout_px_median=9.0, coverage=1.0, is_reference=False)]
    pv = report.pivot(rows, metrics=["heldout_px_median"])
    assert report.pivot.__defaults__[1] == ("clip_type", "ref_cams", "mode", "method", "n_cams")
    assert {(p["ref_cams"], p["n_cams"]): p["median"] for p in pv} == {(3, 3): 2.0, (4, 3): 9.0}
    assert "ref_cams" in report.markdown_table(pv).splitlines()[0]
    ser = report.figure_series(pv)
    assert [k[:2] for k, _ in ser] == [("singles", 3), ("singles", 4)]
    assert all("ref" in lab for _, lab in ser)
    assert "vs full" not in " ".join(report._LABEL.values()) and "vs reference" in report._LABEL["w_mpjpe_cm_median"]


def test_csv_is_reference_parsed_as_bool(tmp_path):
    rows = [_row(subset="a+b", n_cams=2, is_reference=True), _row(subset="a+c", n_cams=2, is_reference=False),
            _row(subset="b+c", n_cams=2, is_reference=None, full_body=0.5)]
    p = tmp_path / "x.csv"
    report.write_csv(rows, str(p))
    back = report.read_csv(str(p))
    assert [b["is_reference"] for b in back] == [True, False, None] and back[2]["full_body"] == 0.5


def test_vs_full_rig_table_filters_ref5_deployment_seen_rows():
    keep = dict(ref_cams=5, test_view="", is_reference=False, n_cams=1, subset="b")
    rows = [_row(method="mono-len-oracle", w_mpjpe_cm_median=10.0, elbow_mae_deg=20.0, **keep),
            _row(method="mono-len-oracle", w_mpjpe_cm_median=20.0, elbow_mae_deg=None, **keep),
            _row(method="mono-len-oracle", w_mpjpe_cm_median=99.0, player_seen=False, **keep),       # unseen
            _row(method="mono-len-oracle", w_mpjpe_cm_median=99.0, ref_cams=4, test_view="t", n_cams=1,  # held-out ref 4
                 is_reference=False, subset="b"),
            _row(method="mono-len-oracle", w_mpjpe_cm_median=99.0, **{**keep, "mode": "e2e"}),
            _row(method="fit", w_mpjpe_cm_median=99.0, **{**keep, "n_cams": 2}),
            _row(method="mono-len-oracle+motionbert-full", clip_type="doubles", w_mpjpe_cm_median=7.0, **keep),
            _row(method="mono-len-oracle+motionbert-full", w_mpjpe_cm_median=5.0, **keep)]
    t = report.vs_full_rig_table(rows)
    assert [(r["clip_type"], r["method"]) for r in t] == [("doubles", "mono-len-oracle+motionbert-full"),
                                                          ("singles", "mono-len-oracle"),
                                                          ("singles", "mono-len-oracle+motionbert-full")]
    s = t[1]
    assert s["n_rows"] == 2 and s["w_mpjpe_cm_median"] == 15.0 and s["elbow_mae_deg"] == 20.0
    assert s["pa_mpjpe_cm_median"] is None and s["name"] == "1 cam (rig lengths)"
    md = report.vs_full_rig_markdown(t)
    assert "| singles | 1 cam (rig lengths) | 2 | 15.0 | - |" in md
    md.encode("ascii")


def test_vs_full_rig_table_finds_every_prior_method():
    keep = dict(ref_cams=5, test_view="", is_reference=False, n_cams=1, subset="b", w_mpjpe_cm_median=1.0)
    rows = [_row(method=m, **keep) for m in ("mono-len-prior+wham", "mono-len-oracle+wham", "mono-len-prior",
                                             "mono-len-oracle+motionbert-full", "mono-len-oracle", "fit",
                                             "mono-len-oracle+gvhmr", "mono-len-oracle+motionbert-lite")]
    assert [r["method"] for r in report.vs_full_rig_table(rows)] == [
        "mono-len-oracle", "mono-len-oracle+motionbert-lite", "mono-len-oracle+motionbert-full",
        "mono-len-oracle+wham", "mono-len-oracle+gvhmr", "mono-len-prior", "mono-len-prior+wham"]
    assert report.mono_methods(rows) == [r["method"] for r in report.vs_full_rig_table(rows)]
    assert [r["method"] for r in report.vs_full_rig_table(rows, methods=("mono-len-oracle+wham",))] == [
        "mono-len-oracle+wham"]
