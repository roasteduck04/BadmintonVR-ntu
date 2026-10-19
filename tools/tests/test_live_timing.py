# tools/tests/test_live_timing.py
from analysis.live import timing

MS = 1_000_000


def _events():
    ev = [{"node": "runner", "ev": "phase", "phase": "start", "t0_ns": 0, "setup_s": 1.0, "source_ready_s": 0.5}]
    for s in range(4):
        ev.append({"node": "runner", "ev": "frames", "step": s, "t_capture_ns": s * 33 * MS,
                   "t_avail_ns": s * 33 * MS, "t_2d_done_ns": s * 33 * MS + 20 * MS, "t_push_ns": s * 33 * MS + 21 * MS,
                   "views": {"a": {"decode_ms": 2.0, "infer_ms": 15.0, "wait_ms": 0.0, "dropped": 0, "n_frames": 1,
                                   "n_dets": 2, "first_call": s == 0},
                             "b": {"decode_ms": 3.0, "infer_ms": 18.0, "wait_ms": 1.0, "dropped": int(s == 3),
                                   "n_frames": 1, "n_dets": 2, "first_call": False}}})
        ev.append({"node": "pose3d", "ev": "step", "step": s, "calib_ns": MS, "track_ns": 2 * MS, "tri_ns": 3 * MS})
        ev.append({"node": "runner", "ev": "out", "step": s, "quality": 1 if s < 2 else 0,
                   "t_capture_ns": s * 33 * MS, "t_written_ns": s * 33 * MS + 500 * MS, "write_ns": MS})
    ev.append({"node": "pose3d", "ev": "window", "k": 0, "start": 2, "stop": 4, "commit_stop": 4, "lrswap_ns": 4 * MS,
               "prep_ns": 2 * MS, "solve_ns": {"P1": 30 * MS, "P2": 40 * MS}, "regress_ns": 4 * MS,
               "status": {"P1": "fitted", "P2": "late"}, "t_ready_ns": 0, "t_done_ns": {}, "lo": {}})
    ev.append({"node": "runner", "ev": "phase", "phase": "end", "loop_s": 2.0})
    return ev


def test_step_rows_derive_every_stage():
    rows = timing.step_rows(_events())
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    r = rows[2]
    assert (r["decode"], r["infer"], r["wait_2d"]) == (3.0, 18.0, 20.0)
    assert (r["calib"], r["track"], r["tri"]) == (1.0, 2.0, 3.0)
    assert r["fit_share"] == (4 + 2 + 40 + 4) / 2 and r["win_status"] == {"P1": "fitted", "P2": "late"}
    assert r["hold"] == 479.0 and r["e2e"] == 500.0 and r["write"] == 1.0
    assert rows[0]["fit_share"] is None and rows[0]["first_call"] and rows[3]["dropped"] == 1


def test_summary_counts_and_rates():
    cfg = {"ref_frames": list(range(0, 8, 2)), "fps": 30.0}
    ev = _events()
    s = timing.summary(timing.step_rows(ev), ev, cfg)
    assert s["written"] == 4 and s["quality_counts"] == {1: 2, 0: 2}
    assert s["window_status"] == {"P1": {"fitted": 1}, "P2": {"late": 1}}
    assert abs(s["wall_s"] - 0.599) < 1e-9 and abs(s["realtime_factor"] - (4 / 30) / 0.599) < 1e-9
    assert s["first_frame_s"] == 0.5 and s["dropped_2d_frames"] == 1 and s["first_2d_call_ms"] == 18.0
    assert s["stages"]["e2e"]["p50"] == 500.0 and s["e2e_fitted"]["n"] == 2
