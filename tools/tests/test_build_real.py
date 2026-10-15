import numpy as np
import pytest
from analysis import build_real, schema, skeleton


def test_bodylocal_centres_xz_keeps_y():
    j = np.zeros((58, 3))
    j[0] = [2.0, 0.95, -3.0]      # pelvis at court (2, -3)
    j[15] = [2.1, 1.60, -3.0]     # head
    local, pos = build_real.world_to_bodylocal_and_position(j)
    assert pos == [2.0, -3.0]
    assert np.allclose(local[0], [0.0, 0.95, 0.0])   # pelvis centred, Y kept
    assert np.allclose(local[15], [0.1, 1.60, 0.0])  # head keeps Y + relative X


def test_pack_joints_flat_length_and_conf():
    local = np.zeros((58, 3))
    conf = np.linspace(0.1, 1.0, 58)
    flat = build_real.pack_joints_flat(local, conf)
    assert len(flat) == schema.NUM_JOINTS * schema.STRIDE == 232
    assert flat[3] == round(float(conf[0]), 4)


def test_build_real_doc_validates():
    frame_count = 5
    frames = []
    for f in range(frame_count):
        local = np.zeros((58, 3))
        local[:, 1] = skeleton._REST[:, 1]    # plausible heights
        flat = build_real.pack_joints_flat(local, np.ones(58))
        frames.append({"joints_flat": flat, "position": [0.5, 3.0]})
    frames[2]["joints_flat"] = None            # a dropped frame is legal
    frames[2]["position"] = None
    doc = build_real.build_real_doc("3-1", 60.0, frame_count, frames, side="far")
    assert schema.validate_analysis(doc) == []
    assert doc["schema_version"] == "5.0"
    assert doc["players"][0]["side"] == "far"
    assert len(doc["players"][0]["frames"]) == frame_count


def _racket_case(mcp_conf, n=30):
    """One player, right hand holding the racket; the triangulated racket continues the hand (MCP seen) or the
    forearm (MCP unseen), so the body-derived and triangulated shafts agree."""
    N = {nm: i for i, nm in enumerate(skeleton.JOINT_NAMES)}
    J = np.zeros((58, 3))
    J[:, 1] = 1.0
    J[N["right_elbow"]] = [0.2, 1.2, 0.0]
    J[N["right_wrist"]] = [0.2, 1.0, 0.2]
    J[N["right_middle1"]] = [0.2, 1.0, 0.3]
    J[N["left_wrist"]] = [-0.6, 0.8, 0.0]
    conf = np.full(58, 0.9)
    conf[N["right_middle1"]] = conf[N["left_middle1"]] = mcp_conf
    flat = build_real.pack_joints_flat(J, conf)
    pd = [{"id": "P1", "pose_frames": [{"position": [0.0, 0.0], "joints_flat": list(flat)} for _ in range(n)]}]
    if mcp_conf >= 0.1:
        g0, d = J[N["right_middle1"]], np.array([0.0, 0.0, 1.0])
    else:
        g0, d = J[N["right_wrist"]], J[N["right_wrist"]] - J[N["right_elbow"]]
    g = np.tile(g0, (n, 1))
    h = g + 0.66 * d / np.linalg.norm(d)
    tracks = {"P1": {"grip": g.copy(), "head": h, "side": h.copy(), "status": ["detected"] * n,
                     "confidence": [0.5] * n, "has_side": [False] * n}}
    return build_real._to_bodylocal_racket_multi(tracks, pd)["P1"], J, N


def test_racket_shaft_falls_back_to_the_forearm_when_the_hand_is_unseen():
    """Review 2 H2: with no confident hand joint (build_real never triangulates 91-132) the body shaft used to be
    a constant -Z; it now follows the forearm (wrist - elbow)."""
    tr, J, N = _racket_case(mcp_conf=0.0)
    fore = J[N["right_wrist"]] - J[N["right_elbow"]]
    fore /= np.linalg.norm(fore)
    for f in (0, 15, 29):
        shaft = tr["head"][f] - tr["grip"][f]
        assert np.dot(shaft / np.linalg.norm(shaft), fore) > 0.99
        assert np.allclose(tr["grip"][f], J[N["right_wrist"]])


def test_racket_shaft_uses_wrist_to_hand_when_the_hand_is_seen():
    tr, J, N = _racket_case(mcp_conf=0.9)
    shaft = tr["head"][10] - tr["grip"][10]
    assert np.dot(shaft / np.linalg.norm(shaft), [0.0, 0.0, 1.0]) > 0.99
    assert np.allclose(tr["grip"][10], J[N["right_middle1"]])


def test_max_bone_table_is_name_based_on_the_58_joint_skeleton():
    """Review 2 L9: the SMPL-24 pair (23, 21) "hand" is left_thumb2 + RIGHT wrist in SMPL-X 58."""
    N = {nm: i for i, nm in enumerate(skeleton.JOINT_NAMES)}
    j = np.zeros((58, 3))
    j[N["left_wrist"]] = [-0.3, 1.0, 0.0]
    j[N["right_wrist"]] = [0.3, 1.0, 0.0]
    j[N["left_thumb1"]] = [-0.32, 1.0, 0.0]
    j[N["left_thumb2"]] = [-0.34, 1.0, 0.0]          # 0.64 m from the right wrist: legal
    j[N["left_middle1"]] = [-0.3, 1.0, 0.3]           # 0.3 m from its wrist: too long a hand
    out, _ = build_real._enforce_bone_lengths(j, np.ones(58), schema.PARENTS)
    assert np.allclose(out[N["left_thumb2"]], j[N["left_thumb2"]])
    assert np.linalg.norm(out[N["left_middle1"]] - out[N["left_wrist"]]) == pytest.approx(0.12)


def test_velocity_cap_is_name_based():
    N = {nm: i for i, nm in enumerate(skeleton.JOINT_NAMES)}
    a = np.zeros((58, 4))
    a[:, 3] = 0.9
    b = a.copy()
    b[N["left_thumb1"], 0] = 0.6                       # SMPL-24 index 22 = left_hand: capped at 0.45 before
    b[N["left_middle1"], 0] = 0.6
    frames = [{"joints_flat": a.ravel().tolist()}, {"joints_flat": b.ravel().tolist()}]
    build_real._clamp_joint_velocities(frames)
    out = np.array(frames[1]["joints_flat"]).reshape(-1, 4)
    assert out[N["left_thumb1"], 0] == pytest.approx(0.6)
    assert out[N["left_middle1"], 0] == pytest.approx(0.45)
    a28 = np.zeros((28, 4))
    a28[:, 3] = 0.9
    b28 = a28.copy()
    b28[22, 0] = 0.6                                   # SMPL-28 left_hand
    frames = [{"joints_flat": a28.ravel().tolist()}, {"joints_flat": b28.ravel().tolist()}]
    build_real._clamp_joint_velocities(frames)
    assert frames[1]["joints_flat"][22 * 4] == pytest.approx(0.45)


def test_enforce_median_bone_lengths_strict():
    """All frames get exactly the median bone length, not just >30% outliers."""
    from analysis.build_real import _enforce_median_bone_lengths

    parents = list(schema.PARENTS)
    rest = np.array(skeleton._REST, dtype=np.float64)

    # Build 20 frames with varying bone lengths
    frames = []
    for i in range(20):
        j = rest.copy()
        # Vary left upper arm (joint 18, parent 16) by ±15%
        scale = 1.0 + 0.15 * np.sin(i * 0.5)
        bone = j[18] - j[16]
        j[18] = j[16] + bone * scale
        conf = np.ones(58)
        flat = build_real.pack_joints_flat(j, conf)
        frames.append({"joints_flat": flat, "position": [0, 0]})

    result = _enforce_median_bone_lengths(frames, parents)

    # After enforcement, all frames should have the same bone length for joint 18
    lengths = []
    for pf in result:
        arr = np.array(pf["joints_flat"]).reshape(-1, 4)
        bl = np.linalg.norm(arr[18, :3] - arr[16, :3])
        lengths.append(bl)
    lengths = np.array(lengths)
    assert np.std(lengths) < 1e-4, f"bone lengths should be constant, got std={np.std(lengths):.6f}"
    assert np.allclose(lengths, np.median(lengths), atol=1e-4)


def test_enforce_median_bone_lengths_root_to_leaf_cascade():
    """Parent correction cascades to children (knee fix moves ankle)."""
    from analysis.build_real import _enforce_median_bone_lengths

    parents = list(schema.PARENTS)
    rest = np.array(skeleton._REST, dtype=np.float64)

    # Build 20 frames: stretch left thigh (joint 4, parent 1) by 50% on frame 0
    frames = []
    for i in range(20):
        j = rest.copy()
        if i == 0:
            bone = j[4] - j[1]
            j[4] = j[1] + bone * 1.5
            # Ankle (7) stays at original position — after enforcement,
            # the shin (7->4) should also be median length
        conf = np.ones(58)
        flat = build_real.pack_joints_flat(j, conf)
        frames.append({"joints_flat": flat, "position": [0, 0]})

    result = _enforce_median_bone_lengths(frames, parents)

    # Thigh length (joint 4) should be consistent across all frames
    thigh_lengths = []
    for pf in result:
        arr = np.array(pf["joints_flat"]).reshape(-1, 4)
        thigh_lengths.append(np.linalg.norm(arr[4, :3] - arr[1, :3]))
    assert np.std(thigh_lengths) < 1e-4


def test_detect_sustained_swaps_nulls_swapped_stretch():
    """A 15-frame stretch where P1 is at P2's anchor gets nulled."""
    from analysis.build_real import _detect_sustained_swaps

    frame_count = 300
    # P1 anchor at Z=3.0, P2 anchor at Z=-3.0
    pd = [
        {"id": "P1", "side": "near", "pose_frames": []},
        {"id": "P2", "side": "far", "pose_frames": []},
    ]

    for f in range(frame_count):
        if 100 <= f < 115:
            # P1 teleports to P2's zone for 15 frames
            pd[0]["pose_frames"].append({
                "joints_flat": [0.0] * 96,
                "position": [0.0, -3.0],  # P2's anchor
            })
        else:
            pd[0]["pose_frames"].append({
                "joints_flat": [0.0] * 96,
                "position": [0.0, 3.0],  # P1's anchor
            })
        pd[1]["pose_frames"].append({
            "joints_flat": [0.0] * 96,
            "position": [0.0, -3.0],
        })

    pd = _detect_sustained_swaps(pd)

    # Frames 100-114 should be nulled for P1
    for f in range(100, 115):
        assert pd[0]["pose_frames"][f]["joints_flat"] is None, f"frame {f} should be nulled"
        assert pd[0]["pose_frames"][f]["position"] is None, f"frame {f} position should be nulled"

    # Frame 99 and 115 should NOT be nulled
    assert pd[0]["pose_frames"][99]["joints_flat"] is not None
    assert pd[0]["pose_frames"][115]["joints_flat"] is not None


def test_detect_sustained_swaps_ignores_short_stretch():
    """A 5-frame excursion (< MIN_SWAP_RUN=10) is left alone."""
    from analysis.build_real import _detect_sustained_swaps

    frame_count = 200
    pd = [
        {"id": "P1", "side": "near", "pose_frames": []},
        {"id": "P2", "side": "far", "pose_frames": []},
    ]

    for f in range(frame_count):
        if 100 <= f < 105:
            pd[0]["pose_frames"].append({
                "joints_flat": [0.0] * 96,
                "position": [0.0, -3.0],
            })
        else:
            pd[0]["pose_frames"].append({
                "joints_flat": [0.0] * 96,
                "position": [0.0, 3.0],
            })
        pd[1]["pose_frames"].append({
            "joints_flat": [0.0] * 96,
            "position": [0.0, -3.0],
        })

    pd = _detect_sustained_swaps(pd)

    # Short stretch should NOT be nulled
    for f in range(100, 105):
        assert pd[0]["pose_frames"][f]["joints_flat"] is not None


def _dropout_frames(nj=58, n=40, j=5, gap=(10, 15)):
    """Constant pose; joint j drops out (conf 0, xyz zeroed by pack_joints_flat) for frames gap[0]..gap[1]-1."""
    frames = []
    for f in range(n):
        a = np.zeros((nj, 4))
        a[:, 0] = np.arange(nj) * 0.01 + 0.3
        a[:, 1] = 1.0
        a[:, 3] = 0.9
        if gap[0] <= f < gap[1]:
            a[j] = 0.0
        frames.append({"joints_flat": a.ravel().tolist(), "position": [0.0, 0.0]})
    return frames


@pytest.mark.parametrize("smooth", [
    lambda fr: build_real._ema_joints(fr, alpha=0.25),
    lambda fr: build_real._sg_smooth_joints(fr, window=11, polyorder=3),
    lambda fr: build_real._bspline_smooth_joints(fr, knot_spacing=5),
])
@pytest.mark.parametrize("nj", [58, 28])
def test_smoothers_do_not_mix_dropped_out_joints_into_neighbours(smooth, nj):
    """Review 2 M2: a joint returning after a dropout was blended with its zeroed (floor) placeholder."""
    out = smooth(_dropout_frames(nj=nj))
    for f in (8, 9, 15, 16, 20):
        a = np.array(out[f]["joints_flat"]).reshape(-1, 4)
        assert a[5, 0] == pytest.approx(0.35, abs=1e-6) and a[5, 1] == pytest.approx(1.0, abs=1e-6), f
    gap = np.array(out[12]["joints_flat"]).reshape(-1, 4)
    assert gap[5, 3] == 0.0 and np.allclose(gap[5, :3], 0.0)       # the dropout itself stays a dropout


def test_sg_smooths_even_length_short_segments():
    """Review 2 L8: seg_len | 1 rounded an 8-frame run up to a 9-frame window, so even runs stayed raw."""
    for n in (7, 8):
        frames = _dropout_frames(nj=28, n=n, gap=(0, 0))
        for f, fr in enumerate(frames):
            fr["joints_flat"][5 * 4] += 0.05 * (-1) ** f            # zig-zag noise on joint 5 x
        out = build_real._sg_smooth_joints(frames, window=11, polyorder=3)
        xs = np.array([fr["joints_flat"][5 * 4] for fr in out])
        assert np.ptp(xs) < 0.09, n                                  # raw zig-zag spans 0.10


@pytest.mark.parametrize("mode", ["bonelen", "sgsmooth", "bspline", "physics_sg", "physics_bsp"])
def test_resmooth_uses_the_docs_own_skeleton(tmp_path, monkeypatch, mode):
    """Review 2 L1: resmooth passed the global 58-joint PARENTS, so every shipped 28-joint base doc crashed.
    (The final facing call is covered too: test_resmooth_facing_uses_the_docs_own_joint_indices.)"""
    import json
    from analysis import skeleton24 as sk
    rng = np.random.default_rng(0)
    frames = _dropout_frames(nj=28, n=30)
    for fr in frames:
        a = np.array(fr["joints_flat"]).reshape(-1, 4)
        a[:, :3] += rng.normal(0, 0.01, (28, 3)) * (a[:, 3:] > 0)
        fr["joints_flat"] = a.ravel().tolist()
    sk28 = sk.skeleton_block("smpl-28")
    doc = {"schema_version": "4.0", "meta": {"fps": 60.0, "frame_count": 30},
           "skeleton": {"type": "smpl-28", "joint_names": sk28["joint_names"], "parents": sk28["parents"]},
           "players": [{"id": "P1", "frames": frames}]}
    src, out = tmp_path / "c.analysis.json", tmp_path / "o.analysis.json"
    src.write_text(json.dumps(doc))
    assert build_real.main(["resmooth", str(src), "--mode", mode, "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert len(res["players"][0]["frames"][0]["joints_flat"]) == 28 * 4


def _rk(hx, hy, s=0.9):
    """5 racket keypoints [top, bottom, handle, left, right] around a handle at (hx, hy)."""
    kp = [[hx, hy - 200, s], [hx, hy - 100, s], [hx, hy, s], [hx - 30, hy - 160, s], [hx + 30, hy - 160, s]]
    order = build_real._RACKET_KP
    named = dict(zip(("top", "bottom", "handle", "left", "right"), kp))
    return [list(named[k]) for k in order]


def test_smooth_racket_2d_accepts_the_single_detection_format():
    """Review 2 L6: a frame of 5 x [x, y, s] (old racket2d) crashed the smoother."""
    frames = [_rk(100.0 + f, 500.0) for f in range(10)]
    out = build_real._smooth_racket_2d({"v": frames})["v"]
    h = build_real._HANDLE_IDX
    assert len(out) == 10 and out[4][0][h][0] == pytest.approx(104.0, abs=0.01)


def test_smooth_racket_2d_follows_rackets_not_candidate_indices():
    """Review 2 L5: candidates are re-sorted by score every frame, so index 0 swaps between two rackets and
    smoothing index 0 blended them."""
    frames = []
    for f in range(20):
        a, b = _rk(100.0, 500.0), _rk(900.0, 500.0)
        frames.append([a, b] if f % 2 else [b, a])
    out = build_real._smooth_racket_2d({"v": frames})["v"]
    h = build_real._HANDLE_IDX
    xs = sorted(c[h][0] for fr in out for c in fr)
    assert xs[0] == pytest.approx(100.0, abs=0.5) and xs[-1] == pytest.approx(900.0, abs=0.5)
    assert all(abs(c[h][0] - 100.0) < 1 or abs(c[h][0] - 900.0) < 1 for fr in out for c in fr)


def test_racket_sticky_bonus_follows_the_racket_not_the_candidate_index():
    def kps(wx, wy):
        k = [[0.0, 0.0, 0.9] for _ in range(17)]
        k[9] = k[10] = [wx, wy, 0.9]
        return k
    a, b = _rk(100.0, 500.0), _rk(160.0, 500.0)
    matched = {"P1": {"v": [kps(100.0, 500.0), kps(150.0, 500.0)]}}     # frame 1: 50 px from A, 10 px from B
    rkp = {"v": [[a, b], [b, a]]}                                              # order swapped in frame 1
    out = build_real._assign_racket_to_players(matched, None, ["v"], 2, ["P1"], use_hungarian=True,
                                               sticky_bonus=50.0, synced_rkp=rkp)
    h = build_real._HANDLE_IDX
    assert out["P1"]["v"][0][h][0] == 100.0 and out["P1"]["v"][1][h][0] == 100.0    # kept racket A


def test_fix_3d_swaps_multi_handles_a_3_cycle():
    """Review 2 L7: the write-back assumed 2-cycles; a rotation 0->1->2 duplicated one frame and lost one."""
    anchors = [(0.0, -5.0), (3.0, 0.0), (-3.0, 5.0)]
    pd = []
    for i, a in enumerate(anchors):
        frames = [{"position": list(a), "tag": (i, f)} for f in range(10)]
        pd.append({"id": f"P{i + 1}", "pose_frames": frames})
    for i in range(3):                                   # frame 7: every player holds the next one's frame
        pd[i]["pose_frames"][7] = {"position": list(anchors[(i + 1) % 3]), "tag": ((i + 1) % 3, 7)}
    build_real._fix_3d_swaps_multi(pd, anchor_window=5)
    tags = [pd[i]["pose_frames"][7]["tag"] for i in range(3)]
    assert tags == [(0, 7), (1, 7), (2, 7)]


def test_height_gate_keeps_overhead_frames():
    """Review 2 L10: a frame was dropped whenever any confident joint sat > 2.0 m above the floor (overhead
    reach, jump smash). Now only implausible joints are dropped, and the frame only for an implausible pelvis."""
    N = {nm: i for i, nm in enumerate(skeleton.JOINT_NAMES)}
    local = np.zeros((58, 3))
    local[:, 1] = 1.0
    local[N["right_wrist"], 1] = 2.45                      # racket arm straight up in a jump
    local[N["left_wrist"], 1] = 4.5                        # garbage triangulation
    conf = np.full(58, 0.9)
    out, c, drop = build_real._height_gate(local, conf)
    assert not drop and c[N["right_wrist"]] == 0.9 and c[N["left_wrist"]] == 0.0
    local[N["pelvis"], 1] = -1.0
    assert build_real._height_gate(local, conf)[2]


# test_58_joint_producers_write_one_skeleton_type removed in the NTU copy: tools/smplx_gpu.py (SMPL-X lane) is not in this copy.
@pytest.mark.parametrize("kind", ["smpl-28", "smpl-24"])
def test_resmooth_facing_uses_the_docs_own_joint_indices(tmp_path, kind):
    """v0.7.5 follow-up of review 2 L1: resmooth's final fill_facing passed the 58-joint face indices, so 28-joint
    docs crashed there. Face / hip / shoulder indices now come from the doc's joint names; the heading uses the
    doc's handedness (mirror-handed real docs: left hip at -X when facing +Z)."""
    import json
    import math
    from analysis import skeleton24 as sk
    names = list(sk.TYPES[kind][0])
    pts = {"pelvis": (0.0, 0.95, 0.0), "left_hip": (-0.1, 0.9, 0.0), "right_hip": (0.1, 0.9, 0.0),
           "left_knee": (-0.1, 0.5, 0.02), "right_knee": (0.1, 0.5, 0.02),
           "left_ankle": (-0.1, 0.08, 0.0), "right_ankle": (0.1, 0.08, 0.0),
           "left_foot": (-0.1, 0.02, 0.12), "right_foot": (0.1, 0.02, 0.12),
           "left_shoulder": (-0.18, 1.42, 0.0), "right_shoulder": (0.18, 1.42, 0.0), "neck": (0.0, 1.5, 0.0),
           "head": (0.0, 1.62, 0.02), "left_eye": (-0.03, 1.6, 0.08), "right_eye": (0.03, 1.6, 0.08),
           "nose_tip": (0.0, 1.57, 0.12)}
    frames = []
    for f in range(30):
        a = np.zeros((len(names), 4))
        for i, n in enumerate(names):
            a[i, :3] = pts.get(n, (0.0, 1.0, 0.0))
            a[i, 3] = 0.9
        frames.append({"joints_flat": a.ravel().tolist(), "position": [0.0, 0.0]})
    sk_block = sk.skeleton_block(kind)
    doc = {"schema_version": "4.0", "meta": {"fps": 60.0, "frame_count": 30},
           "skeleton": {"type": kind, "joint_names": sk_block["joint_names"], "parents": sk_block["parents"]},
           "players": [{"id": "P1", "frames": frames}]}
    src, out = tmp_path / "c.analysis.json", tmp_path / "o.analysis.json"
    src.write_text(json.dumps(doc))
    assert build_real.main(["resmooth", str(src), "--mode", "bonelen", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    fc = [fr["facing"] for fr in res["players"][0]["frames"]]
    assert all(c is not None and abs(math.remainder(c["heading"], 2 * math.pi)) < 0.05 for c in fc)
    if kind == "smpl-28":
        assert all(abs(c["head_heading"]) < 0.05 for c in fc)
    else:
        assert all("head_heading" not in c for c in fc)
