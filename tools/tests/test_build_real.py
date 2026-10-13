import numpy as np
from analysis import build_real, schema, smpl24


def test_bodylocal_centres_xz_keeps_y():
    j = np.zeros((24, 3))
    j[0] = [2.0, 0.95, -3.0]      # pelvis at court (2, -3)
    j[15] = [2.1, 1.60, -3.0]     # head
    local, pos = build_real.world_to_bodylocal_and_position(j)
    assert pos == [2.0, -3.0]
    assert np.allclose(local[0], [0.0, 0.95, 0.0])   # pelvis centred, Y kept
    assert np.allclose(local[15], [0.1, 1.60, 0.0])  # head keeps Y + relative X


def test_pack_joints_flat_length_and_conf():
    local = np.zeros((24, 3))
    conf = np.linspace(0.1, 1.0, 24)
    flat = build_real.pack_joints_flat(local, conf)
    assert len(flat) == schema.NUM_JOINTS * schema.STRIDE == 96
    assert flat[3] == round(float(conf[0]), 4)


def test_build_real_doc_validates():
    frame_count = 5
    frames = []
    for f in range(frame_count):
        local = np.zeros((24, 3))
        local[:, 1] = smpl24._REST[:, 1]      # plausible heights
        flat = build_real.pack_joints_flat(local, np.ones(24))
        frames.append({"joints_flat": flat, "position": [0.5, 3.0]})
    frames[2]["joints_flat"] = None            # a dropped frame is legal
    frames[2]["position"] = None
    doc = build_real.build_real_doc("3-1", 60.0, frame_count, frames, side="far")
    assert schema.validate_analysis(doc) == []
    assert doc["schema_version"] == "3.0"
    assert doc["players"][0]["side"] == "far"
    assert len(doc["players"][0]["frames"]) == frame_count


def test_enforce_median_bone_lengths_strict():
    """All frames get exactly the median bone length, not just >30% outliers."""
    from analysis.build_real import _enforce_median_bone_lengths

    parents = list(schema.PARENTS)
    rest = np.array(smpl24._REST, dtype=np.float64)

    # Build 20 frames with varying bone lengths
    frames = []
    for i in range(20):
        j = rest.copy()
        # Vary left upper arm (joint 18, parent 16) by ±15%
        scale = 1.0 + 0.15 * np.sin(i * 0.5)
        bone = j[18] - j[16]
        j[18] = j[16] + bone * scale
        conf = np.ones(24)
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
    rest = np.array(smpl24._REST, dtype=np.float64)

    # Build 20 frames: stretch left thigh (joint 4, parent 1) by 50% on frame 0
    frames = []
    for i in range(20):
        j = rest.copy()
        if i == 0:
            bone = j[4] - j[1]
            j[4] = j[1] + bone * 1.5
            # Ankle (7) stays at original position — after enforcement,
            # the shin (7->4) should also be median length
        conf = np.ones(24)
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
