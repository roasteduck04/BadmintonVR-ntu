"""skeleton24 + schema24 — the SMPL-24/28 contract (analysis.json schema 4.1)."""
import pytest

from analysis import schema, schema24, skeleton24 as sk


def test_smpl24_is_prefix_of_smpl28_and_parents_valid():
    assert sk.SMPL28_NAMES[:24] == sk.SMPL24_NAMES and sk.SMPL28_PARENTS[:24] == sk.SMPL24_PARENTS
    for names, parents in ((sk.SMPL24_NAMES, sk.SMPL24_PARENTS), (sk.SMPL28_NAMES, sk.SMPL28_PARENTS)):
        assert len(names) == len(parents) == len(set(names))
        assert parents[0] == -1 and all(0 <= p < j for j, p in enumerate(parents) if j)
    assert sk.REST28.shape == (28, 3)
    head = sk.SMPL28_NAMES.index("head")
    assert all(sk.SMPL28_PARENTS[j] == head for j in (24, 25, 26, 27))


def test_aux_indices_point_at_the_right_wholebody_keypoints():
    assert len(sk.AUX_NAMES) == len(sk.AUX_WB_IDX) == 14
    m = dict(zip(sk.AUX_NAMES, sk.AUX_WB_IDX))
    assert m["left_heel"] == 19 and m["right_heel"] == 22 and m["right_big_toe"] == 20
    assert m["left_index_mcp"] == 91 + 5 and m["right_pinky_mcp"] == 112 + 17


def test_skeleton_block_labels():
    b = sk.skeleton_block("smpl-28")
    assert b["type"] == "smpl-28" and b["family"] == "SMPL" and b["label"] == "SMPL-28 (SMPL-24 + 4 face)"
    assert b["joint_names"] == sk.SMPL28_NAMES and b["aux_names"] == sk.AUX_NAMES
    assert sk.skeleton_block("smpl-24")["label"] == "SMPL-24"


def _doc(kind="smpl-28", F=3):
    J = len(sk.TYPES[kind][0])

    def fr(f):
        return {"frame_id": f, "time": f / 60.0, "joints_flat": [0.1] * (J * 4), "position": [0.0, -3.0],
                "interpolated": False, "aux_flat": [0.2] * 56, "racket": None, "grip": None, "stroke": None}
    return {"schema_version": "4.1", "clip_id": "t", "meta": {"fps": 60.0, "frame_count": F},
            "court": schema.court_block(), "skeleton": sk.skeleton_block(kind),
            "players": [{"id": "P1", "side": "near", "frames": [fr(f) for f in range(F)]}],
            "shuttle": {"frames": []}, "rallies": []}


@pytest.mark.parametrize("kind", ["smpl-24", "smpl-28"])
def test_valid_41_doc_passes(kind):
    assert schema24.validate_analysis24(_doc(kind)) == []


def test_wrong_joint_count_nan_and_aux_length_fail():
    d = _doc()
    d["players"][0]["frames"][0]["joints_flat"] = [0.0] * 96
    d["players"][0]["frames"][1]["joints_flat"][5] = float("nan")
    d["players"][0]["frames"][2]["aux_flat"] = [0.0] * 55
    errs = schema24.validate_analysis24(d)
    assert any("frame 0" in e and "joints_flat" in e for e in errs)
    assert any("frame 1" in e and "non-finite" in e for e in errs)
    assert any("frame 2" in e and "aux_flat" in e for e in errs)


def test_frame_count_mismatch_and_missing_interpolated_fail():
    d = _doc()
    d["meta"]["frame_count"] = 4
    del d["players"][0]["frames"][0]["interpolated"]
    errs = schema24.validate_analysis24(d)
    assert any("frame_count" in e for e in errs) and any("interpolated" in e for e in errs)


def test_schema5_rejected_here_but_still_valid_for_schema_py():
    from analysis import synth_analysis
    doc5 = synth_analysis.build_synth_doc(frames=10)
    assert schema.validate_analysis(doc5) == []
    assert schema24.validate_analysis24(doc5) != []


def test_legacy_40_accepted_read_only():
    d = _doc()
    d["schema_version"] = "4.0"
    d["skeleton"] = {"type": "smpl-24", "joint_names": sk.SMPL28_NAMES, "parents": sk.SMPL28_PARENTS}
    for fr in d["players"][0]["frames"]:
        fr.pop("aux_flat")
        fr.pop("interpolated")
    assert schema24.validate_analysis24(d) != []
    assert schema24.validate_analysis24(d, allow_legacy=True) == []


def test_malformed_containers_are_errors_not_exceptions():
    d = _doc()
    d["players"][0]["frames"][0] = "not a frame"
    d["players"][0]["frames"][1]["joints_flat"] = 5
    d["players"][0]["frames"][2]["aux_flat"] = {"x": 1}
    errs = schema24.validate_analysis24(d)
    assert any("frame 0" in e for e in errs) and any("frame 1" in e and "joints_flat" in e for e in errs)
    assert any("frame 2" in e and "aux_flat" in e for e in errs)
    d = _doc()
    d["players"] = ["P1"]
    d["skeleton"] = "smpl-28"
    errs = schema24.validate_analysis24(d)
    assert any("player" in e for e in errs) and any("skeleton" in e for e in errs)


def test_41_frame_needs_the_joints_flat_key_and_frame_count_is_not_negative():
    d = _doc()
    d["players"][0]["frames"][1].pop("joints_flat")
    assert any("frame 1" in e and "joints_flat missing" in e for e in schema24.validate_analysis24(d))
    d = _doc()
    d["meta"]["frame_count"] = -3
    assert any("frame_count" in e for e in schema24.validate_analysis24(d))
