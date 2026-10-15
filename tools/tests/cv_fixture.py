"""Shared fixture for the Unity EditMode tests (Assets/Editor/Tests/CvDebugTests.cs): a real cvdebug/1 sidecar of the
synthetic 4-camera clip + Python-computed case tables. Regenerate after any sidecar format change:
    python tools/tests/cv_fixture.py
The pytest in test_cvdebug_sp4.py fails when the committed file no longer matches what cvdebug.py produces."""
import json
import os
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # tools/
sys.path.insert(0, HERE)                           # tools/tests/ (synth helpers)

from analysis import cvdebug as cd                 # noqa: E402
from analysis import video_io                      # noqa: E402
import cvdebug_synth as cs                         # noqa: E402

FIX = os.path.join(HERE, "fixtures")
F_KEEP = 40                                        # frames kept in the fixture (small file; enough for every state)


def _trim(sc, n):
    """Cut every per-frame array to the first n frames (F x J, V x F, F x 6 layouts)."""
    F, J, V = sc["frame_count"], len(sc["joint_names"]), len(sc["cameras"])
    for p in sc["players"]:
        for k in ("frame_state", "frame_err_px"):
            p[k] = p[k][:n]
        for k in ("seen_mask", "view_mask", "err_dpx", "sigma_mm", "filled"):
            p[k] = p[k][:n * J]
        for k in ("det_index", "lr_swapped"):
            p[k] = np.asarray(p[k]).reshape(V, F)[:, :n].ravel().tolist()
        p["contact"] = p["contact"][:n * 6]
    sc["frame_count"] = n
    return sc


def _mark(sc):
    """Camera c excluded (health 927 px, bits cleared), camera d held out (bits cleared): SP4 plan Review Focus 1."""
    names = [c["name"] for c in sc["cameras"]]
    for c in sc["cameras"]:
        if c["name"] == "c":
            c.update(excluded=True, health_px=927.0)
        if c["name"] == "d":
            c.update(held_out=True)
    clear = ~((1 << names.index("c")) | (1 << names.index("d")))
    for p in sc["players"]:
        p["seen_mask"] = [int(m) & clear for m in p["seen_mask"]]
        p["view_mask"] = [int(m) & clear for m in p["view_mask"]]
    return sc


def _persons(sc):
    n, V = sc["frame_count"], len(sc["cameras"])

    def tr(tid, label, first, last, pid, x, z):
        k = last - first + 1
        xz = []
        for i in range(k):
            xz += ([x, z] if i != 3 else [-9999.0, -9999.0])       # one frame without a candidate
        return {"track_id": tid, "label": label, "player_id": pid, "score": 0.5, "first_frame": first,
                "last_frame": last, "ambiguous": 0, "pinned": 0, "floor_xz": xz, "support": [2] * k,
                "det_index": [-1] * (V * k)}

    pdoc = {"format": "persons/1", "views": [c["name"] for c in sc["cameras"]], "views_used": [],
            "det_shift": [c["det_shift"] for c in sc["cameras"]], "frame_count": n, "source_build_id": "",
            "tracks": [tr(0, "player", 0, n - 1, "P1", -0.5, -3.0), tr(1, "nonplayer", 5, 34, "", 3.4, 0.0),
                       tr(2, "unknown", 10, 39, "", -4.0, 6.0)],
            "stats": {"hotspots": [{"x": 3.5, "z": 0.25, "frames": 120}]}}
    why = cd.attach_persons(sc, pdoc, sc["source_build_id"])
    assert why == "", why
    return sc


def _shift(an, ao, bn, bo):
    d = [a - bo[bn.index(n)] for n, a in zip(an, ao) if n in bn]
    return (float(np.median(d)), True) if d else (0.0, False)


def _cases(sc):
    frames = []
    for f in (0, 1, 7, 100):
        for off, ds in ((0.0, 0), (0.5, 0), (0.5, 1), (-0.5, 0), (-0.5, -1), (-0.7, -1), (-1.7, -2), (2.9, 3),
                        (-3.1, -3), (1.5, 2), (1.5, 1), (0.49, 0)):
            frames.append({"f": f, "offset": off, "det_shift": ds, "expected": video_io.view_frame(f, off, ds)[0]})
    proj, rng = [], np.random.default_rng(0)
    for c in sc["cameras"]:
        for _ in range(3):
            X = np.array([rng.uniform(-2.5, 2.5), rng.uniform(0.1, 1.9), rng.uniform(-6, 6)])
            uv = cd.unity_project(c["unity_pos"], c["unity_rot"], np.asarray(c["K"]).reshape(3, 3), X[None])[0]
            loc = (X - np.asarray(c["unity_pos"])) @ Rotation.from_quat(c["unity_rot"]).as_matrix()   # M^T (p - c)
            if np.isfinite(uv).all() and loc[2] > 0.5:
                proj.append({"camera": c["name"], "point": [float(x) for x in X], "uv": [float(u) for u in uv]})
    shifts = []
    for an, ao, bn, bo in ((["back_41", "backright_41", "frontleft_41"], [0.0, -3.1, -0.4],
                            ["backright_41", "frontleft_41"], [0.0, 2.9]),
                           (["a", "b"], [0.0, 0.5], ["a", "b"], [0.0, 0.5]),
                           (["a"], [0.0], ["b"], [0.0])):
        s, known = _shift(an, ao, bn, bo)
        shifts.append({"a_names": an, "a_offsets": ao, "b_names": bn, "b_offsets": bo, "shift": s, "known": known})
    return {"frames": frames, "projections": proj, "shifts": shifts}


def make(root):
    b = cs.built_clip(root)
    _, sc = cd.build("syn", variant="pose24", analysis_dir=b["adir"], cache_dir=b["cache_dir"], data_dir=b["data_dir"],
                     registry=b["reg"], log=lambda *a: None, write=False)
    sc = _persons(_mark(_trim(sc, F_KEEP)))
    errs = cd.validate_sidecar(sc)
    assert errs == [], errs
    return sc, _cases(sc)


def main():
    os.makedirs(FIX, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sc, cases = make(tmp)
    for name, obj in (("cvdebug_small.json", sc), ("cv_cases.json", cases)):
        with open(os.path.join(FIX, name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, allow_nan=False, separators=(",", ":")))
    print(f"wrote {FIX}")


if __name__ == "__main__":
    main()
