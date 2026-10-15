"""A tiny synthetic 'full pose24 build' debug cache for the camsubsets tests."""
from __future__ import annotations

import numpy as np

import pose24_synth as syn
from analysis.pose24 import doc as docmod
from analysis.pose24 import kpset, mono, track

VIEWS = ["a", "b", "c", "d"]
BUILD_ID = "synthbuild0001"


def full_bundle(F=60, noise_px=0.5):
    cams = syn.rig(VIEWS)
    players = {"P1": syn.motion(F, z0=-3.0), "P2": syn.motion(F, x0=0.5, z0=4.0)}
    b = syn.bundle(players, cams, VIEWS, noise_px=noise_px, seed=3)
    for v in ("a", "b"):                                   # P2 unseen by a and b (SP1 F17 analogue)
        b["obs"]["P2"][v][..., 2] = 0.0
    n = F
    det = {p: np.stack([np.where((b["obs"][p][v][..., 2] > 0).any(-1), 0, -1)[:n] for v in VIEWS]).astype(int)
           for p in players}                                 # (V, F) like real caches: -1 = no detection there
    b.update({"det_index": det,
              "lr_swapped": {p: np.zeros((len(VIEWS), n), bool) for p in players},
              "offsets_all": {v: 0.0 for v in VIEWS}, "time_ref_view": "a", "timeline_of": None,
              "view_health_checked": True, "view_health": {v: 3.0 for v in VIEWS}, "shifts": {v: 0 for v in VIEWS}})
    b["lr_swapped"]["P1"][VIEWS.index("c"), 5:10] = True     # a known pattern (carried only, never fitted)
    return b, players


def full_cache(tmp_path, F=60):
    b, players = full_bundle(F)
    path = track.cache_path("synfull", str(tmp_path))
    per = {p: {"det_index": b["det_index"][p], "lr_swapped": b["lr_swapped"][p]} for p in b["obs"]}
    docmod.save_debug_cache(path, b, per, {"build_id": BUILD_ID, "out_clip": "synfull", "time_ref_view": "a",
                                           "offsets_all": b["offsets_all"], "timeline_of": None,
                                           "view_health_checked": True, "view_health": b["view_health"],
                                           "shifts": b["shifts"], "score_params": {}, "kpset_names": [],
                                           "joint_names": [], "players_lengths": {
                                               p: mono.lengths_from_points(kpset.from133(X) * np.array([1.0, -1.0, 1.0]))
                                               for p, X in players.items()}})
    return path, docmod.load_debug_cache(path), players
