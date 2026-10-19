import json

import pose24_synth as syn
from analysis.live import offline_timing
from analysis.pose24 import doc as docmod
from analysis.pose24 import track


def test_offline_timing_on_a_cached_synthetic_build(tmp_path):
    views = ["a", "b", "c"]
    cams = syn.rig(views)
    b = syn.bundle({"P1": syn.motion(80)}, cams, views, noise_px=0.3)
    cache_dir = str(tmp_path / "cache")
    docmod.save_debug_cache(track.cache_path("synth", cache_dir), b, {}, {"time_ref_view": "a"})
    res = offline_timing.run("synth", ref_range=(10, 70), strides=(1, 2), cache_dir=cache_dir,
                             analysis_dir=str(tmp_path), log=lambda *a: None)
    assert res["runs"]["1"]["frames"] == 60 and res["runs"]["2"]["frames"] == 30
    assert res["runs"]["2"]["fps"] == 30.0 and res["runs"]["1"]["fit_s"] > 0 and res["doc_runtime_s"] is None
    json.dumps(res)
