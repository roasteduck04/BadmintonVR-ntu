# tools/tests/test_live_report.py
import json
import os

import test_live_runner as tr
from analysis.live import report
from analysis.pose24 import build


def test_run_report_and_compare_render_from_a_synthetic_run(tmp_path):
    c = tr._cfg(n=70, warm=30)
    c["paths"] = dict(c["paths"], analysis_dir=str(tmp_path / "analysis"))
    c["reference_clips"] = ["synth_ref", "missing_ref"]
    res, run = tr._run(tmp_path, c)
    os.makedirs(tmp_path / "analysis")
    ref = json.loads((run / res["final_doc"]).read_text(encoding="utf-8"))   # the run itself as reference -> 0 error
    with open(build.output_path(str(tmp_path / "analysis"), "synth_ref"), "w", encoding="utf-8") as fh:
        json.dump(ref, fh)
    md = report.render_run(str(run), log=lambda *a: None)
    txt = open(md, encoding="utf-8").read()
    for s in ("## Summary", "## Seedless start", "P1 acquired", "## Timing", "### vs `synth_ref`",
              "Reference `missing_ref` not found", "## Reference-free checks", "Resolved config"):
        assert s in txt, s
    rj = json.loads((run / "report.json").read_text(encoding="utf-8"))
    g = rj["quality"]["synth_ref"]["groups"]["after warm-up"]
    assert g["w_mpjpe_cm"]["median"] < 1e-6 and rj["timing"]["written"] == 70
    assert (run / "plots" / "latency_per_step.png").exists()
    out = report.render_compare([str(run), str(run)], str(tmp_path / "cmp"), log=lambda *a: None)
    assert json.loads((tmp_path / "cmp" / "compare.json").read_text(encoding="utf-8"))["runs"][0]["mpjpe_cm"] < 1e-6
