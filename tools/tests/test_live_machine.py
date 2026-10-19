# tools/tests/test_live_machine.py
import time

from analysis.live import machine, samplers


def test_machine_report_has_the_core_fields_and_never_raises():
    m = machine.report()
    for k in ("host", "os", "python", "cpu", "cpu_logical", "ram_gb", "gpus", "onnxruntime", "packages", "git"):
        assert k in m
    assert isinstance(m["gpus"], list)
    assert "numpy" in m["packages"]
    assert machine.short(m).startswith(m["host"])


def test_gpu_list_parses_nvidia_smi(monkeypatch):
    outs = {"--query-gpu=name,memory.total,driver_version": "NVIDIA GeForce RTX 3060, 12288, 617.14\n",
            None: "| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4 |"}
    monkeypatch.setattr(machine, "_run", lambda cmd, timeout=10: outs.get(cmd[1] if len(cmd) > 1 else None))
    assert machine.gpus() == [{"name": "NVIDIA GeForce RTX 3060", "memory_mb": 12288, "driver": "617.14", "cuda": "13.4"}]


def test_no_nvidia_smi_means_no_gpus(monkeypatch):
    monkeypatch.setattr(machine, "_run", lambda cmd, timeout=10: None)
    assert machine.gpus() == []


def test_gpu_sampler_without_nvidia_smi_records_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(samplers.shutil, "which", lambda name: None)
    with samplers.GpuSampler(str(tmp_path / "g.csv")) as g:
        pass
    assert g.error == "nvidia-smi not found"
    assert samplers.summarise_gpu(str(tmp_path / "g.csv")) is None


def test_cpu_sampler_writes_rows(tmp_path):
    p = tmp_path / "c.csv"
    with samplers.CpuSampler(str(p), interval_ms=20):
        time.sleep(0.15)
    s = samplers.summarise_cpu(str(p))
    assert s["samples"] >= 2 and s["rss_max_mb"] > 0


def test_gpu_summary(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("timestamp,utilization_gpu,utilization_memory,memory_used,power_draw\n"
                 "t, 5, 1, 600, 20\nt, 40, 9, 1600, 90\nt, 30, 8, 1500, 80\n", encoding="utf-8")
    s = samplers.summarise_gpu(str(p))
    assert s["samples"] == 3 and s["util_max"] == 40 and s["mem_used_max_mb"] == 1600
    assert abs(s["util_busy_frac"] - 2 / 3) < 1e-9
