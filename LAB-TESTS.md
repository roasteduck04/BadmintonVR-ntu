# Lab PC — test campaign (v0.8.3): which settings make the live stream fastest, and what they cost

One command runs a list of tests unattended on the lab RTX 5090 PC and packs one zip to copy back; the laptop then
scores every test against the offline (post-processing) builds. Plan: `configs/live/campaign/lab_5090.yaml`.

- **Stage A (16 tests, ~20 min + TensorRT engine builds):** the 2D step alone, as fast as possible: CUDA vs TensorRT
  FP16 / FP32, full 4K vs shrink-to-1080 / 720 vs real 1080p / 720p video files, detector every 1 / 3 / 5 frames, one
  process per camera vs one for all, GPU video decoding. Scored on the laptop against the offline 2D (pixels).
- **Stage B (16 streams, ~25 min):** whole simulated-live streams: best_live (×2), as fast as possible, pure_live (×2),
  real 720p files (×2), TensorRT + 720p (×2), 30 fps, fit window 30 frames, warm-up 60 frames, both, and 28 s streams.
  Scored on the laptop against the 3-camera and 5-camera offline builds.
- **Unity group (4 streams):** the same streams with Unity in Play mode following them, to measure Unity's cost.

Tests whose requirement is missing (TensorRT, PyNvVideoCodec, the converted videos) are skipped with the reason, not
failed. Everything is resumable.

Paths below: code `~/BadmintonVR` (WSL), Unity project `C:\Users\sports_1\BadmintonVR-unity`
(`/mnt/c/Users/sports_1/BadmintonVR-unity` in WSL).

---

## 1. Update (WSL)

```bash
cd ~/BadmintonVR && source ~/venv-pc/bin/activate
git pull
pip install -r tools/requirements-pc.txt
```

Stop the live server first if it is running (Ctrl+C in its window): the campaign needs the GPU to itself.

## 2. TensorRT (optional but planned)

```bash
pip install tensorrt-cu13
python tools/gpu_check.py --video "data/15-08/singles/0.5/15.8 back 0.5-1.mov" --trt
```

- Must end with `gpu-check: OK (TensorRT)`. The first time it builds the engines (a few minutes); they are cached in
  `~/.cache/badmintonvr/trt/`.
- If `pip install tensorrt-cu13` fails, try `pip install tensorrt`. If gpu_check still says FAIL, paste its output —
  the campaign then simply skips the TensorRT tests.

GPU video decoding (optional, one test): `pip install PyNvVideoCodec` (skip if it fails).

## 3. Real 720p and 1080p videos (~2-4 min)

```bash
python tools/live_run.py transcode --clip singles_05x --preset rtx5090 --short-side 720
python tools/live_run.py transcode --clip singles_05x --preset rtx5090 --short-side 1080
```

Writes `data/15-08/singles/0.5_720p/` and `0.5_1080p/` (H.264, every frame kept, same 60 fps). Each line must say
`1877 frames ... read back 1877 frames`.

## 4. Check the plan (nothing runs)

```bash
python tools/live_run.py campaign run --plan lab_5090 --group all --dry-run --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

Lists every test; `SKIP: ...` shows what is missing.

## 5. Stage A + B — Unity NOT in Play mode (~45-60 min)

Close Unity or stop Play mode, keep the PC awake (Windows sleep pauses WSL), then:

```bash
python tools/live_run.py campaign run --plan lab_5090 --group main --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

- One line per test: `[k/N] <test>: done in 62 s`. Leave the window open.
- Stop early: in another WSL window `touch /mnt/c/Users/sports_1/BadmintonVR-unity/runs/campaign/lab_5090/STOP`
  (it stops after the current test). Run the same command again to continue; finished tests are kept (`--retry-failed`
  re-runs failed ones).

## 6. Unity group — Unity in Play mode (~6 min)

Open the Unity project, scene `badminton`, press **Play**, then:

```bash
python tools/live_run.py campaign run --plan lab_5090 --group unity --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

Unity shows each stream live (● LIVE) while it runs.

## 7. Copy back

Each group ends with `results zip: .../runs/campaign/lab_5090.zip` (the last one has everything). Copy
`C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090.zip` to the laptop (no frame files inside; tens of MB) and
send it. If you rerun later, pack again with `python tools/live_run.py campaign pack /mnt/c/Users/sports_1/BadmintonVR-unity/runs/campaign/lab_5090`.

On the laptop: `python tools/live_run.py campaign report <path to lab_5090.zip>` re-renders every report against the
offline builds, rescores every bench against the offline 2D and writes `runs/campaign_in/lab_5090/summary.md`.

## Problems

| problem | fix |
|---|---|
| a test `failed` | its folder has `campaign.log` and `run.log` / `bench.log`; the zip has them too |
| TensorRT tests skipped | step 2 (`gpu_check --trt` must say OK) |
| 720p / 1080p tests skipped | step 3 |
| unity tests skipped "not in Play mode" | press Play first; the heartbeat must be fresh (the page's pill turns green) |
| campaign stopped when the window closed | run the same command again: it continues where it stopped |
