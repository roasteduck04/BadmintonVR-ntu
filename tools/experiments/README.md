# Round-5 experiments (test branch `test/r5-experiments` only)

Tests ideas for **fewer resources and better quality** without touching the live code, the server or its flow: every
file here is new, and the existing code is only *wrapped* at run time, in processes started with this folder on
`PYTHONPATH` (`sitecustomize.py` attaches `hooks.py` when `analysis.live.pose2d_fast` / `framewriter` load - in the
campaign, each live run and every 2D / fit process they spawn). Without that `PYTHONPATH` nothing changes.

| idea | switch (a config key, set per test in `configs/live/campaign/lab_5090_r5.yaml`) | file |
|---|---|---|
| A smaller input for the same pose model (256x192 instead of 384x288) | `rtmpose.mode=balanced` (already supported) | - |
| B more cameras (4, 5) | `views=[...]` (already supported) | - |
| C INT8 pose model on TensorRT | `rtmpose.exp_int8=true` (model from `int8_calib.py`) | `hooks.py`, `int8_calib.py` |
| D staggered cameras: pose model every 2nd frame per camera, cameras offset, keypoints carried at their velocity in between | `rtmpose.exp_stagger=2` | `hooks.py` |
| F frames over a local UDP socket instead of files (what writing costs) | `output.exp_socket=udp://127.0.0.1:8899` | `hooks.py`, `frame_receiver.py` |

Laptop check (CPU, 2026-10-29): INT8 of every layer broke the model (150 px); INT8 of the Conv layers only (the
default) gave 4.5-5 px median body error vs FP32 with MinMax / entropy calibration (the live 2D budget is ~1.2 px);
percentile ran out of memory on the laptop - the lab run decides. Stagger and socket hooks ran in a CPU stream (60 / 60
frames received over UDP; each camera carried forward on the frames between detections).

E (a small model trained on our own 35 keypoints) is not here: it needs a training setup and hours of GPU time - a
separate project if A-D are not enough.

## Lab results (2026-10-30, `runs/campaign_in/lab_5090_r5/summary.md`)

- **A balanced (256x192):** 2D alone only 9 % faster (11.2 vs 13.3 ms) for 1.7x the 2D error (2.14 vs 1.26 px). Its
  streams (Y02, Y08, Y10, Y14, Y16) are **invalid**: its raw scores are ~5.5x lower (median 0.78 vs 4.30) and
  `pose2d_fast.score_to_wb` passes wholebody scores through unchanged, so every keypoint failed the score gates, no
  player was acquired and nothing was fitted (their speed is the speed of doing no 3D). Fixed for round 5b:
  `score_map_fit.py` fits a quantile map per keypoint group (body / feet / face / hands, 2,977 matched persons from A01
  vs A02 `dets.npz`) into `score_map_wb.json`, and `hooks.py` applies it whenever `rtmpose.mode=balanced` (pose24
  keypoints passing the 2.5 gate: 0 % -> 94 %, performance 94 %). Rerun plan `lab_5090_r5b` (the 5 balanced streams +
  Y01).
- **B cameras:** quality vs the 5-camera offline build (q5): 3 cameras 2.94 cm, back three 2.16 cm, 4 cameras 1.71 cm,
  5 cameras 1.28 cm. Speed: 3 cameras 57 fps (keeps up), 4 cameras 43.6 fps (0.73x), 5 cameras 34.5 fps (0.58x; 28 s
  0.64x), 5 cameras at 30 fps 0.87x (delay grows 1.25 -> 1.99 s). Per camera, decoding the 4K files goes 2.4 -> 10.9
  -> 23.5 ms (3 / 4 / 5 cameras) and the pose model 13 -> 18 -> 21 ms (GPU shared): with 5 cameras the CPU decoding is
  the new limit (threads asked for exceed the 48 logical CPUs; real 1080p camera streams would decode far cheaper).
- **C INT8:** TensorRT rejects the quantized bias (DequantizeLinear of int32) and runs the model in pieces: 2D 17 ms
  (slower than FP16's 13), stream 0.77x, q3 2.02 vs 1.39 cm; with 5 cameras the 5 engine builds exhausted WSL's memory
  and WSL closed (Y11 not run). INT8 check (percentile, 35 crops): body median 2.24 px, all-keypoint p90 32 px. Dropped.
- **D stagger 2:** 2D alone 1.30x real time (9.3 ms) and 1.53 px; stream 3 cameras same speed (it already kept up),
  q3 1.69 vs 1.39 cm (+0.3 cm); 5 cameras 41 vs 34.5 fps - helps, not enough (decoding stays).
- **F UDP:** same delay as files (1.30 s median both) since the v0.8.5 writer fix: no gain.
- 3 cameras, current setup (Y01) reproduces round 4: 57 fps, delay 1.30 s, q3 1.39 cm.

## Lab PC

```bash
cd ~/BadmintonVR && source ~/venv-pc/bin/activate && git fetch && git checkout test/r5-experiments && git pull
export PYTHONPATH="$PWD/tools/experiments"
python tools/experiments/int8_calib.py --device cuda --method percentile   # ~5 min: INT8 model + its 2D error vs FP32
python tools/live_run.py campaign run --plan lab_5090_r5 --group main --dry-run --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
python tools/live_run.py campaign run --plan lab_5090_r5 --group main --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

Round 5b (balanced streams again, with the score map; ~10 min):

```bash
cd ~/BadmintonVR && source ~/venv-pc/bin/activate && git fetch && git checkout test/r5-experiments && git pull
export PYTHONPATH="$PWD/tools/experiments"
python tools/live_run.py campaign run --plan lab_5090_r5b --group main --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

Copy back `C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090_r5b.zip`.

Unity not in Play mode (this round measures the PC). Copy back
`C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090_r5.zip` and `~/.cache/badmintonvr/int8/*.json`
(the INT8 check). Back to the normal code afterwards: `git checkout main` and a new terminal (or `unset PYTHONPATH`).
