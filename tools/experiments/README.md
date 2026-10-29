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

## Lab PC

```bash
cd ~/BadmintonVR && source ~/venv-pc/bin/activate && git fetch && git checkout test/r5-experiments && git pull
export PYTHONPATH="$PWD/tools/experiments"
python tools/experiments/int8_calib.py --device cuda --method percentile   # ~5 min: INT8 model + its 2D error vs FP32
python tools/live_run.py campaign run --plan lab_5090_r5 --group main --dry-run --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
python tools/live_run.py campaign run --plan lab_5090_r5 --group main --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity
```

Unity not in Play mode (this round measures the PC). Copy back
`C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090_r5.zip` and `~/.cache/badmintonvr/int8/*.json`
(the INT8 check). Back to the normal code afterwards: `git checkout main` and a new terminal (or `unset PYTHONPATH`).
