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

## Round 6: racket + shuttle in the live stream (time and quality, nothing added to the server or Unity frames)

Files (all new): `rs_nets.py` (the three networks in plain PyTorch, layer names = the checkpoints), `rs_export.py`
(-> `models/*.onnx`, normalisation inside each graph; `check` vs the stored offline 2D), `rs_live.py` (pre / post
processing in NumPy + OpenCV, onnxruntime CPU / CUDA / TensorRT), `rs_stream.py` (2D inside the live 2D processes, 3D
in a process next to the runner -> `<run>/racket_shuttle.jsonl`), `rs_bench.py` (models alone, CUDA vs TensorRT),
`rs_check2d.py` (laptop: what each way of finding the racket costs in 2D), `rs_replay.py` (the 3D again on a run's
recorded 2D or on the offline stored 2D, in seconds), `rs_report.py` (scores runs against the offline racket /
shuttle). Hooks in `hooks.py` (also `analysis.live.sources` now).

Models (`models/`, MIT licences): RacketVision RTMDet-M racket detector (Hugging Face linfeng302/RacketVision-Models
`epoch_300.pth`; 104 MB ONNX, committed in two parts that `rs_live.model_path` joins), our fine-tuned RTMPose-M racket
keypoint model (`data/racket_finetune_run/best_final_epoch_60.pth`, the offline "bootstrap" racket), TrackNetV3
(`TrackNet_best.pt`). Laptop check of the exported models against the stored offline results (back_051):
racket keypoints on the guide boxes 0.004 px median (54 rackets), detector + keypoints: all 95 offline detector
candidates found, 0.02 px median - i.e. exact; TrackNet (8-frame windows, no InpaintNet, background from 240 frames)
16 px median at 4K vs the stored TrackNetV3 + InpaintNet (which ran every frame with positional weights).

Speed-up options looked at: TensorRT FP16 for all three (this round); no racket detector - boxes from the live wrists
(`wrist`); TrackNet once per 8 frames per camera instead of every frame (`exp_shuttle_stride`). Not built: TrackNetV4
(motion attention, +0.4 F1, slower), TrackNetV5-Lite (2.6 M parameters, weights not confirmed public), a smaller
racket model (RTMPose-S/T trained on our 9.4k auto labels; needs GPU training).

Laptop results (2026-10-31, CPU):
- 2D, what finding the racket costs (`rs_check2d.py`, back_051, 30 frames, 60 offline rackets, 4K px): the live
  1080p alone 0.35 px; no flip test 0.91 px; detector 70 % found, 2.8 px (2.3 px with the second tight pass);
  wrist boxes 57-75 % found, 4.5-17 px (best 1.2 racket lengths pushed 0.3 along the forearm: 73 %, 4.5 px;
  3.5 px with the second pass) -> the second pass (`exp_racket_refine`) is on by default.
- 3D on the offline stored 2D (`rs_replay.py offline`, 3 cameras, frames 300-900, RacketVision detector 2D) vs the
  offline 5-camera racket / shuttle: near player's racket 94 % of frames, grip 2.3 cm, head 3.4 cm, shaft 3.1 deg;
  far player 52 % (its racket is mostly seen by one of these cameras), head 8.5 cm, shaft 10 deg; shuttle 63 % of
  the offline frames, 10.5 cm median.
- 3D speed: it first took 39 ms (racket) + 40 ms (shuttle) per frame on the laptop. Now Gauss-Newton with analytic
  derivatives for both (scipy's numerical derivatives were ~90 % of the time), one solve from the last frame's racket,
  at most 8 short starts otherwise, a failed full fit not retried for 3 steps -> racket ~16 ms median for two
  players incl. the other-hand fallback, shuttle ~1.5 ms (`exp_racket3d_solver=scipy` = the offline solver, ~2x).
  2-camera shuttle points need 8 deg between the rays and <= 12 px (p90 111 -> 70 cm for 68 -> 63 % coverage).
- Racket hand: chosen each frame (the hand whose projection has racket handles nearest; switch only when the other
  is clearly nearer; the other hand if the fit fails) - a hand locked by its label went wrong for the far player,
  whose arms the live pose sometimes swaps (replay of a live stream: P2 7 % -> 87 % of frames, head 38 -> 9 cm).
- Live CPU streams (3 cameras, 30 fps; wrist + shuttle, detector every 3rd + shuttle): everything runs end to end
  (laptop smoke tests of 15-30 frames after the warm-up, not quality results; replayed with the final 3D: both
  rackets 87 % of frames, head ~9.5 cm, shaft ~9 deg; shuttle 14 cm, 4.4 cm one frame later - the offline shuttle
  file's timeline sits one frame off the live / pose24 one, so the report also gives the best whole-frame shift).
  With `exp_rs_async` on the laptop CPU the racket / shuttle thread fell far behind the pose and its results came
  after their frames were written (nothing fitted): the 3D now waits up to 1.5 s for a frame's 2D, and the thread
  drops its oldest step beyond 30 waiting (`worker.dropped` in rs_summary.json); the lab GPU decides.

Every run also records its 3D inputs (`rs_inputs.jsonl.gz`), so 3D settings are compared on the laptop afterwards
(`rs_replay.py run <run> --set shuttle3d=tri ...`) instead of in more lab runs.

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

Rounds 6 + 7 in one session (`lab_5090_r67`: 18 racket / shuttle streams on singles_05x, 11 doubles streams on the three
doubles clips, 1 doubles + racket + shuttle; models alone ~5 min, then ~60 min). This branch now also holds v0.8.6
(doubles, the new server page). The doubles videos must be in the WSL clone first:

```bash
cd ~/BadmintonVR && source ~/venv-pc/bin/activate && git fetch && git checkout test/r5-experiments && git pull
export PYTHONPATH="$PWD/tools/experiments"
for d in "1 side" "2 sides" "2 sides (4 angle)"; do mkdir -p "data/15-08/doubles/$d" && cp -n "/mnt/c/Users/sports_1/BadmintonVR/data/15-08/doubles/$d/"*.mov "data/15-08/doubles/$d/"; done
U=/mnt/c/Users/sports_1/BadmintonVR-unity
python tools/experiments/rs_bench.py --out $U/runs/campaign/lab_5090_r67/rs_bench_3cam
python tools/experiments/rs_bench.py --views 5 --out $U/runs/campaign/lab_5090_r67/rs_bench_5cam
python tools/live_run.py campaign run --plan lab_5090_r67 --group main --unity-project $U
```

The first `rs_bench` builds the three TensorRT engines (a few minutes, cached for the streams). Copy back
`C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090_r67.zip` (it includes the bench folders); on the laptop:
`python tools/experiments/rs_report.py <zip>` plus `campaign report` for the pose columns (doubles quality: the offline
4-player builds are in the archive folder). The separate plans `lab_5090_r6` / `lab_5090_r7` hold the same tests.
Laptop check of the merge: racket / shuttle 3D on the stored doubles_2sides 2D fits all four rackets (P1-P3 80-97 %,
P4 32 % of 300 frames) at ~26 ms per frame (four players; Z19 shows whether it keeps up at 60 fps).

Unity not in Play mode (this round measures the PC). Copy back
`C:\Users\sports_1\BadmintonVR-unity\runs\campaign\lab_5090_r5.zip` and `~/.cache/badmintonvr/int8/*.json`
(the INT8 check). Back to the normal code afterwards: `git checkout main` and a new terminal (or `unset PYTHONPATH`).
