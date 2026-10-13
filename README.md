# BadmintonVR (v0.3.1)

Turns multi-camera phone video of a badminton rally into a **3D digital twin**: body pose,
court position, racket, shuttle, rally segments and facing direction. It then replays the rally in Unity on a
regulation court.

```
5 phone videos ─► 2D pose / racket / shuttle detection  (GPU scripts)
               ─► calibrate cameras + triangulate to 3D  (Python, CPU)
               ─► analysis.json                          (one file per clip)
               ─► Unity viewer                           (replay on a 3D court)
```

**What's included:** all the code, plus the finished 3D results for two recorded singles
rallies:
- `singles_1x`: 944 frames (15.7 s), phones on the standard 1× lens.
- `singles_05x`: 1877 frames (31.3 s), phones on the 0.5× ultra-wide lens.

**Videos:** the original phone footage for all five recorded clips is in `data/15-08/`, about
2.6 GB. Each clip is filmed by several phones at 4K, 60 fps.

| Folder | Clip ID | Cameras | Calibrated? | 3D result included? |
|---|---|---|---|---|
| `data/15-08/singles/1/` | `singles_1x` (1× lens) | 5 | yes (`*_11_cam.json`) | yes |
| `data/15-08/singles/0.5/` | `singles_05x` (0.5× ultra-wide) | 5 | yes (`*_051_cam.json`) | yes |
| `data/15-08/doubles/1 side/` | `doubles_1side` | 5 | yes (`*_21_cam.json`) | no |
| `data/15-08/doubles/2 sides/` | `doubles_2sides` | 5 | yes (`*_31_cam.json`) | no |
| `data/15-08/doubles/2 sides (4 angle)/` | `doubles_2sides_4angle` | 4 | yes (`*_41_cam.json`) | no |

The camera calibrations and player seeds for all five clips are included.

**What's not included:** the intermediate 2D detections (the GPU outputs of steps 2–4) and
any doubles 3D results. You can view the singles results in Unity straight away
(Section 2). Re-running the pipeline from the videos needs an NVIDIA GPU for steps 2–4
(Section 3).

---

## 1. Folder guide

| Folder | What's inside |
|---|---|
| `tools/` | **All Python code.** Scripts you run sit at the top level. `tools/analysis/` holds the pipeline modules. |
| `tools/analysis/` | The core 3D reconstruction: triangulation, player tracking, SMPL-24 skeleton, racket, shuttle, rallies, facing. Also a rule-based stroke classifier (`strokes.py`), which is tested but not yet connected to the build |
| `tools/tests/` | Unit tests (pytest). Not needed to run anything; they check the code is correct. |
| `tools/colab/`, `tools/kaggle/` | Notebook versions of the GPU steps, for running on Google Colab or Kaggle |
| `data/15-08/` | Original phone videos (recorded 15 Aug), one folder per clip (see the table above). Keep this layout: the calibrator looks for the videos here. |
| `data/calib/` | Camera calibrations (`*_11` = singles_1x, `*_051` = singles_05x, `*_21` = doubles_1side, `*_31` = doubles_2sides, `*_41` = doubles_2sides_4angle), player seeds (`*_seeds.json`) and court geometry |
| `Assets/StreamingAssets/analysis/` | **The results**: `singles_1x.analysis.json` and `singles_05x.analysis.json` |
| `Assets/` | Unity project: the viewer scene, C# playback scripts (`Assets/Scripts/Analysis/`) and editor tools |
| `Packages/`, `ProjectSettings/` | Unity project configuration. Leave these alone. |

---

## 2. Quick start: view the results (no video, no GPU)

**a) View in Unity.** Open this folder as a project in **Unity 6000.1.4f1** (URP) and open
`Assets/badminton.unity`. Press **Play**; `singles_05x` loads by default, and you can switch
to `singles_1x` in the clip menu. The stick-figure and solid-body renderers work. The
textured "character" renderer needs a model file that isn't included, so it just logs a
warning.

**b) Python checks.** These need **Python 3.12+**.

```bash
pip install -r tools/requirements.txt pytest
cd tools
```

> **Important:** run every Python command **from inside the `tools/` folder**. The modules
> import each other as `analysis.*`.

Print tracking-quality metrics for a result (coverage, gaps, position jumps per player):

```bash
python -m analysis.eval_tracking ../Assets/StreamingAssets/analysis/singles_1x.analysis.json
```

> **Known issue: ignore `wrong_side_frames` and `total_swap_suspect`.** They will show
> almost every frame (e.g. 944 and 1874). `eval_tracking.py` assumes the near player is at
> +Z, but the rest of the pipeline puts the near side at −Z. The players are actually on
> the correct sides throughout: see `z_range` (P1 negative, P2 positive). All the other
> metrics are correct.

Re-run just the smoothing stage on a result, with a different filter:

```bash
python -m analysis.build_real resmooth ../Assets/StreamingAssets/analysis/singles_1x.analysis.json --mode physics_bsp --out ../Assets/StreamingAssets/analysis/singles_1x_physics_bsp.analysis.json
```

Run the test suite (162 tests, a few seconds):

```bash
python -m pytest tests -q
```

---

## 3. Full pipeline (which scripts, in what order)

Five phones film the court from different angles (back, back-left, back-right, front-left,
front-right). These are the **only scripts needed** to go from video to the Unity replay:

| Step | Script | Runs on | Produces |
|---|---|---|---|
| 1 | `tools/serve_calibrator.py` | laptop | `data/calib/<view>_cam.json` + `<clip>_seeds.json` |
| 2 | `tools/pose2d_gpu.py` | NVIDIA GPU | `data/pose2d/<view>_kp2d.json` |
| 3 | `tools/racket2d_gpu.py` *(optional)* | NVIDIA GPU | `data/racket2d/<view>_racket2d.json` |
| 4 | `tools/shuttle2d_gpu.py` *(optional)* | NVIDIA GPU | `data/shuttle/<view>_shuttle.json` |
| 5 | `tools/analysis/build_real.py multi` | laptop | `<clip>.analysis.json` |
| 6 | Unity | laptop | 3D replay |

For all five clips, step 1 is already done: the calibrations and seeds are in
`data/calib/`. Run every command in this section from the project root, except step 5.

### Step 1: Calibrate cameras and seed players (browser UI)

```bash
python tools/serve_calibrator.py
```

This opens `http://localhost:8001`. For each camera, click known court points (corners,
service lines) and the script solves the camera pose (PnP). Then switch to **Seeds** mode
and click each player once, so the tracker knows who is who. It reads the videos from
`data/15-08/` and can also reopen the included calibrations to check or redo them.

### Steps 2–4: 2D detection (needs an NVIDIA GPU)

```bash
python tools/pose2d_gpu.py    --src data/15-08/singles/1 --clip singles_1x --out data/pose2d
python tools/racket2d_gpu.py  --src data/15-08/singles/1 --clip singles_1x --out data/racket2d --pose2d-dir data/pose2d
python tools/shuttle2d_gpu.py --src data/15-08/singles/1 --clip singles_1x --out data/shuttle --views back_11 backleft_11 backright_11 frontleft_11 frontright_11
```

- For another clip, change `--src` to its folder and `--clip` to its clip ID from the table
  at the top (and the view suffix in `--views`: `_051`, `_21`, `_31` or `_41`). Put quotes around folder
  names with spaces, e.g. `--src "data/15-08/doubles/1 side"`.
- **Shuttle `--views`:** without it, `shuttle2d_gpu.py` only processes the back camera.
  Step 5 needs all of them.
- **Racket `--pose2d-dir`:** run step 2 first. The racket script then also searches a crop
  around each player's wrist, which finds far more rackets.
- **Pose:** RTMPose Wholebody, 133 keypoints per person, multi-person. Needs `pip install rtmlib`.
- **Racket:** RacketVision, 5 keypoints. Needs the OpenMMLab stack; setup is in the docstring
  at the top of `racket2d_gpu.py`.
- **Shuttle:** TrackNetV3. Setup is in the docstring at the top of `shuttle2d_gpu.py`.
- The expected video filenames for each clip are listed in `CLIP_CAMERAS` at the top of each
  script. Add an entry there for new footage.
- No GPU? Use the notebooks in `tools/colab/` (Colab T4 runtime) or `tools/kaggle/`
  instead. They do the same job.

### Step 5: Build the 3D analysis.json (CPU, about 30–60 s per clip)

From inside `tools/`, give one flag **per camera**. Repeat the flag; don't list several
files after one flag. This is the exact command that produced the included
`singles_1x.analysis.json`. For `singles_05x`, replace `_11` with `_051` and `singles_1x`
with `singles_05x`.

```bash
cd tools
python -m analysis.build_real multi \
  --cam back_11=../data/calib/back_11_cam.json \
  --cam backleft_11=../data/calib/backleft_11_cam.json \
  --cam backright_11=../data/calib/backright_11_cam.json \
  --cam frontleft_11=../data/calib/frontleft_11_cam.json \
  --cam frontright_11=../data/calib/frontright_11_cam.json \
  --kp2d-multi back_11=../data/pose2d/back_11_kp2d.json \
  --kp2d-multi backleft_11=../data/pose2d/backleft_11_kp2d.json \
  --kp2d-multi backright_11=../data/pose2d/backright_11_kp2d.json \
  --kp2d-multi frontleft_11=../data/pose2d/frontleft_11_kp2d.json \
  --kp2d-multi frontright_11=../data/pose2d/frontright_11_kp2d.json \
  --racket2d back_11=../data/racket2d/back_11_racket2d.json \
  --racket2d backleft_11=../data/racket2d/backleft_11_racket2d.json \
  --racket2d backright_11=../data/racket2d/backright_11_racket2d.json \
  --racket2d frontleft_11=../data/racket2d/frontleft_11_racket2d.json \
  --racket2d frontright_11=../data/racket2d/frontright_11_racket2d.json \
  --shuttle back_11=../data/shuttle/back_11_shuttle.json \
  --shuttle backleft_11=../data/shuttle/backleft_11_shuttle.json \
  --shuttle backright_11=../data/shuttle/backright_11_shuttle.json \
  --shuttle frontleft_11=../data/shuttle/frontleft_11_shuttle.json \
  --shuttle frontright_11=../data/shuttle/frontright_11_shuttle.json \
  --seeds ../data/calib/singles_1x_seeds.json \
  --mode singles --clip singles_1x --fps 60 \
  --out ../Assets/StreamingAssets/analysis/singles_1x.analysis.json
```

`--racket2d` and `--shuttle` are optional; leave them out if you skipped steps 3–4.

**Doubles:** use the doubles view suffix, `--seeds ../data/calib/<clip>_seeds.json` and
`--mode doubles` (4 players). For example, for `doubles_1side` use views `back_21`,
`backleft_21`, `backright_21`, `frontleft_21`, `frontright_21`. `doubles_2sides_4angle` has
only four cameras (`back_41`, `backright_41`, `frontleft_41`, `frontright_41`). No doubles
results are included, so these haven't been checked in this package.

Internally this runs:
1. Cross-camera player tracking (Kalman filter, body-proportion signature, Hungarian matching).
2. Sub-frame time sync between the phones.
3. RANSAC triangulation of each joint into 3D.
4. Mapping to the SMPL-24 skeleton, bone-length enforcement and smoothing.
5. Facing direction, rally segmentation, racket and shuttle. (Stroke labels are not
   produced yet: `strokes.py` exists but `build_real` doesn't call it, so `stroke` is null.)

### Step 6: View in Unity

Put the output in `Assets/StreamingAssets/analysis/` (the `--out` above already does this),
then press Play and pick the clip.

---

## 4. Other scripts (helpers, not required)

| Command (from `tools/`) | Use it to |
|---|---|
| `python -m analysis.eval_tracking new.json --compare old.json` | Compare tracking quality between two results |
| `python -m analysis.synth_analysis --out <file>` | Generate a synthetic 2-player rally, for testing without footage |
| `python calibrate_court.py <video>` | Older single-camera option: click 4 court corners to get a ground-plane homography |
| `python calibrate_multiview.py <video> --view <id>` | Command-line alternative to the browser calibrator (one camera at a time) |
| `python click_seeds.py --clip <clip> --view <id>=<video>:red` | Command-line alternative to the browser seed clicker |
| `python check_position.py <video> --probe` | Click floor points to check the calibration gives correct real-world distances |

All other `.py` files (the rest of `tools/analysis/`, `byte_tracker.py` and
`racket_smoothing.py`) are **library modules**. They're imported by the scripts above, not
run directly.

---

## 5. Output format: analysis.json (schema v3)

Defined in `tools/analysis/schema.py`:

- **Skeleton:** SMPL-24 joint names and parents (`tools/analysis/smpl24.py`)
- **Per player, per frame:** `joints_flat` (24 joints × x, y, z, confidence; body-local,
  Y-up, metres), court `position` `[x, z]`, `racket`, `grip`, `facing`, and a `stroke`
  slot (null in this version)
- **Rallies:** `rallies` list of `{start, end, duration_s}`
- **Shuttle:** per-frame `[x, y, z]` or null
- **Court frame:** origin at the court centre; X = width (6.10 m), Z = length (13.40 m), Y up

## Requirements summary

- Python 3.12+ with `opencv-python`, `numpy`, `scipy` (`tools/requirements.txt`)
- Unity 6000.1.4f1 with URP (only for the 3D viewer)
- An NVIDIA GPU, Colab or Kaggle, only for the 2D detection steps (2–4)
