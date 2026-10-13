# START HERE — video → 3D → Unity on the GPU PC (offline pipeline, v0.7.x)

**What this does:** takes the 5 recorded videos of the `singles_05x` rally, finds every person's body keypoints in
each video on the NVIDIA GPU (**all 5 cameras at the same time**), builds a 3D body for each player (pose24, SMPL-24),
and writes the file Unity plays. Pose only (no racket, no shuttle). Calibration is already done (`data\calib\`).
No Docker, no ROS: plain Python, like the earlier GPU runs on this PC.

```
 5 videos ──► Step 3: 2D pose, 5 cameras at once (GPU) ──► Step 4: 3D build "pose24" (CPU) ──► Part E: Unity
 data\15-08\singles\0.5\    data\pose2d\<camera>_kp2d.json       Assets\StreamingAssets\analysis\
                                                                  singles_05x_pose24.analysis.json
```

**Time budget:** Python setup 5–20 min (only once) · checks 2 min · 2D pose 10–25 min · 3D build ~5 min ·
first Unity import 10–30 min. **Disk:** ~6 GB free (Python packages ~2 GB, 2D output ~0.6 GB).

**Contents:** Part A PC checks · Part B get the files · Part C Python (Route A or B) · Part D run · Part E Unity ·
Part F send results · Part G problems.

---

## Part A — Check the PC (once)

1. Press **Start**, type `PowerShell`, open **Windows PowerShell**.
2. Run:
   ```powershell
   nvidia-smi
   ```
3. You should see a table with your GPU (e.g. `NVIDIA GeForce RTX 3060`, 12288MiB) and at the top
   **Driver Version 550 or higher**, **CUDA Version 12.4 or higher**.
   *"not recognized"* or an error → install the latest driver from nvidia.com → Drivers, restart, try again.
4. For Part E you need **Unity 6000.1.4f1** (Unity Hub → Installs → Install Editor; if it is not listed: Archive tab).

---

## Part B — Get the project folder from OneDrive

The OneDrive `BadmintonVR` folder already contains the new code (`tools`, `Assets`) and this guide.

1. **Download the project folder** (videos, calibration, Unity project) to **`C:\BadmintonVR`**:
   - **Browser:** select the `BadmintonVR` folder → **Download** → you get `BadmintonVR.zip` in `Downloads`.
     Right-click it → **Extract All…** → type `C:\` → **Extract**. You must end up with `C:\BadmintonVR\tools`,
     `C:\BadmintonVR\data`, `C:\BadmintonVR\Assets`. (If you get `C:\BadmintonVR\BadmintonVR\...`, move the inner
     folder's contents up one level, or use the inner folder as the project folder.)
   - **Synced OneDrive folder:** right-click `BadmintonVR` → **Always keep on this device**, wait for the green ticks,
     then **copy** (not move) it to `C:\`.
   - **Already have an older `C:\BadmintonVR` on the PC?** Download the folder again and replace it (the code changed).

   > Work in `C:\BadmintonVR`, **not inside the OneDrive folder**: OneDrive syncing files while they are written
   > causes locked files and slow runs.

2. Check it in PowerShell:
   ```powershell
   cd C:\BadmintonVR
   dir
   dir "data\15-08\singles\0.5"
   dir tools\pc_run.py, tools\pose2d_parallel.py, tools\gpu_check.py, tools\requirements-pc.txt
   ```
   - `dir` shows `Assets`, `data`, `tools`, `START-HERE.md` (and others).
   - The videos folder has these 5 files:

     | File | Size |
     |---|---|
     | `15.8 back 0.5-1.mov` | 140 MB |
     | `15.8 back left 0.5-1.mov` | 148 MB |
     | `15.8 back right 0.5-1.mov` | 147 MB |
     | `15.8 front left 0.5-1.mov` | 144 MB |
     | `15.8 front right 0.5-1.mov` | 148 MB |
   - The last `dir` lists the 4 new tools. If it says *cannot find*, the OneDrive folder still has the old code.

---

## Part C — Python: try Route A first, use Route B if it fails

Both routes end with the same check (`gpu-check: OK (CUDA)`). Then Part D is the same for both.

### Route A — the PC's existing conda environment `tf-gpu` (in WSL Ubuntu, used for the GPU runs in August)

1. In PowerShell, open the Ubuntu shell:
   ```powershell
   wsl
   ```
   The prompt changes to something like `user@PC:/mnt/c/Users/...$`. (Commands in this route are Linux commands.)
2. Activate the environment and go to the project:
   ```bash
   conda env list
   conda activate tf-gpu
   cd /mnt/c/BadmintonVR
   python --version
   ```
   - `conda env list` must show `tf-gpu`. If `conda` is *not found*, or there is no `tf-gpu` → **use Route B**.
   - `python --version` must be **3.10, 3.11 or 3.12**. Older → **use Route B**.
3. Add the packages the 3D build needs (the 2D packages should already be there):
   ```bash
   pip install scipy networkx matplotlib threadpoolctl tqdm
   ```
4. Check the GPU:
   ```bash
   python tools/gpu_check.py --video "data/15-08/singles/0.5/15.8 back 0.5-1.mov"
   ```
   - **OK** = the last line is `gpu-check: OK (CUDA)` → go to **Part D** (stay in this Ubuntu shell; use the
     `python tools/...` commands with `/` slashes).
   - `FAIL`, `No module named 'rtmlib'` / `'onnxruntime'` / `'cv2'`, or any other error → **use Route B**
     (type `exit` to leave Ubuntu). Do not try to repair `tf-gpu`.

### Route B — a fresh Python environment on Windows (if Route A failed)

1. Check Python in PowerShell:
   ```powershell
   py --list
   ```
   You need **3.10, 3.11 or 3.12** in the list. If `py` is not recognized or none of those is listed: install
   **Python 3.12** from python.org → Downloads → Windows → "Windows installer (64-bit)"; in the installer **tick
   "Add python.exe to PATH"**, then **Install Now**. Close and reopen PowerShell, run `py --list` again.
2. Create the environment in the project folder (replace `3.12` with the version you have):
   ```powershell
   cd C:\BadmintonVR
   py -3.12 -m venv .venv-pc
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv-pc\Scripts\Activate.ps1
   ```
   The prompt now starts with `(.venv-pc)`. **In every new PowerShell window**, repeat the last three lines
   (`cd`, `Set-ExecutionPolicy`, `Activate.ps1`) before running anything.
3. Install the packages (~2 GB download, 5–15 min; includes CUDA 12 and cuDNN as pip packages, so no separate CUDA
   install is needed):
   ```powershell
   python -m pip install --upgrade pip
   pip install -r tools\requirements-pc.txt
   ```
   OK when it ends with `Successfully installed ...` and no red `ERROR`.
4. Check the GPU (the first time it also downloads the pose model, ~300 MB):
   ```powershell
   python tools\gpu_check.py --video "data\15-08\singles\0.5\15.8 back 0.5-1.mov"
   ```
   Expected (the important lines):
   ```
   onnxruntime 1.22.0, package providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
   det_model: ['CUDAExecutionProvider', 'CPUExecutionProvider']
   pose_model: ['CUDAExecutionProvider', 'CPUExecutionProvider']
   first frame (cold): 1500 ms, 4 people
   next 10 frames: median 70 ms per frame (model only)
   gpu-check: OK (CUDA)
   ```
   (the ms numbers are examples.) **OK when** the last line is `gpu-check: OK (CUDA)` → go to **Part D**.
   If it says `FAIL`: send me the whole output (do not continue — on the CPU the 2D step would take hours).

---

## Part D — Run it

Use the shell from Part C (Route A: Ubuntu with `tf-gpu` active, in `/mnt/c/BadmintonVR`; Route B: PowerShell with
`(.venv-pc)`, in `C:\BadmintonVR`). The commands are the same; in PowerShell you may write `\` instead of `/`.

### D1. One command (recommended)

```
python tools/pc_run.py
```

It runs steps 1–5 below in order. For each step it prints `===== <step> =====` and the command, shows the output
live, and ends with a line like `----- pose2d_parallel5: 812.3 s, exit code 0 (log: ...results/pose2d_parallel5.log) -----`.
It **stops at the first failure** (`STOPPED: ... failed`). At the end it prints
`Results are in: ...\results -> zip this folder and upload it to OneDrive.` and a **timing summary** of every
section, also saved to `results/timing_summary.txt`:

```
===== timing summary =====
step                                 seconds  exit
check_nvidia_smi                         0.2  0
gpu_check                                7.3  0
pose2d_parallel5                      1876.9  0
    5 cameras, 5 at a time, 1877 frames each: 199.96 ms per camera frame (incl. start-up)
    model + tracking only: 195.2 ms per frame per camera
    back_051         total   1284.9 s   model load   8.1 s   frames 1877   fps 1.403
    ...                                 (one line per camera)
pose24_build                           683.2  0
    track                               259.0          <- camera sync + player tracking
    lrswap                               12.0          <- left/right fix
    fit                                 452.0          <- 3D body fit
    regress_confidence                    2.0          <- SMPL joints + confidence
    total (build's own clock)           726.0
collect                                  0.1  0
all steps                             2568.2  (42.8 min)
```

(step and stage times are from the 2026-10-12 run on the RTX 3060; the "model + tracking" and "model load" values are
examples — that run predates them. "model load" = loading the pose model onto the GPU.)

Re-run one step: `python tools/pc_run.py --step check` (or `gpu`, `pose2d`, `pose24`, `collect`).
Print the commands without running: `python tools/pc_run.py --dry-run`.

### D1b. Short test — only 1 or a few frames per camera (2D only)

Add **`--max-frames N`** (how many frames per camera) and optionally **`--start-frame F`** (which frame to start at;
default 0 = the first frame):

```
python tools/pc_run.py --max-frames 1 --start-frame 900
```

- Runs check, GPU check and the 2D step on frame 900 of all 5 cameras at the same time, then the timing summary.
- The 3D build is **skipped**: it needs the whole clip (sync, tracking and the body fit work over time).
- Output goes to **`data/pose2d_frames/`**, so the full-clip 2D files in `data/pose2d/` are not overwritten.
- For each camera you get: model load time, the time of **each frame** (`per-frame ms [...]`) and the total. The first
  frame is always slower (GPU warm-up); with `--max-frames 5` you also see the normal per-frame speed.
- To reach frame F the video is decoded up to F first (frame-accurate; a few seconds, not counted in the frame times).
- Same options on the 2D step alone: `python tools/pc_run.py --step pose2d --max-frames 1`, or the launcher
  `python tools/pose2d_parallel.py --src data/15-08/singles/0.5 --clip singles_05x --out data/pose2d_frames --max-frames 1 --start-frame 900`.

### D2. The steps, one by one

#### Step 1 — check (seconds)

```
python tools/pc_run.py --step check
```

Lists the 5 videos (`ok` + size), the Python packages (`ok` + version), saves `results/pc_info.json`
(Python, CPU, cores) and runs `nvidia-smi`. **OK when** nothing says `MISSING` and it ends with `exit code 0`.

#### Step 2 — GPU check (~30 s)

```
python tools/pc_run.py --step gpu
```

The same check as in Part C, saved to `results/gpu_check.log`. **OK when** the last line is `gpu-check: OK (CUDA)`.

#### Step 3 — 2D pose, all 5 cameras at the same time (GPU, ~10–25 min)

```
python tools/pc_run.py --step pose2d
```

(the same as `python tools/pose2d_parallel.py --src data/15-08/singles/0.5 --clip singles_05x --out data/pose2d --parallel 5`)

What happens: 5 separate Python processes start, one per camera. Each loads the pose model (RTMPose Wholebody,
133 keypoints per person) onto the GPU and goes through its video frame by frame: 1877 frames, 4K, 60 fps = 31.3 s
of play. You will see:

```
singles_05x: 5 cameras, 5 at a time
  start back_051
  start backleft_051
  start backright_051
  start frontleft_051
  start frontright_051
                                          (then nothing for several minutes while they work)
  done  frontright_051: 501.2 s ok        (cameras finish at different times: more people in view = slower)
  ...
singles_05x: 5 cameras, 5 at a time -> 812.3 s total
  1877 frames per camera: 86.5 ms per camera frame, 432.7 ms per 5-camera frame
  view              start s  total s  frames    fps
  back_051             0.01    456.7    1877    4.3
  ...
```

(numbers are examples.)

- **Watch the GPU while it runs:** open a second PowerShell window, run `nvidia-smi -l 2` (updates every 2 s;
  Ctrl+C stops it). Expect 5 `python` processes, GPU memory ~6 GB of 12 GB, GPU utilisation 80–100 %.
- **Output:** `data/pose2d/back_051_kp2d.json` … `frontright_051_kp2d.json` (5 files, 50–200 MB each),
  `data/pose2d/singles_05x_parallel_timing.json` (the timing summary), one log per camera in `data/pose2d/_logs/`.
  `pc_run.py` also copies the summary and the logs to `results/pose2d_parallel5/`.
- **OK when** every camera says `ok`. A camera with `FAILED`: the last lines of
  `data/pose2d/_logs/singles_05x_<camera>.log` say why. Out of GPU memory → `python tools/pc_run.py --step pose2d --parallel 3`.
- **Reference (RTX 3090, 2026-10-08):** 5 at once 562 s (60 ms per camera frame), one after another 866 s (92 ms).
  The GPU is full at 2–3 cameras, so 5 at once is ~1.5× faster, not 5×. The RTX 3060 is slower: expect more.

**Optional — one camera after another, for comparison** (another 15–30 min; separate folder, the 3D build still
reads `data/pose2d`):

```
python tools/pc_run.py --step pose2d --parallel 1 --out data/pose2d_seq
```

#### Step 4 — 3D build for Unity (CPU, ~5 min)

```
python tools/pc_run.py --step pose24
```

(the same as `python -m analysis.pose24 build --clip singles_05x` with `PYTHONPATH=tools`)

What it does: syncs the 5 cameras in time, follows the two players across the cameras (and ignores bystanders),
fixes left/right mix-ups, fits a 3D body (SMPL-24) to each player in every frame, and writes the Unity file.
Expected output (test run on 2026-10-12):

```
[pose24] singles_05x: singles, views back_051, backleft_051, backright_051, frontleft_051, frontright_051
    no base doc on this timeline: racket / shuttle / rallies not carried     <- normal (pose only)
    appearance features: none (run analysis.appearance)                      <- normal
    sync backleft_051 vs back_051: +0.5 frames (0.8px, P2)
    ...
    tracker3d: 2366 tracklets (player 53, nonplayer 1685, unknown 628), 3 pinned, 49 bystander hotspots
    P1: pose 100.0% (measured 100.0%), stature ~1.49 m, median sigma 26 mm
    P2: pose 99.9% (measured 99.5%), stature ~1.42 m, median sigma 129 mm
    wrote .../Assets/StreamingAssets/analysis/singles_05x_pose24.analysis.json; runtime track 123s, lrswap 6s, fit 172s, regress_confidence 1s, total 302s
```

- **OK when** the last line says `wrote ... singles_05x_pose24.analysis.json`. Your numbers will be close, not
  identical (fresh 2D detections).
- **Output:** `Assets/StreamingAssets/analysis/singles_05x_pose24.analysis.json` (~7 MB) and
  `Assets/StreamingAssets/analysis/singles_05x.persons.json` (~2 MB); a debug cache in `data/pose_cache/`.

#### Step 5 — collect (seconds)

```
python tools/pc_run.py --step collect
```

Copies the two Unity files into `results/unity_files/`; then the timing summary is printed (see D1).

---

## Part E — View it in Unity

1. **Unity Hub → Projects → Add → Add project from disk** → select `C:\BadmintonVR` → **Add Project**.
2. Open it with **6000.1.4f1** (pick it if Hub shows a version warning). The **first open imports everything:
   10–30 min**.
3. If Unity asks to enter **Safe Mode**: click **Ignore**, open **Window → General → Console**, take a screenshot
   and send it.
4. **Project** window → `Assets` → double-click the scene **`badminton`**.
5. Press **Play** (▶ at the top).
6. In the viewer panel: **clip** → `singles_05x`; **pose** row → **SMPL-24**. Play / pause and drag the timeline to
   scrub; drag in the view to orbit, scroll to zoom.
7. Screenshots with **Win + Shift + S** (e.g. a smash, both players moving) → save them into `C:\BadmintonVR\results\`.

On another computer instead: copy the two files from `results\unity_files\` into that project's
`Assets\StreamingAssets\analysis\`, then steps 4–6.

---

## Part F — Send the results back (OneDrive)

1. Right-click `C:\BadmintonVR\results` → **Send to → Compressed (zipped) folder** → name it
   `results-<PC name>-<date>.zip`.
2. Upload the zip to OneDrive (next to the `BadmintonVR` folder).

`results\` holds: `step_times.txt` (time of every step), `pc_info.json`, one `.log` per step,
`pose2d_parallel5\` (timing summary + per-camera logs), `unity_files\`, your screenshots.
Do **not** upload `data\pose2d\` (~550 MB) unless asked.

---

## Part G — When something goes wrong

| What you see | What to do |
|---|---|
| `nvidia-smi` not recognized | install the NVIDIA driver (Part A), restart |
| Route A: `conda: command not found` / no `tf-gpu` / Python older than 3.10 | use Route B |
| Route A: gpu_check `FAIL` or `No module named ...` | use Route B (don't repair `tf-gpu`) |
| `py` not recognized | install Python 3.12 with "Add python.exe to PATH" ticked (Route B step 1) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first |
| gpu_check: `this onnxruntime has no CUDA provider` | `pip uninstall -y onnxruntime onnxruntime-gpu`, then `pip install -r tools\requirements-pc.txt` |
| gpu_check: `FAIL (a sub-model runs without CUDAExecutionProvider)` | stop, send the output (`results\gpu_check.log`) |
| Step 1: `MISSING data\15-08\...` | videos not in `data\15-08\singles\0.5\` or renamed (Part B step 2) |
| Step 1: `MISSING <package>` | Route B: `pip install -r tools\requirements-pc.txt`; Route A: `pip install <package>` |
| Step 3: a camera `FAILED`, log mentions `memory` / `CUDA` / `allocate` | `python tools/pc_run.py --step pose2d --parallel 3` |
| Step 3 shows nothing for minutes | normal: it prints when a camera finishes; check `nvidia-smi -l 2` (GPU busy = working) |
| Step 4: `FileNotFoundError ... _kp2d.json` | step 3 did not finish for every camera; run it again |
| Unity: clip not in the list | the file must be in `Assets\StreamingAssets\analysis\`; stop and press Play again |
| anything else | zip and send the `results` folder plus a screenshot of the window |
