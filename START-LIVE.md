# Lab PC — first live stream (v0.8.2): videos → one button → Unity

What you will see: a page in the browser with two tabs (1 Calibrate, 2 Stream). You press **Start streaming to
Unity**; the PC plays the singles 0.5x videos as if the cameras were live, builds both players' skeletons frame by
frame, writes one JSON file per frame, and Unity (in Play mode) shows each pose as soon as its file appears, with a red
**● LIVE** line.

Set-up on the lab PC: the code in WSL Ubuntu (`~/BadmintonVR-ntu`, venv `~/venv-pc`) and the Unity project on the
Windows side (below: `C:\Users\<you>\BadmintonVR-unity`). The server runs in WSL (it needs the GPU); Unity reads the
frame files from the Unity project's `runs\` folder. Time: about 30 min the first time.

---

## 1. Update the code (WSL)

Start → **Ubuntu** (or PowerShell: `wsl`). Then:

```bash
cd ~/BadmintonVR-ntu
git pull
source ~/venv-pc/bin/activate
pip install -r tools/requirements-pc.txt
```

`git pull` must end at the commit that adds `tools/pipeline_server/` (check: `ls tools/pipeline_server` shows
`server.py`). `pip` adds `fastapi`, `uvicorn` and `av` (PyAV, the fast video reader).

## 2. Check the GPU and record the machine

```bash
python tools/gpu_check.py --video "data/15-08/singles/0.5/15.8 back 0.5-1.mov"
python tools/live_run.py machine
```

- `gpu_check` must end with **OK (CUDA)**.
- **RTX 50xx:** if it fails with a CUDA / cuDNN / "no kernel image" error, `onnxruntime-gpu==1.22.0` has no RTX 5090
  kernels. Fix: `pip install -U "onnxruntime-gpu[cuda,cudnn]"`, run `gpu_check` again, and note the version it
  installed (`pip show onnxruntime-gpu`).
- Keep the `machine` output (CPU, GPU, RAM) for the report.

## 3. Videos

```bash
ls "data/15-08/singles/0.5"
```

Five `.mov` files must be listed. If the folder is missing but the videos are on the Windows side (e.g. in
`C:\BadmintonVR\data\15-08`), link them instead of copying:

```bash
mkdir -p data && ln -s "/mnt/c/BadmintonVR/data/15-08" data/15-08
```

## 4. Update the Unity project (Windows)

The Unity project needs the new viewer scripts (server follower, live broadcast view) and the start file.

- **If `C:\Users\<you>\BadmintonVR-unity` is a git clone of BadmintonVR-ntu:** PowerShell →
  `cd C:\Users\<you>\BadmintonVR-unity` → `git pull`.
- **Otherwise** copy the changed folders from the WSL clone (PowerShell; `Ubuntu` = your WSL distribution name,
  `wsl -l -v` lists it; `<wsluser>` = your Ubuntu user name):

  ```powershell
  $src = "\\wsl$\Ubuntu\home\<wsluser>\BadmintonVR-ntu"
  $dst = "C:\Users\<you>\BadmintonVR-unity"
  robocopy "$src\Assets\Scripts\Analysis" "$dst\Assets\Scripts\Analysis" /E
  robocopy "$src\Assets\Editor\Tests" "$dst\Assets\Editor\Tests" /E
  robocopy "$src\Assets\Resources\ViewerUI" "$dst\Assets\Resources\ViewerUI" /E
  copy "$src\start-pipeline-server.bat" "$dst\"
  ```

  If `Assets\Scripts\Analysis\AnalysisViewerUI.cs` is still in the Unity project, move it out of `Assets` (it is the
  retired old viewer).

## 5. Unity: open and press Play

Open the project in **Unity 6000.1.4f1**, scene `Assets/badminton.unity`, wait for the import, press **Play**.
The viewer appears; nothing streams yet.

## 6. Start the server

In File Explorer open `C:\Users\<you>\BadmintonVR-unity`, then either:

- PowerShell in that folder: `.\start-pipeline-server.bat wsl`, or
- right-click `start-pipeline-server.bat` → Create shortcut → shortcut Properties → Target: add ` wsl` at the end → OK,
  then double-click the shortcut.

If your WSL distribution is not called `Ubuntu`, or the clone / venv are elsewhere, edit the three `WSL_` lines at the
top of the .bat once.

A black window starts the server and the page opens (http://localhost:8770). Check the top bar:

- **Unity: in Play mode ✓** (green). Red → Unity is not in Play mode, or the server writes to another folder than the
  Unity project (the .bat must be in the Unity project folder).
- **GPU: rtx5090**.

## 7. First stream

Tab **2 Stream**: clip **singles_05x**, preset **best_live** (2D straight from the videos, 60 fps, sync file) or
**pure_live** (the same, but the camera offsets are estimated in the warm-up: nothing from an offline build). The source
line must say "straight from the videos". Press **Start streaming to Unity**.

- The first run downloads the 2D models once (~300 MB) and warms the GPU up; the first frame can take a minute.
- The page shows each phase (models, warm-up, streaming, saving, report) with its time, a bar that counts the frame
  files, and the full path of the folder
  (`C:\Users\<you>\BadmintonVR-unity\runs\live_singles_05x_60fps_live2d_realtime_<date>_<time>\unity_live_frames\`,
  a new folder per stream). **Open folder** opens it in Explorer to watch the JSON files appear one by one.
- When it ends, the **Time report** under Unity shows setup, speed, delay and the time of every stage.
- Unity shows **● LIVE — Broadcasting: processing the game in real time**, frames/s arriving and seconds behind the
  cameras. Every frame is shown in order; if the PC is slower than the cameras the broadcast falls behind (it does not
  skip frames). At the end: **Broadcast ended**, and the timeline comes back.

Then, for comparison, one stream with the clock set to **as fast as possible** (Stream tab → 2D source and output →
Clock), which measures the PC's top speed.

## 8. What to send back

- The `gpu_check` and `machine` outputs (and the onnxruntime-gpu version if you had to update it).
- For each stream: the run folder's `report.md` (Stream tab → Past streams → Report), or the whole run folder without
  `unity_live_frames`.
- A screenshot of Unity while it shows **● LIVE**, and the page's Past streams table.
- Notes: did Unity stay smooth, how far behind it got, anything odd.

## 9. Problems

| problem | fix |
|---|---|
| page does not open | open http://127.0.0.1:8770 by hand; the black window shows the error |
| `No module named 'av'` / `'fastapi'` | step 1 `pip install -r tools/requirements-pc.txt` in `~/venv-pc` |
| gpu_check fails on the RTX 5090 | step 2: newer `onnxruntime-gpu` |
| "Unity is not in Play mode" although it is | the .bat is not in the Unity project folder (the server's runs folder ≠ Unity's), or Unity is paused |
| clip line says NOT calibrated | tab 1 Calibrate (the singles 0.5x calibrations ship with the repo) |
| `sync file ... does not exist` | tab 1 → Camera sync → Make from the offline build, or set Camera offsets to "estimate in the warm-up" |
| videos not found | step 3 |
| Unity behind by many seconds | expected if the PC is slower than the cameras; send the numbers (that is what the test measures) |
