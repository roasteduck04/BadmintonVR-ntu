#!/bin/bash
# v0.8.1 full live runs: the six D2 runs (cached / live 2D, 30 / 60 fps, realtime / asfast) with the v0.8.1 fit settings
# and a 2D stack, so they compare one-to-one with v0.8.0's D2. Run from the repository root:
#   STACK="--set rtmpose.engine=fast ..." bash tools/live_bench/v081_live_runs.sh [preset] [out dir]
# A run folder that already has report.md is skipped (resumable); a STOP file in the out dir skips the rest.
set -u
PRESET=${1:-rtx3060}
OUT=${2:-runs/v081}
STACK=${STACK:-"--set rtmpose.engine=fast"}
FIT="--set stream.solver_tol=1e-4 --set stream.lsmr_maxiter=20"
mkdir -p "$OUT"

close_emulators() {
  [ "${CLOSE_EMULATORS:-0}" = "1" ] || return 0
  for p in MuMuNxDevice.exe MuMuVMMHeadless.exe MuMuNxMain.exe MuMuVMMSVC.exe MuMuNxService.exe MuMuPlayer.exe; do
    taskkill.exe /f /im $p >/dev/null 2>&1 || true
  done
}

run() {
  tag=$1; shift
  [ -f "$OUT/STOP" ] && { echo "STOP: skip $tag"; return; }
  [ -f "$OUT/$tag/report.md" ] && { echo "have $tag"; return; }
  close_emulators
  echo "=== $tag  ($(date +%H:%M:%S))"
  python tools/live_run.py run --clip singles_05x --preset "$PRESET" --run-dir "$OUT/$tag" $FIT "$@" 2>&1 | tail -3
}

run cached30_rt  --set backend=cached --set output_fps=30 --set pace=realtime
run cached60_rt  --set backend=cached --set output_fps=60 --set pace=realtime
run live30_rt    --set backend=live --set output_fps=30 --set pace=realtime $STACK
run live60_rt    --set backend=live --set output_fps=60 --set pace=realtime $STACK
run live30_fast  --set backend=live --set output_fps=30 --set pace=asfast $STACK
run live60_fast  --set backend=live --set output_fps=60 --set pace=asfast $STACK
python tools/live_run.py compare $(ls -d "$OUT"/*/ | sort) --out "$OUT/compare" > /dev/null 2>&1
echo "LIVE-DONE $(date +%H:%M:%S)"
