#!/bin/bash
# v0.8.1 extra live runs after the score-map fix: body_feet again, sync file vs auto, detector every frame vs every 5.
#   [SET=final] bash tools/live_bench/v081_live_extra.sh [preset] [out dir]      (repository root; resumable like v081_live_runs.sh)
set -u
PRESET=${1:-rtx3060}
OUT=${2:-runs/v081x}
FIT="--set stream.solver_tol=1e-4 --set stream.lsmr_maxiter=20"
P="--set rtmpose.engine=fast --set rtmpose.decoder=pyav --set rtmpose.decode_short_side=1080 --set rtmpose.court_filter_m=1.0"
FEET="--set rtmpose.model=body_feet --set rtmpose.mode=balanced"
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
  python tools/live_run.py run --clip singles_05x --preset "$PRESET" --run-dir "$OUT/$tag" --set backend=live $FIT "$@" 2>&1 | tail -2
}
if [ "${SET:-extra}" = "final" ]; then      # after the warm-up drop + shared-frame fixes: the chosen stack (p8) again, auto + file
  run p8_30_fast_auto   --set output_fps=30 --set pace=asfast $P --set rtmpose.det_every=5
  run p8_60_fast_auto   --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=5
  run p8_30_fast_file   --set output_fps=30 --set pace=asfast $P --set rtmpose.det_every=5 --set sync.offsets=file
  run p8_60_fast_file   --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=5 --set sync.offsets=file
  run p8_30_rt_file     --set output_fps=30 --set pace=realtime $P --set rtmpose.det_every=5 --set sync.offsets=file
  run det3_60_fast_auto --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=3
  run det3_60_fast_file --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=3 --set sync.offsets=file
  python tools/live_run.py compare $(ls -d "$OUT"/*/ | sort) --out "$OUT/compare" > /dev/null 2>&1
  echo "EXTRA-DONE $(date +%H:%M:%S)"
  exit 0
fi
run feet30_fast_auto  --set output_fps=30 --set pace=asfast $P --set rtmpose.det_every=5 $FEET
run feet60_fast_auto  --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=5 $FEET
run feet30_fast_file  --set output_fps=30 --set pace=asfast $P --set rtmpose.det_every=5 $FEET --set sync.offsets=file
run feet60_fast_file  --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=5 $FEET --set sync.offsets=file
run feet30_rt_file    --set output_fps=30 --set pace=realtime $P --set rtmpose.det_every=5 $FEET --set sync.offsets=file
run p8_30_fast_file   --set output_fps=30 --set pace=asfast $P --set rtmpose.det_every=5 --set sync.offsets=file
run p8_60_fast_file   --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=5 --set sync.offsets=file
run det1_60_fast_auto --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=1
run det1_60_fast_file --set output_fps=60 --set pace=asfast $P --set rtmpose.det_every=1 --set sync.offsets=file
python tools/live_run.py compare $(ls -d "$OUT"/*/ | sort) --out "$OUT/compare" > /dev/null 2>&1
echo "EXTRA-DONE $(date +%H:%M:%S)"
