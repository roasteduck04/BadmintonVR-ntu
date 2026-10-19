#!/bin/bash
# v0.8.1, round 2: the 2D options again after the batch-padding fix (pose calls padded to 1/2/4/8/16/32 crops and
# warmed before the clock). Same rules as v081_2d_sweep.sh; folders runs/bench2d/v081b/<tag>/.
#   bash tools/live_bench/v081_2d_sweep_b.sh [preset] [seconds]
set -u
PRESET=${1:-rtx3060}
SEC=${2:-5}
OUT=runs/bench2d/v081b
mkdir -p "$OUT"
B="--clip singles_05x --preset $PRESET --seconds $SEC"

close_emulators() {
  [ "${CLOSE_EMULATORS:-0}" = "1" ] || return 0
  for p in MuMuNxDevice.exe MuMuVMMHeadless.exe MuMuNxMain.exe MuMuVMMSVC.exe MuMuNxService.exe MuMuPlayer.exe; do
    taskkill.exe /f /im $p >/dev/null 2>&1 || true
  done
}

bench() {
  tag=$1; shift
  [ -f "$OUT/STOP" ] && { echo "STOP: skip $tag"; return; }
  [ -f "$OUT/$tag/bench.json" ] && { echo "have $tag"; return; }
  close_emulators
  echo "=== $tag  $*  ($(date +%H:%M:%S))"
  python tools/live_run.py bench2d $B --run-dir "$OUT/$tag" "$@" 2>&1 | grep -a -E "steps/s|Error|error|Traceback" | tail -4
}

F="--set sync.offsets=file --set rtmpose.engine=fast"
PY="--set rtmpose.decoder=pyav --set rtmpose.decode_short_side=1080"
ONE="--set rtmpose.batch_views=true"

bench p1_pcam_full    $F                                                        # padded, one process per camera, 4K
bench p2_pcam_1080    $F $PY
bench p3_1proc_1080   $F $PY $ONE                                               # option 3 with padding
bench p4_pcam_court   $F $PY --set rtmpose.court_filter_m=1.0
bench p5_1proc_court  $F $PY $ONE --set rtmpose.court_filter_m=1.0
bench p6_pcam_court_det3  $F $PY --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench p7_1proc_court_det3 $F $PY $ONE --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench p8_pcam_court_det5  $F $PY --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=5
bench p9_pcam_court_det5_feet $F $PY --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=5 --set rtmpose.model=body_feet --set rtmpose.mode=balanced
bench pa_pcam_court_det3_720  $F --set rtmpose.decoder=pyav --set rtmpose.decode_short_side=720 --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench pb_pcam_court_det3_auto --set sync.offsets=auto --set rtmpose.engine=fast $PY --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench p0_rtmlib_ref   --set sync.offsets=file    # v0.8.0 engine, sync file (reference); last: on the remote PC its first two
                                                  # attempts died with WSL itself, so the new settings run first

python tools/live_run.py bench2d-compare $(ls -d "$OUT"/*/ | sort) --out "$OUT/compare.md" > /dev/null
echo "SWEEP-DONE $(date +%H:%M:%S)"
