#!/bin/bash
# v0.8.1: 2D speed options, one bench2d run each (the live 2D step alone; quality vs the offline 2D).
# Run from the repository root on the GPU machine:  bash tools/live_bench/v081_2d_sweep.sh [preset] [seconds]
# Every run gets runs/bench2d/v081/<tag>/ ; a STOP file in that folder skips the remaining runs; a run that already
# has bench.json is skipped (re-run the script to resume). Ends with runs/bench2d/v081/compare.md.
# On a machine with Android emulators (MuMu), set CLOSE_EMULATORS=1 to close them before every run.
set -u
PRESET=${1:-rtx3060}
SEC=${2:-5}
OUT=runs/bench2d/v081
mkdir -p "$OUT"
B="--clip singles_05x --preset $PRESET --seconds $SEC"

close_emulators() {
  [ "${CLOSE_EMULATORS:-0}" = "1" ] || return 0
  for p in MuMuNxDevice.exe MuMuVMMHeadless.exe MuMuNxMain.exe MuMuVMMSVC.exe MuMuNxService.exe MuMuPlayer.exe; do
    taskkill.exe /f /im $p >/dev/null 2>&1 || true
  done
  echo "emulator processes left: $(tasklist.exe 2>/dev/null | grep -ci 'MuMu')"
}

bench() {
  tag=$1; shift
  [ -f "$OUT/STOP" ] && { echo "STOP: skip $tag"; return; }
  [ -f "$OUT/$tag/bench.json" ] && { echo "have $tag"; return; }
  close_emulators
  echo "=== $tag  $*  ($(date +%H:%M:%S))"
  python tools/live_run.py bench2d $B --run-dir "$OUT/$tag" "$@" 2>&1 | grep -a -E "bench2d:|steps/s|Error|error|Traceback" | tail -4
}

F="--set rtmpose.engine=fast"
FILE="--set sync.offsets=file"

# --- baselines: the v0.8.0 2D path
bench b0_v080_auto                                    # exactly v0.8.0 (auto sync: every source frame is posed)
bench b1_v080_file  $FILE                             # same, known offsets: only the frames the steps need

# --- engine: batched pose crops (one pose call per frame instead of one per person)
bench e0_fast       $FILE $F
bench e1_fast_1proc $FILE $F --set rtmpose.batch_views=true          # option 3: all 3 cameras in one process / call

# --- option 1: shrink while decoding (pyav / ffmpeg scale in the decoder; cv2 = decode 4K, then resize)
bench s1_cv2_1080   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=cv2   --set rtmpose.decode_short_side=1080
bench s2_pyav_1080  $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=pyav  --set rtmpose.decode_short_side=1080
bench s3_ffm_1080   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=ffmpeg --set rtmpose.decode_short_side=1080
bench s4_pyav_720   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=pyav  --set rtmpose.decode_short_side=720
bench s5_pyav_540   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=pyav  --set rtmpose.decode_short_side=540

# --- option 4: GPU decode
bench g1_nvdec_1080   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=nvdec   --set rtmpose.decode_short_side=1080
bench g2_nvcodec_1080 $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=nvcodec --set rtmpose.decode_short_side=1080

# --- option 2: detector every N frames (boxes from the last pose in between)
S="$FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=pyav --set rtmpose.decode_short_side=1080"
bench d2_det2   $S --set rtmpose.det_every=2
bench d3_det3   $S --set rtmpose.det_every=3
bench d5_det5   $S --set rtmpose.det_every=5
bench d10_det10 $S --set rtmpose.det_every=10

# --- extra: pose only the persons on the court (bystanders never reach the pose model)
bench c1_court1 $S --set rtmpose.court_filter_m=1.0

# --- option 5: lighter models
bench m1_wb_balanced    $S --set rtmpose.mode=balanced
bench m2_wb_lightweight $S --set rtmpose.mode=lightweight
bench m3_feet_perf      $S --set rtmpose.model=body_feet --set rtmpose.mode=performance
bench m4_feet_balanced  $S --set rtmpose.model=body_feet --set rtmpose.mode=balanced
bench m5_body_balanced  $S --set rtmpose.model=body --set rtmpose.mode=balanced

# --- stacks
bench x1_court_det3       $S --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench x2_court_det3_720   $FILE $F --set rtmpose.batch_views=true --set rtmpose.decoder=pyav --set rtmpose.decode_short_side=720 --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3
bench x3_court_det3_bal   $S --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=3 --set rtmpose.mode=balanced
bench x4_court_det5_feet  $S --set rtmpose.court_filter_m=1.0 --set rtmpose.det_every=5 --set rtmpose.model=body_feet --set rtmpose.mode=balanced

python tools/live_run.py bench2d-compare $(ls -d "$OUT"/*/ | sort) --out "$OUT/compare.md" > /dev/null
echo "SWEEP-DONE $(date +%H:%M:%S)"
