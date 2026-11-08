#!/usr/bin/env bash
set -euo pipefail

# Task01 head-camera visual-only light-flicker demo.  This wrapper deliberately
# keeps the source dataset read-only and refuses to share Isaac Sim with formal
# evaluation processes.
TASK_ROOT="${TASK_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_ROOT="${G1_LIGHT_DEMO_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_LIGHT_DEMO_OUTPUT_ROOT:-$TASK_ROOT/data/augmentation_previews/light_flicker_demo_20260830}"
EPISODE="${G1_LIGHT_DEMO_EPISODE:-0}"
PREVIEW_SECONDS="${G1_LIGHT_DEMO_SECONDS:-15}"
monitor_pid=""

cleanup() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup EXIT INT TERM

assert_runtime_idle() {
  local busy
  busy="$(pgrep -af "[e]valuation/supervisor.py|[p]ython .*sim_main.py" || true)"
  if [[ -n "$busy" ]]; then
    echo "Development Isaac Sim/DDS runtime is occupied; refusing to start demo." >&2
    printf '%s\n' "$busy" >&2
    exit 75
  fi
}

test -d "$DATASET_ROOT/meta"
test -d "$DATASET_ROOT/data"
test -x "$TASK_ROOT/run_3d_dust_preview.sh"
assert_runtime_idle
mkdir -p -- "$OUTPUT_ROOT"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.before.sha256"

run_preview() {
  local output="$1" mode="$2" curve_path="$3"
  local started_ns finished_ns elapsed_seconds
  assert_runtime_idle
  rm -f -- "$output" "$curve_path"
  echo "[light-demo] Rendering $mode -> $output"
  started_ns="$(date +%s%N)"
  G1_DUST_MODE=off \
  G1_DUST_PRESET=medium \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_SEED=260822 \
  G1_DUST_PREVIEW_SECONDS="$PREVIEW_SECONDS" \
  G1_DUST_PREVIEW_OUTPUT="$output" \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_LIGHT_FLICKER="$mode" \
  G1_LIGHT_FLICKER_SEED=260830 \
  G1_LIGHT_FLICKER_CURVE_PATH="$curve_path" \
  G1_LIGHT_FLICKER_FORCE_CAMERA_REFRESH=1 \
  G1_LIGHT_FLICKER_CAMERA_EXPOSURE_FALLBACK=1 \
    "$TASK_ROOT/run_3d_dust_preview.sh"
  test -s "$output"
  if [[ "$mode" != off ]]; then test -s "$curve_path"; fi
  finished_ns="$(date +%s%N)"
  elapsed_seconds="$(awk -v a="$started_ns" -v b="$finished_ns" 'BEGIN { printf "%.6f", (b-a)/1000000000 }')"
  printf '%s\t%s\t%s\n' "$mode" "$elapsed_seconds" "$output" >>"$OUTPUT_ROOT/timing.tsv"
  assert_runtime_idle
}

mkdir -p -- "$OUTPUT_ROOT"
run_preview "$OUTPUT_ROOT/clean_15s.mp4" off "$OUTPUT_ROOT/clean_curve.json"

# Sample GPU memory/utilisation during the changing-light render.  Failure of
# nvidia-smi must not hide the actual preview failure.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 500 >"$OUTPUT_ROOT/gpu.csv" &
monitor_pid=$!
set +e
run_preview "$OUTPUT_ROOT/industrial_flicker_15s.mp4" industrial "$OUTPUT_ROOT/flicker_curve.json"
status=$?
set -e
cleanup
if (( status != 0 )); then exit "$status"; fi

find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" "$OUTPUT_ROOT/source_hashes.after.sha256"

python3 "$TASK_ROOT/make_light_flicker_demo.py" \
  --root "$OUTPUT_ROOT" \
  --hash-before "$OUTPUT_ROOT/source_hashes.before.sha256" \
  --hash-after "$OUTPUT_ROOT/source_hashes.after.sha256"
touch "$OUTPUT_ROOT/ALL_COMPLETE"
echo "[light-demo] Complete: $OUTPUT_ROOT"
