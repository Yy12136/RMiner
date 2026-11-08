#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${G1_DUST_REPLAY_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_FLOW_FINAL_OUTPUT_ROOT:-$TASK_ROOT/data/augmentation_previews/flow_final_20260828}"
BASELINE_ROOT="$TASK_ROOT/data/augmentation_previews/flow_poc_20260827/flow_medium"
CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
  _bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  _bench_rel="$_bench_dir"
  for _bench_i in 1 2 3 4 5 6; do
    if [[ -f "$_bench_rel/env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$_bench_rel/env.sh"
      break
    fi
    _bench_rel="$(cd -- "$_bench_rel/.." && pwd)"
  done
  unset _bench_dir _bench_rel _bench_i
fi
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
EPISODE=0
monitor_pid=""
REQUESTED_VARIANTS="${G1_FLOW_FINAL_VARIANTS:-clean,final_train_low,final_train_medium,final_train_heavy,final_cinematic}"

cleanup_monitor() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

assert_runtime_idle() {
  local busy
  busy="$(pgrep -af "[e]valuation/supervisor.py|[p]ython .*sim_main.py" || true)"
  if [[ -n "$busy" ]]; then
    echo "Development Isaac Sim/DDS runtime is occupied; refusing to share it." >&2
    printf '%s\n' "$busy" >&2
    exit 75
  fi
}

assert_runtime_idle
test -s "$BASELINE_ROOT/episode_000/manifest.json"
mkdir -p -- "$OUTPUT_ROOT"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.before.sha256"

# Preserve the already-rendered, deterministic single-emitter Flow Medium POC
# as the requested "current Flow Medium" comparison baseline.
mkdir -p -- "$OUTPUT_ROOT/current_flow_medium_baseline"
cp -a "$BASELINE_ROOT/." "$OUTPUT_ROOT/current_flow_medium_baseline/"
printf '%s\n' \
  "Reused from flow_poc_20260827/flow_medium: same dataset, episode 0 and seeds." \
  >"$OUTPUT_ROOT/current_flow_medium_baseline/baseline_source.txt"

run_variant() {
  local variant="$1"
  local mode="$2"
  local profile="$3"
  local preset="$4"
  local ambient_multiplier="$5"
  local variant_root="$OUTPUT_ROOT/$variant"
  local gpu_csv="$variant_root/gpu.csv"
  assert_runtime_idle
  mkdir -p -- "$variant_root"
  echo "[flow-final] Rendering $variant: $mode/$profile/$preset ambient=$ambient_multiplier"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >"$gpu_csv" &
  monitor_pid=$!
  set +e
  G1_DUST_MODE="$mode" \
  G1_DUST_PROFILE="$profile" \
  G1_DUST_PRESET="$preset" \
  G1_DUST_AMBIENT_MULTIPLIER="$ambient_multiplier" \
  G1_DUST_SEED=260822 \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_DUST_REPLAY_OUTPUT_ROOT="$variant_root" \
    "$TASK_ROOT/run_3d_dust_replay.sh" "$EPISODE"
  local status=$?
  set -e
  cleanup_monitor
  if (( status != 0 )); then
    echo "[flow-final] Variant $variant failed with status $status" >&2
    return "$status"
  fi
  assert_runtime_idle
  for name in observation.images.head.mp4 observation.images.left.mp4 \
    observation.images.right.mp4 three_camera_grid.mp4 manifest.json; do
    test -s "$variant_root/episode_000/$name"
  done
}

should_run() {
  [[ ",$REQUESTED_VARIANTS," == *",$1,"* ]]
}

should_run clean && run_variant clean off train medium 1.20
should_run final_train_low && run_variant final_train_low flow train low 1.20
should_run final_train_medium && run_variant final_train_medium flow train medium 1.20
should_run final_train_heavy && run_variant final_train_heavy flow train heavy 1.20
should_run final_cinematic && run_variant final_cinematic flow cinematic medium 1.00

find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" \
  "$OUTPUT_ROOT/source_hashes.after.sha256"

source "$CONDA_SH"
conda activate tv
python "$TASK_ROOT/make_dust_final_report.py" --root "$OUTPUT_ROOT" \
  >"$OUTPUT_ROOT/report_stdout.json"
touch "$OUTPUT_ROOT/ALL_COMPLETE"
echo "[flow-final] Complete: $OUTPUT_ROOT/deliverables"
