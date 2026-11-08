#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${G1_HEADLAMP_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_HEADLAMP_OUTPUT_ROOT:-$TASK_ROOT/data/augmentation_previews/headlamp_demo_20260831}"
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
MAX_FRAMES="${G1_HEADLAMP_MAX_FRAMES:-450}"
monitor_pid=""

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
  busy="$(pgrep -af '[e]valuation/supervisor.py|[p]ython .*sim_main.py' || true)"
  if [[ -n "$busy" ]]; then
    echo "Development Isaac Sim/DDS runtime is occupied; refusing to share it." >&2
    printf '%s\n' "$busy" >&2
    exit 75
  fi
}

assert_runtime_idle
mkdir -p -- "$OUTPUT_ROOT/raw"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.before.sha256"
: >"$OUTPUT_ROOT/timing.tsv"

run_variant() {
  local name="$1"
  local dust_mode="$2"
  local headlamp="$3"
  local lamp_intensity="$4"
  local ambient_intensity="$5"
  local variant_root="$OUTPUT_ROOT/raw/$name"
  local start_epoch end_epoch
  assert_runtime_idle
  mkdir -p -- "$variant_root"
  echo "[headlamp-demo] Rendering $name"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >"$variant_root/gpu.csv" &
  monitor_pid=$!
  start_epoch="$(date +%s)"
  set +e
  G1_LIGHT_FLICKER=off \
  G1_HEADLAMP="$headlamp" \
  G1_HEADLAMP_INTENSITY="$lamp_intensity" \
  G1_HEADLAMP_AMBIENT_INTENSITY="$ambient_intensity" \
  G1_HEADLAMP_CAMERA_EXPOSURE=0.0 \
  G1_HEADLAMP_NUREC_EMISSIVE_SCALE=0.14 \
  G1_HEADLAMP_CONE_ANGLE_DEG=44 \
  G1_HEADLAMP_CONE_SOFTNESS=0.50 \
  G1_HEADLAMP_COLOR_TEMP_K=4100 \
  G1_HEADLAMP_RADIUS_M=0.025 \
  G1_HEADLAMP_FOCUS=2.0 \
  G1_DUST_MODE="$dust_mode" \
  G1_DUST_PROFILE=train \
  G1_DUST_PRESET=medium \
  G1_DUST_AMBIENT_MULTIPLIER=1.20 \
  G1_DUST_SEED=260822 \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_DUST_REPLAY_OUTPUT_ROOT="$variant_root" \
  G1_DUST_REPLAY_MAX_FRAMES="$MAX_FRAMES" \
    "$TASK_ROOT/run_3d_dust_replay.sh" 0
  local status=$?
  set -e
  end_epoch="$(date +%s)"
  cleanup_monitor
  if (( status != 0 )); then
    echo "[headlamp-demo] $name failed with status $status" >&2
    return "$status"
  fi
  assert_runtime_idle
  local source_video="$variant_root/episode_000/observation.images.head.mp4"
  test -s "$source_video"
  cp -- "$source_video" "$OUTPUT_ROOT/${name}_15s.mp4"
  printf '%s\t%s\n' "$name" "$((end_epoch - start_epoch))" \
    >>"$OUTPUT_ROOT/timing.tsv"
}

run_variant normal_clean off off 0 1200
run_variant dark_dust_no_headlamp flow on 0 80
run_variant dark_dust_headlamp flow on 2000000 80

find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" \
  "$OUTPUT_ROOT/source_hashes.after.sha256"

source "$CONDA_SH"
conda activate tv
python "$TASK_ROOT/make_headlamp_demo.py" \
  --root "$OUTPUT_ROOT" --dataset "$DATASET_ROOT"
python - "$OUTPUT_ROOT/headlamp_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("all_checks_passed", False):
    raise SystemExit("Headlamp demo did not pass all acceptance checks")
PY
touch "$OUTPUT_ROOT/ALL_COMPLETE"
echo "[headlamp-demo] Complete: $OUTPUT_ROOT"
