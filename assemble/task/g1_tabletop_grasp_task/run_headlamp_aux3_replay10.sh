#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT_OVERRIDE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_ROOT="${G1_AUX_REPLAY_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_AUX_REPLAY_OUTPUT:-$TASK_ROOT/data/augmentation_previews/headlamp_aux3_replay10_20260904}"
SEED="${G1_AUX_LIGHT_SEED:-260904}"
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
MONITOR_PID=""

cleanup() {
  status=$?
  if [[ -n "${MONITOR_PID:-}" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; fi
  exit "$status"
}
trap cleanup EXIT INT TERM

busy="$(pgrep -af '[e]valuation/supervisor.py|[p]ython .*sim_main.py|[r]un_3d_dust' || true)"
if [[ -n "$busy" ]]; then
  echo "Development Isaac Sim/DDS runtime is occupied; refusing to share it." >&2
  printf '%s\n' "$busy" >&2
  exit 75
fi

mkdir -p "$OUTPUT_ROOT/replays"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/source_hashes.before.sha256"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$OUTPUT_ROOT/gpu.csv" &
MONITOR_PID=$!

for episode in 0 1 2 3 4 5 6 7 8 9; do
  episode_dir="$OUTPUT_ROOT/replays/episode_$(printf '%03d' "$episode")"
  mkdir -p "$episode_dir"
  echo "[aux-replay10] episode=$episode"
  G1_HEADLAMP=on \
  G1_HEADLAMP_INTENSITY=30000000 \
  G1_HEADLAMP_AMBIENT_INTENSITY=80 \
  G1_HEADLAMP_CAMERA_EXPOSURE=0 \
  G1_HEADLAMP_NUREC_EMISSIVE_SCALE=0.16 \
  G1_HEADLAMP_CONE_ANGLE_DEG=32 \
  G1_HEADLAMP_CONE_SOFTNESS=0.45 \
  G1_HEADLAMP_COLOR_TEMP_K=4100 \
  G1_HEADLAMP_RADIUS_M=0.025 \
  G1_HEADLAMP_FOCUS=2.5 \
  G1_HEADLAMP_AIM_YAW_DEG=0 \
  G1_HEADLAMP_AIM_PITCH_DEG=0 \
  G1_AUX_LIGHTS=on \
  G1_AUX_LIGHT_COUNT=3 \
  G1_AUX_LIGHT_SEED="$SEED" \
  G1_AUX_LIGHT_EPISODE="$episode" \
  G1_AUX_LIGHT_INTENSITY=12000000 \
  G1_AUX_LIGHT_CONE_ANGLE_DEG=32 \
  G1_AUX_LIGHT_CONE_SOFTNESS=0.45 \
  G1_AUX_LIGHT_FOCUS=2.5 \
  G1_AUX_LIGHT_COLOR_TEMP_K=4100 \
  G1_AUX_LIGHT_RADIUS_M=0.025 \
  G1_AUX_LIGHT_ASSIGNMENT_FILE="$episode_dir/aux_light_assignments.json" \
  G1_DUST_MODE=off \
  G1_LIGHT_FLICKER=off \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_DUST_REPLAY_OUTPUT_ROOT="$OUTPUT_ROOT/replays" \
    "$TASK_ROOT/run_3d_dust_replay.sh" "$episode"
  test -s "$episode_dir/observation.images.head.mp4"
  test -s "$episode_dir/manifest.json"
  test -s "$episode_dir/aux_light_assignments.json"
done

if [[ -n "${MONITOR_PID:-}" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; MONITOR_PID=""; fi
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" "$OUTPUT_ROOT/source_hashes.after.sha256"

source "$CONDA_SH"
conda activate tv
python "$TASK_ROOT/make_aux_light_replay10.py" --root "$OUTPUT_ROOT" --dataset "$DATASET_ROOT"
python - "$OUTPUT_ROOT/replay10_report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("all_checks_passed", False):
    raise SystemExit("auxiliary-light replay validation failed")
PY
touch "$OUTPUT_ROOT/ALL_COMPLETE"
echo "[aux-replay10] Complete: $OUTPUT_ROOT"
