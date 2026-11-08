#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
EPISODE_INDEX="${1:-${G1_DUST_REPLAY_EPISODE:-0}}"
DATASET_ROOT="${G1_DUST_REPLAY_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_DUST_REPLAY_OUTPUT_ROOT:-$TASK_ROOT/data/augmentation_previews/isaacsim_3d_dust_replay}"

if [[ ! "$EPISODE_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Episode index must be a non-negative integer." >&2
  exit 2
fi
OUTPUT_DIR="$OUTPUT_ROOT/episode_$(printf '%03d' "$EPISODE_INDEX")"
READY_FILE="/tmp/g1_3d_dust_replay_${UID}.ready"
LOG_FILE="$OUTPUT_DIR/isaacsim.log"
DDS_SETTLE_SECONDS="${G1_DUST_REPLAY_DDS_SETTLE_SECONDS:-12}"
DUST_MODE="${G1_DUST_MODE:-legacy}"
DUST_PROFILE="${G1_DUST_PROFILE:-train}"
DUST_PRESET="${G1_DUST_PRESET:-medium}"
AMBIENT_MULTIPLIER="${G1_DUST_AMBIENT_MULTIPLIER:-1.0}"
OBJECT_RESET_SEED="${G1_OBJECT_RESET_SEED:-260822}"
RECORDING_MARKER="${G1_DUST_RECORDING_MARKER:-/tmp/g1_dust_recording_${UID}.active}"
MAX_FRAMES="${G1_DUST_REPLAY_MAX_FRAMES:-}"
mkdir -p -- "$OUTPUT_DIR"
rm -f -- "$READY_FILE" "$RECORDING_MARKER"

launcher_pid=""
cleanup_replay() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
  rm -f -- "$READY_FILE" "$RECORDING_MARKER"
  return "$status"
}
trap cleanup_replay EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

G1_ENABLE_3D_DUST=1 \
G1_DUST_MODE="$DUST_MODE" \
G1_DUST_PROFILE="$DUST_PROFILE" \
G1_DUST_PRESET="$DUST_PRESET" \
G1_DUST_AMBIENT_MULTIPLIER="$AMBIENT_MULTIPLIER" \
G1_DISABLE_SUCCESS_CHECK=1 \
G1_SHOW_CAMERA_VIEWPORTS=0 \
G1_SIM_READY_FILE="$READY_FILE" \
G1_DUST_SEED="$((260822 + EPISODE_INDEX))" \
G1_OBJECT_RESET_SEED="$OBJECT_RESET_SEED" \
G1_DUST_RECORDING_MARKER="$RECORDING_MARKER" \
  "$TASK_ROOT/run_task.sh" --headless >"$LOG_FILE" 2>&1 &
launcher_pid=$!

for _ in {1..300}; do
  [[ -f "$READY_FILE" ]] && break
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "Isaac Sim exited before replay. See $LOG_FILE" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ ! -f "$READY_FILE" ]]; then
  echo "Timed out waiting for Isaac Sim. See $LOG_FILE" >&2
  exit 1
fi
echo "Isaac Sim ready; waiting ${DDS_SETTLE_SECONDS}s for DDS discovery..."
sleep "$DDS_SETTLE_SECONDS"

source "$CONDA_SH"
conda activate tv
replay_args=(
  --dataset "$DATASET_ROOT"
  --episode "$EPISODE_INDEX"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$MAX_FRAMES" ]]; then
  if [[ ! "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_DUST_REPLAY_MAX_FRAMES must be a positive integer" >&2
    exit 2
  fi
  replay_args+=(--max-frames "$MAX_FRAMES")
fi
python "$TASK_ROOT/replay_lerobot_episode_with_3d_dust.py" "${replay_args[@]}"

echo "Isaac Sim 3D-dust replay generated: $OUTPUT_DIR"
