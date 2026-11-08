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
READY_FILE="/tmp/g1_3d_dust_preview_${UID}.ready"
OUTPUT="${G1_DUST_PREVIEW_OUTPUT:-$TASK_ROOT/data/augmentation_previews/isaacsim_3d_dust/isaacsim_head_3d_dust.mp4}"
PREVIEW_DURATION_SECONDS="${G1_DUST_PREVIEW_SECONDS:-15}"
DUST_MODE="${G1_DUST_MODE:-legacy}"
DUST_PROFILE="${G1_DUST_PROFILE:-train}"
DUST_PRESET="${G1_DUST_PRESET:-medium}"
AMBIENT_MULTIPLIER="${G1_DUST_AMBIENT_MULTIPLIER:-1.0}"
LOG_FILE="${OUTPUT%.mp4}.sim.log"

mkdir -p -- "$(dirname -- "$OUTPUT")"
rm -f -- "$READY_FILE"

launcher_pid=""
cleanup_preview() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
  rm -f -- "$READY_FILE"
  return "$status"
}
trap cleanup_preview EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

G1_ENABLE_3D_DUST=1 \
G1_DUST_MODE="$DUST_MODE" \
G1_DUST_PROFILE="$DUST_PROFILE" \
G1_DUST_PRESET="$DUST_PRESET" \
G1_DUST_AMBIENT_MULTIPLIER="$AMBIENT_MULTIPLIER" \
G1_DUST_SEED="${G1_DUST_SEED:-260822}" \
G1_OBJECT_RESET_SEED="${G1_OBJECT_RESET_SEED:-260822}" \
G1_SHOW_CAMERA_VIEWPORTS=0 \
G1_SIM_READY_FILE="$READY_FILE" \
  "$TASK_ROOT/run_task.sh" --headless >"$LOG_FILE" 2>&1 &
launcher_pid=$!

for _ in {1..240}; do
  [[ -f "$READY_FILE" ]] && break
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "Isaac Sim exited before becoming ready. See $LOG_FILE" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ ! -f "$READY_FILE" ]]; then
  echo "Timed out waiting for Isaac Sim. See $LOG_FILE" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate tv
python "$TASK_ROOT/record_3d_dust_preview.py" \
  --output "$OUTPUT" \
  --seconds "$PREVIEW_DURATION_SECONDS"

echo "3D dust preview generated: $OUTPUT"
