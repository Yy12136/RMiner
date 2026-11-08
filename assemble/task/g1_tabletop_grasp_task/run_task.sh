#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE_ROOT="$(cd -- "$TASK_ROOT/../.." && pwd)"
SIM_ROOT="$ASSEMBLE_ROOT/third_party/unitree_sim_isaaclab"
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
CAMERA_VIEWER="$TASK_ROOT/show_g1_cameras.py"
TASK_ID="${G1_TABLETOP_TASK_ID:-Isaac-TabletopGrasp-G129-Dex1-Joint}"
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_tabletop_grasp_${UID}.ready}"
SIM_LOCK_FILE="${G1_SIM_LOCK_FILE:-/tmp/g1_tabletop_grasp_${UID}.lock}"
SUCCESS_MARKER_FILE="${G1_TASK_SUCCESS_FILE:-/tmp/g1_tabletop_grasp_${UID}.success}"
BENCHMARK_RESET_REQUEST_FILE="${G1_BENCHMARK_RESET_REQUEST_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_reset.json}"
BENCHMARK_RESET_STATUS_FILE="${G1_BENCHMARK_RESET_STATUS_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_status.json}"
BENCHMARK_ACTIVE_FILE="${G1_BENCHMARK_ACTIVE_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_active}"
INFERENCE_PAUSE_REQUEST_FILE="${G1_INFERENCE_PAUSE_REQUEST_FILE:-/tmp/g1_tabletop_grasp_${UID}.lingbot_pause_request}"
INFERENCE_PAUSE_STATUS_FILE="${G1_INFERENCE_PAUSE_STATUS_FILE:-/tmp/g1_tabletop_grasp_${UID}.lingbot_pause_status}"

exec 9>"$SIM_LOCK_FILE"
if ! flock -n 9; then
  echo "Another G1 tabletop simulator instance is already running." >&2
  echo "Stop the existing ./run_task.sh before starting a new one." >&2
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$SIM_ROOT/sim_main.py" ]]; then
  echo "Unitree IsaacLab simulator not found: $SIM_ROOT" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate unitree_sim_env
cd "$SIM_ROOT"
export UNITREE_SIM_READY_FILE="$SIM_READY_FILE"
export G1_TASK_SUCCESS_FILE="$SUCCESS_MARKER_FILE"
export G1_BENCHMARK_RESET_REQUEST_FILE="$BENCHMARK_RESET_REQUEST_FILE"
export G1_BENCHMARK_RESET_STATUS_FILE="$BENCHMARK_RESET_STATUS_FILE"
export G1_BENCHMARK_ACTIVE_FILE="$BENCHMARK_ACTIVE_FILE"
export G1_INFERENCE_PAUSE_REQUEST_FILE="$INFERENCE_PAUSE_REQUEST_FILE"
export G1_INFERENCE_PAUSE_STATUS_FILE="$INFERENCE_PAUSE_STATUS_FILE"
rm -f -- "$SIM_READY_FILE" "$SUCCESS_MARKER_FILE" \
  "$BENCHMARK_RESET_REQUEST_FILE" "$BENCHMARK_RESET_STATUS_FILE" \
  "$BENCHMARK_ACTIVE_FILE" "$INFERENCE_PAUSE_REQUEST_FILE" \
  "$INFERENCE_PAUSE_STATUS_FILE"

camera_viewer_pid=""
sim_pid=""
sim_pgid=""
cleanup_started=0

cleanup_task() {
  cleanup_status=$?
  if (( cleanup_started )); then
    return "$cleanup_status"
  fi
  cleanup_started=1
  trap - EXIT INT TERM
  set +e

  if [[ -n "$sim_pgid" ]] && kill -0 -- "-$sim_pgid" 2>/dev/null; then
    echo "Stopping Isaac Sim (PID $sim_pid)..."
    kill -TERM -- "-$sim_pgid" 2>/dev/null || kill -TERM "$sim_pid" 2>/dev/null
    for _ in {1..20}; do
      kill -0 -- "-$sim_pgid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 -- "-$sim_pgid" 2>/dev/null; then
      echo "Isaac Sim did not stop within 10 seconds; forcing shutdown."
      kill -KILL -- "-$sim_pgid" 2>/dev/null || kill -KILL "$sim_pid" 2>/dev/null
    fi
  fi
  if [[ -n "$sim_pid" ]]; then
    wait "$sim_pid" 2>/dev/null || true
  fi

  rm -f -- "$SIM_READY_FILE" "$SUCCESS_MARKER_FILE" \
    "$BENCHMARK_RESET_REQUEST_FILE" "$BENCHMARK_RESET_STATUS_FILE" \
    "$BENCHMARK_ACTIVE_FILE" "$INFERENCE_PAUSE_REQUEST_FILE" \
    "$INFERENCE_PAUSE_STATUS_FILE"
  if [[ -n "$camera_viewer_pid" ]] && kill -0 "$camera_viewer_pid" 2>/dev/null; then
    kill "$camera_viewer_pid" 2>/dev/null || true
    wait "$camera_viewer_pid" 2>/dev/null || true
  fi

  return "$cleanup_status"
}
trap cleanup_task EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${G1_SHOW_CAMERA_VIEWPORTS:-1}" != 0 ]]; then
  python "$CAMERA_VIEWER" &
  camera_viewer_pid=$!
fi

app_launcher_args=()
if [[ "${G1_DUST_MODE:-off}" == "flow" ]]; then
  # Flow must be loaded before RTX initializes; enabling it later creates the
  # USD schemas but does not reliably composite volumes into render products.
  app_launcher_args+=(
    --rendering_mode quality
    --kit_args "--enable omni.flowusd --/rtx/flow/enabled=true --/rtx/flow/maxBlocks=8192 --/rtx/translucency/enabled=true --/rtx/reflections/enabled=true --/rtx/indirectDiffuse/enabled=true"
  )
fi

setsid python sim_main.py \
  --device "${G1_TABLETOP_DEVICE:-${G1_DRILLING_DEVICE:-cuda}}" \
  --enable_cameras \
  --task "$TASK_ID" \
  --enable_dex1_dds \
  --robot_type g129 \
  "${app_launcher_args[@]}" \
  "$@" &
sim_pid=$!
sim_pgid=$sim_pid

set +e
wait "$sim_pid"
sim_status=$?
set -e
exit "$sim_status"
