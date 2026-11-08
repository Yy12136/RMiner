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
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_valve_handwheel_${UID}.ready}"
SIM_LOCK_FILE="${G1_SIM_LOCK_FILE:-/tmp/g1_valve_handwheel_${UID}.lock}"
SUCCESS_MARKER_FILE="${G1_TASK_SUCCESS_FILE:-/tmp/g1_valve_handwheel_${UID}.success}"

exec 9>"$SIM_LOCK_FILE"
if ! flock -n 9; then
  echo "Another G1 valve-handwheel simulator is already running." >&2
  exit 1
fi
if [[ ! -f "$CONDA_SH" || ! -f "$SIM_ROOT/sim_main.py" ]]; then
  echo "Isaac Sim environment is incomplete." >&2
  exit 1
fi
existing_simulators="$(pgrep -af 'python([0-9.]*)? .*sim_main\.py' || true)"
if [[ -n "$existing_simulators" ]]; then
  echo "Another Unitree/Isaac simulator process is already running:" >&2
  echo "$existing_simulators" >&2
  echo "Stop that simulator before starting this task." >&2
  exit 1
fi
if ss -H -ltn 'sport = :60000' 2>/dev/null | grep -q '.'; then
  echo "Image-server port 60000 is already in use." >&2
  echo "Stop the previously running Unitree/Isaac simulator, then retry." >&2
  ss -H -ltnp 'sport = :60000' 2>/dev/null >&2 || true
  exit 1
fi

source "$CONDA_SH"
conda activate unitree_sim_env
cd "$SIM_ROOT"
export UNITREE_SIM_READY_FILE="$SIM_READY_FILE"
export G1_TASK_SUCCESS_FILE="$SUCCESS_MARKER_FILE"
rm -f -- "$SIM_READY_FILE" "$SUCCESS_MARKER_FILE"

camera_viewer_pid=""
simulator_pid=""
cleanup() {
  rm -f -- "$SIM_READY_FILE" "$SUCCESS_MARKER_FILE"
  if [[ -n "$simulator_pid" ]] && kill -0 "$simulator_pid" 2>/dev/null; then
    child_pids="$(pgrep -P "$simulator_pid" 2>/dev/null || true)"
    if [[ -n "$child_pids" ]]; then
      kill -TERM $child_pids 2>/dev/null || true
    fi
    kill -TERM -- "-$simulator_pid" 2>/dev/null || true
    for _cleanup_try in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$simulator_pid" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "$simulator_pid" 2>/dev/null; then
      kill -KILL -- "-$simulator_pid" 2>/dev/null || true
    fi
    wait "$simulator_pid" 2>/dev/null || true
  fi
  if [[ -n "$camera_viewer_pid" ]] && kill -0 "$camera_viewer_pid" 2>/dev/null; then
    kill "$camera_viewer_pid" 2>/dev/null || true
    wait "$camera_viewer_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${G1_SHOW_CAMERA_VIEWPORTS:-1}" != 0 ]]; then
  python "$CAMERA_VIEWER" &
  camera_viewer_pid=$!
fi

simulator_prefix=()
if [[ "${G1_PROCESS_GROUP_MANAGED:-0}" != 1 ]]; then simulator_prefix=(setsid); fi
"${simulator_prefix[@]}" python sim_main.py \
  --device "${G1_VALVE_HANDWHEEL_DEVICE:-${G1_EVAL_SIM_DEVICE:-cuda}}" \
  --enable_cameras \
  --task Isaac-ValveHandwheel-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  "$@" &
simulator_pid=$!
wait "$simulator_pid"
