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
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_hand_cart_rolling_demo_${UID}.ready}"
SIM_LOCK_FILE="${G1_SIM_LOCK_FILE:-/tmp/g1_hand_cart_rolling_demo_${UID}.lock}"

exec 9>"$SIM_LOCK_FILE"
if ! flock -n 9; then
  echo "Another rolling hand-cart demo simulator is already running." >&2
  exit 1
fi
if [[ ! -f "$CONDA_SH" || ! -f "$SIM_ROOT/sim_main.py" ]]; then
  echo "Isaac Sim environment is incomplete." >&2
  exit 1
fi
if ss -H -ltn 'sport = :60000' 2>/dev/null | grep -q '.'; then
  echo "Image-server port 60000 is already in use." >&2
  ss -H -ltnp 'sport = :60000' 2>/dev/null >&2 || true
  exit 1
fi

source "$CONDA_SH"
conda activate unitree_sim_env
cd "$SIM_ROOT"
export UNITREE_SIM_READY_FILE="$SIM_READY_FILE"
export G1_CAMERA_WINDOW_NAME="G1 Rolling Hand-Cart Demo | Head - Left Wrist - Right Wrist"
export G1_HAND_CART_DEMO_MAX_SPEED_MPS="${G1_HAND_CART_DEMO_MAX_SPEED_MPS:-0.18}"
export G1_HAND_CART_DEMO_MAX_ACCEL_MPS2="${G1_HAND_CART_DEMO_MAX_ACCEL_MPS2:-0.30}"
rm -f -- "$SIM_READY_FILE"

camera_viewer_pid=""
simulator_pid=""
cleanup() {
  rm -f -- "$SIM_READY_FILE"
  if [[ -n "$simulator_pid" ]] && kill -0 "$simulator_pid" 2>/dev/null; then
    child_pids="$(pgrep -P "$simulator_pid" 2>/dev/null || true)"
    if [[ -n "$child_pids" ]]; then kill -TERM $child_pids 2>/dev/null || true; fi
    kill -TERM -- "-$simulator_pid" 2>/dev/null || true
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
  --device "${G1_HAND_CART_DEVICE:-${G1_EVAL_SIM_DEVICE:-cuda}}" \
  --enable_cameras \
  --task Isaac-HandCartRollingDemo-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  "$@" &
simulator_pid=$!
wait "$simulator_pid"
