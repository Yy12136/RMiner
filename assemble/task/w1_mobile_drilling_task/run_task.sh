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
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_mobile_drilling_${UID}.ready}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$SIM_ROOT/sim_main.py" ]]; then
  echo "Unitree IsaacLab simulator not found: $SIM_ROOT" >&2
  exit 1
fi

# Keep the same simulator, DDS, cameras, Dex1 controller, and official XR IK
# used by the standard G1-29 teleoperation example. Only the Gym task changes.
source "$CONDA_SH"
conda activate unitree_sim_env
cd "$SIM_ROOT"
export UNITREE_SIM_READY_FILE="$SIM_READY_FILE"
rm -f -- "$SIM_READY_FILE"

camera_viewer_pid=""
cleanup_camera_viewer() {
  rm -f -- "$SIM_READY_FILE"
  if [[ -n "$camera_viewer_pid" ]] && kill -0 "$camera_viewer_pid" 2>/dev/null; then
    kill "$camera_viewer_pid" 2>/dev/null || true
    wait "$camera_viewer_pid" 2>/dev/null || true
  fi
}
trap cleanup_camera_viewer EXIT

if [[ "${G1_SHOW_CAMERA_VIEWPORTS:-1}" != 0 ]]; then
  python "$CAMERA_VIEWER" &
  camera_viewer_pid=$!
fi

python sim_main.py \
  --device "${G1_DRILLING_DEVICE:-cuda}" \
  --enable_cameras \
  --task Isaac-MobileDrilling-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  "$@"
