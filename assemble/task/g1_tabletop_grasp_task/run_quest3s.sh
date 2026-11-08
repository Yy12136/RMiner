#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE_ROOT="$(cd -- "$TASK_ROOT/../.." && pwd)"
TELEOP_ROOT="$ASSEMBLE_ROOT/third_party/xr_teleoperate/teleop"
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
IMAGE_SERVER_IP="${G1_IMAGE_SERVER_IP:-192.168.2.161}"
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_tabletop_grasp_${UID}.ready}"
SIM_READY_TIMEOUT="${G1_SIM_READY_TIMEOUT:-180}"
SIM_DDS_SETTLE_SECONDS="${G1_SIM_DDS_SETTLE_SECONDS:-15}"
LEROBOT_DATASET_DIR="${G1_LEROBOT_DATASET_DIR:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_replay_ready}"
TASK_INSTRUCTION="${G1_TASK_INSTRUCTION:-Insert the detonator lead wire into the detonator opening.}"
SUCCESS_MARKER_FILE="${G1_TASK_SUCCESS_FILE:-/tmp/g1_tabletop_grasp_${UID}.success}"
LEROBOT_TARGET_EPISODES="${G1_LEROBOT_TARGET_EPISODES:-50}"
LEROBOT_RESUME="${G1_LEROBOT_RESUME:-1}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$TELEOP_ROOT/teleop_hand_and_arm.py" ]]; then
  echo "Official Unitree XR teleoperation program not found: $TELEOP_ROOT" >&2
  exit 1
fi
case "$LEROBOT_RESUME" in
  1|true|TRUE|yes|YES) LEROBOT_RESUME_ARG="--lerobot-resume" ;;
  0|false|FALSE|no|NO) LEROBOT_RESUME_ARG="--no-lerobot-resume" ;;
  *)
    echo "G1_LEROBOT_RESUME must be 1/0, true/false, or yes/no." >&2
    exit 1
    ;;
esac

echo "Waiting for the tabletop simulator DDS controller to become ready..."
deadline=$((SECONDS + SIM_READY_TIMEOUT))
while [[ ! -f "$SIM_READY_FILE" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Simulator did not become ready within ${SIM_READY_TIMEOUT}s." >&2
    echo "Start ./run_task.sh first and wait for its Isaac Sim window." >&2
    exit 1
  fi
  sleep 0.5
done
echo "Simulator DDS controller is ready."
echo "Waiting ${SIM_DDS_SETTLE_SECONDS}s for CycloneDDS discovery..."
sleep "$SIM_DDS_SETTLE_SECONDS"

source "$CONDA_SH"
conda activate tv
cd "$TELEOP_ROOT"

echo "============================================================"
echo "G1 tabletop grasp | Quest 3/3S teleoperation"
echo "The robot will NOT follow the controllers immediately."
echo "Press right controller [B] to start teleoperation and record a new episode."
echo "Press right controller [A] to save a successful episode and reset all."
echo "Press [s] to discard the current episode and reset all."
echo "Press [q] in this terminal to stop and return the arms home."
echo "LeRobot Dataset v3.0: $LEROBOT_DATASET_DIR"
echo "Saved-episode target: $LEROBOT_TARGET_EPISODES"
echo "Resume existing dataset: $LEROBOT_RESUME"
echo "Quest URL: https://${IMAGE_SERVER_IP}:8012/?ws=wss://${IMAGE_SERVER_IP}:8012"
echo "============================================================"

exec python teleop_hand_and_arm.py \
  --input-mode=controller \
  --arm=G1_29 \
  --ee=dex1 \
  --sim \
  --img-server-ip="$IMAGE_SERVER_IP" \
  --lerobot-v30 \
  --lerobot-replay-snapshot-task="${G1_TABLETOP_TASK_ID:-Isaac-TabletopGrasp-G129-Dex1-Joint}" \
  --lerobot-dataset-dir="$LEROBOT_DATASET_DIR" \
  --lerobot-target-episodes="$LEROBOT_TARGET_EPISODES" \
  "$LEROBOT_RESUME_ARG" \
  --success-marker-file="$SUCCESS_MARKER_FILE" \
  --instruction="$TASK_INSTRUCTION" \
  "$@"
