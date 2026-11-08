#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
XR_ROOT="$TASK_ROOT/../third_party/xr_teleoperate"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${XR_DEX_ENV:-xr_dex3}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
IMAGE_SERVER_IP="${G1_IMAGE_SERVER_IP:-192.168.2.161}"
END_EFFECTOR="${XR_EE:-dex3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$CONDA_SH"
conda activate "$ENV_NAME"
if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "找不到 Isaac Sim python.sh：$ISAAC_SIM_ROOT/python.sh" >&2
  echo "请设置 ISAAC_SIM_ROOT 后重试。" >&2
  exit 1
fi

echo "启动 G1(29DoF)+Dex3-1 仿真与 Quest 原生手势追踪。"
echo "Quest 浏览器打开官方显示的 HTTPS 地址后，允许 Hand Tracking。"
echo "捏合/张开/弯曲手指会由 xr_teleoperate retarget 到 Dex3-1。"
cd "$XR_ROOT/teleop"
bridge_pid=""
if [[ "$END_EFFECTOR" == "brainco" ]]; then
  python "$SCRIPT_DIR/brainco_inspire_dds_bridge.py" &
  bridge_pid=$!
  trap '[[ -z "$bridge_pid" ]] || kill "$bridge_pid" 2>/dev/null || true' EXIT INT TERM
fi
python teleop_hand_and_arm.py \
  --input-mode=hand \
  --arm=G1_29 \
  --ee="$END_EFFECTOR" \
  --sim \
  --img-server-ip="$IMAGE_SERVER_IP" \
  "$@"
