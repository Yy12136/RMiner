#!/usr/bin/env bash
set -euo pipefail

# Isolated environment for Quest hand-tracking + Unitree G1/Dex3-1 simulation.
ENV_NAME="${XR_DEX_ENV:-xr_dex3}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "找不到 Conda 初始化脚本：$CONDA_SH" >&2
  echo "请设置 CONDA_SH，或先安装 Miniconda。" >&2
  exit 1
fi
source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
fi
conda activate "$ENV_NAME"

XR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../third_party/xr_teleoperate" && pwd)"
if [[ ! -d "$XR_ROOT/teleop" ]]; then
  echo "缺少 xr_teleoperate：$XR_ROOT" >&2
  exit 1
fi

python -m pip install -r "$XR_ROOT/requirements.txt"
# SDK is vendored in this workspace and is required by teleop_hand_and_arm.py
# even in simulation mode (DDS channel definitions are imported at startup).
SDK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../third_party/unitree_sdk2_python" && pwd)"
python -m pip install -e "$SDK_ROOT"
# vuer 0.0.60 still imports Flag from params-proto; newer 3.x releases
# removed that symbol, so keep the compatible 2.x API in this isolated env.
python -m pip install 'params-proto>=2.13,<3.0'
if [[ -d "$XR_ROOT/teleop/teleimager" ]]; then
  python -m pip install -e "$XR_ROOT/teleop/teleimager" --no-deps
fi
if [[ -d "$XR_ROOT/teleop/televuer" ]]; then
  python -m pip install -e "$XR_ROOT/teleop/televuer"
fi
RETARGET_ROOT="$XR_ROOT/teleop/robot_control/dex-retargeting"
if [[ -f "$RETARGET_ROOT/pyproject.toml" ]]; then
  python -m pip install -e "$RETARGET_ROOT"
fi
echo "环境已就绪：$ENV_NAME"
