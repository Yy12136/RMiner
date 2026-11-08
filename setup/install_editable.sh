#!/usr/bin/env bash
# Point the editable ("-e") Python packages at this bundle's third_party tree.
#
# Run this AFTER activating the matching conda env. Use it when the conda env
# already exists on the new machine (e.g. you recreated it from the exported
# yml, or you already had an equivalent Isaac Sim env) but the editable
# packages still reference the old absolute paths.
#
#   conda activate unitree_sim_env && ./install_editable.sh unitree_sim_env
#   conda activate tv              && ./install_editable.sh tv
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE="$(cd -- "$HERE/../assemble" && pwd)"
TP="$ASSEMBLE/third_party"
kind="${1:-${CONDA_DEFAULT_ENV:-}}"

echo "Active python: $(command -v python || true)"
echo "Active env   : ${CONDA_DEFAULT_ENV:-<none>}"

case "$kind" in
  unitree_sim_env)
    # IsaacLab meta-packages (isaaclab, isaaclab_assets, _tasks, _rl, ...)
    ( cd "$TP/IsaacLab" && ./isaaclab.sh --install )
    python -m pip install -e "$TP/unitree_sdk2_python"
    python -m pip install -e "$TP/unitree_sim_isaaclab/teleimager"
    python -m pip install -r "$TP/unitree_sim_isaaclab/requirements.txt"
    ;;
  tv)
    python -m pip install -e "$TP/unitree_sdk2_python"
    python -m pip install -e "$TP/xr_teleoperate/teleop/teleimager"
    python -m pip install -e "$TP/xr_teleoperate/teleop/televuer"
    python -m pip install -e "$TP/xr_teleoperate/teleop/robot_control/dex-retargeting"
    ;;
  *)
    echo "usage: install_editable.sh {unitree_sim_env|tv}" >&2
    echo "       (activate the target conda env first)" >&2
    exit 2
    ;;
esac
echo "Editable packages now point at: $TP"
