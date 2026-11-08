#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Recreate the two conda environments used by the benchmark from scratch.
#
#   ./setup_envs.sh                 # both envs
#   ./setup_envs.sh unitree_sim_env # simulator env only
#   ./setup_envs.sh tv              # Quest teleop env only
#
# NOTE: This mirrors third_party/unitree_sim_isaaclab/auto_setup_env.sh and the
# xr_teleoperate README. It needs internet access (PyTorch / NVIDIA / conda
# indices) and an NVIDIA driver compatible with the pinned Isaac Sim version.
# If the target machine ALREADY has working `unitree_sim_env` / `tv` envs, you
# do NOT need this: just run ./install_editable.sh <env> to repoint the editable
# packages at this bundle. The exported yml files in setup/ are the exact
# package snapshots produced on the source machine.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE="$(cd -- "$HERE/../assemble" && pwd)"
TP="$ASSEMBLE/third_party"
export CYCLONEDDS_HOME="$TP/cyclonedds/install"

if [[ -z "${CONDA_SH:-}" ]]; then
  for _c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    [[ -f "$_c/etc/profile.d/conda.sh" ]] && { CONDA_SH="$_c/etc/profile.d/conda.sh"; break; }
  done
fi
[[ -f "${CONDA_SH:-}" ]] || { echo "conda.sh not found; set CONDA_SH=/path/to/etc/profile.d/conda.sh"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"

[[ -f "$CYCLONEDDS_HOME/lib/libddsc.so" ]] || { echo "==> Building CycloneDDS"; "$HERE/build_cyclonedds.sh"; }

setup_sim_env() {
  echo "==> Creating conda env: unitree_sim_env"
  conda create -n unitree_sim_env python=3.11 -c conda-forge -y
  source "$CONDA_SH"; conda activate unitree_sim_env
  python -m pip install --upgrade pip

  echo "==> Installing PyTorch 2.7.0 (cu128)"
  python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

  echo "==> Installing Isaac Sim 5.1.0"
  python -m pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

  echo "==> Installing remaining simulator requirements"
  python -m pip install -r "$HERE/requirements-unitree_sim_env.txt" \
    --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 || true

  echo "==> Installing editable packages (IsaacLab, sdk2py, teleimager)"
  "$HERE/install_editable.sh" unitree_sim_env

  conda install -y -c conda-forge libstdcxx-ng
  conda env config vars set CYCLONEDDS_HOME="$CYCLONEDDS_HOME" -n unitree_sim_env
  conda deactivate
  echo "==> unitree_sim_env ready"
}

setup_tv_env() {
  echo "==> Creating conda env: tv"
  conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge -y
  source "$CONDA_SH"; conda activate tv
  python -m pip install --upgrade pip

  echo "==> Installing tv requirements (torch 2.3.0 + teleop stack)"
  python -m pip install -r "$HERE/requirements-tv.txt" || true

  echo "==> Installing editable packages (sdk2py, teleimager, televuer, dex-retargeting)"
  "$HERE/install_editable.sh" tv

  conda env config vars set CYCLONEDDS_HOME="$CYCLONEDDS_HOME" -n tv
  conda deactivate
  echo "==> tv ready"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(unitree_sim_env tv)
for t in "${targets[@]}"; do
  case "$t" in
    unitree_sim_env) setup_sim_env ;;
    tv) setup_tv_env ;;
    *) echo "unknown target: $t"; exit 2 ;;
  esac
done
echo "All requested environments are ready."
