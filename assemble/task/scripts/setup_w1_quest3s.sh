#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dependency_dir="$script_dir/.quest_deps"
isaac_sim_dir="${ISAACSIM_PATH:-/home/Isaacsim}"

if [[ ! -x "$isaac_sim_dir/python.sh" && -x ${ISAACSIM_ROOT:-$HOME/isaacsim}/python.sh ]]; then
  isaac_sim_dir=${ISAACSIM_ROOT:-$HOME/isaacsim}
fi
if [[ ! -x "$isaac_sim_dir/python.sh" ]]; then
  echo "Isaac Sim python.sh not found. Set ISAACSIM_PATH first." >&2
  exit 1
fi

mkdir -p "$dependency_dir"
env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u VIRTUAL_ENV \
  -u LD_LIBRARY_PATH \
  "$isaac_sim_dir/python.sh" -m pip install \
  --target "$dependency_dir" \
  --upgrade \
  "vuer==0.1.6" \
  "Robotic_Arm==1.1.6"

echo "Quest 3S and RealMan IK dependencies installed in $dependency_dir"
