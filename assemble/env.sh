#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Central path configuration for the portable Benchmark bundle.
#
# The bundle keeps the original `assemble/` layout so that the Isaac Lab task
# configs (which locate assets via `Path(__file__).parents[5]`) keep working
# unchanged:
#
#   <bundle>/assemble/
#       task/                # task packages + shared controllers + evaluator
#       third_party/         # IsaacLab, unitree_sim_isaaclab, xr_teleoperate, ...
#       assets-new/          # shared USD assets used by the task configs
#       objects/             # extra USD objects (hand cart, valves, ...)
#
# Any launcher can `source` this file. Everything is overridable through the
# environment so nothing depends on a specific machine layout.
# ---------------------------------------------------------------------------

# Directory this file lives in == <bundle>/assemble
BENCH_ASSEMBLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_ASSEMBLE_ROOT
export BENCH_BUNDLE_ROOT="$(cd -- "$BENCH_ASSEMBLE_ROOT/.." && pwd)"

# --- Conda -----------------------------------------------------------------
# Priority: an already-exported $CONDA_SH, then `conda` on PATH, then the
# usual Miniconda/Anaconda/Miniforge install locations.
if [[ -z "${CONDA_SH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
  fi
fi
if [[ ! -f "${CONDA_SH:-}" ]]; then
  unset CONDA_SH
  for _cand in \
      "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
      "$HOME/mambaforge" "$HOME/micromamba" \
      "/opt/conda" "/opt/miniconda3" "/usr/local/miniconda3" \
      "/usr/local/anaconda3" "/usr/local/miniforge3"; do
    if [[ -f "$_cand/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="$_cand/etc/profile.d/conda.sh"
      break
    fi
  done
fi
export CONDA_SH

# --- Python / policy (only needed for `benchmark.sh eval` and training) -----
export LEROBOT_ROOT="${LEROBOT_ROOT:-$BENCH_BUNDLE_ROOT/policy/lerobot}"
export POLICY_PYTHON="${POLICY_PYTHON:-$LEROBOT_ROOT/.venv/bin/python}"

# --- Optional: Isaac Sim standalone asset root used by a few authoring tools -
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-$BENCH_BUNDLE_ROOT/assets/isaacsim_assets}"
