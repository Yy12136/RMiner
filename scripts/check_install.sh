#!/usr/bin/env bash
# Sanity-check that the RMiner tree is complete enough to launch a task.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ok=0
warn=0

check() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf '  [ok]   %s\n' "$label"
  else
    printf '  [MISS] %s (%s)\n' "$label" "$path"
    warn=$((warn + 1))
  fi
}

echo "RMiner install check: $ROOT"
echo
echo "Code:"
check "assemble/env.sh"                        "$ROOT/assemble/env.sh"
check "unitree_sim_isaaclab/sim_main.py"       "$ROOT/assemble/third_party/unitree_sim_isaaclab/sim_main.py"
check "IsaacLab source"                        "$ROOT/assemble/third_party/IsaacLab/source"
check "xr_teleoperate"                         "$ROOT/assemble/third_party/xr_teleoperate/teleop"
check "task evaluation frame"                  "$ROOT/assemble/task/evaluation"

echo
echo "Shared assets:"
check "assets-new/"                            "$ROOT/assemble/assets-new"
check "objects/"                               "$ROOT/assemble/objects"
check "RealScene04"                            "$ROOT/assemble/assets-new/scene/RealScene04"

echo
echo "Per-task sources:"
for d in "$ROOT"/assemble/task/*/; do
  [[ -f "$d/run_task.sh" || -f "$d/run_rolling_demo.sh" ]] || continue
  printf '  [ok]   %s\n' "$(basename "$d")"
done

echo
echo "Conda envs (informational):"
if command -v conda >/dev/null 2>&1; then
  conda env list 2>/dev/null | grep -E '^(unitree_sim_env|tv)\s' || \
    echo "  [warn] unitree_sim_env / tv not found; run ./setup/setup_envs.sh"
else
  echo "  [warn] conda not on PATH"
fi

echo
if [[ "$warn" -eq 0 ]]; then
  echo "All checks passed."
else
  echo "$warn check(s) missing. Run ./scripts/download_assets.sh to fetch assets."
  exit 1
fi
