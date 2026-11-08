#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$TASK_ROOT/assets/robot/Forklift_W1_Final/tools/run_with_isaacsim.sh"
SHARED_TELEOP="$TASK_ROOT/../scripts/x7s_mobile_drilling_teleop.py"
QUEST_DEPS="$TASK_ROOT/../scripts/.quest_deps/vuer"
QUEST_SETUP="$TASK_ROOT/../scripts/setup_w1_quest3s.sh"
X7S_ASSET="$TASK_ROOT/assets/robot/X7S/x7s.usd"
X7S_URDF="$TASK_ROOT/assets/robot/X7S/urdf/X7S.urdf"

if [[ ! -x "$RUNNER" ]]; then
  echo "Isaac Sim runner not found: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$SHARED_TELEOP" ]]; then
  echo "Shared X7S teleoperation script not found: $SHARED_TELEOP" >&2
  exit 1
fi
if [[ ! -f "$X7S_ASSET" || ! -f "$X7S_URDF" ]]; then
  echo "X7S assets are incomplete under: $TASK_ROOT/assets/robot/X7S" >&2
  exit 1
fi
if [[ "${W1_QUEST_ENABLED:-1}" != 0 ]] && [[ ! -d "$QUEST_DEPS" ]]; then
  if [[ ! -x "$QUEST_SETUP" ]]; then
    echo "Quest setup script not found: $QUEST_SETUP" >&2
    exit 1
  fi
  "$QUEST_SETUP"
fi

export W1_TASK_ROOT="$TASK_ROOT"
exec "$RUNNER" "$SHARED_TELEOP" "$@"
