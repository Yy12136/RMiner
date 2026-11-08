#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$TASK_ROOT/../w1_mobile_drilling_task/assets/robot/Forklift_W1_Final/tools/run_with_isaacsim.sh"
SHARED_TELEOP="$TASK_ROOT/../scripts/tracked_upperbody_teleop.py"
QUEST_DEPS="$TASK_ROOT/../scripts/.quest_deps/vuer"
QUEST_SETUP="$TASK_ROOT/../scripts/setup_w1_quest3s.sh"
CONFIG="$TASK_ROOT/config.json"

if [[ ! -x "$RUNNER" ]]; then
  echo "Isaac Sim runner not found: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$SHARED_TELEOP" ]]; then
  echo "Shared tracked teleoperation script not found: $SHARED_TELEOP" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Teleop config not found: $CONFIG" >&2
  exit 1
fi
if [[ "${TRACKED_QUEST_ENABLED:-1}" != 0 ]] && [[ ! -d "$QUEST_DEPS" ]]; then
  if [[ ! -x "$QUEST_SETUP" ]]; then
    echo "Quest setup script not found: $QUEST_SETUP" >&2
    exit 1
  fi
  "$QUEST_SETUP"
fi

export TRACKED_TELEOP_CONFIG="$CONFIG"
exec "$RUNNER" "$SHARED_TELEOP" "$@"