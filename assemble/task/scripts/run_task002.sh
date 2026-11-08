#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ENTRYPOINT="$SCRIPT_DIR/../w1_mobile_drilling_task/run_task.sh"

if [[ ! -x "$TASK_ENTRYPOINT" ]]; then
  echo "W1 Mobile Drilling Task entrypoint not found: $TASK_ENTRYPOINT" >&2
  exit 1
fi

# Task002 uses the complete drilling scene with keyboard-only control.  Normal
# PD targets are the default so PhysX contacts can oppose arm motion; callers
# can still explicitly set W1_DRILLING_STRICT_JOINT_HOLD=1 when needed.
export W1_QUEST_ENABLED=0
export W1_KEYBOARD_UPPER_BODY=1
export W1_DRILLING_STRICT_JOINT_HOLD="${W1_DRILLING_STRICT_JOINT_HOLD:-0}"

exec "$TASK_ENTRYPOINT" "$@"
