#!/usr/bin/env bash
set -euo pipefail

# Preserved launcher for the previous tracked W1 task implementation.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$TASK_ROOT/assets/robot/Forklift_W1_Final/tools/run_with_isaacsim.sh"
SHARED_TELEOP="$TASK_ROOT/../scripts/w1_mobile_manipulator_teleop.py"
QUEST_DEPS="$TASK_ROOT/../scripts/.quest_deps/vuer"
QUEST_SETUP="$TASK_ROOT/../scripts/setup_w1_quest3s.sh"
IK_BACKEND="${W1_QUEST_IK_BACKEND:-pinocchio}"

if [[ ! -x "$RUNNER" ]]; then
  echo "Isaac Sim runner not found: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$SHARED_TELEOP" ]]; then
  echo "Shared W1 teleoperation script not found: $SHARED_TELEOP" >&2
  exit 1
fi
if [[ "${W1_QUEST_ENABLED:-1}" != 0 ]] && \
  { [[ ! -d "$QUEST_DEPS" ]] || \
    { [[ "$IK_BACKEND" == realman ]] && \
      [[ ! -f "$TASK_ROOT/../scripts/.quest_deps/Robotic_Arm/rm_robot_interface.py" ]]; }; }; then
  if [[ ! -x "$QUEST_SETUP" ]]; then
    echo "Quest setup script not found: $QUEST_SETUP" >&2
    exit 1
  fi
  "$QUEST_SETUP"
fi

export W1_TASK_ROOT="$TASK_ROOT"
exec "$RUNNER" "$SHARED_TELEOP" "$@"
