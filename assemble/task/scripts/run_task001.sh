#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEW_ENTRYPOINT="$SCRIPT_DIR/../w1_mobile_drilling_task/run_task.sh"

if [[ ! -x "$NEW_ENTRYPOINT" ]]; then
  echo "W1 Mobile Drilling Task entrypoint not found: $NEW_ENTRYPOINT" >&2
  exit 1
fi

exec "$NEW_ENTRYPOINT" "$@"
