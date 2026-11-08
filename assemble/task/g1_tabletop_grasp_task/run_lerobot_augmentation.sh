#!/usr/bin/env bash
set -euo pipefail

TASKS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$TASKS_ROOT/augmentation/run_lerobot_augmentation.sh" "$@"
