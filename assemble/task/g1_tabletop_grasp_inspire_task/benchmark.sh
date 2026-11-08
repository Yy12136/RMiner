#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
G1 tabletop (Inspire 5-finger hand) sim + collect benchmark

Usage:
  ./benchmark.sh sim [simulator arguments]
  ./benchmark.sh collect [teleoperation arguments]

Note: LeRobot recording / training / policy eval are Dex1-only in the
upstream Unitree teleoperator, so only sim and collect are provided here.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command="$1"
shift
case "$command" in
  sim) exec "$TASK_ROOT/run_task.sh" "$@" ;;
  collect) exec "$TASK_ROOT/run_quest3s.sh" "$@" ;;
  help|--help|-h) usage ;;
  *)
    echo "Unknown benchmark command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
