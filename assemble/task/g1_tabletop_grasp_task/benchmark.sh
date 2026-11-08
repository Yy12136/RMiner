#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
G1 tabletop collect-train-eval benchmark

Usage:
  ./benchmark.sh sim [simulator arguments]
  ./benchmark.sh collect [teleoperation arguments]
  ./benchmark.sh train [--policy act] [--steps 100000] [--batch-size 16] [--save-freq 10000] [--wandb]
  ./benchmark.sh eval --checkpoint /path/to/pretrained_model [--episodes 20] [--timeout 60] [--seed 42] [--initial-gripper 4.0] [--inference sync|rtc]
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
  train) exec "$TASK_ROOT/run_benchmark_train.sh" "$@" ;;
  eval) exec "$TASK_ROOT/run_policy_eval.sh" "$@" ;;
  help|--help|-h) usage ;;
  *)
    echo "Unknown benchmark command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
