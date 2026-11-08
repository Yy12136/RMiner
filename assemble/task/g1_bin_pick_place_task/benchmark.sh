#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/../../env.sh" ]]; then source "$ROOT/../../env.sh"; fi
case "${1:-}" in
  eval) shift; exec ${POLICY_PYTHON:-python3} "$ROOT/../evaluation/supervisor.py" --task task03 "$@" ;;
  sim) shift; exec "$ROOT/run_task.sh" "$@" ;;
  *) echo "Usage: ./benchmark.sh {eval|sim} [arguments]" >&2; exit 2 ;;
esac
