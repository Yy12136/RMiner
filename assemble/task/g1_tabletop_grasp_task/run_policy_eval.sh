#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE_ROOT="$(cd -- "$TASK_ROOT/../.." && pwd)"
TELEOP_ROOT="$ASSEMBLE_ROOT/third_party/xr_teleoperate/teleop"
LEROBOT_ROOT="${LEROBOT_ROOT:-$HOME/Benchmark/policy/lerobot}"
PYTHON="$LEROBOT_ROOT/.venv/bin/python"
CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
  _bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  _bench_rel="$_bench_dir"
  for _bench_i in 1 2 3 4 5 6; do
    if [[ -f "$_bench_rel/env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$_bench_rel/env.sh"
      break
    fi
    _bench_rel="$(cd -- "$_bench_rel/.." && pwd)"
  done
  unset _bench_dir _bench_rel _bench_i
fi
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SIM_READY_FILE="${G1_SIM_READY_FILE:-/tmp/g1_tabletop_grasp_${UID}.ready}"
SUCCESS_FILE="${G1_TASK_SUCCESS_FILE:-/tmp/g1_tabletop_grasp_${UID}.success}"
RESET_REQUEST_FILE="${G1_BENCHMARK_RESET_REQUEST_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_reset.json}"
RESET_STATUS_FILE="${G1_BENCHMARK_RESET_STATUS_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_status.json}"
BENCHMARK_ACTIVE_FILE="${G1_BENCHMARK_ACTIVE_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval_active}"
SOCKET_PATH="${G1_EVAL_SOCKET:-/tmp/g1_tabletop_grasp_${UID}.eval.sock}"
EVAL_LOCK_FILE="${G1_EVAL_LOCK_FILE:-/tmp/g1_tabletop_grasp_${UID}.eval.lock}"
OUTPUT_ROOT="$TASK_ROOT/outputs/eval"
INITIAL_GRIPPER=4.0
NETWORK_INTERFACE=""
CHECKPOINT=""
EVAL_ARGS=()

usage() {
  echo "Usage: ./benchmark.sh eval --checkpoint /path/to/pretrained_model [--episodes 20] [--timeout 60] [--seed 42] [--initial-gripper 4.0] [--inference sync|rtc]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --initial-gripper) INITIAL_GRIPPER="$2"; shift 2 ;;
    --network-interface) NETWORK_INTERFACE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) EVAL_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$SIM_READY_FILE" ]]; then
  echo "The tabletop simulator is not ready." >&2
  echo "Run './benchmark.sh sim' in terminal 1 and wait for its ready message." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "LeRobot environment is not ready: $LEROBOT_ROOT/.venv" >&2
  exit 1
fi

exec 8>"$EVAL_LOCK_FILE"
if ! flock -n 8; then
  echo "Another tabletop policy evaluation is already running." >&2
  exit 1
fi

AUTHKEY="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
rm -f -- "$SOCKET_PATH" "$RESET_REQUEST_FILE" "$RESET_STATUS_FILE"
touch "$BENCHMARK_ACTIVE_FILE"
BRIDGE_ARGS=(
  --socket "$SOCKET_PATH"
  --authkey "$AUTHKEY"
  --reset-request-file "$RESET_REQUEST_FILE"
  --reset-status-file "$RESET_STATUS_FILE"
  --initial-gripper "$INITIAL_GRIPPER"
)
if [[ -n "$NETWORK_INTERFACE" ]]; then
  BRIDGE_ARGS+=(--network-interface "$NETWORK_INTERFACE")
fi

bridge_pid=""
cleanup() {
  if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
    kill "$bridge_pid" 2>/dev/null || true
    wait "$bridge_pid" 2>/dev/null || true
  fi
  rm -f -- "$SOCKET_PATH" "$RESET_REQUEST_FILE" "$BENCHMARK_ACTIVE_FILE"
}
trap cleanup EXIT INT TERM

(
  source "$CONDA_SH"
  conda activate tv
  cd "$TELEOP_ROOT"
  exec python "$TASK_ROOT/policy_eval_dds_bridge.py" "${BRIDGE_ARGS[@]}"
) &
bridge_pid=$!

deadline=$((SECONDS + 45))
while [[ ! -S "$SOCKET_PATH" ]]; do
  if ! kill -0 "$bridge_pid" 2>/dev/null; then
    echo "DDS evaluation bridge exited before becoming ready." >&2
    wait "$bridge_pid" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "DDS evaluation bridge did not become ready within 45 seconds." >&2
    exit 1
  fi
  sleep 0.2
done

set +e
"$PYTHON" "$TASK_ROOT/policy_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --authkey "$AUTHKEY" \
  --output-root "$OUTPUT_ROOT" \
  --success-marker-file "$SUCCESS_FILE" \
  --initial-gripper "$INITIAL_GRIPPER" \
  "${EVAL_ARGS[@]}"
eval_status=$?
set -e

if [[ -n "$bridge_pid" ]]; then
  # A normal evaluator exit asks the bridge to home both arms and restore the
  # grippers before it exits. Give that safe shutdown enough time to finish.
  for _ in $(seq 1 80); do
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$bridge_pid" 2>/dev/null; then
    kill "$bridge_pid" 2>/dev/null || true
  fi
fi
wait "$bridge_pid" 2>/dev/null || true
bridge_pid=""
exit "$eval_status"
