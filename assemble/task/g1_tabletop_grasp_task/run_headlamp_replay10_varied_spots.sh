#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${G1_REPLAY10_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_REPLAY10_OUTPUT:-$TASK_ROOT/data/augmentation_previews/headlamp_replay10_varied_spots_20260901}"
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
monitor_pid=""

episodes=(0 1 2 3 4 5 6 7 8 9)
slots=(upper_right upper_mid_right upper_center upper_mid_left upper_left lower_right lower_mid_right lower_center lower_mid_left lower_left)
yaws=(8 4 0 -4 -8 8 4 0 -4 -8)
pitches=(-3 -3 -3 -3 -3 3 3 3 3 3)

cleanup_monitor() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

busy="$(pgrep -af '[e]valuation/supervisor.py|[p]ython .*sim_main.py' || true)"
if [[ -n "$busy" ]]; then
  echo "Development Isaac Sim/DDS runtime is occupied; refusing to share it." >&2
  printf '%s\n' "$busy" >&2
  exit 75
fi

mkdir -p -- "$OUTPUT_ROOT/replays"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.before.sha256"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 500 >"$OUTPUT_ROOT/gpu.csv" &
monitor_pid=$!

for index in "${!episodes[@]}"; do
  episode="${episodes[$index]}"
  slot="${slots[$index]}"
  yaw="${yaws[$index]}"
  pitch="${pitches[$index]}"
  episode_dir="$OUTPUT_ROOT/replays/episode_$(printf '%03d' "$episode")"
  echo "[replay10] episode=$episode slot=$slot yaw=$yaw pitch=$pitch"
  G1_LIGHT_FLICKER=off \
  G1_HEADLAMP=on \
  G1_HEADLAMP_INTENSITY=30000000 \
  G1_HEADLAMP_AMBIENT_INTENSITY=80 \
  G1_HEADLAMP_CAMERA_EXPOSURE=0 \
  G1_HEADLAMP_NUREC_EMISSIVE_SCALE=0.16 \
  G1_HEADLAMP_CONE_ANGLE_DEG=32 \
  G1_HEADLAMP_CONE_SOFTNESS=0.45 \
  G1_HEADLAMP_COLOR_TEMP_K=4100 \
  G1_HEADLAMP_RADIUS_M=0.025 \
  G1_HEADLAMP_FOCUS=2.5 \
  G1_HEADLAMP_AIM_YAW_DEG="$yaw" \
  G1_HEADLAMP_AIM_PITCH_DEG="$pitch" \
  G1_DUST_MODE=off \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_DUST_REPLAY_OUTPUT_ROOT="$OUTPUT_ROOT/replays" \
    "$TASK_ROOT/run_3d_dust_replay.sh" "$episode"
  test -s "$episode_dir/observation.images.head.mp4"
  test -s "$episode_dir/manifest.json"
  grep -q "aim_yaw=$(printf '%.2f' "$yaw")deg, aim_pitch=$(printf '%.2f' "$pitch")deg" \
    "$episode_dir/isaacsim.log"
  touch "$episode_dir/.RENDER_COMPLETE"
done

cleanup_monitor
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" \
  "$OUTPUT_ROOT/source_hashes.after.sha256"

source "$CONDA_SH"
conda activate tv
python "$TASK_ROOT/make_headlamp_replay10.py" \
  --root "$OUTPUT_ROOT" --dataset "$DATASET_ROOT"
python - "$OUTPUT_ROOT/replay10_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("all_checks_passed", False):
    raise SystemExit("10-episode varied-spot replay did not pass validation")
PY
touch "$OUTPUT_ROOT/ALL_COMPLETE"
echo "[replay10] Complete: $OUTPUT_ROOT"
