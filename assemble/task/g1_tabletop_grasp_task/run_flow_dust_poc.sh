#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${G1_DUST_REPLAY_DATASET:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion_50_20260822}"
OUTPUT_ROOT="${G1_FLOW_POC_OUTPUT_ROOT:-$TASK_ROOT/data/augmentation_previews/flow_poc_20260827}"
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
EPISODE=0
monitor_pid=""

cleanup_monitor() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

assert_development_runtime_idle() {
  local busy
  busy="$(pgrep -af "[p]ython .*sim_main.py|[e]valuation/supervisor.py" || true)"
  if [[ -n "$busy" ]]; then
    echo "An Isaac Sim task or formal evaluation supervisor is active." >&2
    echo "Refusing to interrupt it or share the development GPU/DDS runtime." >&2
    printf '%s\n' "$busy" >&2
    exit 75
  fi
}

assert_development_runtime_idle

mkdir -p -- "$OUTPUT_ROOT"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.before.sha256"

PREVIEW_DIR="$OUTPUT_ROOT/preflight"
mkdir -p -- "$PREVIEW_DIR"
echo "[flow-poc] Running 5-second Flow visibility preflight..."
G1_DUST_MODE=flow \
G1_DUST_PRESET=medium \
G1_OBJECT_RESET_SEED=260822 \
G1_DUST_PREVIEW_SECONDS=5 \
G1_DUST_PREVIEW_OUTPUT="$PREVIEW_DIR/flow_medium_5s.mp4" \
  "$TASK_ROOT/run_3d_dust_preview.sh"
ffprobe -v error -show_entries stream=width,height,r_frame_rate \
  -show_entries format=duration -of json "$PREVIEW_DIR/flow_medium_5s.mp4" \
  >"$PREVIEW_DIR/ffprobe.json"
grep -q "\[flow-dust-3d\] Enabled" "$PREVIEW_DIR/flow_medium_5s.sim.log"

run_variant() {
  local variant="$1"
  local mode="$2"
  local preset="$3"
  local variant_root="$OUTPUT_ROOT/$variant"
  local gpu_csv="$variant_root/gpu.csv"
  assert_development_runtime_idle
  mkdir -p -- "$variant_root"
  echo "[flow-poc] Rendering $variant (mode=$mode preset=$preset)..."
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >"$gpu_csv" &
  monitor_pid=$!
  set +e
  G1_DUST_MODE="$mode" \
  G1_DUST_PRESET="$preset" \
  G1_OBJECT_RESET_SEED=260822 \
  G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 \
  G1_DUST_REPLAY_DATASET="$DATASET_ROOT" \
  G1_DUST_REPLAY_OUTPUT_ROOT="$variant_root" \
    "$TASK_ROOT/run_3d_dust_replay.sh" "$EPISODE"
  local status=$?
  set -e
  cleanup_monitor
  if (( status != 0 )); then
    echo "[flow-poc] Variant $variant failed with status $status" >&2
    return "$status"
  fi
  assert_development_runtime_idle
  for name in observation.images.head.mp4 observation.images.left.mp4 \
    observation.images.right.mp4 three_camera_grid.mp4 manifest.json; do
    test -s "$variant_root/episode_000/$name"
  done
  python3 - "$variant_root/episode_000" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in (
    "observation.images.head.mp4",
    "observation.images.left.mp4",
    "observation.images.right.mp4",
):
    path = root / name
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 640,
        "height": 480,
        "r_frame_rate": "30/1",
    }
    mismatches = {
        key: (stream.get(key), value)
        for key, value in expected.items()
        if stream.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Unexpected video format for {path}: {mismatches}")
PY
}

run_variant clean off medium
run_variant legacy legacy medium
run_variant flow_low flow low
run_variant flow_medium flow medium
run_variant flow_heavy flow heavy

find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" \
  "$OUTPUT_ROOT/source_hashes.after.sha256"

source "$CONDA_SH"
conda activate tv
mkdir -p -- "$OUTPUT_ROOT/deliverables"
python "$TASK_ROOT/make_dust_poc_report.py" --root "$OUTPUT_ROOT" \
  >"$OUTPUT_ROOT/deliverables/report_stdout.json"

python - "$OUTPUT_ROOT/deliverables/poc_report.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
peak=max(v["gpu"].get("gpu_memory_peak_mib",0) for v in report["variants"].values())
minimum=min(v["effective_replay_fps"] for v in report["variants"].values())
if peak > 14848:
    raise SystemExit(f"Peak GPU memory {peak:.1f} MiB exceeds 14.5 GiB")
if minimum < 20:
    raise SystemExit(f"Effective replay rate {minimum:.2f} FPS is below 20 FPS")
print(f"Acceptance passed: peak={peak:.1f} MiB, minimum_fps={minimum:.2f}")
PY

echo "[flow-poc] Complete: $OUTPUT_ROOT/deliverables"
