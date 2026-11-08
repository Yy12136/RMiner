#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="${TASK_ROOT_OVERRIDE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_ROOT="${G1_SOFT075_DEMO_DATASET:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/data/lerobot_v30/detonator_wire_insertion_replay_ready}"
OUTPUT_ROOT="${G1_SOFT075_DEMO_OUTPUT:-$TASK_ROOT/data/augmentation_previews/flow_train_medium_soft_single_double075_demo_20260905}"
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

busy="$(pgrep -af '[e]valuation/supervisor.py|[p]ython .*sim_main.py|[r]un_3d_dust' || true)"
if [[ -n "$busy" ]]; then echo "Development Isaac Sim/DDS runtime is occupied; refusing to share it." >&2; printf '%s\n' "$busy" >&2; exit 75; fi
mkdir -p "$OUTPUT_ROOT"
find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/source_hashes.before.sha256"

run_variant() {
  local name="$1" mode="$2" layout="$3" ambient="$4" opacity="$5"; local root="$OUTPUT_ROOT/$name"; mkdir -p "$root"; echo "[soft075-demo] rendering $name"
  G1_HEADLAMP=off G1_AUX_LIGHTS=off G1_DUST_MODE="$mode" G1_DUST_PROFILE=train G1_DUST_PRESET=medium G1_DUST_WIDE=0 G1_DUST_LAYOUT="$layout" G1_DUST_AMBIENT_MULTIPLIER="$ambient" G1_FLOW_OPACITY="$opacity" G1_DUST_SEED=260822 G1_OBJECT_RESET_SEED=260822 G1_DUST_REPLAY_DDS_SETTLE_SECONDS=12 G1_DUST_REPLAY_DATASET="$DATASET_ROOT" G1_DUST_REPLAY_OUTPUT_ROOT="$root" "$TASK_ROOT/run_3d_dust_replay.sh" 0
  for f in observation.images.head.mp4 observation.images.left.mp4 observation.images.right.mp4 three_camera_grid.mp4 manifest.json; do test -s "$root/episode_000/$f"; done
}

run_variant clean off default 1.0 0.0
run_variant final_train_medium flow default 1.20 0.40
run_variant final_train_medium_single1m_soft flow single_1m_soft 0.55 0.055
run_variant final_train_medium_double075 flow left_right_075m 0.68 0.12

find "$DATASET_ROOT/meta" "$DATASET_ROOT/data" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/source_hashes.after.sha256"
cmp "$OUTPUT_ROOT/source_hashes.before.sha256" "$OUTPUT_ROOT/source_hashes.after.sha256"
ffmpeg -hide_banner -loglevel error -y -i "$OUTPUT_ROOT/clean/episode_000/observation.images.head.mp4" -i "$OUTPUT_ROOT/final_train_medium/episode_000/observation.images.head.mp4" -i "$OUTPUT_ROOT/final_train_medium_single1m_soft/episode_000/observation.images.head.mp4" -i "$OUTPUT_ROOT/final_train_medium_double075/episode_000/observation.images.head.mp4" -filter_complex '[0:v]scale=320:240[c];[1:v]scale=320:240[m];[2:v]scale=320:240[s];[3:v]scale=320:240[d];[c][m][s][d]hstack=inputs=4[out]' -map '[out]' -an -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -movflags +faststart "$OUTPUT_ROOT/head_comparison_4x1.mp4"
source "$CONDA_SH"; conda activate tv; python "$TASK_ROOT/make_soft_single_double075_demo.py" --root "$OUTPUT_ROOT" --dataset "$DATASET_ROOT"; touch "$OUTPUT_ROOT/ALL_COMPLETE"; echo "[soft075-demo] complete: $OUTPUT_ROOT"
