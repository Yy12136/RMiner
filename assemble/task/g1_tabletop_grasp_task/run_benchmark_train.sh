#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$HOME/Benchmark/policy/lerobot}"
TRAIN="$LEROBOT_ROOT/.venv/bin/lerobot-train"
DATASET_ROOT="${G1_LEROBOT_DATASET_DIR:-$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion}"
POLICY="act"
STEPS=100000
BATCH_SIZE=16
SAVE_FREQ=10000
NUM_WORKERS=0
WANDB=false
WANDB_PROJECT="g1-tabletop-grasp"
EXTRA_ARGS=()

usage() {
  echo "Usage: ./benchmark.sh train [--policy act] [--steps 100000] [--batch-size 16] [--save-freq 10000] [--wandb] [-- <extra lerobot-train args>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --wandb) WANDB=true; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -x "$TRAIN" ]]; then
  echo "LeRobot environment is not ready: $LEROBOT_ROOT/.venv" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "LeRobot Dataset v3.0 not found: $DATASET_ROOT" >&2
  exit 1
fi
for value in "$STEPS" "$BATCH_SIZE" "$SAVE_FREQ"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "steps, batch-size, and save-freq must be positive integers." >&2
    exit 1
  fi
done

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${POLICY}_detonator_wire_insertion_${STEPS}steps_${RUN_STAMP}_bs${BATCH_SIZE}"
OUTPUT_DIR="$LEROBOT_ROOT/outputs/train/$RUN_NAME"
LOG_DIR="$LEROBOT_ROOT/outputs/train/logs"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$LOG_DIR"

echo "Starting LeRobot training"
echo "  policy:  $POLICY"
echo "  dataset: $DATASET_ROOT"
echo "  steps:   $STEPS"
echo "  batch:   $BATCH_SIZE"
echo "  output:  $OUTPUT_DIR"
echo "  wandb:   $WANDB"

cd "$LEROBOT_ROOT"
"$TRAIN" \
  --policy.type="$POLICY" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/detonator_wire_insertion \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --persistent_workers=false \
  --steps="$STEPS" \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable="$WANDB" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.mode=online \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo "Training completed: $OUTPUT_DIR"

