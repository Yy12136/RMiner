#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$HOME/Benchmark/policy/lerobot}"
PYTHON="$LEROBOT_ROOT/.venv/bin/python"
TRAIN="$LEROBOT_ROOT/.venv/bin/lerobot-train"
DATASET_ROOT="$TASK_ROOT/data/lerobot_v30/detonator_wire_insertion"
RUN_STAMP="${LEROBOT_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${LEROBOT_TRAIN_STEPS:-500}"
SAVE_FREQ="${LEROBOT_SAVE_FREQ:-250}"
WANDB_ENABLE="${LEROBOT_WANDB_ENABLE:-false}"
WANDB_PROJECT="${LEROBOT_WANDB_PROJECT:-g1-tabletop-grasp}"
WANDB_MODE="${LEROBOT_WANDB_MODE:-online}"
JOB_NAME="${LEROBOT_JOB_NAME:-act_detonator_wire_insertion_${TRAIN_STEPS}steps}"
RUN_PREFIX="act_detonator_wire_insertion_${TRAIN_STEPS}steps_${RUN_STAMP}"
LOG_DIR="$LEROBOT_ROOT/outputs/train/logs"

if [[ ! -x "$PYTHON" || ! -x "$TRAIN" ]]; then
  echo "LeRobot training environment is not ready: $LEROBOT_ROOT/.venv" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "LeRobot Dataset v3.0 dataset not found: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$LEROBOT_ROOT"

for batch_size in 8 4 2; do
  output_dir="$LEROBOT_ROOT/outputs/train/${RUN_PREFIX}_bs${batch_size}"
  log_file="$LOG_DIR/${RUN_PREFIX}_bs${batch_size}.log"

  echo "Starting ACT smoke training"
  echo "  dataset: $DATASET_ROOT"
  echo "  output:  $output_dir"
  echo "  log:     $log_file"
  echo "  batch:   $batch_size"
  echo "  steps:   $TRAIN_STEPS"
  echo "  wandb:   $WANDB_ENABLE ($WANDB_PROJECT)"

  set +e
  "$TRAIN" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/detonator_wire_insertion \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$output_dir" \
    --job_name="$JOB_NAME" \
    --batch_size="$batch_size" \
    --num_workers=0 \
    --persistent_workers=false \
    --steps="$TRAIN_STEPS" \
    --log_freq=10 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --wandb.enable="$WANDB_ENABLE" \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.mode="$WANDB_MODE" \
    2>&1 | tee "$log_file"
  train_status=${PIPESTATUS[0]}
  set -e

  if [[ $train_status -eq 0 ]]; then
    echo "ACT training completed: $output_dir"
    exit 0
  fi

  if grep -Eiq 'CUDA out of memory|OutOfMemoryError' "$log_file"; then
    echo "CUDA out of memory with batch size $batch_size; retrying with a smaller batch." >&2
    continue
  fi

  echo "Training failed for a reason other than CUDA OOM. See: $log_file" >&2
  exit "$train_status"
done

echo "Training exhausted batch sizes 8, 4, and 2 because of CUDA OOM." >&2
exit 1
