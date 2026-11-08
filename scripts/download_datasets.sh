#!/usr/bin/env bash
# Download RMiner LeRobot datasets from Hugging Face and place each one back at
# the path the benchmark expects:
#
#   assemble/task/<task>/data/<lerobot_version>/<dataset_name>/
#
#   ./scripts/download_datasets.sh                       # all tasks (~24 GB)
#   ./scripts/download_datasets.sh g1_valve_handwheel_task
#   ./scripts/download_datasets.sh g1_valve_handwheel_task/lerobot_v30
#
# Requires: pip install -U huggingface_hub
set -euo pipefail

REPO_ID="${RMiner_HF_REPO:-lyang116/RMiner}"
REPO_TYPE="${RMiner_HF_REPO_TYPE:-dataset}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/.hf_datasets"

command -v hf >/dev/null 2>&1 || {
  echo "hf CLI not found. Install it with: pip install -U huggingface_hub" >&2
  exit 1
}

SELECTOR="${1:-}"
if [[ -n "$SELECTOR" ]]; then
  SELECTOR="${SELECTOR#datasets/}"
  PATTERN="datasets/${SELECTOR%/}/**"
else
  PATTERN="datasets/**"
fi

echo "==> Downloading $PATTERN from $REPO_ID"
hf download "$REPO_ID" --repo-type "$REPO_TYPE" \
  --include "$PATTERN" --local-dir "$CACHE"

echo "==> Installing datasets into assemble/task/*/data/"
shopt -s nullglob
for task_dir in "$CACHE"/datasets/*/; do
  task="$(basename "$task_dir")"
  for ver_dir in "$task_dir"*/; do
    ver="$(basename "$ver_dir")"
    for ds in "$ver_dir"*/; do
      name="$(basename "$ds")"
      dst="$ROOT/assemble/task/$task/data/$ver/$name"
      mkdir -p "$(dirname "$dst")"
      echo "  $task/$ver/$name"
      rsync -a --info=progress2 "$ds" "$dst/"
    done
  done
done

echo "==> Done."
