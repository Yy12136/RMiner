#!/usr/bin/env bash
# Download the RMiner 3D assets from Hugging Face and place them in the
# directory layout expected by the Isaac Lab task configs.
#
#   ./scripts/download_assets.sh                 # everything (~8 GB)
#   ./scripts/download_assets.sh --no-task      # skip per-task assets
#
# Requires: pip install -U huggingface_hub
set -euo pipefail

REPO_ID="${RMiner_HF_REPO:-lyang116/RMiner}"
REPO_TYPE="${RMiner_HF_REPO_TYPE:-dataset}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/.hf_assets"

WITH_TASK=1
for arg in "$@"; do
  case "$arg" in
    --no-task) WITH_TASK=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v hf >/dev/null 2>&1 || {
  echo "hf CLI not found. Install it with: pip install -U huggingface_hub" >&2
  exit 1
}

INCLUDE=(
  --include "assets/assemble/assets-new/**"
  --include "assets/assemble/objects/**"
  --include "assets/assemble/third_party/unitree_sim_isaaclab/assets/**"
)
if [[ "$WITH_TASK" == 1 ]]; then
  INCLUDE+=(--include "assets/assemble/task/**")
fi

echo "==> Downloading assets from $REPO_ID to $CACHE"
hf download "$REPO_ID" --repo-type "$REPO_TYPE" "${INCLUDE[@]}" --local-dir "$CACHE"

echo "==> Installing assets into the repository layout"
mkdir -p "$ROOT/assemble" "$ROOT/assemble/task"

copy_dir() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    rsync -a --info=progress2 "$src/" "$dst/"
  fi
}

copy_dir "$CACHE/assets/assemble/assets-new" "$ROOT/assemble/assets-new"
copy_dir "$CACHE/assets/assemble/objects" "$ROOT/assemble/objects"

if [[ "$WITH_TASK" == 1 ]]; then
  for d in "$CACHE"/assets/assemble/task/*/; do
    [[ -d "$d/assets" ]] || continue
    task="$(basename "$d")"
    copy_dir "$d/assets" "$ROOT/assemble/task/$task/assets"
  done
fi

copy_dir "$CACHE/assets/assemble/third_party/unitree_sim_isaaclab/assets" \
         "$ROOT/assemble/third_party/unitree_sim_isaaclab/assets"

echo "==> Done."
echo "    assets-new : $ROOT/assemble/assets-new"
echo "    objects    : $ROOT/assemble/objects"
echo "    task assets: $ROOT/assemble/task/*/assets"
