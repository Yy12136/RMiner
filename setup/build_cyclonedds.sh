#!/usr/bin/env bash
# Build the bundled CycloneDDS into third_party/cyclonedds/install and print the
# CYCLONEDDS_HOME export. Required by both conda envs.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSEMBLE="$(cd -- "$HERE/../assemble" && pwd)"
SRC="$ASSEMBLE/third_party/cyclonedds"

command -v cmake >/dev/null 2>&1 || { echo "cmake is required (apt install cmake build-essential)"; exit 1; }

cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_INSTALL_PREFIX="$SRC/install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_DDSPERF=OFF
cmake --build "$SRC/build" -j"$(nproc)"
cmake --install "$SRC/build"

echo
echo "CycloneDDS built. Add this to your shell / activate hooks:"
echo "  export CYCLONEDDS_HOME=$SRC/install"
