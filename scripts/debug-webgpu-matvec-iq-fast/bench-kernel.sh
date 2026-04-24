#!/usr/bin/env bash
# Isolated kernel benchmark via test-backend-ops perf for MUL_MAT.
# Filters to MUL_MAT(type_a=iq*, m=4096, n=1, k=14336) in post-processing.
# Writes: scripts/debug-webgpu-matvec-iq-fast/results/perf-<tag>.txt
#
# Usage: bash bench-kernel.sh <tag>
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: $0 <tag>   (e.g. master, branch)" >&2
  exit 2
fi

BUILD_DIR="${BUILD_DIR:-build-webgpu-profile}"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/perf-$TAG.txt"

BIN="$BUILD_DIR/bin/test-backend-ops"
if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found. Run build.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "==> $TAG: running test-backend-ops perf MUL_MAT on WebGPU (this takes a few minutes)"
LD_LIBRARY_PATH="$BUILD_DIR/bin" "$BIN" perf -b WebGPU -o MUL_MAT 2>&1 \
  | grep -E "type_a=iq[0-9].*m=4096,n=1,k=14336" \
  | sed 's/\x1b\[[0-9;]*m//g' \
  > "$OUT"

lines=$(wc -l < "$OUT")
echo "==> wrote $OUT ($lines iquant rows)"

if [ "$lines" -lt 9 ]; then
  echo "warning: expected 9 iquant rows, got $lines" >&2
fi
