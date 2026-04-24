#!/usr/bin/env bash
# End-to-end model benchmark via llama-bench.
# Writes: scripts/debug-webgpu-matvec-iq-fast/results/llama-bench-<tag>.csv
#
# Usage: bash bench-models.sh <tag>   (tag is typically "master" or "branch")
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: $0 <tag>   (e.g. master, branch)" >&2
  exit 2
fi

BUILD_DIR="${BUILD_DIR:-build-webgpu-profile}"
MODEL_DIR="${MODEL_DIR:-models/llama-3.2-1b}"
REPS="${REPS:-3}"
NGL="${NGL:-99}"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/llama-bench-$TAG.csv"

MODELS=(
  "Llama-3.2-1B-Instruct-UD-IQ1_S.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ1_M.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf"
  "Llama-3.2-1B-Instruct-IQ4_NL.gguf"
  "Llama-3.2-1B-Instruct-IQ4_XS.gguf"
)

BENCH="$BUILD_DIR/bin/llama-bench"
if [ ! -x "$BENCH" ]; then
  echo "error: $BENCH not found. Run build.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
: > "$OUT"

for m in "${MODELS[@]}"; do
  mpath="$MODEL_DIR/$m"
  if [ ! -s "$mpath" ]; then
    echo "skip  $m (missing — run fetch-models.sh)" >&2
    continue
  fi
  echo "==> $TAG: $m"
  LD_LIBRARY_PATH="$BUILD_DIR/bin" "$BENCH" \
    -m "$mpath" -p 64 -n 128 -r "$REPS" -ngl "$NGL" -o csv 2>/dev/null >> "$OUT"
done

lines=$(grep -c . "$OUT" || true)
echo "==> wrote $OUT ($lines lines)"
