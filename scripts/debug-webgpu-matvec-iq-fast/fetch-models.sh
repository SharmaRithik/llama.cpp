#!/usr/bin/env bash
# Download the 6 i-quant Llama-3.2-1B GGUFs used by the bench harness.
# Idempotent: skips files that already exist and are non-empty.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/llama-3.2-1b}"
BASE_URL="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main"

FILES=(
  "Llama-3.2-1B-Instruct-UD-IQ1_S.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ1_M.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf"
  "Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf"
  "Llama-3.2-1B-Instruct-IQ4_NL.gguf"
  "Llama-3.2-1B-Instruct-IQ4_XS.gguf"
)

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

for f in "${FILES[@]}"; do
  if [ -s "$f" ]; then
    echo "skip  $f (already present, $(du -h "$f" | cut -f1))"
    continue
  fi
  echo "fetch $f"
  wget --no-verbose --show-progress --continue "$BASE_URL/$f"
done

echo "done. models in $(pwd)"
