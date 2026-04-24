#!/usr/bin/env bash
# Configure + build llama-bench and test-backend-ops for the WebGPU backend.
# Reuses an existing build dir if already configured.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-webgpu-profile}"
DAWN_PREFIX="${DAWN_PREFIX:-$HOME/dawn-install}"

if [ ! -d "$DAWN_PREFIX" ]; then
  echo "error: DAWN_PREFIX=$DAWN_PREFIX does not exist." >&2
  echo "       Install Dawn and set DAWN_PREFIX, e.g.:" >&2
  echo "         DAWN_PREFIX=/opt/dawn bash $0" >&2
  exit 1
fi

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "==> configuring $BUILD_DIR"
  cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$DAWN_PREFIX" \
    -DGGML_WEBGPU=ON \
    -DGGML_WEBGPU_CPU_PROFILE=OFF \
    -DGGML_WEBGPU_GPU_PROFILE=OFF \
    -DGGML_BUILD_TESTS=ON
fi

echo "==> building llama-bench + test-backend-ops in $BUILD_DIR"
cmake --build "$BUILD_DIR" --target llama-bench test-backend-ops -j"$(nproc)"

echo "==> built:"
ls -lh "$BUILD_DIR/bin/llama-bench" "$BUILD_DIR/bin/test-backend-ops"
