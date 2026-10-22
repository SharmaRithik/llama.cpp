# On-demand MLIR matmul compilation

This native prototype sends a high-level matmul to an MLIR compiler when
llama.cpp needs a new WebGPU pipeline. It uses no Python kernel builder and
embeds no pre-generated matmul shaders. The ordinary build and execution path
remain available when the compiler option is disabled.

## Flow

1. `get_mul_mat_fast_pipeline` or `get_mul_mat_vec_pipeline` receives the ggml tensor descriptions and device capabilities.
2. On a cache miss, `ggml_webgpu_mlir_matmul_source` describes the operation with upstream Tensor, Arith and Linalg operations. The input contains no GPU IDs, tiling, copies, barriers or vector instructions.
3. `mlirWgslCompileMatmul` parses and verifies that input. The compiler owns workgroup tiling, vectorized copies, shared-memory layout and padding, boundary handling, prefetching, final-iteration peeling, buffer reuse, and the choice between MMA and register arithmetic.
4. For tiled matmuls, upstream Linalg vectorization turns the tile computation into `vector.contract`. Upstream VectorToGPU lowers the MMA path; upstream vector/SCF patterns plus loop-carried vector splitting lower the register path. GEMV instead distributes the reduction across subgroups using Arith, Vector, SCF and GPU ops.
5. The WGSL emitter returns source and launch metadata. llama.cpp gives the source to Dawn and caches the resulting pipeline.

The public compiler boundary is `mlir/Target/WGSL/MatmulCompiler.h` in the paired
LLVM tree. The LLVM tool `mlir-wgsl-compile` uses the same library and can replay
a dumped request independently.

## Supported contract

- `C[m,n] = sum_k A[m,k] * half(B[n,k])`, with FP16 operands and FP32 output. An FP16 B needs no conversion. MMA supports FP16 or FP32 accumulation; register tiling uses FP32. Conversion and accumulation precision are explicit in the input MLIR.
- Static positive M/N/K, contiguous inner dimensions and contiguous output. Row and batch strides, broadcast batches, and overlapping FP16 input ranges are supported. Logical C is stored at `C[n*M+m]` within each batch.
- MMA uses 32-lane subgroups, 16x16x16 fragments and optionally 16x8x16 fragments. The adapter passes the actual device's matrix capabilities, workgroup/shared-memory limits, storage alignment and native vendor policy to the compiler.
- The compiler selects a schedule that fits these limits. The optimized native NVIDIA schedules vary with shape and precision; this is a bounded policy, not an auto-tuner.
- Offset, stride and broadcast uniforms remain runtime values. The existing 16-word ggml ABI is used; matrix dimensions and batch count are specialized. Batching independently applies the rank-2 MLIR program.

Requested unsupported FP16 matmuls fail explicitly. Quantized types retain their normal dispatch. The compiler lowers the supported matmul graph, not the complete model graph.

### Generated GEMV

`GGML_WEBGPU_MLIR_GEMV=1` independently enables compiler generation for the F16-weight N <= 4 matvec path. The adapter sends an ordinary high-level Linalg matmul with FP32 accumulation and no activation truncation. The compiler assigns one output row to each subgroup, vectorizes aligned loads, accumulates in FP32 registers, and emits `gpu.subgroup_reduce add`, which becomes WGSL `subgroupAdd`. It uses no shared-memory reduction or workgroup barriers. For N=2..4, each weight load feeds all activation vectors.

This requires subgroups with a fixed device size, but does not require the experimental subgroup matrix extension. F16/F32 activations, row/batch strides, broadcast batches, partial dimensions and overlapping F16 inputs are supported. GEMV always uses FP32 accumulation, independently of the MMA accumulation setting. Leave the flag unset or set it to `0` for the shipped GEMV path. The flag can be combined with `GGML_WEBGPU_MLIR=mma` or `regtile`, or used alone to isolate GEMV in a comparison.

## Build

From the workspace containing both repositories:

```sh
nice -n 19 ninja -C llvm-project/build -j4 mlir-wgsl-compile mlir-opt mlir-translate
cmake -S llama.cpp -B llama.cpp/build-webgpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_WEBGPU=ON \
  -DDawn_DIR=/usr/local/lib/cmake/Dawn \
  -DGGML_WEBGPU_MLIR_ROOT="$PWD/llvm-project"
nice -n 19 ninja -C llama.cpp/build-webgpu -j4 test-backend-ops llama-bench
```

`GGML_WEBGPU_MLIR_ROOT` is optional and points to the LLVM source tree with its
compiler library built under `build/lib`. Leaving it empty removes the LLVM
include and link dependency. Browser builds do not support this native library
integration.

## Run and inspect

Select the desired native WebGPU adapter, then run:

```sh
GGML_WEBGPU_MLIR=mma GGML_WEBGPU_MLIR_DUMP_DIR=/tmp/mlir-mma \
  llama.cpp/build-webgpu/bin/test-backend-ops test \
  -b WebGPU -o MUL_MAT --test-file llama.cpp/tests/test-webgpu-mlir.txt
```

Use `GGML_WEBGPU_MLIR=regtile` for the other compiler schedule. Leave the variable
unset to use llama.cpp's shipped tunable kernels. For a register-only baseline, also set `GGML_WEBGPU_DISABLE_SUBGROUP_MATRIX=1` to disable the default MMA path.

| Variable | Meaning |
| --- | --- |
| `GGML_WEBGPU_MLIR=mma\|regtile` | Request compiler-generated matmuls |
| `GGML_WEBGPU_MLIR_GEMV=0\|1` | Independently enable generated F16-weight GEMV for N <= 4; default 0, FP32 accumulation |
| `GGML_WEBGPU_MLIR_ACC=f16\|f32` | Accumulation precision; defaults to FP16 for MMA and FP32 for register tiling. A ggml FP32 precision request always takes precedence. |
| `GGML_WEBGPU_MLIR_OPT_LEVEL=0\|1\|2` | Reference schedule, tiled/vectorized schedule, or optimized target policy; default 2 |
| `GGML_WEBGPU_MLIR_DUMP_DIR` | Save input, scheduled/lowered MLIR and WGSL for compiler requests |
| `GGML_WEBGPU_SHADER_DUMP_DIR` | Save the exact WGSL passed to Dawn, including shipped shaders, to audit selection |

Each cache miss logs the compilation
and writes four artifacts when the dump directory is set:

- `.input.mlir`: mathematical operation, required activation conversion and zero initialization.
- `.scheduled.mlir`: tiled GPU kernel with vector contractions, or distributed GEMV with subgroup reductions.
- `.lowered.mlir`: subgroup MMA operations, register arithmetic, or GEMV arithmetic and subgroup reductions.
- `.wgsl`: source passed to Dawn.

The 35-case fixture includes both activation types, partial M/N/K tiles, 14 GEMV cases, and a
repeated request. With both generation switches enabled it causes 20 tiled matmul and 14 GEMV compilations and reuses the
pipeline for the duplicate. The backend test suite also covers merged GEMV inputs, broadcasts and padded batch/row strides. The cache belongs to one device and process; the
key includes M/N/K, activation type, strategy, precision, optimization level, alignment, batch count and binding layout. There is no persistent cache.
Logged compilation time covers MLIR through WGSL emission, excluding Dawn's
pipeline creation. Kernel performance tests run after pipeline preparation.
