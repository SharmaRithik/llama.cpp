# debug-webgpu-matvec-iq-fast

Reproduction harness for the fast i-quant mat-vec kernel work
(`webgpu-matvec-iq-fast` branch / PR in `ggml/src/ggml-webgpu`).

Measures **master vs branch** for two things:

1. **Isolated kernel**: `MUL_MAT(type_a=iq*, m=4096, n=1, k=14336)` via
   `test-backend-ops perf`. Pure GFLOPS per kernel, no model overhead.
2. **End-to-end model**: Llama-3.2-1B in 6 i-quant formats (IQ1_S, IQ1_M,
   IQ2_XXS, IQ3_XXS, IQ4_NL, IQ4_XS) via `llama-bench`, reporting pp64
   (prompt) and tg128 (decode) tokens/second.

## Models used

All six models are 1B Llama-3.2-Instruct GGUFs from the Unsloth HF repo:
`huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF`. Total ~3.7 GB.

| file | exercises |
|---|---|
| `Llama-3.2-1B-Instruct-UD-IQ1_S.gguf`   | IQ1_S  kernel |
| `Llama-3.2-1B-Instruct-UD-IQ1_M.gguf`   | IQ1_M  kernel |
| `Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf` | IQ2_XXS kernel |
| `Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf` | IQ3_XXS kernel |
| `Llama-3.2-1B-Instruct-IQ4_NL.gguf`     | IQ4_NL kernel |
| `Llama-3.2-1B-Instruct-IQ4_XS.gguf`     | IQ4_XS kernel |

The `UD-` prefixed ones are Unsloth "dynamic" quants — dominant quant is
as named, but some sensitive tensors (embed/output) use Q4_K / Q6_K.
Good enough to exercise the named IQ kernel as the hot path but not a
pure single-quant workload.

The isolated-kernel benchmark (`test-backend-ops perf`) generates its own
synthetic tensors and does NOT need any model file — it covers IQ2_XS,
IQ2_S, and IQ3_S (the three formats with no ready-made HF model) in
addition to the six above.

## Requirements

- A WebGPU-capable machine (tested on Intel Arc B580, Mesa 25.2.8).
- Dawn installed. Set `DAWN_PREFIX` if it's not at `~/dawn-install`.
- Python 3 with `venv` (scripts auto-create a venv for matplotlib).
- ~8 GB free disk for the 6 model files.
- `wget` + ~5 GB bandwidth on first run (models are downloaded from
  Hugging Face if missing).

## One-command repro

From the repo root:

```bash
bash scripts/debug-webgpu-matvec-iq-fast/run-all.sh
```

That script does:

1. Download the 6 i-quant GGUFs into `models/llama-3.2-1b/` (skips any
   already present).
2. Check out `master`, build `llama-bench` + `test-backend-ops`, run both
   benchmarks, save CSVs tagged `master`.
3. Check out `webgpu-matvec-iq-fast`, rebuild, re-run, save CSVs tagged
   `branch`.
4. Render grouped-bar PNGs and a markdown summary into
   `scripts/debug-webgpu-matvec-iq-fast/results/`.

It will leave you on the branch you started on.

Total runtime on a warm cache: ~5 min (two builds @ ~60s each, plus
benchmarks). Model downloads on a cold cache add ~5–10 min.

## Individual steps

If you'd rather drive the pieces yourself:

| script | what it does |
|---|---|
| `fetch-models.sh` | Download the 6 GGUFs, idempotent |
| `build.sh [dir]` | Configure + build `llama-bench`, `test-backend-ops` in `build-webgpu-profile` (override via `BUILD_DIR=`) |
| `bench-models.sh <tag>` | Run `llama-bench -p 64 -n 128 -r 3` on the 6 models, write `results/llama-bench-<tag>.csv` |
| `bench-kernel.sh <tag>` | Run `test-backend-ops perf` for MUL_MAT, write `results/perf-<tag>.txt` |
| `plot.py` | Ingest the two pairs of CSV/txt, emit PNGs + `results/summary.md` |

Typical manual flow:

```bash
bash scripts/debug-webgpu-matvec-iq-fast/fetch-models.sh
git checkout master
bash scripts/debug-webgpu-matvec-iq-fast/build.sh
bash scripts/debug-webgpu-matvec-iq-fast/bench-models.sh master
bash scripts/debug-webgpu-matvec-iq-fast/bench-kernel.sh master
git checkout webgpu-matvec-iq-fast
bash scripts/debug-webgpu-matvec-iq-fast/build.sh
bash scripts/debug-webgpu-matvec-iq-fast/bench-models.sh branch
bash scripts/debug-webgpu-matvec-iq-fast/bench-kernel.sh branch
python3 scripts/debug-webgpu-matvec-iq-fast/plot.py
```

## Output

After `run-all.sh` completes, `scripts/debug-webgpu-matvec-iq-fast/results/`
contains:

- `llama-bench-master.csv`, `llama-bench-branch.csv`
- `perf-master.txt`, `perf-branch.txt`
- `chart-kernel.png` — GFLOPS per iquant, the kernel-level win
- `chart-decode.png` — tg128 tokens/second per iquant, the user-visible win
- `chart-prompt.png` — pp64 tokens/second (should be flat; this is the
  "no regression" chart)
- `summary.md` — numbers table + chart embeds, ready to paste into a PR

## Environment overrides

- `DAWN_PREFIX` — Dawn install path (default `$HOME/dawn-install`)
- `BUILD_DIR` — build directory (default `build-webgpu-profile`)
- `MODEL_DIR` — where GGUFs live / are fetched to (default `models/llama-3.2-1b`)
- `REPS` — `llama-bench -r` value (default `3`)
- `MASTER_REF` — ref to compare against (default `master`)
- `BRANCH_REF` — branch under test (default `webgpu-matvec-iq-fast`)

## Caveats

- `test-backend-ops perf` does not report stddev. Runs are long (thousands
  of kernel invocations averaged), so single-shot numbers are stable in
  practice, but don't over-interpret sub-5% differences on the kernel
  chart.
- `llama-bench -r 3` emits avg ± stddev directly. Error bars are shown
  on the model charts.
- The Unsloth "UD-" GGUFs are dynamic quants: the file name describes
  the dominant quant but some tensors use higher precision (typically
  Q4_K/Q6_K for embed/output). Decode numbers reflect that mix, not a
  pure single-quant workload.
- This harness assumes `-ngl 99` (all layers on GPU). Adjust via
  `NGL=` env var if needed.
