#!/usr/bin/env bash
# End-to-end WebGPU bounds-check experiment.
#
# 1. Build two WebGPU backends:
#    - build-webgpu-bounds-off/  source as-is (Dawn "disable_robustness"
#      toggle present -> bounds checks OFF; this is the native default)
#    - build-webgpu-bounds-on/   patched (toggle removed -> bounds checks ON)
# 2. Download the 13 (model, quant) GGUF pairs into models/<model>/
#    if any are missing.
# 3. Sweep llama-bench (-p 512 -n 128 -r 3) across all 13 pairs on both
#    builds.
# 4. Render boundscheck-experiment.txt with the Intel Arc B580 column
#    filled in and AVG rows per GPU sub-table. Other GPU sub-tables
#    remain as placeholders for future runs.
#
# Re-runnable: build dirs are reused if present (incremental); GGUFs are
# only downloaded if absent. Pass --rebuild to force a clean configure,
# --no-bench to skip the sweep, --no-render to skip the table update.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WEBGPU_SRC="ggml/src/ggml-webgpu/ggml-webgpu.cpp"
BUILD_OFF="build-webgpu-bounds-off"
BUILD_ON="build-webgpu-bounds-on"
MODELS_ROOT="$REPO_ROOT/models"
TABLE="$REPO_ROOT/boundscheck-experiment.txt"
RESULT_OFF="$REPO_ROOT/arc-bench-bounds-off.json"
RESULT_ON="$REPO_ROOT/arc-bench-bounds-on.json"
STDERR_LOG="$REPO_ROOT/arc-bench-stderr.log"

DAWN_PREFIX="${DAWN_PREFIX:-$HOME/dawn-install}"
JOBS="$(nproc 2>/dev/null || echo 4)"

REBUILD=0
DO_BENCH=1
DO_RENDER=1
for arg in "$@"; do
    case "$arg" in
        --rebuild)   REBUILD=1 ;;
        --no-bench)  DO_BENCH=0 ;;
        --no-render) DO_RENDER=0 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '\033[1;34m[bc]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bc]\033[0m %s\n' "$*" >&2; exit 1; }

# (model_dir, gguf_basename, hf_repo, hf_filename)
PAIRS=(
    "Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-F16.gguf|unsloth/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-F16.gguf"
    "Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q2_K.gguf|unsloth/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q2_K.gguf"
    "Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q4_K_M.gguf|unsloth/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    "Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q8_0.gguf|unsloth/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q8_0.gguf"
    "gemma-3-270m-it|gemma-3-270m-it-Q4_K_M.gguf|unsloth/gemma-3-270m-it-GGUF|gemma-3-270m-it-Q4_K_M.gguf"
    "Qwen3-0.6B|Qwen3-0.6B-Q4_K_M.gguf|unsloth/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q4_K_M.gguf"
    "LFM2.5-350M|LFM2.5-350M-Q4_K_M.gguf|LiquidAI/LFM2.5-350M-GGUF|LFM2.5-350M-Q4_K_M.gguf"
    "SmolLM3-3B|SmolLM3-3B-Q4_K_M.gguf|unsloth/SmolLM3-3B-GGUF|SmolLM3-3B-Q4_K_M.gguf"
    "Ministral-3-3B-Instruct-2512|Ministral-3-3B-Instruct-2512-Q4_K_M.gguf|unsloth/Ministral-3-3B-Instruct-2512-GGUF|Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
    "Qwen3.5-2B|Qwen3.5-2B-Q4_K_M.gguf|unsloth/Qwen3.5-2B-GGUF|Qwen3.5-2B-Q4_K_M.gguf"
    "gemma-4-E2B-it|gemma-4-E2B-it-Q4_K_M.gguf|unsloth/gemma-4-E2B-it-GGUF|gemma-4-E2B-it-Q4_K_M.gguf"
    "granite-4.0-h-1b|granite-4.0-h-1b-Q4_K_M.gguf|ibm-granite/granite-4.0-h-1b-GGUF|granite-4.0-h-1b-Q4_K_M.gguf"
    "Bonsai-1.7B|Bonsai-1.7B-Q1_0.gguf|prism-ml/Bonsai-1.7B-gguf|Bonsai-1.7B-Q1_0.gguf"
)

# ---------------------------------------------------------------------------
# 1. Patch helpers (toggle "disable_robustness" off in the source).
# ---------------------------------------------------------------------------
patch_bounds_on() {
    python3 - "$WEBGPU_SRC" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
new = src.replace(
    '{ "disable_robustness", "disable_workgroup_init",\n'
    '                                                            "disable_polyfills_on_integer_div_and_mod" }',
    '{ "disable_workgroup_init",\n'
    '                                                            "disable_polyfills_on_integer_div_and_mod" }',
)
new = new.replace('deviceTogglesDesc.enabledToggleCount  = 3;',
                  'deviceTogglesDesc.enabledToggleCount  = 2;')
if new == src:
    sys.exit("patch_bounds_on: source did not change — has the toggle list moved?")
p.write_text(new)
PY
}
revert_patch() { git checkout -- "$WEBGPU_SRC"; }

build_variant() {
    local build_dir="$1" label="$2"
    if [[ $REBUILD -eq 1 ]]; then rm -rf "$build_dir"; fi
    log "configuring $label in $build_dir"
    cmake -S . -B "$build_dir" \
        -DGGML_WEBGPU=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$DAWN_PREFIX" \
        -DDawn_DIR="$DAWN_PREFIX/lib/cmake/Dawn" >/dev/null
    log "building $label (test-backend-ops, llama-bench)"
    cmake --build "$build_dir" --target test-backend-ops llama-bench -j "$JOBS" >/dev/null
}

trap 'revert_patch 2>/dev/null || true' EXIT

[[ -f "$DAWN_PREFIX/lib/cmake/Dawn/DawnConfig.cmake" ]] \
    || die "Dawn install not found at $DAWN_PREFIX (set DAWN_PREFIX=...)."
log "using Dawn install at $DAWN_PREFIX"

# ---------------------------------------------------------------------------
# 2. Build both variants. (OFF first; no patch needed.)
# ---------------------------------------------------------------------------
log "=== variant 1/2: bounds-OFF (default native behaviour) ==="
build_variant "$BUILD_OFF" "bounds-OFF"

log "=== variant 2/2: bounds-ON (Dawn robustness enabled) ==="
patch_bounds_on
build_variant "$BUILD_ON" "bounds-ON"
revert_patch

# ---------------------------------------------------------------------------
# 3. Download missing GGUFs.
# ---------------------------------------------------------------------------
have_hf_cli() { command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; }
hf_dl() {
    local repo="$1" file="$2" outdir="$3"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" "$file" --local-dir "$outdir" >/dev/null
    else
        huggingface-cli download "$repo" "$file" --local-dir "$outdir" >/dev/null
    fi
}
mkdir -p "$MODELS_ROOT"

# First pass: classify each pair as present or missing, no downloads yet.
declare -a PRESENT=() MISSING=()
for entry in "${PAIRS[@]}"; do
    IFS='|' read -r model_dir file _repo _hf <<< "$entry"
    if [[ -f "$MODELS_ROOT/$model_dir/$file" ]]; then
        PRESENT+=("$model_dir/$file")
    else
        MISSING+=("$entry")
    fi
done

log "model inventory: ${#PRESENT[@]}/${#PAIRS[@]} present, ${#MISSING[@]} missing"
for p in "${PRESENT[@]}"; do log "  [present] $p"; done
for entry in "${MISSING[@]}"; do
    IFS='|' read -r model_dir file _repo _hf <<< "$entry"
    log "  [missing] $model_dir/$file"
done

# Second pass: download only the missing ones.
if [[ ${#MISSING[@]} -gt 0 ]]; then
    have_hf_cli || die "neither 'hf' nor 'huggingface-cli' on PATH; install huggingface_hub"
    i=0
    for entry in "${MISSING[@]}"; do
        i=$((i+1))
        IFS='|' read -r model_dir file repo hf_file <<< "$entry"
        log "downloading [$i/${#MISSING[@]}] $model_dir/$file from $repo"
        mkdir -p "$MODELS_ROOT/$model_dir"
        hf_dl "$repo" "$hf_file" "$MODELS_ROOT/$model_dir"
        [[ -f "$MODELS_ROOT/$model_dir/$file" ]] \
            || die "download finished but $MODELS_ROOT/$model_dir/$file is missing"
    done
fi

if [[ $DO_BENCH -eq 0 ]]; then
    log "--no-bench: skipping llama-bench sweep and table render"
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. llama-bench sweep on both builds.
# ---------------------------------------------------------------------------
: > "$STDERR_LOG"

run_sweep() {
    local build_dir="$1" out_json="$2" label="$3"
    local bin="$build_dir/bin/llama-bench"
    [[ -x "$bin" ]] || die "$bin not found"
    : > "$out_json"
    log "$label: sweep over ${#PAIRS[@]} GGUFs"
    local i=0
    for entry in "${PAIRS[@]}"; do
        IFS='|' read -r model_dir file _repo _hf <<< "$entry"
        i=$((i+1))
        local gguf="$MODELS_ROOT/$model_dir/$file"
        log "  [$label][$i/${#PAIRS[@]}] $model_dir / $file"
        if ! "$bin" -m "$gguf" -ngl 99 -p 512 -n 128 -r 3 -o json 2>>"$STDERR_LOG" \
                | python3 -c "
import sys, json, os
data = json.load(sys.stdin)
for r in data:
    # Strip absolute path so the saved JSON does not leak the runner's
    # filesystem layout (we add structured fields for the table render).
    if 'model_filename' in r:
        r['model_filename'] = os.path.basename(r['model_filename'])
    r['__model_dir'] = '$model_dir'
    r['__file']      = '$file'
    print(json.dumps(r))
" \
                >> "$out_json"; then
            log "  [$label] FAILED on $model_dir / $file (see $STDERR_LOG)"
        fi
    done
    log "$label: $(wc -l <"$out_json") rows -> $out_json"
}

run_sweep "$BUILD_OFF" "$RESULT_OFF" "bounds-OFF"
run_sweep "$BUILD_ON"  "$RESULT_ON"  "bounds-ON"

if [[ $DO_RENDER -eq 0 ]]; then
    log "--no-render: skipping table update"
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Render boundscheck-experiment.txt (regenerate from scratch, fill Arc
#    B580 column from this run, leave other GPU columns as placeholders).
# ---------------------------------------------------------------------------
log "rendering $TABLE"
python3 - "$RESULT_OFF" "$RESULT_ON" "$TABLE" <<'PY'
import json, sys, pathlib

off_path, on_path, out_path = sys.argv[1:4]

def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            key = (r['__model_dir'], r['__file'],
                   int(r.get('n_prompt', 0)), int(r.get('n_gen', 0)))
            rows[key] = float(r['avg_ts'])
    return rows

off = load(off_path); on = load(on_path)

# Table contents.
pairs = [
    ("Llama-3.2-1B-Instruct",        "F16",    "Llama-3.2-1B-Instruct-F16.gguf"),
    ("Llama-3.2-1B-Instruct",        "Q2_K",   "Llama-3.2-1B-Instruct-Q2_K.gguf"),
    ("Llama-3.2-1B-Instruct",        "Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("Llama-3.2-1B-Instruct",        "Q8_0",   "Llama-3.2-1B-Instruct-Q8_0.gguf"),
    ("gemma-3-270m-it",              "Q4_K_M", "gemma-3-270m-it-Q4_K_M.gguf"),
    ("Qwen3-0.6B",                   "Q4_K_M", "Qwen3-0.6B-Q4_K_M.gguf"),
    ("LFM2.5-350M",                  "Q4_K_M", "LFM2.5-350M-Q4_K_M.gguf"),
    ("SmolLM3-3B",                   "Q4_K_M", "SmolLM3-3B-Q4_K_M.gguf"),
    ("Ministral-3-3B-Instruct-2512", "Q4_K_M", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    ("Qwen3.5-2B",                   "Q4_K_M", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("gemma-4-E2B-it",               "Q4_K_M", "gemma-4-E2B-it-Q4_K_M.gguf"),
    ("granite-4.0-h-1b",             "Q4_K_M", "granite-4.0-h-1b-Q4_K_M.gguf"),
    ("Bonsai-1.7B",                  "Q1_0",   "Bonsai-1.7B-Q1_0.gguf"),
]
gpus = ["NVIDIA RTX 5080", "AMD RX 9700 XT", "Intel Arc B580", "Apple M2"]

W_MODEL, W_QUANT, W_CELL = 30, 8, 12
COLS = ['pp512 OFF', 'pp512 ON', 'pp delta', 'tg128 OFF', 'tg128 ON', 'tg delta']
def hline():
    parts = ['-'*(W_MODEL+2), '-'*(W_QUANT+2)] + ['-'*(W_CELL+2)]*6
    return '+' + '+'.join(parts) + '+'
def dline():
    parts = ['='*(W_MODEL+2), '='*(W_QUANT+2)] + ['='*(W_CELL+2)]*6
    return '+' + '+'.join(parts) + '+'
def row_cells(label_left, label_right, cells):
    parts = [' ' + label_left.ljust(W_MODEL) + ' ',
             ' ' + label_right.ljust(W_QUANT) + ' ']
    for v in cells: parts.append(' ' + v.rjust(W_CELL) + ' ')
    return '|' + '|'.join(parts) + '|'

def fmt_n(v): return '-' if v is None else f'{v:.2f}'
def fmt_d(a, b):
    if a and b: return f'{(b-a)/a*100:+.1f}%'
    return '-'

def cells_for(gpu, model_dir, quant, file):
    if gpu != 'Intel Arc B580':
        return ['-']*6
    pp_off = off.get((model_dir, file, 512, 0))
    pp_on  = on .get((model_dir, file, 512, 0))
    tg_off = off.get((model_dir, file, 0, 128))
    tg_on  = on .get((model_dir, file, 0, 128))
    return [fmt_n(pp_off), fmt_n(pp_on), fmt_d(pp_off, pp_on),
            fmt_n(tg_off), fmt_n(tg_on), fmt_d(tg_off, tg_on)]

def avg_cells(rows_cells):
    cols = list(zip(*rows_cells))  # transpose
    out = []
    for ci, col in enumerate(cols):
        vals = []
        for v in col:
            v = v.strip()
            if v in ('-', ''): continue
            if v.endswith('%'): v = v[:-1]
            try: vals.append(float(v))
            except ValueError: pass
        if not vals:
            out.append('-')
        elif ci in (2, 5):
            out.append(f'{sum(vals)/len(vals):+.1f}%')
        else:
            out.append(f'{sum(vals)/len(vals):.2f}')
    return out

def render_gpu(gpu):
    out = [f'== {gpu} ==', dline(),
           row_cells('model', 'quant', COLS), dline()]
    rows_cells = []
    for model_dir, quant, file in pairs:
        cells = cells_for(gpu, model_dir, quant, file)
        rows_cells.append(cells)
        out.append(row_cells(model_dir, quant, cells))
        out.append(hline())
    # Replace the trailing hline with: hline, AVG row, dline.
    out.pop()
    out.append(hline())
    out.append(row_cells('AVG', '', avg_cells(rows_cells)))
    out.append(dline())
    return '\n'.join(out)

lines = [
    'WebGPU bounds-check perf comparison across GPUs',
    'Per cell: llama-bench throughput in tokens/sec (-p 512 -n 128 -r 3, -ngl 99)',
    'OFF = bounds checks disabled (default native build, "disable_robustness" Dawn toggle present)',
    'ON  = bounds checks enabled  (patched build, "disable_robustness" toggle removed)',
    'pp512 = prompt-processing throughput (compute-bound)',
    'tg128 = token-generation throughput (memory-bound)',
    'delta = (ON - OFF) / OFF; negative = throughput drops when bounds checks are enabled',
    '',
]
for gpu in gpus:
    lines.append(render_gpu(gpu))
    lines.append('')
pathlib.Path(out_path).write_text('\n'.join(lines))
print(f'wrote {out_path}')
PY

log "done."
log "  table:    $TABLE"
log "  raw json: $RESULT_OFF , $RESULT_ON"
log "  builds:   $BUILD_OFF , $BUILD_ON"
