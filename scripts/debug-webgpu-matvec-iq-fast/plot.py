#!/usr/bin/env python3
"""Render comparison charts + summary.md from bench results.

Inputs (in scripts/debug-webgpu-matvec-iq-fast/results/):
    llama-bench-master.csv, llama-bench-branch.csv
    perf-master.txt, perf-branch.txt

Outputs (same dir):
    chart-kernel.png, chart-decode.png, chart-prompt.png
    summary.md
"""

from __future__ import annotations
import csv
import os
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.stderr.write(
        "matplotlib/numpy not available. Activate the plot venv:\n"
        "    python3 -m venv /tmp/plot-venv && /tmp/plot-venv/bin/pip install matplotlib numpy\n"
        "    /tmp/plot-venv/bin/python scripts/debug-webgpu-matvec-iq-fast/plot.py\n"
    )
    sys.exit(1)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

MODEL_LABEL = [
    ("Llama-3.2-1B-Instruct-UD-IQ1_S.gguf",   "IQ1_S"),
    ("Llama-3.2-1B-Instruct-UD-IQ1_M.gguf",   "IQ1_M"),
    ("Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf", "IQ2_XXS"),
    ("Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf", "IQ3_XXS"),
    ("Llama-3.2-1B-Instruct-IQ4_NL.gguf",     "IQ4_NL"),
    ("Llama-3.2-1B-Instruct-IQ4_XS.gguf",     "IQ4_XS"),
]

KERNEL_ORDER = ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
                "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs"]


def load_bench_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("build_commit") or len(r) < 41:
                continue
            rows.append(r)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        mfn = r[5].split("/")[-1]
        try:
            np_ = int(r[33])
            avg = float(r[39])
            sd = float(r[40])
        except (ValueError, IndexError):
            continue
        kind = "pp" if np_ > 0 else "tg"
        out.setdefault(mfn, {})[kind] = (avg, sd)
    return out


def load_perf_txt(path: Path):
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.search(
            r"type_a=(iq\w+).*?([\d.]+) us/run.*?([\d.]+) GFLOPS", line
        )
        if m:
            out[m.group(1)] = {
                "us": float(m.group(2)),
                "gflops": float(m.group(3)),
            }
    return out


def make_grouped_bars(labels, mast, mast_err, br, br_err, *,
                       title, ylabel, outfile, annotate="pct"):
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5.5))
    ax.bar(x - width / 2, mast, width, yerr=mast_err, capsize=3,
           label="master", color="#4c78a8",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, br, width, yerr=br_err, capsize=3,
           label="webgpu-matvec-iq-fast", color="#f58518",
           edgecolor="black", linewidth=0.5)

    for i, (mv, bv) in enumerate(zip(mast, br)):
        if annotate == "pct":
            val = (bv - mv) / mv * 100 if mv else 0
            s = f"{val:+.1f}%"
            color = "black" if abs(val) < 3 else (
                "#1a7a1a" if val > 0 else "#b01010")
        else:  # speedup
            val = bv / mv if mv else float("inf")
            s = f"{val:.2f}×"
            color = "#1a7a1a"
        ax.text(x[i], max(mv, bv) * 1.03, s,
                ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right" if annotate == "speedup" else "upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(mast), max(br)) * 1.18)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outfile}")


def main():
    bench_m = load_bench_csv(RESULTS / "llama-bench-master.csv")
    bench_b = load_bench_csv(RESULTS / "llama-bench-branch.csv")
    perf_m = load_perf_txt(RESULTS / "perf-master.txt")
    perf_b = load_perf_txt(RESULTS / "perf-branch.txt")

    missing = []
    for f, _ in MODEL_LABEL:
        if f not in bench_m or f not in bench_b:
            missing.append(f)
    if missing:
        print("warning: missing bench data for:", *missing, sep="\n  ")

    # Kernel chart (GFLOPS, speedup labels)
    if perf_m and perf_b:
        mgf = [perf_m[t]["gflops"] for t in KERNEL_ORDER if t in perf_m and t in perf_b]
        bgf = [perf_b[t]["gflops"] for t in KERNEL_ORDER if t in perf_m and t in perf_b]
        lbls = [t.upper() for t in KERNEL_ORDER if t in perf_m and t in perf_b]
        make_grouped_bars(
            lbls, mgf, None, bgf, None,
            title="MUL_MAT(m=4096, n=1, k=14336) — test-backend-ops perf",
            ylabel="GFLOPS (higher is better)",
            outfile=RESULTS / "chart-kernel.png",
            annotate="speedup",
        )
    else:
        print("skip  chart-kernel.png (perf-{master,branch}.txt missing)")

    # Decode + prompt charts
    if bench_m and bench_b:
        lbls = [lab for _, lab in MODEL_LABEL
                if MODEL_LABEL[0][0] in bench_m]  # just a truthy guard
        lbls = [lab for f, lab in MODEL_LABEL if f in bench_m and f in bench_b]
        files = [f for f, _ in MODEL_LABEL if f in bench_m and f in bench_b]

        for kind, title_suffix, outfile in [
            ("tg", "decode throughput (tg128, -r 3)", "chart-decode.png"),
            ("pp", "prompt processing (pp64, -r 3)",  "chart-prompt.png"),
        ]:
            mast = [bench_m[f][kind][0] for f in files]
            mast_err = [bench_m[f][kind][1] for f in files]
            br = [bench_b[f][kind][0] for f in files]
            br_err = [bench_b[f][kind][1] for f in files]
            make_grouped_bars(
                lbls, mast, mast_err, br, br_err,
                title=f"Llama-3.2-1B i-quant {title_suffix}",
                ylabel="tokens / second",
                outfile=RESULTS / outfile,
                annotate="pct",
            )

    # Summary markdown
    write_summary(bench_m, bench_b, perf_m, perf_b)


def write_summary(bench_m, bench_b, perf_m, perf_b):
    md = ["# Results", ""]
    md.append("Generated by `scripts/debug-webgpu-matvec-iq-fast/plot.py`.")
    md.append("")

    if perf_m and perf_b:
        md.append("## Isolated kernel: MUL_MAT(m=4096, n=1, k=14336)")
        md.append("")
        md.append("![](chart-kernel.png)")
        md.append("")
        md.append("| type | master (µs) | branch (µs) | speedup | master (GFLOPS) | branch (GFLOPS) |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for t in KERNEL_ORDER:
            if t not in perf_m or t not in perf_b:
                continue
            m = perf_m[t]; b = perf_b[t]
            spd = m["us"] / b["us"] if b["us"] else float("inf")
            md.append(f"| {t.upper()} | {m['us']:.1f} | {b['us']:.1f} | **{spd:.2f}×** | {m['gflops']:.1f} | {b['gflops']:.1f} |")
        md.append("")

    if bench_m and bench_b:
        md.append("## End-to-end model: Llama-3.2-1B i-quants")
        md.append("")
        md.append("**Decode (tg128):**")
        md.append("")
        md.append("![](chart-decode.png)")
        md.append("")
        md.append("| model | master tok/s | branch tok/s | Δ |")
        md.append("|---|---:|---:|---:|")
        for f, lab in MODEL_LABEL:
            if f not in bench_m or f not in bench_b:
                continue
            m = bench_m[f]["tg"]; b = bench_b[f]["tg"]
            d = (b[0] - m[0]) / m[0] * 100
            md.append(f"| {lab} | {m[0]:.2f} ± {m[1]:.2f} | {b[0]:.2f} ± {b[1]:.2f} | **{d:+.1f}%** |")
        md.append("")
        md.append("**Prompt processing (pp64):**")
        md.append("")
        md.append("![](chart-prompt.png)")
        md.append("")
        md.append("| model | master tok/s | branch tok/s | Δ |")
        md.append("|---|---:|---:|---:|")
        for f, lab in MODEL_LABEL:
            if f not in bench_m or f not in bench_b:
                continue
            m = bench_m[f]["pp"]; b = bench_b[f]["pp"]
            d = (b[0] - m[0]) / m[0] * 100
            md.append(f"| {lab} | {m[0]:.2f} ± {m[1]:.2f} | {b[0]:.2f} ± {b[1]:.2f} | **{d:+.1f}%** |")
        md.append("")

    md.append("## Models used")
    md.append("")
    md.append("All GGUFs downloaded by `fetch-models.sh` from")
    md.append("`huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF`:")
    md.append("")
    for f, lab in MODEL_LABEL:
        md.append(f"- `{f}` → exercises **{lab}**")
    md.append("")

    (RESULTS / "summary.md").write_text("\n".join(md))
    print(f"saved {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
