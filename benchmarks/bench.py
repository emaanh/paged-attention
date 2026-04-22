#!/usr/bin/env python3
"""
Benchmark naive attention across a sweep of sequence lengths.

Usage (from repo root, after building):
    python benchmarks/bench.py [--binary build/bench] [--heads 8] [--head-dim 64]
                               [--warmup 5] [--runs 20] [--output bench_results.csv]

Produces:
  - CSV file with (seq_len, naive_ms) columns
  - PNG plot of latency vs sequence length (if matplotlib is available)
"""

import argparse
import csv
import os
import subprocess
import sys

import matplotlib.pyplot as plt

SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]


def run_bench(binary, num_heads, head_dim, warmup, runs, seq_lens):
    """Call the C++ bench binary and parse its CSV output."""
    cmd = [
        binary,
        str(num_heads),
        str(head_dim),
        str(warmup),
        str(runs),
    ] + [str(s) for s in seq_lens]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: bench binary failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    rows = []
    lines = result.stdout.strip().splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append({
            "seq_len":   int(row["seq_len"]),
            "naive_ms":  float(row["naive_ms"]),
        })
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "naive_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(rows, path):
    seq_lens  = [r["seq_len"]  for r in rows]
    naive_ms  = [r["naive_ms"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, naive_ms, marker="o", label="Naive attention")
    plt.xlabel("Sequence length")
    plt.ylabel("Latency (ms)")
    plt.title("Attention kernel latency vs sequence length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention kernels.")
    parser.add_argument("--binary",   default="build/bench",
                        help="Path to bench binary (default: build/bench)")
    parser.add_argument("--heads",    type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--head-dim", type=int, default=64,
                        help="Head dimension (default: 64)")
    parser.add_argument("--warmup",   type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--runs",     type=int, default=20,
                        help="Timed iterations per config (default: 20)")
    parser.add_argument("--output",   default="bench_results.csv",
                        help="Output CSV path (default: bench_results.csv)")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS,
                        help=f"Sequence lengths to sweep (default: {SEQ_LENS})")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary '{args.binary}' not found. Build the project first.",
              file=sys.stderr)
        sys.exit(1)

    rows = run_bench(
        binary=args.binary,
        num_heads=args.heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        runs=args.runs,
        seq_lens=args.seq_lens,
    )

    print("\nResults:")
    print(f"  {'seq_len':>8}  {'naive_ms':>10}")
    print(f"  {'-'*8}  {'-'*10}")
    for r in rows:
        print(f"  {r['seq_len']:>8}  {r['naive_ms']:>10.4f}")

    save_csv(rows, args.output)
    plot(rows, args.output.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
