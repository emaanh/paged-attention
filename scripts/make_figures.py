"""Regenerate the README figures from results/*.csv.

    uv run --with matplotlib --with pandas scripts/make_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE, GREEN = "#2A78D6", "#EB6834", "#1BAF7A"
INK, INK_SOFT, GRID = "#0B0B0B", "#52514E", "#D8D7D2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})


def shape_label(r):
    b, t = int(r.batch), int(r.seq_len)
    return f"{b}×{t // 1024}K" if t >= 1024 else f"{b}×{t}"


def bandwidth():
    df = pd.read_csv(RES / "h100_sweep.csv")
    copy = pd.read_csv(RES / "h100_copy.csv").pct_peak.max()
    fp16 = df[df.dtype == "fp16"].reset_index(drop=True)
    fp8 = df[df.dtype == "fp8_e4m3"].reset_index(drop=True)
    labels = [shape_label(r) for r in fp16.itertuples()]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], fp16.pct_peak, w, color=BLUE, label="FP16 KV")
    ax.bar([i + w / 2 + 0.01 for i in x], fp8.pct_peak, w, color=ORANGE, label="FP8 KV")
    ax.axhline(copy, color=INK_SOFT, lw=1, ls="--")
    ax.text(len(labels) - 0.5, copy + 1, f"plain streaming read, {copy:.0f}%", ha="right",
            va="bottom", fontsize=8, color=INK_SOFT)
    ax.set_xticks(list(x), labels, rotation=40, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of peak DRAM bandwidth")
    ax.set_xlabel("batch × context length (Hq=32, Hkv=8, d=128, 16-token pages)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "h100_bandwidth.png")


def flashinfer():
    df = pd.read_csv(RES / "h100_flashinfer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, dt, title in [(axes[0], "fp16", "FP16 KV cache"), (axes[1], "fp8_e4m3", "FP8 KV cache")]:
        d = df[df.dtype == dt].reset_index(drop=True)
        labels = [shape_label(r) for r in d.itertuples()]
        rel = d.ours_us / d.flashinfer_us
        ax.bar(range(len(labels)), rel, 0.7, color=BLUE)
        ax.axhline(1.0, color=ORANGE, lw=2)
        ax.text(len(labels) - 0.5, 1.004, "FlashInfer", ha="right", va="bottom", fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.8, 1.15)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("our latency / FlashInfer latency\n(below 1 = we are faster)")
    fig.savefig(OUT / "h100_flashinfer.png")


def pagesize():
    df = pd.read_csv(RES / "h100_pagesize.csv")
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for dt, color, name in [("fp16", BLUE, "FP16"), ("fp8_e4m3", ORANGE, "FP8")]:
        d = df[df.dtype == dt]
        ax.plot(d.page, d.pct_peak, color=color, lw=2, marker="o", ms=4, label=name)
        last = d.iloc[-1]
        ax.annotate(name, (last.page, last.pct_peak), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ticks = sorted(df.page.unique())
    ax.set_xticks(ticks, [str(t) if t < 8192 else "8192\n(no paging)" for t in ticks], fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("tokens per page (batch 32 × 8K context)")
    ax.set_ylabel("% of peak DRAM bandwidth")
    ax.legend(loc="lower right")
    fig.savefig(OUT / "h100_pagesize.png")


def capacity():
    df = pd.read_csv(RES / "h100_capacity.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    series = [(("fp16", "contiguous"), INK_SOFT, "contiguous FP16 (reserves 8K)"),
              (("fp16", "paged"), BLUE, "paged FP16"),
              (("fp8_e4m3", "paged"), ORANGE, "paged FP8")]
    lens = sorted(df.actual_len.unique())
    w = 0.26
    for k, ((dt, layout), color, name) in enumerate(series):
        d = df[(df.dtype == dt) & (df.layout == layout)].set_index("actual_len").loc[lens]
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(lens))]
        ax.bar(xs, d.sequences, w, color=color, label=name)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(range(len(lens)), [str(l) for l in lens])
    ax.set_xlabel("actual tokens per sequence")
    ax.set_ylabel("sequences in 1 GiB of KV")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "h100_capacity.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bandwidth()
    flashinfer()
    pagesize()
    capacity()
    print("wrote", *sorted(p.name for p in OUT.glob("h100_*.png")))
