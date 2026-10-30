"""Draw the README figures from the result values.

The values are copied from screen.py and measure_latency.py (batch=100),
because data/ is not in the repository. Edit them here if a value changes.

    pip install -r requirements.txt
    python scripts/plot_results.py        # writes docs/figures/*.png

Figures:
  gate_mrr.png            screening: the gate as two groups, MRR as bars
  decision_speed.png      the decision: single-query latency and batch throughput side by side

One look across the figures: the chosen model is in color, the others that
passed the gate in grey, and the excluded ones hatched.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

CHOSEN = "embeddinggemma"
CHOSEN_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"

# model, MRR, passed the gate (screen.py, n=185)
MRR = [("embeddinggemma", 0.741, True), ("qwen3-embedding", 0.731, True),
       ("snowflake-arctic-embed2", 0.705, True), ("bge-m3", 0.700, True),
       ("nomic-embed-text（gbrainの既定）", 0.394, False), ("all-minilm", 0.149, False)]
# model, single-query latency in ms, throughput in chunks/s (measure_latency.py 100)
SPEED = [("embeddinggemma", 2104, 1.89), ("snowflake-arctic-embed2", 2199, 0.78),
         ("bge-m3", 2195, 0.78), ("qwen3-embedding", 2076, 0.29)]


def style():
    plt.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "savefig.facecolor": SURFACE,
    })


def title(fig, text, sub):
    """Title and subtitle at fixed distances from the top, whatever the figure height."""
    h = fig.get_figheight()
    fig.suptitle(text, x=0.02, y=1 - 0.12 / h, ha="left", va="top", fontsize=12.5, color=INK,
                 fontweight="bold").set_in_layout(False)
    fig.text(0.02, 1 - 0.5 / h, sub, ha="left", va="top", fontsize=9, color=INK_2)
    return 1 - 0.85 / h     # top of the plotting area


def footer(fig, line):
    fig.text(0.02, 0.01 + 0.03 / fig.get_figheight(), line, fontsize=8, color=INK_2)
    return 0.3 / fig.get_figheight()


def gate_mrr():
    """Gate as two groups, MRR as bars."""
    passed = [r for r in MRR if r[2]]
    excluded = [r for r in MRR if not r[2]]
    rows = len(MRR) + 2                              # two group headers
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.42 * rows))
    y = rows - 1
    ticks, labels = [], []
    for header, group in (("足切りを通った", passed), ("足切りで除外", excluded)):
        ax.text(-0.02, y, f"{header}（{len(group)}モデル）", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
        if group is passed:
            ax.text(0.008, y, "この順位では選んでいない → hit@10のQ検定（p=0.449、差を検出できず）→ スループットで決着",
                    va="center", fontsize=9, color=INK_2, zorder=3,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        y -= 1
        for name, value, ok in group:
            chosen = name == CHOSEN
            if ok:
                ax.barh(y, value, height=0.62, color=CHOSEN_COLOR if chosen else MUTED, zorder=2)
            else:
                ax.barh(y, value, height=0.62, color=SURFACE, edgecolor="#8f8e88", hatch="///", linewidth=1.0, zorder=2)
            ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=10, color=INK if ok else INK_2)
            ticks.append(y)
            labels.append(name)
            y -= 1
        if group is passed:
            ax.axhline(y + 0.5, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(ticks, labels, color=INK)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.50", "0.75"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("MRR（n=185）　右ほど良い", loc="left", fontsize=10.5)
    top = title(fig, "足切り：上位4モデルのMRRは0.70〜0.74に固まり、4位と5位の間は0.306空いている",
                "破線で上下に分けた。線の位置は結果を見て決めた。青＝後の段階で選んだモデル")
    fig.tight_layout(rect=(0.02, 0, 1, top))
    fig.savefig(OUT / "gate_mrr.png", dpi=150)
    plt.close(fig)


def decision_speed():
    """Latency and throughput side by side: only the second tells the models apart."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    y = list(range(len(SPEED)))[::-1]
    for ax, col, name, limit, ticks, fmt, note in (
            (axes[0], 1, "単発のレイテンシ（ms、中央値）　左ほど良い", 2900, [0, 1000, 2000], "{:,.0f}", "4モデルの差は6%以内"),
            (axes[1], 2, "バッチのスループット（chunks/s）　右ほど良い", 2.4, [0, 0.5, 1, 1.5, 2], "{:.2f}", "1位は2位の2.4倍（各1回の測定）")):
        for yi, row in zip(y, SPEED):
            chosen = row[0] == CHOSEN
            ax.barh(yi, row[col], height=0.62, color=CHOSEN_COLOR if chosen else MUTED, zorder=2)
            # Bold only on the metric that decided: qwen3-embedding is the fastest single query.
            ax.text(row[col] + limit * 0.015, yi, fmt.format(row[col]), va="center", fontsize=10, color=INK,
                    fontweight="bold" if chosen and col == 2 else "normal")
        ax.set_xlim(0, limit)
        ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.set_xlabel(note, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r[0] for r in SPEED], color=INK)
    top = title(fig, "決着：単発のレイテンシでは見分けられず、スループットは2位の2.4倍だった",
                "足切りを通った4モデル。200チャンクをbatch=100で投入し、各1回測った。どちらの指標で決めるかは測った後に決めた。青＝選んだモデル")
    bottom = footer(fig, "CPU推論（Ryzen 5 7520U、メモリ16GB、Ollama 0.34.3）")
    fig.tight_layout(rect=(0.02, bottom, 1, top))
    fig.savefig(OUT / "decision_speed.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    gate_mrr()
    decision_speed()


if __name__ == "__main__":
    main()
