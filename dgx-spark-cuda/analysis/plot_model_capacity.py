"""DGX Spark 上 GPT-OSS-20B 與 120B 的容量對比。

輸入 dataset-cell-summary.csv（由 build_cross_platform.py 產生），取兩個
closed-loop C1–C64 campaign：

    formal-20b-s064-20260818-dgx    GPT-OSS-20B
    formal-120b-s064-20260818-dgx   GPT-OSS-120B

輸出三面板：總吞吐、每使用者吞吐、SLO 達成率。三張放在一起才看得出
「總吞吐一直在漲，但服務品質早就崩了」。

樣式與 plot_cross_platform.py 一致。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "dataset-cell-summary.csv"
OUT = HERE / "figures"

CAMPAIGNS = {
    "GPT-OSS-20B": "formal-20b-s064-20260818-dgx",
    "GPT-OSS-120B": "formal-120b-s064-20260818-dgx",
}

# chat SLO profile，取自 scripts/benchmark-config.json
SLO_TTFT_MS = 1000
SLO_ITL_MS = 50
GOOD_FRACTION_PCT = 95.0

C = {
    "20b": "#2a78d6",
    "120b": "#eb6834",
    "pass": "#0ca30c",
    "warn": "#fab219",
    "invalid": "#d03b3b",
    "surface": "#ffffff",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "band": "#f4f4f1",
}
STYLE = {"GPT-OSS-20B": (C["20b"], "s"), "GPT-OSS-120B": (C["120b"], "o")}

FOOTER = (
    "Formal runs only · DGX Spark (GB10) · vLLM, same image digest for both models · "
    f"ShareGPT closed-loop, 128 output tokens · chat SLO: TTFT <= {SLO_TTFT_MS} ms and ITL <= {SLO_ITL_MS} ms · "
    "3 valid repeats per cell · co-located benchmark; not representative of networked service latency"
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": C["surface"],
            "axes.facecolor": C["surface"],
            "savefig.facecolor": C["surface"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 26,
            "axes.labelsize": 11,
            "axes.labelcolor": C["ink2"],
            "axes.edgecolor": C["axis"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": C["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": C["ink2"],
            "ytick.color": C["ink2"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": C["ink"],
            "legend.frameon": False,
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
        }
    )


def hint(ax: plt.Axes, text: str) -> None:
    ax.text(0.0, 1.02, text, transform=ax.transAxes,
            fontsize=9.5, color=C["muted"], ha="left", va="bottom")


def concurrency_axis(ax: plt.Axes, xs) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_xlabel("Concurrency")


def last_passing(d: pd.DataFrame) -> float | None:
    """最後一個仍達 SLO 門檻的併發數。"""
    ok = d[d.slo_attainment_pct_median >= GOOD_FRACTION_PCT]
    return float(ok.concurrency.max()) if len(ok) else None


def panel_throughput(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    for name, d in data.items():
        colour, marker = STYLE[name]
        ax.plot(d.concurrency, d.output_token_throughput_median,
                marker=marker, color=colour, label=name, linewidth=2)
        cut = last_passing(d)
        if cut is not None:
            y = float(d[d.concurrency == cut].output_token_throughput_median.iloc[0])
            ax.plot([cut], [y], marker="X", markersize=13, color=colour,
                    markeredgecolor=C["surface"], markeredgewidth=1.4, zorder=5)
    concurrency_axis(ax, sorted(next(iter(data.values())).concurrency))
    ax.set_ylabel("Output tokens/s (median)")
    ax.set_title("Aggregate throughput keeps rising")
    hint(ax, "X marks the last concurrency still meeting the SLO")
    ax.legend(loc="upper left")


def panel_per_user(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    for name, d in data.items():
        colour, marker = STYLE[name]
        ax.plot(d.concurrency, d.output_token_throughput_per_user_median,
                marker=marker, color=colour, label=name, linewidth=2)
    d120 = data["GPT-OSS-120B"]
    lo, hi = d120.output_token_throughput_per_user_median.iloc[-1], d120.output_token_throughput_per_user_median.iloc[0]
    ax.annotate(f"{hi:.1f} -> {lo:.1f} tok/s\nper user",
                xy=(float(d120.concurrency.iloc[-1]), lo),
                xytext=(-96, 34), textcoords="offset points",
                fontsize=9.5, color=C["120b"], fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": C["120b"], "lw": 1.2})
    concurrency_axis(ax, sorted(d120.concurrency))
    ax.set_ylabel("Output tokens/s per user (median)")
    ax.set_title("What each user actually gets, collapses")
    hint(ax, "the same runs as the left panel, divided by concurrency")
    ax.legend(loc="upper right")


def panel_slo(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    ax.axhspan(GOOD_FRACTION_PCT, 100, color=C["pass"], alpha=0.08, zorder=0)
    ax.axhline(GOOD_FRACTION_PCT, color=C["pass"], linestyle="--", linewidth=1.2)
    ax.text(0.995, GOOD_FRACTION_PCT, f" {GOOD_FRACTION_PCT:.0f}% ",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=9, color=C["pass"], fontweight="bold")
    for name, d in data.items():
        colour, marker = STYLE[name]
        ax.plot(d.concurrency, d.slo_attainment_pct_median,
                marker=marker, color=colour, label=name, linewidth=2)
        cut = last_passing(d)
        if cut is not None:
            ax.axvline(cut, color=colour, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(cut, 52, f" C{int(cut)} ceiling", rotation=90,
                    fontsize=9, color=colour, fontweight="bold", va="bottom")
    concurrency_axis(ax, sorted(next(iter(data.values())).concurrency))
    ax.set_ylim(-4, 108)
    ax.set_ylabel("Good request fraction (%)")
    ax.set_title("Where each model's capacity actually ends")
    hint(ax, "a request is good only if it meets BOTH the TTFT and ITL SLO")
    ax.legend(loc="lower left")


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"找不到 {CSV}；請先執行 build_cross_platform.py")
    df = pd.read_csv(CSV)
    data = {}
    for name, campaign in CAMPAIGNS.items():
        d = df[(df.campaign == campaign) & (df.status == "valid")].sort_values("concurrency")
        if d.empty:
            raise SystemExit(f"{campaign} 沒有 valid cell")
        data[name] = d

    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    panel_throughput(axes[0], data)
    panel_per_user(axes[1], data)
    panel_slo(axes[2], data)
    fig.suptitle("GPT-OSS-20B vs 120B on a single DGX Spark", y=0.99)
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom",
             fontsize=8.5, color=C["muted"], wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))

    for ext in ("png", "svg"):
        path = OUT / f"02-model-capacity.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None, facecolor=C["surface"])
        print(f"wrote {path}")
    plt.close(fig)

    for name, d in data.items():
        cut = last_passing(d)
        print(f"  {name:14s} SLO ceiling: C{int(cut) if cut else '-'}"
              f"  peak aggregate {d.output_token_throughput_median.max():.1f} tok/s")


if __name__ == "__main__":
    main()
