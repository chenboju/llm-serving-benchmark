"""RTX 4090 vs DGX Spark 跨平台對比圖。

輸入 cross-platform-comparison.csv（由 build_cross_platform.py 產生），
輸出三面板圖：closed-loop 吞吐、open-loop 飽和點、各 suite 吞吐比。

樣式沿用 ../../analysis/plot_dataset_figures.py 的調色盤與 rcParams。該處的
#2a78d6 / #eb6834 是驗證過的相鄰 categorical 配色（CVD dE 24.7、對比 >= 3:1）；
本圖兩邊都是 vLLM，不與 backend 配色同時出現，因此改配給 platform 不會混淆。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "cross-platform-comparison.csv"
OUT = HERE / "figures"

C = {
    "rtx4090": "#2a78d6",
    "dgx": "#eb6834",
    "pass": "#0ca30c",
    "warn": "#fab219",
    "invalid": "#d03b3b",
    "surface": "#ffffff",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

FOOTER = (
    "Formal runs only · GPT-OSS-20B · same vLLM image digest on both platforms · context limit 65,536 tokens · "
    "the same 37 cells on both, all valid (>= 3 repeats, request errors <= 1%, no unexpected GPU OOM) · "
    "co-located benchmark; not representative of networked service latency"
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
    ax.text(
        0.0, 1.02, text, transform=ax.transAxes,
        fontsize=9.5, color=C["muted"], ha="left", va="bottom",
    )


def panel_closed_loop(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df[df.suite == "sharegpt_closed"].sort_values("concurrency")
    x = d.concurrency
    ax.plot(x, d.output_token_throughput_median_rtx4090, marker="s",
            color=C["rtx4090"], label="RTX 4090 24 GB", linewidth=2)
    ax.plot(x, d.output_token_throughput_median_dgx, marker="o",
            color=C["dgx"], label="DGX Spark (GB10)", linewidth=2)
    for _, r in d.iterrows():
        ax.annotate(f"{r.output_token_throughput_ratio_dgx_over_4090:.2f}×",
                    (r.concurrency, r.output_token_throughput_median_dgx),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=C["dgx"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output tokens/s (median)")
    ax.set_title("ShareGPT closed-loop throughput")
    # 標註掛在 DGX 點下方，需留白否則會被 x 軸切掉
    ax.set_ylim(bottom=-0.08 * d.output_token_throughput_median_rtx4090.max())
    hint(ax, "higher is better · labels show DGX / 4090 ratio")
    ax.legend(loc="upper left")


def panel_open_loop(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df[df.suite == "sharegpt_open"].sort_values("request_rate")
    x = d.request_rate
    ax.plot(x, d.output_token_throughput_median_rtx4090, marker="s",
            color=C["rtx4090"], label="RTX 4090 24 GB", linewidth=2)
    ax.plot(x, d.output_token_throughput_median_dgx, marker="o",
            color=C["dgx"], label="DGX Spark (GB10)", linewidth=2)
    for _, r in d.iterrows():
        slo = r.slo_attainment_pct_median_dgx
        colour = C["pass"] if slo >= 95 else (C["warn"] if slo >= 50 else C["invalid"])
        ax.annotate(f"SLO {slo:.0f}%",
                    (r.request_rate, r.output_token_throughput_median_dgx),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=9, color=colour, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlabel("Target arrival rate (requests/s)")
    ax.set_ylabel("Output tokens/s (median)")
    ax.set_title("ShareGPT open-loop: parity is an artefact")
    ax.set_ylim(bottom=-0.16 * d.output_token_throughput_median_rtx4090.max())
    hint(ax, "close at low rates only because neither is saturated · labels show DGX SLO attainment")
    ax.legend(loc="upper left")


def panel_ratio_by_suite(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = ["sharegpt_closed", "sharegpt_natural", "sharegpt_open", "rag", "ruler_performance"]
    labels, data = [], []
    for s in order:
        v = df[df.suite == s]["output_token_throughput_ratio_dgx_over_4090"].dropna()
        if len(v):
            pretty = (s.replace("sharegpt_", "ShareGPT ")
                       .replace("ruler_performance", "RULER perf")
                       .replace("rag", "RAG"))
            labels.append(f"{pretty} (n={len(v)})")
            data.append(v.values)
    # tick label 用 set_xticklabels 設定：boxplot 的 labels= 在 matplotlib 3.9
    # 改名為 tick_labels，並於 3.11 移除；requirements 允許 >= 3.8，兩邊都要能跑。
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops={"color": C["ink"], "linewidth": 1.8})
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for box in bp["boxes"]:
        box.set(facecolor=C["dgx"], alpha=0.35, edgecolor=C["dgx"])
    ax.axhline(1.0, color=C["muted"], linestyle="--", linewidth=1)
    ax.text(0.995, 1.0, " parity", transform=ax.get_yaxis_transform(),
            va="center", fontsize=9, color=C["muted"])
    ax.set_ylabel("Throughput ratio DGX / 4090")
    ax.set_title("Throughput ratio by suite")
    hint(ax, "1.0 = parity · high open-loop values = unsaturated load, not an advantage")
    ax.tick_params(axis="x", labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(28)
        lbl.set_ha("right")


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"找不到 {CSV}；請先執行 build_cross_platform.py")
    df = pd.read_csv(CSV)
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    panel_closed_loop(axes[0], df)
    panel_open_loop(axes[1], df)
    panel_ratio_by_suite(axes[2], df)
    fig.suptitle("RTX 4090 vs DGX Spark — GPT-OSS-20B on vLLM", y=0.99)
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom",
             fontsize=8.5, color=C["muted"], wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))

    for ext in ("png", "svg"):
        path = OUT / f"01-cross-platform-throughput.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None, facecolor=C["surface"])
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
