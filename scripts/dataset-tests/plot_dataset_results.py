#!/usr/bin/env python3
"""Create llama.cpp and vLLM dataset benchmark comparison charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis" / "dataset-output"
DEFAULT_OUTPUT = ROOT / "analysis" / "dataset-charts"
COLORS = {"llama.cpp": "#D97706", "vLLM": "#2563EB"}
MARKERS = {"llama.cpp": "o", "vLLM": "s"}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft JhengHei",
                "Noto Sans CJK TC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#E2E8F0",
        }
    )


def finish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.7)
    ax.grid(False, axis="x")


def backend_label(value: str) -> str:
    return "vLLM" if value == "vllm" else "llama.cpp"


def add_engine_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    ylabel: str,
    title: str,
    ci_metric: str | None = None,
) -> None:
    for backend in ("llama.cpp", "vLLM"):
        subset = frame[frame["engine"] == backend].sort_values(x)
        if subset.empty:
            continue
        ax.plot(
            subset[x],
            subset[y],
            color=COLORS[backend],
            marker=MARKERS[backend],
            linewidth=2.3,
            markersize=6,
            label=backend,
        )
        if ci_metric:
            low = subset[f"{ci_metric}_ci95_low"]
            high = subset[f"{ci_metric}_ci95_high"]
            if low.notna().any() and high.notna().any():
                ax.fill_between(subset[x], low, high, color=COLORS[backend], alpha=0.15)
    ax.set(title=title, xlabel=x.replace("_", " "), ylabel=ylabel)
    finish_axis(ax)
    ax.legend(frameon=False)


def add_run_metadata(fig: plt.Figure, frame: pd.DataFrame) -> None:
    """Add compact actual-sample metadata without crowding every data point."""
    summaries = []
    for engine in ("llama.cpp", "vLLM"):
        subset = frame[frame["engine"] == engine]
        if subset.empty:
            continue

        def numeric(column: str) -> pd.Series:
            if column not in subset:
                return pd.Series(dtype=float)
            return pd.to_numeric(subset[column], errors="coerce").dropna()

        request_count = int(numeric("request_count").sum())
        input_tokens = numeric("input_tokens_p50")
        output_tokens = numeric("output_tokens_p50")
        errors = numeric("request_error_rate_pct_median")

        def span(values: pd.Series) -> str:
            if values.empty:
                return "N/A"
            low, high = values.min(), values.max()
            return f"{low:.0f}" if low == high else f"{low:.0f}–{high:.0f}"

        error = errors.max() if not errors.empty else np.nan
        summaries.append(
            f"{engine}: n={request_count:,}, input={span(input_tokens)} tok/req, "
            f"output={span(output_tokens)} tok/req, max error={error:.2f}%"
            if np.isfinite(error)
            else f"{engine}: n={request_count:,}, input={span(input_tokens)} tok/req, "
            f"output={span(output_tokens)} tok/req, max error=N/A"
        )
    if summaries:
        fig.text(0.5, 0.012, "   |   ".join(summaries), ha="center", va="bottom", fontsize=8.5, color="#475569")


def fill_repeat_ci(
    ax: plt.Axes, x: pd.Series, subset: pd.DataFrame, metric: str, color: str
) -> None:
    low_name = f"{metric}_ci95_low"
    high_name = f"{metric}_ci95_high"
    if low_name not in subset or high_name not in subset:
        return
    low = pd.to_numeric(subset[low_name], errors="coerce")
    high = pd.to_numeric(subset[high_name], errors="coerce")
    if low.notna().any() and high.notna().any():
        ax.fill_between(x, low, high, color=color, alpha=0.13)


def mark_vllm_only_region(ax: plt.Axes, boundary: float = 16) -> None:
    """Show where the fair llama.cpp/vLLM range ends."""
    _, xmax = ax.get_xlim()
    if xmax <= boundary:
        return
    ax.axvline(boundary, color="#64748B", linewidth=1.1, linestyle=":")
    ax.axvspan(boundary, xmax, color="#DBEAFE", alpha=0.22, zorder=0)
    ax.text(
        boundary,
        0.98,
        "  vLLM-only stress",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1D4ED8",
    )


def plot_sharegpt_closed(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "sharegpt_closed"].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    add_engine_lines(
        axes[0], frame, "concurrency", "output_token_throughput_median", "token/s",
        "輸出吞吐量", "output_token_throughput"
    )
    add_engine_lines(
        axes[1], frame, "concurrency", "ttft_p95_median", "ms/request", "P95 TTFT", "ttft_p95"
    )
    add_engine_lines(
        axes[2], frame, "concurrency", "goodput_req_s_median", "req/s", "SLO goodput", "goodput_req_s"
    )
    for ax in axes:
        mark_vllm_only_region(ax, 16)
    fig.suptitle("ShareGPT closed-loop：並發擴展", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "01_sharegpt_closed_loop.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sharegpt_open(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "sharegpt_open"].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    add_engine_lines(
        axes[0], frame, "request_rate", "request_throughput_median", "req/s", "完成吞吐量", "request_throughput"
    )
    add_engine_lines(
        axes[1], frame, "request_rate", "ttft_p95_median", "ms/request", "P95 TTFT", "ttft_p95"
    )
    add_engine_lines(
        axes[2], frame, "request_rate", "slo_attainment_pct_median", "%", "SLO 達成率", "slo_attainment_pct"
    )
    for ax in axes:
        ax.set_xlabel("目標到達率 (req/s)")
        mark_vllm_only_region(ax, 8)
    fig.suptitle("ShareGPT open-loop：Poisson 到達率", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "02_sharegpt_open_loop.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sharegpt_natural(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "sharegpt_natural"].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    add_engine_lines(
        axes[0], frame, "request_rate", "request_throughput_median", "req/s",
        "完成吞吐量", "request_throughput"
    )
    add_engine_lines(
        axes[1], frame, "request_rate", "ttft_p95_median", "ms/request",
        "P95 TTFT", "ttft_p95"
    )
    add_engine_lines(
        axes[2], frame, "request_rate", "slo_attainment_pct_median", "%",
        "SLO 達成率", "slo_attainment_pct"
    )
    for ax in axes:
        ax.set_xlabel("目標到達率 (req/s)")
        mark_vllm_only_region(ax, 4)
    fig.suptitle("ShareGPT 真實輸出長度：vLLM 高壓區", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "02b_sharegpt_natural_output.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rag(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "rag"].copy()
    if frame.empty:
        return
    datasets = [("squad", "SQuAD"), ("hotpotqa", "HotpotQA")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    styles = {1: "-", 8: "--", 16: ":", 32: "-."}
    for column, (dataset, title) in enumerate(datasets):
        subset = frame[frame["cell_id"].str.contains(f"rag-{dataset}-")]
        for backend in ("llama.cpp", "vLLM"):
            for concurrency in sorted(int(value) for value in subset["concurrency"].dropna().unique()):
                part = subset[(subset["engine"] == backend) & (subset["concurrency"] == concurrency)].sort_values(
                    "bucket_tokens"
                )
                if part.empty:
                    continue
                label = f"{backend}, C={concurrency}"
                axes[0, column].plot(
                    part["bucket_tokens"] / 1024,
                    part["output_token_throughput_median"],
                    color=COLORS[backend],
                    marker=MARKERS[backend],
                    linestyle=styles[concurrency],
                    linewidth=2,
                    label=label,
                )
                fill_repeat_ci(
                    axes[0, column], part["bucket_tokens"] / 1024, part,
                    "output_token_throughput", COLORS[backend]
                )
                axes[1, column].plot(
                    part["bucket_tokens"] / 1024,
                    part["ttft_p95_median"],
                    color=COLORS[backend],
                    marker=MARKERS[backend],
                    linestyle=styles[concurrency],
                    linewidth=2,
                    label=label,
                )
                fill_repeat_ci(
                    axes[1, column], part["bucket_tokens"] / 1024, part,
                    "ttft_p95", COLORS[backend]
                )
        axes[0, column].set_title(f"{title}：輸出吞吐量")
        axes[1, column].set_title(f"{title}：P95 TTFT")
        axes[0, column].set_ylabel("token/s")
        axes[1, column].set_ylabel("ms/request")
        axes[1, column].set_xlabel("輸入長度 (K tokens)")
        for row in range(2):
            finish_axis(axes[row, column])
            axes[row, column].legend(frameon=False, fontsize=9)
    fig.suptitle("真實 RAG context 長度曲線", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(output / "03_rag_context_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prefix(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "prefix"].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, backend in zip(axes, ("llama.cpp", "vLLM")):
        subset = frame[frame["engine"] == backend]
        for cache_mode, linestyle in (("off", "--"), ("on", "-")):
            concurrency_markers = {1: "o", 8: "s", 16: "^", 32: "D"}
            for concurrency in sorted(int(value) for value in subset["concurrency"].dropna().unique()):
                marker = concurrency_markers.get(concurrency, "x")
                part = subset[
                    (subset["cache_mode"] == cache_mode) & (subset["concurrency"] == concurrency)
                ].sort_values("bucket_tokens")
                if part.empty:
                    continue
                ax.plot(
                    part["bucket_tokens"] / 1024,
                    part["ttft_p95_median"],
                    color=COLORS[backend],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.2,
                    label=f"cache {cache_mode}, C={concurrency}",
                )
                fill_repeat_ci(
                    ax, part["bucket_tokens"] / 1024, part, "ttft_p95", COLORS[backend]
                )
        ax.set(title=backend, xlabel="共用前綴長度 (K tokens)", ylabel="P95 TTFT (ms/request)")
        finish_axis(ax)
        ax.legend(frameon=False)
    fig.suptitle("SQuAD 分組前綴快取", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(output / "04_prefix_cache.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_long_context(data: pd.DataFrame, output: Path) -> None:
    frame = data[data["suite"] == "ruler_performance"].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    concurrency_styles = {1: "-", 2: "--", 4: "-.", 8: ":", 16: (0, (3, 1, 1, 1)), 32: (0, (1, 1))}
    for backend in ("llama.cpp", "vLLM"):
        backend_frame = frame[frame["engine"] == backend]
        for concurrency in sorted(int(value) for value in backend_frame["concurrency"].dropna().unique()):
            linestyle = concurrency_styles.get(concurrency, "-")
            subset = frame[(frame["engine"] == backend) & (frame["concurrency"] == concurrency)].sort_values(
                "bucket_tokens"
            )
            if subset.empty:
                continue
            x = subset["bucket_tokens"] / 1024
            label = f"{backend}, C={concurrency}"
            axes[0].plot(x, subset["output_token_throughput_median"], color=COLORS[backend], linestyle=linestyle,
                         marker=MARKERS[backend], linewidth=2.2, label=label)
            fill_repeat_ci(axes[0], x, subset, "output_token_throughput", COLORS[backend])
            axes[1].plot(x, subset["ttft_p95_median"], color=COLORS[backend], linestyle=linestyle,
                         marker=MARKERS[backend], linewidth=2.2, label=label)
            fill_repeat_ci(axes[1], x, subset, "ttft_p95", COLORS[backend])
            axes[2].plot(x, subset["gpu_memory_peak_gib_median"], color=COLORS[backend], linestyle=linestyle,
                         marker=MARKERS[backend], linewidth=2.2, label=label)
            fill_repeat_ci(axes[2], x, subset, "gpu_memory_peak_gib", COLORS[backend])
    for ax, title, ylabel in zip(
        axes,
        ("輸出吞吐量", "P95 TTFT", "GPU 記憶體峰值"),
        ("token/s", "ms/request", "GiB"),
    ):
        ax.set(title=title, xlabel="輸入長度 (K tokens)", ylabel=ylabel)
        finish_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("RULER 長上下文效能與部署上限", fontsize=17, fontweight="bold")
    add_run_metadata(fig, frame)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "05_ruler_long_context_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(accuracy: pd.DataFrame, output: Path) -> None:
    if accuracy.empty:
        return
    frame = accuracy[accuracy["task"] == "overall-13-task-mean"].copy()
    if frame.empty:
        return
    frame["engine"] = frame["backend"].map(backend_label)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for backend in ("llama.cpp", "vLLM"):
        subset = frame[frame["engine"] == backend].sort_values("length_tokens")
        ax.plot(
            subset["length_tokens"] / 1024,
            subset["accuracy_pct"],
            color=COLORS[backend],
            marker=MARKERS[backend],
            linewidth=2.4,
            label=backend,
        )
    ax.set(title="RULERv1 官方 13 任務平均正確率", xlabel="上下文長度 (K tokens)", ylabel="Accuracy (%)")
    ax.set_ylim(0, 105)
    finish_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "06_ruler_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slo_heatmap(data: pd.DataFrame, output: Path) -> None:
    if data.empty or "slo_attainment_pct_median" not in data:
        return
    frame = data.copy()
    frame["label"] = frame["suite"] + ":" + frame["cell_id"]
    pivot = frame.pivot_table(index="label", columns="engine", values="slo_attainment_pct_median", aggfunc="first")
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return
    height = max(7, min(20, 0.28 * len(pivot)))
    fig, ax = plt.subplots(figsize=(8.5, height))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "SLO attainment (%)"},
        ax=ax,
    )
    ax.set(title="各測試格 SLO 達成率", xlabel="推論引擎", ylabel="測試格")
    fig.tight_layout()
    fig.savefig(output / "07_slo_attainment_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_path = args.input_dir / "dataset-cell-summary.csv"
    if not data_path.exists():
        raise SystemExit(f"Run analyze_dataset_results.py first: {data_path}")
    data = pd.read_csv(data_path)
    if data.empty:
        print("No dataset cells to plot.")
        return
    data["engine"] = data["backend"].map(backend_label)
    if "status" in data:
        data = data[data["status"].astype(str).str.startswith("valid")].copy()
    if data.empty:
        print("No valid dataset cells to plot.")
        return
    accuracy_path = args.input_dir / "ruler-accuracy.csv"
    accuracy = pd.read_csv(accuracy_path) if accuracy_path.exists() else pd.DataFrame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_sharegpt_closed(data, args.output_dir)
    plot_sharegpt_open(data, args.output_dir)
    plot_sharegpt_natural(data, args.output_dir)
    plot_rag(data, args.output_dir)
    plot_prefix(data, args.output_dir)
    plot_long_context(data, args.output_dir)
    plot_accuracy(accuracy, args.output_dir)
    plot_slo_heatmap(data, args.output_dir)
    print(f"Charts: {args.output_dir}")


if __name__ == "__main__":
    main()
