from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_ROOT.parent
RESULTS = REPO_ROOT / "results"
ANALYSIS_OUTPUT = PLATFORM_ROOT / "analysis" / "output"
CHARTS = PLATFORM_ROOT / "analysis" / "charts"

COLORS = {"llama-cpp": "#D97706", "vLLM": "#2563EB"}
MARKERS = {"llama-cpp": "o", "vLLM": "s"}


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
            "grid.linewidth": 0.8,
        }
    )


def read_summary(backend: str, suite: str) -> pd.DataFrame:
    files = sorted((RESULTS / backend / suite).glob("*/summary.csv"))
    frames = [(path, pd.read_csv(path)) for path in files]
    frames = [(path, frame) for path, frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    frame = frames[-1][1].copy()
    frame["backend"] = "vLLM" if backend == "vllm" else "llama-cpp"
    return frame


def finish_axis(ax: plt.Axes, x_values: list[float] | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    if x_values is not None:
        ax.set_xticks(x_values)


def add_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    backend: str,
    end_label: bool = True,
) -> None:
    subset = data[data["backend"] == backend].sort_values(x)
    ax.plot(
        subset[x],
        subset[y],
        label=backend,
        color=COLORS[backend],
        marker=MARKERS[backend],
        linewidth=2.4,
        markersize=6,
    )
    if end_label and not subset.empty:
        last = subset.iloc[-1]
        ax.annotate(
            f"{last[y]:,.0f}",
            (last[x], last[y]),
            xytext=(6, 0),
            textcoords="offset points",
            color=COLORS[backend],
            va="center",
            fontsize=9,
            fontweight="bold",
        )


def plot_core() -> None:
    concurrency = pd.concat(
        [read_summary("llama-cpp", "concurrency"), read_summary("vllm", "concurrency")],
        ignore_index=True,
    )
    context = pd.concat(
        [read_summary("llama-cpp", "context"), read_summary("vllm", "context")],
        ignore_index=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], concurrency, "concurrency", "decode_total_tps_mean", "Concurrency：總 Decode throughput", "tokens/s"),
        (axes[0, 1], concurrency, "concurrency", "e2e_ttft_ms_mean", "Concurrency：TTFT", "ms"),
        (axes[1, 0], context, "context_depth_tokens", "prefill_tps_mean", "Context：Prefill throughput", "tokens/s"),
        (axes[1, 1], context, "context_depth_tokens", "e2e_ttft_ms_mean", "Context：TTFT", "ms"),
    ]
    for ax, frame, x, y, title, ylabel in panels:
        for backend in ["llama-cpp", "vLLM"]:
            add_line(ax, frame, x, y, backend)
        values = sorted(frame[x].unique())
        finish_axis(ax, values)
        ax.set_title(title)
        ax.set_xlabel("Concurrency" if x == "concurrency" else "既有 Context tokens")
        ax.set_ylabel(ylabel)
        if x == "context_depth_tokens":
            ax.set_xticklabels(["0" if v == 0 else f"{int(v / 1024)}K" for v in values])
        ax.legend(frameon=False)

    fig.suptitle("vLLM vs llama.cpp：基礎 Scaling 比較", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CHARTS / "01_core_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prefix_cache() -> None:
    frame = pd.concat(
        [read_summary("llama-cpp", "prefix-cache"), read_summary("vllm", "prefix-cache")],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [
        (axes[0], "cache_off_ttft_ms", "Cache off：TTFT"),
        (axes[1], "cache_on_ttft_ms", "Cache on：TTFT"),
    ]
    for ax, metric, title in panels:
        for backend in ["llama-cpp", "vLLM"]:
            add_line(ax, frame, "context_depth_tokens", metric, backend)
        values = sorted(frame["context_depth_tokens"].unique())
        finish_axis(ax, values)
        ax.set_xticklabels([f"{int(v / 1024)}K" for v in values])
        ax.set_xlabel("Context tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("Prefix cache 效果", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CHARTS / "02_prefix_cache.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aiperf_workload(aiperf: pd.DataFrame, workload: str, filename: str) -> None:
    frame = aiperf[aiperf["workload"] == workload].copy()
    frame["backend"] = frame["backend"].replace({"vllm": "vLLM"})
    slo = float(frame["ttft_slo_ms"].iloc[0])
    titles = {
        "interactive": "AIPerf Interactive",
        "rag": "AIPerf RAG",
        "long-context": "AIPerf Long-context",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for backend in ["llama-cpp", "vLLM"]:
        add_line(axes[0], frame, "concurrency", "output_token_throughput_avg", backend)
        add_line(axes[1], frame, "concurrency", "time_to_first_token_p95", backend)
        add_line(axes[2], frame, "concurrency", "slo_attainment_pct", backend)

    x_values = sorted(frame["concurrency"].unique())
    finish_axis(axes[0], x_values)
    axes[0].set(title="Output throughput", xlabel="Concurrency", ylabel="tokens/s")

    finish_axis(axes[1], x_values)
    axes[1].axhline(slo, color="#64748B", linestyle="--", linewidth=1.5, label=f"SLO {slo:,.0f} ms")
    axes[1].set(title="P95 TTFT", xlabel="Concurrency", ylabel="ms")
    max_ttft = frame["time_to_first_token_p95"].max()
    min_ttft = frame["time_to_first_token_p95"].min()
    if max_ttft / min_ttft >= 20:
        axes[1].set_yscale("log")
        axes[1].set_ylabel("ms（對數刻度）")

    finish_axis(axes[2], x_values)
    axes[2].set(title="SLO 達成率", xlabel="Concurrency", ylabel="%", ylim=(-3, 105))

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False)

    fig.suptitle(f"{titles[workload]}：vLLM vs llama.cpp", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CHARTS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(aiperf: pd.DataFrame) -> None:
    frame = aiperf.copy()
    frame["backend"] = frame["backend"].replace({"vllm": "vLLM"})
    workloads = ["interactive", "rag", "long-context"]
    labels = {"interactive": "Interactive", "rag": "RAG", "long-context": "Long-context"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    concurrency = pd.concat(
        [read_summary("llama-cpp", "concurrency"), read_summary("vllm", "concurrency")],
        ignore_index=True,
    )
    for backend in ["llama-cpp", "vLLM"]:
        add_line(axes[0, 0], concurrency, "concurrency", "decode_total_tps_mean", backend)
    finish_axis(axes[0, 0], sorted(concurrency["concurrency"].unique()))
    axes[0, 0].set(title="基礎測試：總 Decode throughput", xlabel="Concurrency", ylabel="tokens/s")
    axes[0, 0].legend(frameon=False)

    for ax, workload in zip([axes[0, 1], axes[1, 0], axes[1, 1]], workloads):
        subset = frame[frame["workload"] == workload]
        for backend in ["llama-cpp", "vLLM"]:
            add_line(ax, subset, "concurrency", "output_token_throughput_avg", backend)
        finish_axis(ax, sorted(subset["concurrency"].unique()))
        ax.set(title=f"AIPerf {labels[workload]}：Output throughput", xlabel="Concurrency", ylabel="tokens/s")
        ax.legend(frameon=False)

    fig.suptitle("vLLM 與 llama.cpp 吞吐量走線總覽", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CHARTS / "00_throughput_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    CHARTS.mkdir(parents=True, exist_ok=True)
    aiperf = pd.read_csv(ANALYSIS_OUTPUT / "aiperf_key_metrics.csv")
    plot_dashboard(aiperf)
    plot_core()
    plot_prefix_cache()
    plot_aiperf_workload(aiperf, "interactive", "03_aiperf_interactive.png")
    plot_aiperf_workload(aiperf, "rag", "04_aiperf_rag.png")
    plot_aiperf_workload(aiperf, "long-context", "05_aiperf_long_context.png")
    print(f"Wrote 6 charts to {CHARTS}")


if __name__ == "__main__":
    main()
