#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a clean context-scaling summary and PNG."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pp", required=True, type=int)
    parser.add_argument("--tg", required=True, type=int)
    parser.add_argument("--depths", required=True, nargs="+", type=int)
    parser.add_argument(
        "--title",
        default="llama.cpp RTX 4090 — Context Scaling",
    )
    return parser.parse_args()


def parse_depth(test_name):
    match = re.search(r"@ d(\d+)", test_name)
    return int(match.group(1)) if match else 0


def load_results(raw_csv):
    results = {}

    with raw_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "test_name",
            "t_s_mean",
            "t_s_std",
            "e2e_ttft_mean",
            "e2e_ttft_std",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise RuntimeError(f"Raw CSV is missing columns: {missing}")

        for row in reader:
            test_name = row["test_name"]
            depth = parse_depth(test_name)
            result = results.setdefault(depth, {})

            if test_name.startswith("pp"):
                result["prefill_mean"] = float(row["t_s_mean"])
                result["prefill_std"] = float(row["t_s_std"])
                result["ttft_mean"] = float(row["e2e_ttft_mean"])
                result["ttft_std"] = float(row["e2e_ttft_std"])

            elif test_name.startswith("tg"):
                result["decode_mean"] = float(row["t_s_mean"])
                result["decode_std"] = float(row["t_s_std"])

    return results


def validate_results(results, depths):
    required_metrics = {
        "prefill_mean",
        "prefill_std",
        "decode_mean",
        "decode_std",
        "ttft_mean",
        "ttft_std",
    }

    for depth in depths:
        if depth not in results:
            raise RuntimeError(f"Missing benchmark result for depth {depth}")

        missing_metrics = required_metrics - set(results[depth])
        if missing_metrics:
            missing = ", ".join(sorted(missing_metrics))
            raise RuntimeError(
                f"Depth {depth} is missing metrics: {missing}"
            )


def write_summary(summary_csv, results, depths, pp_tokens, tg_tokens):
    fields = [
        "context_depth_tokens",
        "prompt_tokens",
        "output_tokens",
        "prefill_tps_mean",
        "prefill_tps_std",
        "decode_tps_mean",
        "decode_tps_std",
        "e2e_ttft_ms_mean",
        "e2e_ttft_ms_std",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for depth in depths:
            item = results[depth]

            writer.writerow(
                {
                    "context_depth_tokens": depth,
                    "prompt_tokens": pp_tokens,
                    "output_tokens": tg_tokens,
                    "prefill_tps_mean": f"{item['prefill_mean']:.2f}",
                    "prefill_tps_std": f"{item['prefill_std']:.2f}",
                    "decode_tps_mean": f"{item['decode_mean']:.2f}",
                    "decode_tps_std": f"{item['decode_std']:.2f}",
                    "e2e_ttft_ms_mean": f"{item['ttft_mean']:.2f}",
                    "e2e_ttft_ms_std": f"{item['ttft_std']:.2f}",
                }
            )


def format_depth(depth):
    if depth == 0:
        return "0"
    if depth % 1024 == 0:
        return f"{depth // 1024}K"
    return f"{depth:,}"


def add_y_padding(axis, values):
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum

    if span == 0:
        span = max(abs(maximum) * 0.1, 1)

    axis.set_ylim(
        minimum - span * 0.18,
        maximum + span * 0.28,
    )


def create_plot(
    plot_png,
    results,
    depths,
    pp_tokens,
    tg_tokens,
    title,
):
    labels = [format_depth(depth) for depth in depths]

    prefill = [results[d]["prefill_mean"] for d in depths]
    prefill_std = [results[d]["prefill_std"] for d in depths]
    decode = [results[d]["decode_mean"] for d in depths]
    decode_std = [results[d]["decode_std"] for d in depths]
    ttft = [results[d]["ttft_mean"] for d in depths]
    ttft_std = [results[d]["ttft_std"] for d in depths]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    series = [
        (
            prefill,
            prefill_std,
            "Prefill throughput",
            "Tokens/s",
            "#2563eb",
        ),
        (
            decode,
            decode_std,
            "Decode throughput",
            "Tokens/s",
            "#16a34a",
        ),
        (
            ttft,
            ttft_std,
            "End-to-end TTFT",
            "Milliseconds",
            "#dc2626",
        ),
    ]

    maximum_depth = max(depths)
    x_margin = maximum_depth * 0.04 if maximum_depth else 1

    for axis, (values, errors, subtitle, ylabel, color) in zip(
        axes, series
    ):
        axis.errorbar(
            depths,
            values,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
        )

        axis.set_xlim(-x_margin, maximum_depth + x_margin)
        axis.set_xticks(depths, labels)
        axis.set_xlabel("Existing context depth")
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle)
        axis.grid(axis="y", alpha=0.25)

        add_y_padding(axis, values)

        for depth, value in zip(depths, values):
            axis.annotate(
                f"{value:,.0f}",
                (depth, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    fig.suptitle(
        f"{title} (PP {pp_tokens} / TG {tg_tokens})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plot_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    run_dir = args.run_dir.resolve()
    raw_csv = run_dir / "raw" / "llama-benchy.csv"
    summary_csv = run_dir / "summary.csv"
    plot_png = run_dir / "context-scaling.png"

    if not raw_csv.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    results = load_results(raw_csv)
    validate_results(results, args.depths)

    write_summary(
        summary_csv,
        results,
        args.depths,
        args.pp,
        args.tg,
    )

    create_plot(
        plot_png,
        results,
        args.depths,
        args.pp,
        args.tg,
        args.title,
    )

    print(f"Clean summary: {summary_csv}")
    print(f"Plot:          {plot_png}")


if __name__ == "__main__":
    main()