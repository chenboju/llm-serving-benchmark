#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pp", required=True, type=int)
    parser.add_argument("--tg", required=True, type=int)
    parser.add_argument(
        "--concurrencies",
        required=True,
        nargs="+",
        type=int,
    )
    return parser.parse_args()


def load_results(raw_csv):
    results = {}

    with raw_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            test_name = row["test_name"]
            match = re.search(r"\(c(\d+)\)", test_name)

            if not match:
                continue

            concurrency = int(match.group(1))
            result = results.setdefault(concurrency, {})

            if test_name.startswith("pp"):
                result["prefill_total_mean"] = float(row["t_s_mean"])
                result["prefill_total_std"] = float(row["t_s_std"])
                result["prefill_req_mean"] = float(row["t_s_req_mean"])
                result["prefill_req_std"] = float(row["t_s_req_std"])
                result["ttft_mean"] = float(row["e2e_ttft_mean"])
                result["ttft_std"] = float(row["e2e_ttft_std"])

            elif test_name.startswith("tg"):
                result["decode_total_mean"] = float(row["t_s_mean"])
                result["decode_total_std"] = float(row["t_s_std"])
                result["decode_req_mean"] = float(row["t_s_req_mean"])
                result["decode_req_std"] = float(row["t_s_req_std"])
                result["decode_peak_mean"] = float(row["peak_ts_mean"])
                result["decode_peak_std"] = float(row["peak_ts_std"])

    return results


def validate_results(results, concurrencies):
    required = {
        "prefill_total_mean",
        "prefill_total_std",
        "prefill_req_mean",
        "prefill_req_std",
        "decode_total_mean",
        "decode_total_std",
        "decode_req_mean",
        "decode_req_std",
        "decode_peak_mean",
        "decode_peak_std",
        "ttft_mean",
        "ttft_std",
    }

    for concurrency in concurrencies:
        if concurrency not in results:
            raise RuntimeError(
                f"Missing result for concurrency {concurrency}"
            )

        missing = required - set(results[concurrency])
        if missing:
            raise RuntimeError(
                f"Concurrency {concurrency} is missing: "
                + ", ".join(sorted(missing))
            )


def write_summary(
    summary_csv,
    results,
    concurrencies,
    pp_tokens,
    tg_tokens,
):
    fields = [
        "concurrency",
        "prompt_tokens",
        "output_tokens",
        "prefill_total_tps_mean",
        "prefill_total_tps_std",
        "prefill_request_tps_mean",
        "prefill_request_tps_std",
        "decode_total_tps_mean",
        "decode_total_tps_std",
        "decode_request_tps_mean",
        "decode_request_tps_std",
        "decode_peak_tps_mean",
        "decode_peak_tps_std",
        "e2e_ttft_ms_mean",
        "e2e_ttft_ms_std",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for concurrency in concurrencies:
            item = results[concurrency]

            writer.writerow(
                {
                    "concurrency": concurrency,
                    "prompt_tokens": pp_tokens,
                    "output_tokens": tg_tokens,
                    "prefill_total_tps_mean": (
                        f"{item['prefill_total_mean']:.2f}"
                    ),
                    "prefill_total_tps_std": (
                        f"{item['prefill_total_std']:.2f}"
                    ),
                    "prefill_request_tps_mean": (
                        f"{item['prefill_req_mean']:.2f}"
                    ),
                    "prefill_request_tps_std": (
                        f"{item['prefill_req_std']:.2f}"
                    ),
                    "decode_total_tps_mean": (
                        f"{item['decode_total_mean']:.2f}"
                    ),
                    "decode_total_tps_std": (
                        f"{item['decode_total_std']:.2f}"
                    ),
                    "decode_request_tps_mean": (
                        f"{item['decode_req_mean']:.2f}"
                    ),
                    "decode_request_tps_std": (
                        f"{item['decode_req_std']:.2f}"
                    ),
                    "decode_peak_tps_mean": (
                        f"{item['decode_peak_mean']:.2f}"
                    ),
                    "decode_peak_tps_std": (
                        f"{item['decode_peak_std']:.2f}"
                    ),
                    "e2e_ttft_ms_mean": (
                        f"{item['ttft_mean']:.2f}"
                    ),
                    "e2e_ttft_ms_std": (
                        f"{item['ttft_std']:.2f}"
                    ),
                }
            )


def configure_x_axis(axis, concurrencies):
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        concurrencies,
        [str(value) for value in concurrencies],
    )
    axis.set_xlim(
        min(concurrencies) * 0.8,
        max(concurrencies) * 1.2,
    )
    axis.set_xlabel("Concurrency")
    axis.grid(axis="y", alpha=0.25)


def annotate_values(axis, x_values, y_values, formatter):
    for x_value, y_value in zip(x_values, y_values):
        axis.annotate(
            formatter(y_value),
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )


def create_plot(
    plot_png,
    results,
    concurrencies,
    pp_tokens,
    tg_tokens,
):
    prefill_total = [
        results[c]["prefill_total_mean"] for c in concurrencies
    ]
    prefill_total_std = [
        results[c]["prefill_total_std"] for c in concurrencies
    ]
    prefill_req = [
        results[c]["prefill_req_mean"] for c in concurrencies
    ]
    prefill_req_std = [
        results[c]["prefill_req_std"] for c in concurrencies
    ]

    decode_total = [
        results[c]["decode_total_mean"] for c in concurrencies
    ]
    decode_total_std = [
        results[c]["decode_total_std"] for c in concurrencies
    ]
    decode_req = [
        results[c]["decode_req_mean"] for c in concurrencies
    ]
    decode_req_std = [
        results[c]["decode_req_std"] for c in concurrencies
    ]

    ttft = [results[c]["ttft_mean"] for c in concurrencies]
    ttft_std = [results[c]["ttft_std"] for c in concurrencies]

    best_index = max(
        range(len(concurrencies)),
        key=lambda index: decode_total[index],
    )
    best_concurrency = concurrencies[best_index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].errorbar(
        concurrencies,
        prefill_total,
        yerr=prefill_total_std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Aggregate",
    )
    axes[0].errorbar(
        concurrencies,
        prefill_req,
        yerr=prefill_req_std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Per request",
    )
    axes[0].set_title("Prefill throughput")
    axes[0].set_ylabel("Tokens/s")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    configure_x_axis(axes[0], concurrencies)
    annotate_values(
        axes[0],
        concurrencies,
        prefill_total,
        lambda value: f"{value:,.0f}",
    )

    axes[1].errorbar(
        concurrencies,
        decode_total,
        yerr=decode_total_std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Aggregate",
    )
    axes[1].errorbar(
        concurrencies,
        decode_req,
        yerr=decode_req_std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Per request",
    )
    axes[1].set_title(
        f"Decode throughput (best sustained: c{best_concurrency})"
    )
    axes[1].set_ylabel("Tokens/s")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()
    configure_x_axis(axes[1], concurrencies)
    annotate_values(
        axes[1],
        concurrencies,
        decode_total,
        lambda value: f"{value:,.0f}",
    )

    axes[2].errorbar(
        concurrencies,
        ttft,
        yerr=ttft_std,
        marker="o",
        linewidth=2,
        capsize=4,
        color="#dc2626",
    )
    axes[2].set_title("End-to-end TTFT")
    axes[2].set_ylabel("Milliseconds")
    axes[2].set_ylim(bottom=0)
    configure_x_axis(axes[2], concurrencies)
    annotate_values(
        axes[2],
        concurrencies,
        ttft,
        lambda value: (
            f"{value / 1000:.2f}s"
            if value >= 1000
            else f"{value:.0f}ms"
        ),
    )

    fig.suptitle(
        "llama.cpp RTX 4090 — "
        f"Concurrency Scaling (PP {pp_tokens} / TG {tg_tokens})",
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
    plot_png = run_dir / "concurrency-scaling.png"

    if not raw_csv.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    results = load_results(raw_csv)
    validate_results(results, args.concurrencies)

    write_summary(
        summary_csv,
        results,
        args.concurrencies,
        args.pp,
        args.tg,
    )

    create_plot(
        plot_png,
        results,
        args.concurrencies,
        args.pp,
        args.tg,
    )

    print(f"Clean summary: {summary_csv}")
    print(f"Plot:          {plot_png}")


if __name__ == "__main__":
    main()