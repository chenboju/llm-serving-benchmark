#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pp", required=True, type=int)
    parser.add_argument("--tg", required=True, type=int)
    parser.add_argument("--concurrencies", required=True, nargs="+", type=int)
    return parser.parse_args()


def load_results(path):
    results = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            match = re.search(r"\(c(\d+)\)", row["test_name"])
            if not match:
                continue
            item = results.setdefault(int(match.group(1)), {})
            if row["test_name"].startswith("pp"):
                item.update(
                    prefill_total_mean=float(row["t_s_mean"]),
                    prefill_total_std=float(row["t_s_std"]),
                    prefill_req_mean=float(row["t_s_req_mean"]),
                    prefill_req_std=float(row["t_s_req_std"]),
                    ttft_mean=float(row["e2e_ttft_mean"]),
                    ttft_std=float(row["e2e_ttft_std"]),
                )
            elif row["test_name"].startswith("tg"):
                item.update(
                    decode_total_mean=float(row["t_s_mean"]),
                    decode_total_std=float(row["t_s_std"]),
                    decode_req_mean=float(row["t_s_req_mean"]),
                    decode_req_std=float(row["t_s_req_std"]),
                    decode_peak_mean=float(row["peak_ts_mean"]),
                    decode_peak_std=float(row["peak_ts_std"]),
                )
    return results


def validate(results, concurrencies):
    required = {
        "prefill_total_mean", "prefill_total_std", "prefill_req_mean", "prefill_req_std",
        "decode_total_mean", "decode_total_std", "decode_req_mean", "decode_req_std",
        "decode_peak_mean", "decode_peak_std", "ttft_mean", "ttft_std",
    }
    for concurrency in concurrencies:
        missing = required - set(results.get(concurrency, {}))
        if missing:
            raise RuntimeError(f"Concurrency {concurrency} is missing: {', '.join(sorted(missing))}")


def write_summary(path, results, concurrencies, pp, tg):
    fields = [
        "concurrency", "prompt_tokens", "output_tokens", "prefill_total_tps_mean",
        "prefill_total_tps_std", "prefill_request_tps_mean", "prefill_request_tps_std",
        "decode_total_tps_mean", "decode_total_tps_std", "decode_request_tps_mean",
        "decode_request_tps_std", "decode_peak_tps_mean", "decode_peak_tps_std",
        "e2e_ttft_ms_mean", "e2e_ttft_ms_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for concurrency in concurrencies:
            item = results[concurrency]
            writer.writerow({
                "concurrency": concurrency, "prompt_tokens": pp, "output_tokens": tg,
                "prefill_total_tps_mean": f"{item['prefill_total_mean']:.2f}",
                "prefill_total_tps_std": f"{item['prefill_total_std']:.2f}",
                "prefill_request_tps_mean": f"{item['prefill_req_mean']:.2f}",
                "prefill_request_tps_std": f"{item['prefill_req_std']:.2f}",
                "decode_total_tps_mean": f"{item['decode_total_mean']:.2f}",
                "decode_total_tps_std": f"{item['decode_total_std']:.2f}",
                "decode_request_tps_mean": f"{item['decode_req_mean']:.2f}",
                "decode_request_tps_std": f"{item['decode_req_std']:.2f}",
                "decode_peak_tps_mean": f"{item['decode_peak_mean']:.2f}",
                "decode_peak_tps_std": f"{item['decode_peak_std']:.2f}",
                "e2e_ttft_ms_mean": f"{item['ttft_mean']:.2f}",
                "e2e_ttft_ms_std": f"{item['ttft_std']:.2f}",
            })


def configure_axis(axis, concurrencies):
    axis.set_xscale("log", base=2)
    axis.set_xticks(concurrencies, [str(value) for value in concurrencies])
    axis.set_xlim(min(concurrencies) * 0.8, max(concurrencies) * 1.2)
    axis.set_xlabel("Concurrency")
    axis.grid(axis="y", alpha=0.25)


def annotate(axis, x_values, y_values):
    for x_value, y_value in zip(x_values, y_values):
        axis.annotate(f"{y_value:,.0f}", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)


def plot(path, results, concurrencies, pp, tg):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specifications = [
        ("prefill_total_mean", "prefill_total_std", "prefill_req_mean", "prefill_req_std", "Prefill throughput", "Tokens/s"),
        ("decode_total_mean", "decode_total_std", "decode_req_mean", "decode_req_std", "Decode throughput", "Tokens/s"),
    ]
    for axis, (total_key, total_std_key, req_key, req_std_key, title, unit) in zip(axes[:2], specifications):
        totals = [results[c][total_key] for c in concurrencies]
        requests = [results[c][req_key] for c in concurrencies]
        axis.errorbar(concurrencies, totals, yerr=[results[c][total_std_key] for c in concurrencies], marker="o", linewidth=2, capsize=4, label="Aggregate")
        axis.errorbar(concurrencies, requests, yerr=[results[c][req_std_key] for c in concurrencies], marker="o", linewidth=2, capsize=4, label="Per request")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_ylim(bottom=0)
        axis.legend()
        configure_axis(axis, concurrencies)
        annotate(axis, concurrencies, totals)
    ttft = [results[c]["ttft_mean"] for c in concurrencies]
    axes[2].errorbar(concurrencies, ttft, yerr=[results[c]["ttft_std"] for c in concurrencies], marker="o", linewidth=2, capsize=4, color="#dc2626")
    axes[2].set_title("End-to-end TTFT")
    axes[2].set_ylabel("Milliseconds")
    axes[2].set_ylim(bottom=0)
    configure_axis(axes[2], concurrencies)
    annotate(axes[2], concurrencies, ttft)
    figure.suptitle(f"vLLM RTX 4090 - Concurrency Scaling (PP {pp} / TG {tg})", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = arguments()
    run_dir = args.run_dir.resolve()
    raw_csv = run_dir / "raw" / "llama-benchy.csv"
    if not raw_csv.is_file():
        raise FileNotFoundError(raw_csv)
    results = load_results(raw_csv)
    validate(results, args.concurrencies)
    write_summary(run_dir / "summary.csv", results, args.concurrencies, args.pp, args.tg)
    plot(run_dir / "concurrency-scaling.png", results, args.concurrencies, args.pp, args.tg)
    print(f"Clean summary: {run_dir / 'summary.csv'}")
    print(f"Plot:          {run_dir / 'concurrency-scaling.png'}")


if __name__ == "__main__":
    main()
