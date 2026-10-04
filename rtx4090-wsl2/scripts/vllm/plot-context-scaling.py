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
    parser.add_argument("--depths", required=True, nargs="+", type=int)
    parser.add_argument("--title", default="vLLM RTX 4090 - Context Scaling")
    return parser.parse_args()


def depth_from_name(test_name):
    match = re.search(r"@ d(\d+)", test_name)
    return int(match.group(1)) if match else 0


def load_results(path):
    results = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"test_name", "t_s_mean", "t_s_std", "e2e_ttft_mean", "e2e_ttft_std"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError("Raw CSV is missing: " + ", ".join(sorted(missing)))
        for row in reader:
            item = results.setdefault(depth_from_name(row["test_name"]), {})
            if row["test_name"].startswith("pp"):
                item.update(
                    prefill_mean=float(row["t_s_mean"]),
                    prefill_std=float(row["t_s_std"]),
                    ttft_mean=float(row["e2e_ttft_mean"]),
                    ttft_std=float(row["e2e_ttft_std"]),
                )
            elif row["test_name"].startswith("tg"):
                item.update(
                    decode_mean=float(row["t_s_mean"]),
                    decode_std=float(row["t_s_std"]),
                )
    return results


def validate(results, depths):
    required = {"prefill_mean", "prefill_std", "decode_mean", "decode_std", "ttft_mean", "ttft_std"}
    for depth in depths:
        missing = required - set(results.get(depth, {}))
        if missing:
            raise RuntimeError(f"Depth {depth} is missing: {', '.join(sorted(missing))}")


def write_summary(path, results, depths, pp, tg):
    fields = [
        "context_depth_tokens", "prompt_tokens", "output_tokens",
        "prefill_tps_mean", "prefill_tps_std", "decode_tps_mean",
        "decode_tps_std", "e2e_ttft_ms_mean", "e2e_ttft_ms_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for depth in depths:
            item = results[depth]
            writer.writerow({
                "context_depth_tokens": depth,
                "prompt_tokens": pp,
                "output_tokens": tg,
                "prefill_tps_mean": f"{item['prefill_mean']:.2f}",
                "prefill_tps_std": f"{item['prefill_std']:.2f}",
                "decode_tps_mean": f"{item['decode_mean']:.2f}",
                "decode_tps_std": f"{item['decode_std']:.2f}",
                "e2e_ttft_ms_mean": f"{item['ttft_mean']:.2f}",
                "e2e_ttft_ms_std": f"{item['ttft_std']:.2f}",
            })


def plot(path, results, depths, pp, tg, title):
    labels = ["0" if depth == 0 else f"{depth // 1024}K" for depth in depths]
    series = [
        ("prefill_mean", "prefill_std", "Prefill throughput", "Tokens/s", "#2563eb"),
        ("decode_mean", "decode_std", "Decode throughput", "Tokens/s", "#16a34a"),
        ("ttft_mean", "ttft_std", "End-to-end TTFT", "Milliseconds", "#dc2626"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    maximum_depth = max(depths)
    margin = maximum_depth * 0.04 if maximum_depth else 1
    for axis, (value_key, error_key, subtitle, unit, color) in zip(axes, series):
        values = [results[depth][value_key] for depth in depths]
        errors = [results[depth][error_key] for depth in depths]
        axis.errorbar(depths, values, yerr=errors, marker="o", linewidth=2, capsize=4, color=color)
        axis.set_xlim(-margin, maximum_depth + margin)
        axis.set_xticks(depths, labels)
        axis.set_xlabel("Existing context depth")
        axis.set_ylabel(unit)
        axis.set_title(subtitle)
        axis.grid(axis="y", alpha=0.25)
        span = max(values) - min(values) or max(abs(max(values)) * 0.1, 1)
        axis.set_ylim(min(values) - span * 0.18, max(values) + span * 0.28)
        for depth, value in zip(depths, values):
            axis.annotate(f"{value:,.0f}", (depth, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    figure.suptitle(f"{title} (PP {pp} / TG {tg})", fontsize=14)
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
    validate(results, args.depths)
    write_summary(run_dir / "summary.csv", results, args.depths, args.pp, args.tg)
    plot(run_dir / "context-scaling.png", results, args.depths, args.pp, args.tg, args.title)
    print(f"Clean summary: {run_dir / 'summary.csv'}")
    print(f"Plot:          {run_dir / 'context-scaling.png'}")


if __name__ == "__main__":
    main()
