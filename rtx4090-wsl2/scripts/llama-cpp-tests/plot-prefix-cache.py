#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pp", required=True, type=int)
    parser.add_argument("--tg", required=True, type=int)
    parser.add_argument("--depths", required=True, nargs="+", type=int)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_row(rows, prefix):
    for row in rows:
        if row["test_name"].startswith(prefix):
            return row
    raise RuntimeError(f"Missing row beginning with: {prefix}")


def number(row, field):
    return float(row[field])


def load_results(run_dir, depths):
    raw_dir = run_dir / "raw"
    results = {}

    for depth in depths:
        off_rows = read_rows(raw_dir / f"cache-off-d{depth}.csv")
        on_rows = read_rows(raw_dir / f"cache-on-d{depth}.csv")

        off_pp = find_row(off_rows, "pp")
        off_tg = find_row(off_rows, "tg")
        context_pp = find_row(on_rows, "ctx_pp")
        on_pp = find_row(on_rows, "pp")
        on_tg = find_row(on_rows, "tg")

        off_ttft = number(off_pp, "e2e_ttft_mean")
        on_ttft = number(on_pp, "e2e_ttft_mean")
        off_prompt = number(off_pp, "est_ppt_mean")
        on_prompt = number(on_pp, "est_ppt_mean")

        results[depth] = {
            "off_prefill_tps": number(off_pp, "t_s_mean"),
            "off_prefill_tps_std": number(off_pp, "t_s_std"),
            "on_prefill_tps": number(on_pp, "t_s_mean"),
            "on_prefill_tps_std": number(on_pp, "t_s_std"),
            "off_decode_tps": number(off_tg, "t_s_mean"),
            "off_decode_tps_std": number(off_tg, "t_s_std"),
            "on_decode_tps": number(on_tg, "t_s_mean"),
            "on_decode_tps_std": number(on_tg, "t_s_std"),
            "off_prompt_ms": off_prompt,
            "off_prompt_ms_std": number(off_pp, "est_ppt_std"),
            "on_prompt_ms": on_prompt,
            "on_prompt_ms_std": number(on_pp, "est_ppt_std"),
            "off_ttft_ms": off_ttft,
            "off_ttft_ms_std": number(off_pp, "e2e_ttft_std"),
            "on_ttft_ms": on_ttft,
            "on_ttft_ms_std": number(on_pp, "e2e_ttft_std"),
            "context_load_ms": number(
                context_pp,
                "e2e_ttft_mean",
            ),
            "context_load_ms_std": number(
                context_pp,
                "e2e_ttft_std",
            ),
            "context_load_prompt_ms": number(
                context_pp,
                "est_ppt_mean",
            ),
            "ttft_saved_ms": off_ttft - on_ttft,
            "ttft_reduction_pct": (
                100 * (1 - on_ttft / off_ttft)
            ),
            "prompt_saved_ms": off_prompt - on_prompt,
            "prompt_reduction_pct": (
                100 * (1 - on_prompt / off_prompt)
            ),
        }

    return results


def write_summary(path, results, depths, pp_tokens, tg_tokens):
    fields = [
        "context_depth_tokens",
        "prompt_tokens",
        "output_tokens",
        "context_load_ttft_ms",
        "cache_off_ttft_ms",
        "cache_on_ttft_ms",
        "ttft_saved_ms",
        "ttft_reduction_pct",
        "cache_off_prompt_ms",
        "cache_on_prompt_ms",
        "prompt_saved_ms",
        "prompt_reduction_pct",
        "cache_off_prefill_tps",
        "cache_on_new_prompt_tps",
        "cache_off_decode_tps",
        "cache_on_decode_tps",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for depth in depths:
            item = results[depth]

            writer.writerow(
                {
                    "context_depth_tokens": depth,
                    "prompt_tokens": pp_tokens,
                    "output_tokens": tg_tokens,
                    "context_load_ttft_ms": (
                        f"{item['context_load_ms']:.2f}"
                    ),
                    "cache_off_ttft_ms": (
                        f"{item['off_ttft_ms']:.2f}"
                    ),
                    "cache_on_ttft_ms": (
                        f"{item['on_ttft_ms']:.2f}"
                    ),
                    "ttft_saved_ms": (
                        f"{item['ttft_saved_ms']:.2f}"
                    ),
                    "ttft_reduction_pct": (
                        f"{item['ttft_reduction_pct']:.2f}"
                    ),
                    "cache_off_prompt_ms": (
                        f"{item['off_prompt_ms']:.2f}"
                    ),
                    "cache_on_prompt_ms": (
                        f"{item['on_prompt_ms']:.2f}"
                    ),
                    "prompt_saved_ms": (
                        f"{item['prompt_saved_ms']:.2f}"
                    ),
                    "prompt_reduction_pct": (
                        f"{item['prompt_reduction_pct']:.2f}"
                    ),
                    "cache_off_prefill_tps": (
                        f"{item['off_prefill_tps']:.2f}"
                    ),
                    "cache_on_new_prompt_tps": (
                        f"{item['on_prefill_tps']:.2f}"
                    ),
                    "cache_off_decode_tps": (
                        f"{item['off_decode_tps']:.2f}"
                    ),
                    "cache_on_decode_tps": (
                        f"{item['on_decode_tps']:.2f}"
                    ),
                }
            )


def depth_label(depth):
    if depth % 1024 == 0:
        return f"{depth // 1024}K"
    return f"{depth:,}"


def add_bar_labels(axis, bars, suffix=""):
    labels = [
        f"{bar.get_height():,.0f}{suffix}"
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=9)


def create_plot(path, results, depths, pp_tokens, tg_tokens):
    labels = [depth_label(depth) for depth in depths]
    positions = list(range(len(depths)))

    off_ttft = [results[d]["off_ttft_ms"] for d in depths]
    off_ttft_std = [
        results[d]["off_ttft_ms_std"] for d in depths
    ]
    context_load = [
        results[d]["context_load_ms"] for d in depths
    ]
    context_load_std = [
        results[d]["context_load_ms_std"] for d in depths
    ]
    on_ttft = [results[d]["on_ttft_ms"] for d in depths]
    on_ttft_std = [
        results[d]["on_ttft_ms_std"] for d in depths
    ]

    off_prompt = [results[d]["off_prompt_ms"] for d in depths]
    off_prompt_std = [
        results[d]["off_prompt_ms_std"] for d in depths
    ]
    on_prompt = [results[d]["on_prompt_ms"] for d in depths]
    on_prompt_std = [
        results[d]["on_prompt_ms_std"] for d in depths
    ]

    ttft_reduction = [
        results[d]["ttft_reduction_pct"] for d in depths
    ]
    prompt_reduction = [
        results[d]["prompt_reduction_pct"] for d in depths
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    width = 0.24

    off_bars = axes[0].bar(
        [x - width for x in positions],
        off_ttft,
        width,
        yerr=off_ttft_std,
        capsize=4,
        label="Cache off",
    )
    load_bars = axes[0].bar(
        positions,
        context_load,
        width,
        yerr=context_load_std,
        capsize=4,
        label="Initial cache load",
    )
    on_bars = axes[0].bar(
        [x + width for x in positions],
        on_ttft,
        width,
        yerr=on_ttft_std,
        capsize=4,
        label="Cached follow-up",
    )
    axes[0].set_title("End-to-end TTFT")
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_xticks(positions, labels)
    axes[0].set_xlabel("Context depth")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[0].set_ylim(
        0,
        max(off_ttft + context_load) * 1.22,
    )
    add_bar_labels(axes[0], off_bars)
    add_bar_labels(axes[0], load_bars)
    add_bar_labels(axes[0], on_bars)

    prompt_width = 0.32

    off_prompt_bars = axes[1].bar(
        [x - prompt_width / 2 for x in positions],
        off_prompt,
        prompt_width,
        yerr=off_prompt_std,
        capsize=4,
        label="Cache off",
    )
    on_prompt_bars = axes[1].bar(
        [x + prompt_width / 2 for x in positions],
        on_prompt,
        prompt_width,
        yerr=on_prompt_std,
        capsize=4,
        label="Cached follow-up",
    )
    axes[1].set_title("Prompt processing time")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_xticks(positions, labels)
    axes[1].set_xlabel("Context depth")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].set_ylim(0, max(off_prompt) * 1.22)
    add_bar_labels(axes[1], off_prompt_bars)
    add_bar_labels(axes[1], on_prompt_bars)

    reduction_width = 0.32

    ttft_bars = axes[2].bar(
        [x - reduction_width / 2 for x in positions],
        ttft_reduction,
        reduction_width,
        label="TTFT reduction",
    )
    prompt_bars = axes[2].bar(
        [x + reduction_width / 2 for x in positions],
        prompt_reduction,
        reduction_width,
        label="Prompt-time reduction",
    )
    axes[2].set_title("Prefix-cache benefit")
    axes[2].set_ylabel("Reduction (%)")
    axes[2].set_xticks(positions, labels)
    axes[2].set_xlabel("Context depth")
    axes[2].set_ylim(0, 110)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    add_bar_labels(axes[2], ttft_bars, "%")
    add_bar_labels(axes[2], prompt_bars, "%")

    fig.suptitle(
        "llama.cpp RTX 4090 — "
        f"Prefix Cache (PP {pp_tokens} / TG {tg_tokens})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    results = load_results(run_dir, args.depths)

    summary_csv = run_dir / "summary.csv"
    plot_png = run_dir / "prefix-cache.png"

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
    )

    print(f"Clean summary: {summary_csv}")
    print(f"Plot:          {plot_png}")


if __name__ == "__main__":
    main()