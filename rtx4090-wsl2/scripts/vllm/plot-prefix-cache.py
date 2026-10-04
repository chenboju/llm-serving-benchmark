#!/usr/bin/env python3
import argparse
import csv
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
    return parser.parse_args()


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def first(items, prefix):
    try:
        return next(row for row in items if row["test_name"].startswith(prefix))
    except StopIteration as error:
        raise RuntimeError(f"Missing row beginning with: {prefix}") from error


def load_results(run_dir, depths):
    results = {}
    for depth in depths:
        off_rows = rows(run_dir / "raw" / f"cache-off-d{depth}.csv")
        on_rows = rows(run_dir / "raw" / f"cache-on-d{depth}.csv")
        off_pp, off_tg = first(off_rows, "pp"), first(off_rows, "tg")
        context_pp, on_pp, on_tg = first(on_rows, "ctx_pp"), first(on_rows, "pp"), first(on_rows, "tg")
        off_ttft, on_ttft = float(off_pp["e2e_ttft_mean"]), float(on_pp["e2e_ttft_mean"])
        off_prompt, on_prompt = float(off_pp["est_ppt_mean"]), float(on_pp["est_ppt_mean"])
        results[depth] = {
            "off_ttft": off_ttft, "off_ttft_std": float(off_pp["e2e_ttft_std"]),
            "on_ttft": on_ttft, "on_ttft_std": float(on_pp["e2e_ttft_std"]),
            "load_ttft": float(context_pp["e2e_ttft_mean"]),
            "load_ttft_std": float(context_pp["e2e_ttft_std"]),
            "off_prompt": off_prompt, "off_prompt_std": float(off_pp["est_ppt_std"]),
            "on_prompt": on_prompt, "on_prompt_std": float(on_pp["est_ppt_std"]),
            "off_prefill": float(off_pp["t_s_mean"]), "on_prefill": float(on_pp["t_s_mean"]),
            "off_decode": float(off_tg["t_s_mean"]), "on_decode": float(on_tg["t_s_mean"]),
            "ttft_saved": off_ttft - on_ttft,
            "ttft_reduction": 100 * (1 - on_ttft / off_ttft),
            "prompt_saved": off_prompt - on_prompt,
            "prompt_reduction": 100 * (1 - on_prompt / off_prompt),
        }
    return results


def write_summary(path, results, depths, pp, tg):
    fields = [
        "context_depth_tokens", "prompt_tokens", "output_tokens", "context_load_ttft_ms",
        "cache_off_ttft_ms", "cache_on_ttft_ms", "ttft_saved_ms", "ttft_reduction_pct",
        "cache_off_prompt_ms", "cache_on_prompt_ms", "prompt_saved_ms", "prompt_reduction_pct",
        "cache_off_prefill_tps", "cache_on_new_prompt_tps", "cache_off_decode_tps", "cache_on_decode_tps",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for depth in depths:
            item = results[depth]
            writer.writerow({
                "context_depth_tokens": depth, "prompt_tokens": pp, "output_tokens": tg,
                "context_load_ttft_ms": f"{item['load_ttft']:.2f}",
                "cache_off_ttft_ms": f"{item['off_ttft']:.2f}",
                "cache_on_ttft_ms": f"{item['on_ttft']:.2f}",
                "ttft_saved_ms": f"{item['ttft_saved']:.2f}",
                "ttft_reduction_pct": f"{item['ttft_reduction']:.2f}",
                "cache_off_prompt_ms": f"{item['off_prompt']:.2f}",
                "cache_on_prompt_ms": f"{item['on_prompt']:.2f}",
                "prompt_saved_ms": f"{item['prompt_saved']:.2f}",
                "prompt_reduction_pct": f"{item['prompt_reduction']:.2f}",
                "cache_off_prefill_tps": f"{item['off_prefill']:.2f}",
                "cache_on_new_prompt_tps": f"{item['on_prefill']:.2f}",
                "cache_off_decode_tps": f"{item['off_decode']:.2f}",
                "cache_on_decode_tps": f"{item['on_decode']:.2f}",
            })


def label_bars(axis, bars, suffix=""):
    axis.bar_label(bars, labels=[f"{bar.get_height():,.0f}{suffix}" for bar in bars], padding=3, fontsize=9)


def plot(path, results, depths, pp, tg):
    labels = [f"{depth // 1024}K" for depth in depths]
    positions = list(range(len(depths)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    width = 0.24
    specs = [
        ("off_ttft", -width, "Cache off"),
        ("load_ttft", 0, "Initial cache load"),
        ("on_ttft", width, "Cached follow-up"),
    ]
    for key, offset, name in specs:
        bars = axes[0].bar([x + offset for x in positions], [results[d][key] for d in depths], width, yerr=[results[d][key + "_std"] for d in depths], capsize=4, label=name)
        label_bars(axes[0], bars)
    axes[0].set_title("End-to-end TTFT")
    axes[0].set_ylabel("Milliseconds")
    axes[0].legend()

    prompt_width = 0.32
    for key, offset, name in [("off_prompt", -prompt_width / 2, "Cache off"), ("on_prompt", prompt_width / 2, "Cached follow-up")]:
        bars = axes[1].bar([x + offset for x in positions], [results[d][key] for d in depths], prompt_width, yerr=[results[d][key + "_std"] for d in depths], capsize=4, label=name)
        label_bars(axes[1], bars)
    axes[1].set_title("Prompt processing time")
    axes[1].set_ylabel("Milliseconds")
    axes[1].legend()

    reduction_width = 0.32
    for key, offset, name in [("ttft_reduction", -reduction_width / 2, "TTFT reduction"), ("prompt_reduction", reduction_width / 2, "Prompt-time reduction")]:
        bars = axes[2].bar([x + offset for x in positions], [results[d][key] for d in depths], reduction_width, label=name)
        label_bars(axes[2], bars, "%")
    axes[2].set_title("Prefix-cache benefit")
    axes[2].set_ylabel("Reduction (%)")
    axes[2].set_ylim(0, 110)
    axes[2].legend()

    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Context depth")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"vLLM RTX 4090 - Prefix Cache (PP {pp} / TG {tg})", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = arguments()
    run_dir = args.run_dir.resolve()
    results = load_results(run_dir, args.depths)
    write_summary(run_dir / "summary.csv", results, args.depths, args.pp, args.tg)
    plot(run_dir / "prefix-cache.png", results, args.depths, args.pp, args.tg)
    print(f"Clean summary: {run_dir / 'summary.csv'}")
    print(f"Plot:          {run_dir / 'prefix-cache.png'}")


if __name__ == "__main__":
    main()
