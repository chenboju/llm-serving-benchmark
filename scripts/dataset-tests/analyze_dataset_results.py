#!/usr/bin/env python3
"""Pool AIPerf records, validate cells, and produce comparison CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_matrix import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "analysis" / "dataset-output"
GB_TO_GIB = 1_000_000_000 / (1024**3)
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000


METRIC_PREFIXES = {
    "output_token_throughput": "Output Token Throughput (",
    "output_token_throughput_per_user": "Output Token Throughput Per User (",
    "request_throughput": "Request Throughput (",
    "request_count": "Request Count",
    "request_error_rate_pct": "Request Error Rate",
    "benchmark_duration_seconds": "Benchmark Duration",
    "good_request_count": "Good Request Count",
    "input_tokens_avg": "Input Sequence Length",
    "output_tokens_avg": "Output Sequence Length",
}


def find_latest_campaigns(backend: str) -> list[Path]:
    """Return latest comparison, vLLM-stress, and prefix campaigns."""
    root = ROOT / "results" / backend / "dataset"
    candidates = [path.parent for path in root.glob("*/campaign.json")]
    categories: dict[str, list[Path]] = {
        "comparison": [],
        "vllm_stress": [],
        "prefix": [],
    }
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "campaign.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = [str(value) for value in metadata.get("selected_cells", [])]
        scopes = set(str(value) for value in metadata.get("selected_scopes", []))
        if selected and all(value.startswith("prefix-") for value in selected):
            category = "prefix"
        elif scopes == {"vllm_stress"}:
            category = "vllm_stress"
        else:
            category = "comparison"
        categories[category].append(candidate)
    return [
        max(group, key=lambda path: path.stat().st_mtime)
        for group in categories.values()
        if group
    ]


def find_one(root: Path, filename: str) -> Path | None:
    candidates = [path for path in root.rglob(filename) if "warmup" not in path.parts]
    return sorted(candidates)[-1] if candidates else None


def read_metric_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_value(rows: list[dict[str, str]], prefix: str, statistic: str = "avg") -> float:
    for row in rows:
        if row.get("Metric", "").startswith(prefix):
            value = row.get(statistic, "")
            try:
                return float(value)
            except (TypeError, ValueError):
                return math.nan
    return math.nan


def read_records(repeat_dir: Path) -> list[dict[str, Any]]:
    path = find_one(repeat_dir, "profile_export.jsonl")
    if path is None:
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        phase = record.get("metadata", {}).get("benchmark_phase")
        if phase in {None, "profiling"}:
            records.append(record)
    return records


def record_metric(record: dict[str, Any], name: str) -> float:
    value = record.get("metrics", {}).get(name, {}).get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def percentile(values: list[float], q: float) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.percentile(values, q)) if values else math.nan


def median_ci95(values: list[float]) -> tuple[float, float, float]:
    """Return median and a deterministic percentile-bootstrap 95% CI."""
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan, math.nan
    median = statistics.median(values)
    if len(values) == 1:
        return median, math.nan, math.nan
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(array, size=(BOOTSTRAP_RESAMPLES, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return median, float(low), float(high)


def telemetry_summary(
    repeat_dir: Path, request_count: int, output_token_count: float = 0
) -> dict[str, float]:
    path = find_one(repeat_dir, "gpu_telemetry_export.jsonl")
    if path is None:
        return {
            "gpu_utilization_avg_pct": math.nan,
            "gpu_memory_peak_gib": math.nan,
            "gpu_power_avg_w": math.nan,
            "gpu_temperature_peak_c": math.nan,
            "energy_j": math.nan,
            "energy_j_per_request": math.nan,
            "energy_j_per_token": math.nan,
        }
    telemetry = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            telemetry.append({"timestamp_ns": row.get("timestamp_ns"), **row.get("telemetry_data", {})})
    telemetry.sort(key=lambda row: row.get("timestamp_ns") or 0)

    def values(*names: str) -> list[float]:
        found = []
        for row in telemetry:
            for name in names:
                if row.get(name) is not None:
                    found.append(float(row[name]))
                    break
        return found

    power = values("nvidia_power_usage", "gpu_power_usage")
    utilization = values("nvidia_gpu_utilization", "gpu_utilization")
    # AIPerf 0.12 normalizes pynvml/DCGM memory to decimal GB. The report
    # promises binary GiB, so convert explicitly instead of relabeling it.
    memory_gib = [value * GB_TO_GIB for value in values("nvidia_memory_used", "gpu_memory_used")]
    temperature = values("nvidia_temperature", "gpu_temperature")
    energy_j = 0.0
    energy_samples = 0
    for left, right in zip(telemetry, telemetry[1:]):
        left_power = left.get("nvidia_power_usage", left.get("gpu_power_usage"))
        right_power = right.get("nvidia_power_usage", right.get("gpu_power_usage"))
        if left_power is None or right_power is None:
            continue
        dt = (float(right["timestamp_ns"]) - float(left["timestamp_ns"])) / 1e9
        if 0 < dt <= 30:
            energy_j += (float(left_power) + float(right_power)) * 0.5 * dt
            energy_samples += 1
    if energy_samples == 0:
        energy_j = math.nan
    return {
        "gpu_utilization_avg_pct": statistics.mean(utilization) if utilization else math.nan,
        "gpu_memory_peak_gib": max(memory_gib) if memory_gib else math.nan,
        "gpu_power_avg_w": statistics.mean(power) if power else math.nan,
        "gpu_temperature_peak_c": max(temperature) if temperature else math.nan,
        "energy_j": energy_j,
        "energy_j_per_request": energy_j / request_count if request_count and math.isfinite(energy_j) else math.nan,
        "energy_j_per_token": (
            energy_j / output_token_count
            if output_token_count > 0 and math.isfinite(energy_j)
            else math.nan
        ),
    }


def repeat_summary(repeat_dir: Path, slo: dict[str, float]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    csv_path = find_one(repeat_dir, "profile_export_aiperf.csv")
    if csv_path is None:
        raise FileNotFoundError(f"Missing AIPerf summary under {repeat_dir}")
    rows = read_metric_csv(csv_path)
    summary = {
        key: metric_value(rows, prefix) for key, prefix in METRIC_PREFIXES.items()
    }
    records = read_records(repeat_dir)
    summary["request_count"] = summary["request_count"] if math.isfinite(summary["request_count"]) else len(records)
    ttft = [record_metric(record, "time_to_first_token") for record in records]
    itl = [record_metric(record, "inter_token_latency") for record in records]
    latency = [record_metric(record, "request_latency") for record in records]
    input_tokens = [record_metric(record, "input_sequence_length") for record in records]
    output_tokens = [record_metric(record, "output_sequence_length") for record in records]
    for name, values_ in (
        ("ttft", ttft),
        ("itl", itl),
        ("request_latency", latency),
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        summary[f"{name}_p50"] = percentile(values_, 50)
        summary[f"{name}_p95"] = percentile(values_, 95)
        summary[f"{name}_p99"] = percentile(values_, 99)
    good = sum(
        math.isfinite(ttft_value)
        and math.isfinite(itl_value)
        and ttft_value <= slo["ttft_ms"]
        and itl_value <= slo["itl_ms"]
        for ttft_value, itl_value in zip(ttft, itl)
    )
    duration = summary["benchmark_duration_seconds"]
    summary["slo_attainment_pct"] = good / len(records) * 100 if records else math.nan
    summary["goodput_req_s"] = good / duration if duration and math.isfinite(duration) else math.nan
    output_token_count = sum(value for value in output_tokens if math.isfinite(value))
    summary.update(telemetry_summary(repeat_dir, len(records), output_token_count))
    return summary, records


def aggregate_cell(backend: str, campaign: Path, cell_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    validation_path = cell_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    slo = config["slo"][cell["slo_profile"]]
    repeat_validity: dict[int, bool] = {}
    repeat_metrics_path = cell_dir / "repeat-metrics.csv"
    if repeat_metrics_path.exists():
        for item in read_metric_csv(repeat_metrics_path):
            try:
                repeat_number = int(item["repeat"])
            except (KeyError, TypeError, ValueError):
                continue
            repeat_validity[repeat_number] = str(item.get("valid", "")).lower() == "true"
    repeat_rows = []
    pooled_records: list[dict[str, Any]] = []
    for repeat_dir in sorted(cell_dir.glob("repeat-*")):
        if not repeat_dir.is_dir():
            continue
        try:
            repeat_number = int(repeat_dir.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if repeat_validity and not repeat_validity.get(repeat_number, False):
            continue
        try:
            summary, records = repeat_summary(repeat_dir, slo)
        except FileNotFoundError:
            continue
        if summary["request_error_rate_pct"] <= config["campaign"]["max_error_rate_pct"]:
            repeat_rows.append(summary)
            pooled_records.extend(records)
    required_repeats = int(config["campaign"]["profile_runs"])
    if len(repeat_rows) < required_repeats:
        oom_repeats = int(validation.get("oom_repeats", 0) or 0)
        return {
            "backend": backend,
            "campaign": campaign.name,
            **cell,
            "status": "invalid",
            "failure_reason": "oom" if oom_repeats else "errors-or-insufficient-valid-repeats",
            "valid_repeats": len(repeat_rows),
            "invalid_repeats": int(validation.get("invalid_repeats", 0) or 0),
            "oom_repeats": oom_repeats,
            "request_count": len(pooled_records),
            "p99_eligible": False,
        }

    row: dict[str, Any] = {
        "backend": backend,
        "campaign": campaign.name,
        **cell,
        "status": validation.get("status", "unvalidated"),
        "valid_repeats": len(repeat_rows),
        "invalid_repeats": int(validation.get("invalid_repeats", 0) or 0),
        "oom_repeats": int(validation.get("oom_repeats", 0) or 0),
        "failure_reason": "",
        "request_count": len(pooled_records),
        "p99_eligible": len(pooled_records) >= config["campaign"]["p99_min_samples"],
    }
    repeat_metrics = [
        "output_token_throughput",
        "output_token_throughput_per_user",
        "request_throughput",
        "request_error_rate_pct",
        "goodput_req_s",
        "slo_attainment_pct",
        "ttft_p95",
        "itl_p95",
        "request_latency_p95",
        "input_tokens_p50",
        "output_tokens_p50",
        "gpu_utilization_avg_pct",
        "gpu_memory_peak_gib",
        "gpu_power_avg_w",
        "gpu_temperature_peak_c",
        "energy_j_per_request",
        "energy_j_per_token",
    ]
    for metric in repeat_metrics:
        values = [float(repeat[metric]) for repeat in repeat_rows]
        finite = [value for value in values if math.isfinite(value)]
        median, low, high = median_ci95(finite)
        row[f"{metric}_median"] = median
        row[f"{metric}_mean"] = statistics.mean(finite) if finite else math.nan
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high

    pooled = {
        "ttft_ms": [record_metric(record, "time_to_first_token") for record in pooled_records],
        "itl_ms": [record_metric(record, "inter_token_latency") for record in pooled_records],
        "request_latency_ms": [record_metric(record, "request_latency") for record in pooled_records],
        "input_tokens": [record_metric(record, "input_sequence_length") for record in pooled_records],
        "output_tokens": [record_metric(record, "output_sequence_length") for record in pooled_records],
    }
    for metric, values in pooled.items():
        row[f"{metric}_p50"] = percentile(values, 50)
        row[f"{metric}_p95"] = percentile(values, 95)
        row[f"{metric}_p99"] = percentile(values, 99) if row["p99_eligible"] else math.nan
    return row


def build_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    if "scope" in summary:
        summary = summary[summary["scope"] == "comparison"].copy()
    if summary.empty:
        return pd.DataFrame()
    keys = ["cell_id", "suite", "workload", "concurrency", "bucket_tokens", "request_rate", "cache_mode"]
    metrics = [
        "output_token_throughput_median",
        "output_token_throughput_per_user_median",
        "request_throughput_median",
        "ttft_ms_p95",
        "itl_ms_p95",
        "goodput_req_s_median",
        "gpu_memory_peak_gib_median",
        "energy_j_per_request_median",
        "energy_j_per_token_median",
    ]
    available = [metric for metric in metrics if metric in summary]
    metadata = summary[keys].drop_duplicates("cell_id")
    llama = summary[summary["backend"] == "llama-cpp"][["cell_id", *available]].rename(
        columns={metric: f"{metric}_llama_cpp" for metric in available}
    )
    vllm = summary[summary["backend"] == "vllm"][["cell_id", *available]].rename(
        columns={metric: f"{metric}_vllm" for metric in available}
    )
    merged = metadata.merge(llama, on="cell_id", how="left").merge(vllm, on="cell_id", how="left")
    for metric in available:
        denominator = merged[f"{metric}_llama_cpp"]
        merged[f"{metric}_vllm_over_llama"] = merged[f"{metric}_vllm"] / denominator.replace(0, np.nan)
    return merged


def build_concurrency_scaling(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cn/C1-style scaling within otherwise identical conditions."""
    if summary.empty:
        return pd.DataFrame()
    frame = summary.copy()
    frame["scaling_condition"] = frame["cell_id"].map(
        lambda value: re.sub(r"-c\d+$", "", str(value))
    )
    rows: list[dict[str, Any]] = []
    for (backend, condition), group in frame.groupby(
        ["backend", "scaling_condition"], dropna=False
    ):
        group = group.sort_values("concurrency")
        if group["concurrency"].nunique() < 2:
            continue
        baseline = group.iloc[0]
        for _, current in group.iterrows():
            output_base = baseline.get("output_token_throughput_median")
            request_base = baseline.get("request_throughput_median")
            rows.append(
                {
                    "backend": backend,
                    "suite": current.get("suite"),
                    "scaling_condition": condition,
                    "cell_id": current.get("cell_id"),
                    "baseline_concurrency": baseline.get("concurrency"),
                    "concurrency": current.get("concurrency"),
                    "output_throughput_scaling_x": (
                        current.get("output_token_throughput_median") / output_base
                        if pd.notna(output_base) and output_base != 0
                        else math.nan
                    ),
                    "request_throughput_scaling_x": (
                        current.get("request_throughput_median") / request_base
                        if pd.notna(request_base) and request_base != 0
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_prefix_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary[summary["suite"] == "prefix"].copy()
    if subset.empty:
        return pd.DataFrame()
    rows = []
    for (backend, bucket, concurrency), group in subset.groupby(
        ["backend", "bucket_tokens", "concurrency"], dropna=False
    ):
        modes = group.set_index("cache_mode")
        if "off" not in modes.index or "on" not in modes.index:
            continue
        off, on = modes.loc["off"], modes.loc["on"]
        rows.append(
            {
                "backend": backend,
                "bucket_tokens": bucket,
                "concurrency": concurrency,
                "ttft_off_ms": off["ttft_ms_p95"],
                "ttft_on_ms": on["ttft_ms_p95"],
                "ttft_reduction_pct": (off["ttft_ms_p95"] - on["ttft_ms_p95"]) / off["ttft_ms_p95"] * 100,
                "throughput_off_tok_s": off["output_token_throughput_median"],
                "throughput_on_tok_s": on["output_token_throughput_median"],
                "throughput_gain_pct": (on["output_token_throughput_median"] / off["output_token_throughput_median"] - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


def report_markdown(summary: pd.DataFrame, prefix: pd.DataFrame, accuracy: pd.DataFrame) -> str:
    lines = [
        "# llama.cpp 與 vLLM 資料集基準報告",
        "",
        "本報告以每格 3 次有效測量為基準；變異過大時使用 5 次。P95 是主要尾端指標，P99 僅在 pooled request count ≥ 1,000 時顯示。",
        "",
        "## 有效性",
        "",
    ]
    if summary.empty:
        lines += ["尚未找到正式 dataset campaign 結果。", ""]
        return "\n".join(lines)
    validity = summary.groupby(["backend", "status"], dropna=False).size().reset_index(name="cells")
    lines += [validity.to_markdown(index=False), "", "## 核心結果", ""]
    view_columns = [
        "backend",
        "cell_id",
        "output_token_throughput_median",
        "ttft_p95_median",
        "itl_p95_median",
        "goodput_req_s_median",
        "slo_attainment_pct_median",
        "request_count",
        "p99_eligible",
    ]
    view = summary[[column for column in view_columns if column in summary]].copy()
    lines += [view.to_markdown(index=False, floatfmt=".2f"), ""]
    if "scope" in summary:
        stress = summary[
            (summary["backend"] == "vllm") & (summary["scope"] == "vllm_stress")
        ]
        if not stress.empty:
            stress_columns = [
                "cell_id",
                "status",
                "concurrency",
                "request_rate",
                "output_token_throughput_median",
                "ttft_p95_median",
                "goodput_req_s_median",
                "oom_repeats",
                "failure_reason",
            ]
            lines += [
                "## vLLM 高壓容量區",
                "",
                "這些格不參與兩引擎公平勝負；invalid/OOM 表示目前部署的容量邊界。",
                "",
                stress[[column for column in stress_columns if column in stress]].to_markdown(
                    index=False, floatfmt=".2f"
                ),
                "",
            ]
    if not prefix.empty:
        lines += ["## Prefix cache 改善", "", prefix.to_markdown(index=False, floatfmt=".2f"), ""]
    if not accuracy.empty:
        overall = accuracy[accuracy["task"] == "overall-13-task-mean"]
        lines += ["## RULER 正確率", "", overall.to_markdown(index=False, floatfmt=".2f"), ""]
    lines += [
        "## 單位",
        "",
        "- TTFT、ITL、request latency：ms；ITL 表示每個 token gap。",
        "- Throughput：token/s 或 req/s；goodput 是同時符合 TTFT 與 ITL SLO 的 req/s。",
        "- GPU memory：GiB；power：W；energy efficiency：J/request、J/output token。",
        "- RULER accuracy、SLO attainment、error rate、cache improvement：%。",
        "- P50/P95/P99 是分位數，沿用原指標單位。",
        "",
        "RAG 結果只包含已組裝 context 的模型推論，不包含 retrieval、reranker、資料庫與網路延遲。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--llama-campaign", type=Path, action="append")
    parser.add_argument("--vllm-campaign", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(args.config)
    campaigns = {
        "llama-cpp": args.llama_campaign or find_latest_campaigns("llama-cpp"),
        "vllm": args.vllm_campaign or find_latest_campaigns("vllm"),
    }
    rows = []
    for backend, backend_campaigns in campaigns.items():
        for campaign in backend_campaigns:
            if not campaign.is_dir():
                raise SystemExit(f"Campaign directory not found: {campaign}")
            for cell_dir in sorted(campaign.iterdir()):
                if cell_dir.is_dir() and (cell_dir / "cell.json").exists():
                    row = aggregate_cell(backend, campaign, cell_dir, config)
                    row["_campaign_mtime"] = campaign.stat().st_mtime_ns
                    rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = (
            summary.sort_values("_campaign_mtime")
            .drop_duplicates(["backend", "cell_id"], keep="last")
            .drop(columns="_campaign_mtime")
            .reset_index(drop=True)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "dataset-cell-summary.csv", index=False, encoding="utf-8")
    vllm_stress = (
        summary[(summary.get("backend") == "vllm") & (summary.get("scope") == "vllm_stress")]
        if not summary.empty and "scope" in summary
        else pd.DataFrame()
    )
    vllm_stress.to_csv(
        args.output_dir / "vllm-stress-summary.csv", index=False, encoding="utf-8"
    )
    comparisons = build_comparisons(summary)
    comparisons.to_csv(args.output_dir / "engine-comparisons.csv", index=False, encoding="utf-8")
    scaling = build_concurrency_scaling(summary)
    scaling.to_csv(args.output_dir / "concurrency-scaling.csv", index=False, encoding="utf-8")
    prefix = build_prefix_improvement(summary) if not summary.empty else pd.DataFrame()
    prefix.to_csv(args.output_dir / "prefix-cache-improvement.csv", index=False, encoding="utf-8")

    accuracy_files = [
        ROOT / "results" / backend / "dataset" / "ruler-accuracy" / "ruler-accuracy.csv"
        for backend in ("llama-cpp", "vllm")
    ]
    accuracy_frames = [pd.read_csv(path) for path in accuracy_files if path.exists()]
    accuracy = pd.concat(accuracy_frames, ignore_index=True) if accuracy_frames else pd.DataFrame()
    accuracy.to_csv(args.output_dir / "ruler-accuracy.csv", index=False, encoding="utf-8")
    (args.output_dir / "dataset-benchmark-report.md").write_text(
        report_markdown(summary, prefix, accuracy), encoding="utf-8"
    )
    print(f"Cells: {len(summary)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
