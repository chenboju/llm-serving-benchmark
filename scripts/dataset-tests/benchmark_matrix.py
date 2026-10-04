#!/usr/bin/env python3
"""Expand and validate the shared comparison plus vLLM stress matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("benchmark-config.json")


@dataclass(frozen=True)
class BenchmarkCell:
    cell_id: str
    suite: str
    workload: str
    dataset: str
    dataset_type: str
    concurrency: int
    slo_profile: str
    output_tokens: int | None = None
    bucket_tokens: int | None = None
    request_rate: float | None = None
    arrival_pattern: str | None = None
    cache_mode: str | None = None
    fixed_output: bool = True
    scope: str = "comparison"
    backends: tuple[str, ...] = ("llama-cpp", "vllm")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema: {config.get('schema_version')!r}")
    return config


def _rate_label(value: float) -> str:
    return str(value).replace(".", "p")


def _bucket_label(tokens: int) -> str:
    return f"{tokens // 1024:03d}k"


def expand_matrix(config: dict[str, Any]) -> list[BenchmarkCell]:
    matrix = config["matrix"]
    datasets = config["datasets"]
    cells: list[BenchmarkCell] = []

    spec = matrix["sharegpt_closed"]
    for concurrency in spec["concurrency"]:
        cells.append(
            BenchmarkCell(
                cell_id=f"sharegpt-closed-c{concurrency:02d}",
                suite="sharegpt_closed",
                workload="sharegpt",
                dataset=datasets["sharegpt_fixed"],
                dataset_type="multi_turn",
                concurrency=concurrency,
                slo_profile="chat",
                output_tokens=spec["output_tokens"],
            )
        )
    for concurrency in spec.get("vllm_stress_concurrency", []):
        cells.append(
            BenchmarkCell(
                cell_id=f"sharegpt-closed-c{concurrency:02d}",
                suite="sharegpt_closed",
                workload="sharegpt",
                dataset=datasets["sharegpt_fixed"],
                dataset_type="multi_turn",
                concurrency=concurrency,
                slo_profile="chat",
                output_tokens=spec["output_tokens"],
                scope="vllm_stress",
                backends=("vllm",),
            )
        )

    spec = matrix["sharegpt_open"]
    for rate in spec["request_rate"]:
        cells.append(
            BenchmarkCell(
                cell_id=f"sharegpt-open-r{_rate_label(rate)}-c{spec['concurrency']:02d}",
                suite="sharegpt_open",
                workload="sharegpt",
                dataset=datasets["sharegpt_fixed"],
                dataset_type="multi_turn",
                concurrency=spec["concurrency"],
                request_rate=float(rate),
                arrival_pattern=spec["arrival_pattern"],
                slo_profile="chat",
                output_tokens=spec["output_tokens"],
            )
        )
    for rate in spec.get("vllm_stress_request_rate", []):
        concurrency = int(spec["vllm_stress_concurrency"])
        cells.append(
            BenchmarkCell(
                cell_id=f"sharegpt-open-r{_rate_label(rate)}-c{concurrency:02d}",
                suite="sharegpt_open",
                workload="sharegpt",
                dataset=datasets["sharegpt_fixed"],
                dataset_type="multi_turn",
                concurrency=concurrency,
                request_rate=float(rate),
                arrival_pattern=spec["arrival_pattern"],
                slo_profile="chat",
                output_tokens=spec["output_tokens"],
                scope="vllm_stress",
                backends=("vllm",),
            )
        )

    spec = matrix["sharegpt_natural"]
    cells.append(
        BenchmarkCell(
            cell_id=f"sharegpt-natural-r{_rate_label(spec['request_rate'])}-c{spec['concurrency']:02d}",
            suite="sharegpt_natural",
            workload="sharegpt",
            dataset=datasets["sharegpt_natural"],
            dataset_type="multi_turn",
            concurrency=spec["concurrency"],
            request_rate=float(spec["request_rate"]),
            arrival_pattern=spec["arrival_pattern"],
            slo_profile="chat",
            fixed_output=False,
        )
    )
    for rate in spec.get("vllm_stress_request_rate", []):
        concurrency = int(spec["vllm_stress_concurrency"])
        cells.append(
            BenchmarkCell(
                cell_id=f"sharegpt-natural-r{_rate_label(rate)}-c{concurrency:02d}",
                suite="sharegpt_natural",
                workload="sharegpt",
                dataset=datasets["sharegpt_natural"],
                dataset_type="multi_turn",
                concurrency=concurrency,
                request_rate=float(rate),
                arrival_pattern=spec["arrival_pattern"],
                slo_profile="chat",
                fixed_output=False,
                scope="vllm_stress",
                backends=("vllm",),
            )
        )

    spec = matrix["rag"]
    for dataset, bucket, concurrency in product(
        spec["dataset"], spec["bucket_tokens"], spec["concurrency"]
    ):
        cells.append(
            BenchmarkCell(
                cell_id=f"rag-{dataset}-b{_bucket_label(bucket)}-c{concurrency:02d}",
                suite="rag",
                workload="rag",
                dataset=datasets["rag_pattern"].format(dataset=dataset, bucket=bucket),
                dataset_type="single_turn",
                concurrency=concurrency,
                bucket_tokens=bucket,
                slo_profile="rag",
                output_tokens=spec["output_tokens"],
            )
        )
    for dataset, bucket, concurrency in product(
        spec["dataset"], spec["bucket_tokens"], spec.get("vllm_stress_concurrency", [])
    ):
        cells.append(
            BenchmarkCell(
                cell_id=f"rag-{dataset}-b{_bucket_label(bucket)}-c{concurrency:02d}",
                suite="rag",
                workload="rag",
                dataset=datasets["rag_pattern"].format(dataset=dataset, bucket=bucket),
                dataset_type="single_turn",
                concurrency=concurrency,
                bucket_tokens=bucket,
                slo_profile="rag",
                output_tokens=spec["output_tokens"],
                scope="vllm_stress",
                backends=("vllm",),
            )
        )

    spec = matrix["prefix"]
    for bucket, cache_mode, concurrency in product(
        spec["bucket_tokens"], spec["cache_mode"], spec["concurrency"]
    ):
        cells.append(
            BenchmarkCell(
                cell_id=f"prefix-b{_bucket_label(bucket)}-{cache_mode}-c{concurrency:02d}",
                suite="prefix",
                workload="prefix",
                dataset=datasets["prefix_pattern"].format(
                    bucket=bucket, cache_mode=cache_mode
                ),
                dataset_type="single_turn",
                concurrency=concurrency,
                bucket_tokens=bucket,
                cache_mode=cache_mode,
                slo_profile="rag",
                output_tokens=spec["output_tokens"],
            )
        )
    for bucket, cache_mode, concurrency in product(
        spec["bucket_tokens"],
        spec["cache_mode"],
        spec.get("vllm_stress_concurrency", []),
    ):
        cells.append(
            BenchmarkCell(
                cell_id=f"prefix-b{_bucket_label(bucket)}-{cache_mode}-c{concurrency:02d}",
                suite="prefix",
                workload="prefix",
                dataset=datasets["prefix_pattern"].format(
                    bucket=bucket, cache_mode=cache_mode
                ),
                dataset_type="single_turn",
                concurrency=concurrency,
                bucket_tokens=bucket,
                cache_mode=cache_mode,
                slo_profile="rag",
                output_tokens=spec["output_tokens"],
                scope="vllm_stress",
                backends=("vllm",),
            )
        )

    spec = matrix["ruler_performance"]
    for bucket, concurrency in product(spec["bucket_tokens"], spec["concurrency"]):
        cells.append(
            BenchmarkCell(
                cell_id=f"ruler-perf-b{_bucket_label(bucket)}-c{concurrency:02d}",
                suite="ruler_performance",
                workload="long_context",
                dataset=datasets["ruler_pattern"].format(bucket=bucket),
                dataset_type="single_turn",
                concurrency=concurrency,
                bucket_tokens=bucket,
                slo_profile="long_context",
                output_tokens=spec["output_tokens"],
            )
        )
    for bucket in spec["bucket_tokens"]:
        for concurrency in spec.get("vllm_stress_concurrency_by_bucket", {}).get(
            str(bucket), []
        ):
            cells.append(
                BenchmarkCell(
                    cell_id=f"ruler-perf-b{_bucket_label(bucket)}-c{concurrency:02d}",
                    suite="ruler_performance",
                    workload="long_context",
                    dataset=datasets["ruler_pattern"].format(bucket=bucket),
                    dataset_type="single_turn",
                    concurrency=concurrency,
                    bucket_tokens=bucket,
                    slo_profile="long_context",
                    output_tokens=spec["output_tokens"],
                    scope="vllm_stress",
                    backends=("vllm",),
                )
            )

    validate_cells(cells, config)
    return cells


def validate_cells(cells: Iterable[BenchmarkCell], config: dict[str, Any]) -> None:
    cells = list(cells)
    ids = [cell.cell_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark cell IDs are not unique")
    comparison = [cell for cell in cells if cell.scope == "comparison"]
    stress = [cell for cell in cells if cell.scope == "vllm_stress"]
    if len(comparison) != 45 or len(stress) != 34:
        raise ValueError(
            f"Expected 45 comparison + 34 vLLM stress conditions, got "
            f"{len(comparison)} + {len(stress)}"
        )
    for cell in cells:
        if not cell.backends or not set(cell.backends) <= set(config["backends"]):
            raise ValueError(f"Invalid backend scope in {cell.cell_id}: {cell.backends}")
        maximum = min(int(config["backends"][backend]["max_concurrency"]) for backend in cell.backends)
        if cell.concurrency < 1 or cell.concurrency > maximum:
            raise ValueError(f"Invalid concurrency in {cell.cell_id}")
        if cell.bucket_tokens and cell.output_tokens:
            if cell.bucket_tokens + cell.output_tokens >= 65536:
                raise ValueError(f"Context safety margin violated by {cell.cell_id}")
        if cell.slo_profile not in config["slo"]:
            raise ValueError(f"Unknown SLO profile in {cell.cell_id}")


def select_cells(
    cells: Iterable[BenchmarkCell], suites: list[str] | None, cell_ids: list[str] | None
) -> list[BenchmarkCell]:
    selected = list(cells)
    if suites:
        selected = [cell for cell in selected if cell.suite in suites]
    if cell_ids:
        wanted = set(cell_ids)
        selected = [cell for cell in selected if cell.cell_id in wanted]
        missing = wanted - {cell.cell_id for cell in selected}
        if missing:
            raise ValueError("Unknown cell IDs: " + ", ".join(sorted(missing)))
    return selected


def select_backend_cells(
    cells: Iterable[BenchmarkCell], backend: str, scopes: list[str] | None = None
) -> list[BenchmarkCell]:
    selected = [cell for cell in cells if backend in cell.backends]
    if scopes:
        selected = [cell for cell in selected if cell.scope in scopes]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", action="append", help="Filter by suite; repeatable")
    parser.add_argument("--cell", action="append", help="Filter by cell ID; repeatable")
    parser.add_argument("--backend", choices=["llama-cpp", "vllm"])
    parser.add_argument("--scope", action="append", choices=["comparison", "vllm_stress"])
    parser.add_argument("--json", action="store_true", help="Emit JSON Lines")
    args = parser.parse_args()

    config = load_config(args.config)
    cells = select_cells(expand_matrix(config), args.suite, args.cell)
    if args.backend:
        cells = select_backend_cells(cells, args.backend, args.scope)
    elif args.scope:
        cells = [cell for cell in cells if cell.scope in args.scope]
    for cell in cells:
        if args.json:
            print(json.dumps(asdict(cell), ensure_ascii=False, sort_keys=True))
        else:
            rate = "-" if cell.request_rate is None else f"{cell.request_rate:g} req/s"
            bucket = "-" if cell.bucket_tokens is None else str(cell.bucket_tokens)
            print(
                f"{cell.cell_id:38s} suite={cell.suite:19s} "
                f"c={cell.concurrency:2d} rate={rate:9s} bucket={bucket} scope={cell.scope}"
            )
    print(f"# {len(cells)} condition(s)")


if __name__ == "__main__":
    main()
