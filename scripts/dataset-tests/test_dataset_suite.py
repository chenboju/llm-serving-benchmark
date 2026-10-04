#!/usr/bin/env python3
"""Offline regression tests for the dataset benchmark framework."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analyze_dataset_results import (
    build_comparisons,
    build_concurrency_scaling,
    build_prefix_improvement,
    median_ci95,
    telemetry_summary,
)
from benchmark_matrix import expand_matrix, load_config, select_backend_cells
from convert_ruler_for_nemo import main as _unused_convert_main
from prepare_datasets import (
    ApproximateEncoder,
    QAExample,
    load_hotpot,
    load_squad,
    pack_context,
    prepare_prefix,
    prepare_rag_dataset,
    prepare_ruler,
    prepare_sharegpt,
    read_json_records,
)
from plot_dataset_results import (
    plot_accuracy,
    plot_long_context,
    plot_prefix,
    plot_rag,
    plot_sharegpt_closed,
    plot_sharegpt_natural,
    plot_sharegpt_open,
    plot_slo_heatmap,
    setup_style,
)
from run_benchmarks import (
    llama_cache_args,
    log_has_oom,
    required_prefix_cache_mode,
    summarize_cell,
    validate_llama_cache_log,
    write_repeat_csv,
)
from validate_prepared import read_jsonl


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DatasetSuiteTests(unittest.TestCase):
    def test_invalid_repeats_do_not_trigger_variability_retries(self) -> None:
        campaign = {
            "profile_runs": 3,
            "throughput_cv_limit": 0.10,
            "p95_ttft_cv_limit": 0.15,
            "p99_min_samples": 1000,
        }
        cell = next(
            cell
            for cell in expand_matrix(load_config())
            if cell.cell_id == "rag-hotpotqa-b016k-c08"
        )
        summary = summarize_cell(
            cell,
            [{"valid": False, "oom_detected": False} for _ in range(3)],
            campaign,
        )
        self.assertEqual(summary["status"], "invalid")
        self.assertFalse(summary["unstable"])

    def test_ruler_session_ids_are_unique_across_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "4096"
            for task in ("task_a", "task_b"):
                path = source / task / "test.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "index": 0,
                            "question": f"prompt for {task}",
                            "expected_answer": ["answer"],
                            "length": 4096,
                            "task": task,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            output = root / "prepared"
            with patch("prepare_datasets.DEFAULT_MANIFESTS", root / "manifests"):
                prepare_ruler(source, output, ApproximateEncoder())
            rows = [
                json.loads(line)
                for line in (output / "ruler-4096.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len({row["session_id"] for row in rows}), 2)

    def test_llama_cache_off_disables_ram_prompt_cache(self) -> None:
        self.assertEqual(
            llama_cache_args("off"),
            ["--no-cache-prompt", "--cache-ram", "0"],
        )
        validate_llama_cache_log("server is listening", "off")
        with self.assertRaises(RuntimeError):
            validate_llama_cache_log("prompt cache is enabled", "off")
        validate_llama_cache_log("prompt cache is enabled", "on")

    def test_validator_preserves_unicode_line_separator_inside_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared.jsonl"
            path.write_text(
                '{"text":"before\u2028after","output_length":128}\n'
                '{"text":"ok","output_length":128}\n',
                encoding="utf-8",
            )
            records = read_jsonl(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["text"], "before\u2028after")

    def test_jsonl_reader_preserves_unicode_line_separator_inside_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(
                '{"id":"first","text":"before\u2028after"}\n'
                '{"id":"second","text":"ok"}\n',
                encoding="utf-8",
            )
            records = read_json_records(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["text"], "before\u2028after")

    def test_repeat_csv_accepts_failed_then_successful_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "repeat-metrics.csv"
            write_repeat_csv(
                path,
                [
                    {"repeat": 1, "valid": False},
                    {"repeat": 2, "valid": True, "request_count": 100.0},
                ],
            )
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["request_count"], "100.0")

    def test_bootstrap_median_and_telemetry_units(self) -> None:
        median, low, high = median_ci95([10.0, 20.0, 30.0])
        self.assertEqual(median, 20.0)
        self.assertLessEqual(low, median)
        self.assertGreaterEqual(high, median)
        with tempfile.TemporaryDirectory() as temporary:
            repeat = Path(temporary)
            telemetry = repeat / "gpu_telemetry_export.jsonl"
            rows = [
                {
                    "timestamp_ns": 1_000_000_000,
                    "telemetry_data": {
                        "nvidia_power_usage": 100.0,
                        "nvidia_memory_used": 10.0,
                        "nvidia_gpu_utilization": 90.0,
                        "nvidia_temperature": 70.0,
                    },
                },
                {
                    "timestamp_ns": 2_000_000_000,
                    "telemetry_data": {
                        "nvidia_power_usage": 100.0,
                        "nvidia_memory_used": 10.0,
                        "nvidia_gpu_utilization": 100.0,
                        "nvidia_temperature": 75.0,
                    },
                },
            ]
            telemetry.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = telemetry_summary(repeat, request_count=2, output_token_count=200)
            self.assertAlmostEqual(summary["gpu_memory_peak_gib"], 9.313225746, places=6)
            self.assertAlmostEqual(summary["energy_j_per_request"], 50.0)
            self.assertAlmostEqual(summary["energy_j_per_token"], 0.5)

    def test_oom_log_detection_is_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "aiperf.log"
            log.write_text("request failed: CUDA out of memory\n", encoding="utf-8")
            self.assertTrue(log_has_oom(log))
            log.write_text("benchmark completed; headroom is 20%\n", encoding="utf-8")
            self.assertFalse(log_has_oom(log))

    def test_matrix_has_common_and_vllm_stress_cells(self) -> None:
        cells = expand_matrix(load_config())
        self.assertEqual(len(cells), 79)
        self.assertEqual(len({cell.cell_id for cell in cells}), 79)
        self.assertEqual(len(select_backend_cells(cells, "llama-cpp")), 45)
        self.assertEqual(len(select_backend_cells(cells, "vllm")), 79)
        self.assertEqual(sum(cell.scope == "vllm_stress" for cell in cells), 34)
        self.assertTrue(
            all(cell.concurrency <= 16 for cell in select_backend_cells(cells, "llama-cpp"))
        )
        self.assertEqual(
            max(cell.concurrency for cell in select_backend_cells(cells, "vllm")), 64
        )
        self.assertTrue(all((cell.bucket_tokens or 0) + (cell.output_tokens or 0) < 65536 for cell in cells))
        self.assertEqual(
            required_prefix_cache_mode([cell for cell in cells if cell.suite == "rag"]),
            "off",
        )
        self.assertEqual(
            required_prefix_cache_mode(
                [cell for cell in cells if cell.suite == "prefix" and cell.cache_mode == "on"]
            ),
            "on",
        )
        with self.assertRaises(RuntimeError):
            required_prefix_cache_mode([cell for cell in cells if cell.suite == "prefix"])

    def test_sharegpt_is_deterministic_and_capped_to_four_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sharegpt.json"
            conversations = []
            for index in range(6):
                messages = []
                for turn in range(5):
                    messages += [
                        {"from": "human", "value": f"question {index} {turn}"},
                        {"from": "gpt", "value": f"answer {index} {turn} words"},
                    ]
                conversations.append({"id": str(index), "conversations": messages})
            write_json(source, conversations)
            encoder = ApproximateEncoder()
            first = root / "first"
            second = root / "second"
            with patch("prepare_datasets.DEFAULT_MANIFESTS", root / "manifests"):
                prepare_sharegpt(source, first, encoder, count=5, max_turns=4)
                prepare_sharegpt(source, second, encoder, count=5, max_turns=4)
            first_text = (first / "sharegpt-fixed-128.jsonl").read_text(encoding="utf-8")
            second_text = (second / "sharegpt-fixed-128.jsonl").read_text(encoding="utf-8")
            self.assertEqual(first_text, second_text)
            rows = [json.loads(line) for line in first_text.splitlines()]
            self.assertTrue(all(len(row["turns"]) == 4 for row in rows))
            self.assertTrue(all(turn["output_length"] == 128 for row in rows for turn in row["turns"]))

    def test_rag_buckets_keep_question_last(self) -> None:
        encoder = ApproximateEncoder()
        rng = __import__("random").Random(42)
        prompt, _, count = pack_context(
            ["Relevant document contains the answer Taiwan."],
            ["distractor text " * 1200],
            "What contains the answer?",
            1024,
            encoder,
            rng,
        )
        self.assertGreaterEqual(count, int(1024 * 0.95))
        self.assertLessEqual(count, int(1024 * 1.05))
        self.assertTrue(prompt.rstrip().endswith("Answer:"))
        self.assertLess(prompt.rfind("Relevant document"), prompt.rfind("Question:"))

    def test_squad_and_hotpot_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            squad = root / "squad.json"
            write_json(
                squad,
                {
                    "data": [
                        {
                            "title": "T",
                            "paragraphs": [
                                {
                                    "context": "Taipei is in Taiwan.",
                                    "qas": [
                                        {"id": str(i), "question": f"Question {i}?", "answers": [{"text": "Taipei"}]}
                                        for i in range(4)
                                    ],
                                }
                            ],
                        }
                    ]
                },
            )
            hotpot = root / "hotpot.jsonl"
            hotpot.write_text(
                json.dumps(
                    {
                        "id": "h1",
                        "question": "Where?",
                        "answer": "Taiwan",
                        "context": {"title": ["A", "B"], "sentences": [["Taipei is in Taiwan."], ["Noise."]]},
                        "supporting_facts": {"title": ["A"], "sent_id": [0]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            squad_examples = load_squad(squad)
            hotpot_examples = load_hotpot(hotpot)
            self.assertEqual(len(squad_examples), 4)
            self.assertEqual(len(hotpot_examples), 1)
            self.assertIn("Taipei", hotpot_examples[0].relevant_documents[0])

    def test_comparison_and_prefix_math(self) -> None:
        rows = []
        for backend, factor in (("llama-cpp", 1.0), ("vllm", 2.0)):
            for cache, ttft, throughput in (("off", 100.0, 10.0), ("on", 25.0, 15.0)):
                rows.append(
                    {
                        "backend": backend,
                        "cell_id": f"prefix-b004k-{cache}-c01",
                        "suite": "prefix",
                        "workload": "prefix",
                        "concurrency": 1,
                        "bucket_tokens": 4096,
                        "request_rate": float("nan"),
                        "cache_mode": cache,
                        "output_token_throughput_median": throughput * factor,
                        "request_throughput_median": 1 * factor,
                        "ttft_ms_p95": ttft / factor,
                        "itl_ms_p95": 10 / factor,
                        "goodput_req_s_median": 0.5 * factor,
                        "gpu_memory_peak_gib_median": 20,
                        "energy_j_per_request_median": 50 / factor,
                    }
                )
        frame = pd.DataFrame(rows)
        comparisons = build_comparisons(frame)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue((comparisons["output_token_throughput_median_vllm_over_llama"] == 2).all())
        prefix = build_prefix_improvement(frame)
        self.assertTrue((prefix["ttft_reduction_pct"] == 75).all())
        self.assertTrue((prefix["throughput_gain_pct"] == 50).all())

        scaling_frame = pd.DataFrame(
            [
                {
                    "backend": "llama-cpp",
                    "suite": "sharegpt_closed",
                    "cell_id": f"sharegpt-closed-c{concurrency:02d}",
                    "concurrency": concurrency,
                    "output_token_throughput_median": throughput,
                    "request_throughput_median": throughput / 100,
                }
                for concurrency, throughput in ((1, 100.0), (8, 600.0))
            ]
        )
        scaling = build_concurrency_scaling(scaling_frame)
        self.assertEqual(scaling.loc[scaling["concurrency"] == 8, "output_throughput_scaling_x"].iloc[0], 6.0)

    def test_all_chart_functions_write_png(self) -> None:
        rows = []

        def add_row(
            engine: str,
            suite: str,
            cell_id: str,
            concurrency: int,
            bucket: float = float("nan"),
            rate: float = float("nan"),
            cache: str | float = float("nan"),
            scale: float = 1.0,
        ) -> None:
            rows.append(
                {
                    "backend": "vllm" if engine == "vLLM" else "llama-cpp",
                    "engine": engine,
                    "suite": suite,
                    "cell_id": cell_id,
                    "concurrency": concurrency,
                    "bucket_tokens": bucket,
                    "request_rate": rate,
                    "cache_mode": cache,
                    "output_token_throughput_median": 100 * scale,
                    "output_token_throughput_ci95_low": 95 * scale,
                    "output_token_throughput_ci95_high": 105 * scale,
                    "request_throughput_median": 2 * scale,
                    "request_throughput_ci95_low": 1.8 * scale,
                    "request_throughput_ci95_high": 2.2 * scale,
                    "ttft_ms_p95": 800 / scale,
                    "ttft_p95_median": 800 / scale,
                    "ttft_p95_ci95_low": 750 / scale,
                    "ttft_p95_ci95_high": 850 / scale,
                    "itl_ms_p95": 30 / scale,
                    "goodput_req_s_median": 1.5 * scale,
                    "goodput_req_s_ci95_low": 1.3 * scale,
                    "goodput_req_s_ci95_high": 1.7 * scale,
                    "slo_attainment_pct_median": min(100, 80 * scale),
                    "slo_attainment_pct_ci95_low": min(100, 75 * scale),
                    "slo_attainment_pct_ci95_high": min(100, 85 * scale),
                    "gpu_memory_peak_gib_median": 18 + bucket / 16384 if bucket == bucket else 18,
                    "gpu_memory_peak_gib_ci95_low": 17 + bucket / 16384 if bucket == bucket else 17,
                    "gpu_memory_peak_gib_ci95_high": 19 + bucket / 16384 if bucket == bucket else 19,
                    "request_count": 1200,
                    "input_tokens_p50": bucket if bucket == bucket else 512,
                    "output_tokens_p50": 128,
                    "request_error_rate_pct_median": 0.0,
                }
            )

        for engine, scale in (("llama.cpp", 1.0), ("vLLM", 1.5)):
            for concurrency in (1, 2, 4):
                add_row(engine, "sharegpt_closed", f"closed-{concurrency}", concurrency, scale=scale)
            for rate in (0.5, 1, 2):
                add_row(engine, "sharegpt_open", f"open-{rate}", 16, rate=rate, scale=scale)
            for rate in ((4,) if engine == "llama.cpp" else (4, 8, 16)):
                add_row(engine, "sharegpt_natural", f"natural-{rate}", 16 if rate == 4 else 64, rate=rate, scale=scale)
            for dataset in ("squad", "hotpotqa"):
                for bucket in (1024, 4096):
                    for concurrency in (1, 8):
                        add_row(
                            engine,
                            "rag",
                            f"rag-{dataset}-b{bucket}-c{concurrency}",
                            concurrency,
                            bucket=bucket,
                            scale=scale,
                        )
            for bucket in (4096, 16384):
                for cache in ("off", "on"):
                    for concurrency in (1, 8):
                        add_row(
                            engine,
                            "prefix",
                            f"prefix-{bucket}-{cache}-{concurrency}",
                            concurrency,
                            bucket=bucket,
                            cache=cache,
                            scale=scale * (1.2 if cache == "on" else 1),
                        )
            for bucket in (4096, 8192, 16384):
                for concurrency in (1, 2):
                    add_row(
                        engine,
                        "ruler_performance",
                        f"ruler-{bucket}-{concurrency}",
                        concurrency,
                        bucket=bucket,
                        scale=scale,
                    )

        accuracy = pd.DataFrame(
            [
                {
                    "backend": "vllm" if engine == "vLLM" else "llama-cpp",
                    "length_tokens": length,
                    "task": "overall-13-task-mean",
                    "accuracy_pct": 90 if engine == "vLLM" else 88,
                }
                for engine in ("llama.cpp", "vLLM")
                for length in (4096, 8192, 16384)
            ]
        )
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            setup_style()
            plot_sharegpt_closed(frame, output)
            plot_sharegpt_open(frame, output)
            plot_sharegpt_natural(frame, output)
            plot_rag(frame, output)
            plot_prefix(frame, output)
            plot_long_context(frame, output)
            plot_accuracy(accuracy, output)
            plot_slo_heatmap(frame, output)
            self.assertEqual(len(list(output.glob("*.png"))), 8)


if __name__ == "__main__":
    unittest.main()
