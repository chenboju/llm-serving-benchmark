#!/usr/bin/env python3
"""Invoke NeMo-Skills' official RULER evaluator and summarize 13-task accuracy."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_ruler_accuracy import TASKS
from benchmark_matrix import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def match_type(task: str) -> str:
    return "part" if task.startswith("qa_") else "all"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["llama-cpp", "vllm"])
    parser.add_argument("--nemo-skills-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--length", type=int, action="append", choices=[4096, 8192, 16384, 32768])
    args = parser.parse_args()

    config = load_config(args.config)
    expected_revision = config["matrix"]["ruler_accuracy"]["nemo_skills_revision"]
    try:
        actual_revision = subprocess.check_output(
            ["git", "-C", str(args.nemo_skills_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Cannot verify NeMo-Skills revision: {error}") from error
    if actual_revision != expected_revision:
        raise SystemExit(
            f"NeMo-Skills revision is {actual_revision}; expected {expected_revision}"
        )

    sys.path.insert(0, str(args.nemo_skills_root))
    module = importlib.import_module("nemo_skills.evaluation.evaluator.ruler")
    evaluator = module.eval_ruler
    predictions_root = args.predictions_dir or (
        ROOT / "results" / args.backend / "dataset" / "ruler-accuracy"
    )
    lengths = args.length or [4096, 8192, 16384, 32768]
    expected_samples = int(config["matrix"]["ruler_accuracy"]["samples_per_task"])
    rows: list[dict[str, Any]] = []
    for length in lengths:
        for task in TASKS:
            path = predictions_root / str(length) / task / "predictions.jsonl"
            if not path.exists():
                raise SystemExit(f"Missing predictions: {path}")
            evaluator({"input_file": str(path), "match_type": match_type(task)})
            predictions = read_jsonl(path)
            if len(predictions) != expected_samples:
                raise SystemExit(
                    f"Expected {expected_samples} predictions for {length}/{task}, "
                    f"got {len(predictions)}"
                )
            accuracy = statistics.mean(float(row["is_correct"]) for row in predictions) * 100
            rows.append(
                {
                    "backend": args.backend,
                    "length_tokens": length,
                    "task": task,
                    "match_type": match_type(task),
                    "samples": len(predictions),
                    "accuracy_pct": accuracy,
                }
            )

    for length in lengths:
        selected = [row["accuracy_pct"] for row in rows if row["length_tokens"] == length]
        rows.append(
            {
                "backend": args.backend,
                "length_tokens": length,
                "task": "overall-13-task-mean",
                "match_type": "official-group-mean",
                "samples": sum(
                    int(row["samples"])
                    for row in rows
                    if row["length_tokens"] == length and row["task"] in TASKS
                ),
                "accuracy_pct": statistics.mean(selected),
            }
        )

    output = predictions_root / "ruler-accuracy.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (predictions_root / "evaluator-metadata.json").write_text(
        json.dumps(
            {
                "nemo_skills_root": str(args.nemo_skills_root),
                "nemo_skills_revision": actual_revision,
                "samples_per_task": expected_samples,
                "tasks": TASKS,
                "lengths": lengths,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Accuracy: {output}")


if __name__ == "__main__":
    main()
