#!/usr/bin/env python3
"""Convert pinned RULER generator output to NeMo-Skills' official schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def find_task_file(source_dir: Path, task: str) -> Path:
    exact = source_dir / task / "test.jsonl"
    if exact.exists():
        return exact
    candidates = [
        path
        for path in source_dir.rglob("*.jsonl")
        if task in str(path.parent) and path.name == "test.jsonl"
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one generated test.jsonl for {task}; got {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=100,
        help="Expected sample count for each task (default: 100)",
    )
    args = parser.parse_args()

    if args.expected_samples <= 0:
        parser.error("--expected-samples must be greater than zero")

    for task in TASKS:
        source = find_task_file(args.source_dir, task)
        output = args.output_dir / task / "test.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open(encoding="utf-8") as input_handle, output.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            count = 0
            for line in input_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                converted = {
                    "index": record["index"],
                    "question": record["input"] + record.get("answer_prefix", ""),
                    "expected_answer": record["outputs"],
                    "length": record.get("length", args.length),
                    "task": task,
                }
                output_handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
        if count != args.expected_samples:
            raise ValueError(
                f"{task}: expected {args.expected_samples} official samples, got {count}"
            )
        print(f"{task}: {count} samples -> {output}")


if __name__ == "__main__":
    main()
