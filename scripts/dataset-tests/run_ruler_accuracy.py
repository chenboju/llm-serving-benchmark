#!/usr/bin/env python3
"""Generate natural-EOS predictions for official RULERv1 tasks."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_matrix import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[2]
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_indices(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("index")) for row in read_jsonl(path) if row.get("generation") is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["llama-cpp", "vllm"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "datasets" / "raw" / "ruler-official"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--length", type=int, action="append", choices=[4096, 8192, 16384, 32768])
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    backend = config["backends"][args.backend]
    lengths = args.length or config["matrix"]["ruler_accuracy"]["lengths"]
    tasks = args.task or TASKS
    output_root = args.output_dir or (
        ROOT / "results" / args.backend / "dataset" / "ruler-accuracy"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "backend": args.backend,
        "backend_config": backend,
        "started_at": utc_now(),
        "lengths": lengths,
        "tasks": tasks,
        "temperature": 0,
        "max_completion_tokens": args.max_completion_tokens,
        "natural_eos": True,
        "ruler_branch": config["matrix"]["ruler_accuracy"]["branch"],
        "ruler_revision": config["matrix"]["ruler_accuracy"]["revision"],
    }
    (output_root / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    endpoint = backend["url"].rstrip("/") + "/v1/chat/completions"
    expected_samples = int(config["matrix"]["ruler_accuracy"]["samples_per_task"])
    total = 0
    for length in lengths:
        for task in tasks:
            path = args.data_dir / str(length) / task / "test.jsonl"
            if not path.exists():
                raise SystemExit(f"Missing official RULER task data: {path}")
            count = len(read_jsonl(path))
            if count != expected_samples:
                raise SystemExit(
                    f"Expected {expected_samples} examples for {length}/{task}, got {count}"
                )
            total += count
    completed = 0
    for length in lengths:
        for task in tasks:
            source = args.data_dir / str(length) / task / "test.jsonl"
            destination = output_root / str(length) / task / "predictions.jsonl"
            if destination.exists() and not args.resume:
                raise SystemExit(f"Output exists; use --resume or remove it: {destination}")
            done = completed_indices(destination) if args.resume else set()
            for row in read_jsonl(source):
                completed += 1
                if str(row.get("index")) in done:
                    continue
                started = time.perf_counter()
                response = post_json(
                    endpoint,
                    {
                        "model": backend["model"],
                        "messages": [{"role": "user", "content": row["question"]}],
                        "max_completion_tokens": args.max_completion_tokens,
                        "temperature": 0,
                        "stream": False,
                    },
                    backend["api_key"],
                    args.timeout_seconds,
                )
                generation = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                output = {
                    **row,
                    "generation": generation,
                    "backend": args.backend,
                    "request_latency_ms": (time.perf_counter() - started) * 1000,
                    "usage_prompt_tokens": usage.get("prompt_tokens"),
                    "usage_completion_tokens": usage.get("completion_tokens"),
                }
                append_jsonl(destination, output)
                print(f"[{completed}/{total}] {args.backend} length={length} task={task} index={row.get('index')}")

    metadata["completed_at"] = utc_now()
    (output_root / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Predictions: {output_root}")


if __name__ == "__main__":
    main()
