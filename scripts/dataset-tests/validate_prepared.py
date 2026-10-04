#!/usr/bin/env python3
"""Validate prepared AIPerf JSONL and its reproducibility sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_matrix import DEFAULT_CONFIG, expand_matrix, load_config, select_cells


ROOT = Path(__file__).resolve().parents[2]
SINGLE_KEYS = {
    "text",
    "texts",
    "image",
    "images",
    "audio",
    "audios",
    "video",
    "videos",
    "timestamp",
    "delay",
    "role",
    "session_id",
    "output_length",
    "extra",
}
MULTI_KEYS = {"session_id", "turns"}
EXACT_BUCKET_DATASETS = {"squad", "hotpotqa", "squad-prefix"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: row must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def validate_single_row(row: dict[str, Any], location: str) -> None:
    unknown = set(row) - SINGLE_KEYS
    if unknown:
        raise ValueError(f"{location}: unsupported AIPerf keys: {sorted(unknown)}")
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"{location}: non-empty text is required")
    if row.get("output_length") is not None and (
        not isinstance(row["output_length"], int) or row["output_length"] <= 0
    ):
        raise ValueError(f"{location}: output_length must be a positive integer")
    if row.get("extra") is not None and not isinstance(row["extra"], dict):
        raise ValueError(f"{location}: extra must be an object")


def validate_dataset(path: Path, dataset_type: str, formal: bool) -> dict[str, Any]:
    rows = read_jsonl(path)
    if dataset_type == "single_turn":
        for index, row in enumerate(rows):
            validate_single_row(row, f"{path}:{index + 1}")
    elif dataset_type == "multi_turn":
        for index, row in enumerate(rows):
            unknown = set(row) - MULTI_KEYS
            if unknown:
                raise ValueError(f"{path}:{index + 1}: unsupported multi-turn keys: {unknown}")
            if not row.get("session_id") or not isinstance(row.get("turns"), list):
                raise ValueError(f"{path}:{index + 1}: session_id and turns are required")
            if not 1 <= len(row["turns"]) <= 4:
                raise ValueError(f"{path}:{index + 1}: expected 1..4 turns")
            for turn_index, turn in enumerate(row["turns"]):
                validate_single_row(turn, f"{path}:{index + 1}:turn {turn_index}")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    index_path = path.with_suffix(".index.jsonl")
    if not index_path.exists():
        raise ValueError(f"Missing index sidecar: {index_path}")
    index_rows = read_jsonl(index_path)
    if len(index_rows) != len(rows):
        raise ValueError(f"{path}: {len(rows)} rows but {len(index_rows)} index rows")

    manifest_path = ROOT / "datasets" / "manifests" / f"{path.stem}.manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError(f"SHA256 mismatch: {path}")
    if manifest.get("index_sha256") != sha256_file(index_path):
        raise ValueError(f"SHA256 mismatch: {index_path}")
    tokenizer_kind = str(manifest.get("tokenizer", {}).get("kind", ""))
    if formal and tokenizer_kind == "approximate":
        raise ValueError(f"{path}: approximate tokenization is not valid for formal runs")

    dataset_name = str(manifest.get("dataset", ""))
    bucket = manifest.get("bucket_tokens")
    actual_counts = [row.get("actual_input_tokens") for row in index_rows]
    actual_counts = [int(value) for value in actual_counts if isinstance(value, int)]
    # RAG/prefix prompts are packed to a target bucket and must stay within
    # +/-5%. RULER lengths are official maximum sequence-length settings;
    # individual generated tasks may legitimately be shorter.
    if dataset_name in EXACT_BUCKET_DATASETS and bucket and actual_counts:
        lower, upper = int(bucket * 0.95), int(bucket * 1.05)
        outside = [value for value in actual_counts if not lower <= value <= upper]
        if outside:
            raise ValueError(
                f"{path}: {len(outside)} token counts outside {lower}..{upper}; first={outside[0]}"
            )

    if manifest.get("dataset") == "squad-prefix":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in index_rows:
            groups[str(row["group_id"])].append(row)
        expected = int(manifest["questions_per_group"])
        if len(groups) != int(manifest["groups"]):
            raise ValueError(f"{path}: prefix group count mismatch")
        for group_id, group_rows in groups.items():
            orders = sorted(int(row["turn_order"]) for row in group_rows)
            if orders != list(range(expected)):
                raise ValueError(f"{path}: invalid order for {group_id}: {orders}")
            roles = Counter(str(row["cache_role"]) for row in group_rows)
            if roles != Counter({"cold": 1, "warm": expected - 1}):
                raise ValueError(f"{path}: invalid cold/warm roles for {group_id}")

    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "tokenizer": manifest.get("tokenizer"),
        "bucket_tokens": bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", action="append")
    parser.add_argument("--cell", action="append")
    parser.add_argument("--allow-approximate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    cells = select_cells(expand_matrix(config), args.suite, args.cell)
    unique: dict[tuple[str, str], None] = {}
    for cell in cells:
        unique[(cell.dataset, cell.dataset_type)] = None

    results = []
    for relative, dataset_type in unique:
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"Missing prepared dataset: {path}")
        results.append(validate_dataset(path, dataset_type, formal=not args.allow_approximate))

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"OK {result['path']} rows={result['rows']} sha256={result['sha256'][:12]}")
        print(f"Validated {len(results)} unique file(s) for {len(cells)} condition(s).")


if __name__ == "__main__":
    main()
