#!/usr/bin/env python3
"""Download, freeze, tokenize, and materialize AIPerf dataset JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "datasets" / "raw"
DEFAULT_PREPARED = ROOT / "datasets" / "prepared"
DEFAULT_MANIFESTS = ROOT / "datasets" / "manifests"
BUCKETS = (1024, 4096, 8192, 16384)
SEED = 42

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"

RAG_SYSTEM = (
    "You are a retrieval-augmented question answering assistant. "
    "Answer from the supplied documents. If the answer is not present, say so."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, value: str) -> str:
    short = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{short}"


class TokenEncoder:
    name: str
    kind: str

    def encode(self, text: str) -> list[int] | list[str]:
        raise NotImplementedError

    def decode(self, tokens: list[int] | list[str]) -> str:
        raise NotImplementedError

    def count(self, text: str) -> int:
        return len(self.encode(text))


class ApproximateEncoder(TokenEncoder):
    """Test-only tokenizer; formal manifests reject this unless explicitly allowed."""

    def __init__(self) -> None:
        self.name = "approximate-whitespace"
        self.kind = "approximate"

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def decode(self, tokens: list[int] | list[str]) -> str:
        return " ".join(str(token) for token in tokens)


class HuggingFaceEncoder(TokenEncoder):
    def __init__(self, name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SystemExit(
                "transformers is required for formal preparation. Install "
                "scripts/dataset-tests/requirements.txt or use "
                "--approximate-tokenizer only for tests."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True, local_files_only=Path(name_or_path).exists()
        )
        self.name = str(getattr(self.tokenizer, "name_or_path", name_or_path))
        self.kind = self.tokenizer.__class__.__name__

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int] | list[str]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def make_encoder(name_or_path: str, approximate: bool) -> TokenEncoder:
    return ApproximateEncoder() if approximate else HuggingFaceEncoder(name_or_path)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported JSON structure: {path}")


def atomic_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing file: {destination}")
        return
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temp:
        temp_path = Path(temp.name)
    try:
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response, temp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def download_sources(raw_dir: Path, hotpot_revision: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sharegpt = raw_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"
    squad = raw_dir / "squad-train-v1.1.json"
    hotpot = raw_dir / "hotpotqa-distractor-train.jsonl"
    atomic_download(SHAREGPT_URL, sharegpt)
    atomic_download(SQUAD_URL, squad)

    if not hotpot.exists():
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SystemExit("datasets is required to download HotpotQA") from exc
        print("Downloading hotpotqa/hotpot_qa (distractor/train)")
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="train",
            revision=hotpot_revision,
        )
        with hotpot.open("w", encoding="utf-8", newline="\n") as handle:
            for record in dataset:
                handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (sharegpt, squad, hotpot)
        },
        "hotpot_revision_requested": hotpot_revision,
    }
    write_json(DEFAULT_MANIFESTS / "raw-download.json", manifest)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def output_manifest(
    output: Path,
    index: Path,
    encoder: TokenEncoder,
    source_paths: list[Path],
    extra: dict[str, Any],
) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path.resolve())

    with output.open(encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "output": display_path(output),
        "output_sha256": sha256_file(output),
        "index": display_path(index),
        "index_sha256": sha256_file(index),
        "rows": row_count,
        "tokenizer": {"name": encoder.name, "kind": encoder.kind},
        "sources": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        **extra,
    }
    manifest_path = DEFAULT_MANIFESTS / f"{output.stem}.manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def sharegpt_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    messages = record.get("conversations") or record.get("messages") or []
    pairs: list[tuple[str, str]] = []
    for index in range(len(messages) - 1):
        current, following = messages[index], messages[index + 1]
        current_role = str(current.get("from") or current.get("role") or "").lower()
        next_role = str(following.get("from") or following.get("role") or "").lower()
        if current_role not in {"human", "user"} or next_role not in {"gpt", "assistant"}:
            continue
        user_text = str(current.get("value") or current.get("content") or "").strip()
        assistant_text = str(following.get("value") or following.get("content") or "").strip()
        if user_text and assistant_text:
            pairs.append((user_text, assistant_text))
    return pairs


def prepare_sharegpt(
    source: Path,
    output_dir: Path,
    encoder: TokenEncoder,
    count: int = 5000,
    max_turns: int = 4,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    records = read_json_records(source)
    eligible = [(record, sharegpt_pairs(record)) for record in records]
    eligible = [(record, pairs[:max_turns]) for record, pairs in eligible if pairs]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:count]
    if len(selected) < count:
        raise ValueError(f"ShareGPT has only {len(selected)} eligible conversations; need {count}")

    fixed_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    fixed_index: list[dict[str, Any]] = []
    natural_index: list[dict[str, Any]] = []
    for position, (record, pairs) in enumerate(selected):
        source_id = str(record.get("id") or record.get("conversation_id") or position)
        session_id = stable_id("sharegpt", source_id + pairs[0][0])
        fixed_turns: list[dict[str, Any]] = []
        natural_turns: list[dict[str, Any]] = []
        turn_meta: list[dict[str, Any]] = []
        for turn_index, (prompt, answer) in enumerate(pairs):
            natural_osl = max(1, encoder.count(answer))
            fixed_turns.append(
                {
                    "text": prompt,
                    "output_length": 128,
                    "extra": {"temperature": 0, "min_tokens": 128, "ignore_eos": True},
                }
            )
            natural_turns.append(
                {
                    "text": prompt,
                    "output_length": natural_osl,
                    "extra": {"temperature": 0},
                }
            )
            turn_meta.append(
                {
                    "turn_index": turn_index,
                    "prompt_tokens": encoder.count(prompt),
                    "recorded_assistant_tokens": natural_osl,
                }
            )
        fixed_rows.append({"session_id": session_id, "turns": fixed_turns})
        natural_rows.append({"session_id": session_id, "turns": natural_turns})
        base_meta = {"sample_id": session_id, "dataset": "sharegpt", "turns": turn_meta}
        fixed_index.append({**base_meta, "output_mode": "fixed", "output_tokens": 128})
        natural_index.append({**base_meta, "output_mode": "recorded_distribution"})

    outputs = [
        ("sharegpt-fixed-128.jsonl", fixed_rows, fixed_index, "fixed-128"),
        ("sharegpt-natural-osl.jsonl", natural_rows, natural_index, "recorded-osl"),
    ]
    manifests: list[dict[str, Any]] = []
    for filename, rows, index_rows, mode in outputs:
        output = output_dir / filename
        index = output.with_suffix(".index.jsonl")
        write_jsonl(output, rows)
        write_jsonl(index, index_rows)
        manifests.append(
            output_manifest(
                output,
                index,
                encoder,
                [source],
                {"dataset": "sharegpt", "seed": seed, "max_turns": max_turns, "mode": mode},
            )
        )
    return manifests


@dataclass
class QAExample:
    sample_id: str
    question: str
    relevant_documents: list[str]
    all_documents: list[str]
    answers: list[str]
    group_questions: list[str]


def load_squad(path: Path) -> list[QAExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    articles = payload.get("data", payload) if isinstance(payload, dict) else payload
    examples: list[QAExample] = []
    for article in articles:
        title = str(article.get("title", "SQuAD"))
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = str(paragraph.get("context", "")).strip()
            questions = [str(qa.get("question", "")).strip() for qa in paragraph.get("qas", [])]
            questions = [question for question in questions if question]
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                if not question or not context:
                    continue
                answers = [str(item.get("text", "")) for item in qa.get("answers", [])]
                sample_id = str(qa.get("id") or stable_id("squad", question + context[:200]))
                examples.append(
                    QAExample(
                        sample_id=sample_id,
                        question=question,
                        relevant_documents=[f"{title}\n{context}"],
                        all_documents=[context],
                        answers=answers,
                        group_questions=questions,
                    )
                )
    return examples


def _hotpot_documents(record: dict[str, Any]) -> list[tuple[str, str]]:
    context = record.get("context", [])
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [
            (str(title), " ".join(str(sentence) for sentence in body))
            for title, body in zip(titles, sentences)
        ]
    documents: list[tuple[str, str]] = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            body = item[1]
            if isinstance(body, list):
                body = " ".join(str(sentence) for sentence in body)
            documents.append((str(item[0]), str(body)))
    return documents


def load_hotpot(path: Path) -> list[QAExample]:
    records = read_json_records(path)
    examples: list[QAExample] = []
    for position, record in enumerate(records):
        documents = _hotpot_documents(record)
        if not documents:
            continue
        supporting = record.get("supporting_facts", {})
        if isinstance(supporting, dict):
            support_titles = {str(title) for title in supporting.get("title", [])}
        else:
            support_titles = {
                str(item[0]) for item in supporting if isinstance(item, (list, tuple)) and item
            }
        relevant = [f"{title}\n{text}" for title, text in documents if title in support_titles]
        if not relevant:
            relevant = [f"{title}\n{text}" for title, text in documents[:2]]
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        sample_id = str(record.get("id") or record.get("_id") or position)
        answer = str(record.get("answer", ""))
        examples.append(
            QAExample(
                sample_id=sample_id,
                question=question,
                relevant_documents=relevant,
                all_documents=[f"{title}\n{text}" for title, text in documents],
                answers=[answer] if answer else [],
                group_questions=[question],
            )
        )
    return examples


def render_rag_prompt(context: str, question: str) -> str:
    return (
        f"System:\n{RAG_SYSTEM}\n\n"
        f"Documents:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def make_document_block(documents: list[str]) -> str:
    return "\n\n".join(f"[Document {index + 1}]\n{text}" for index, text in enumerate(documents))


def pack_context(
    relevant_documents: list[str],
    distractors: list[str],
    question: str,
    target_tokens: int,
    encoder: TokenEncoder,
    rng: random.Random,
) -> tuple[str, str, int]:
    relevant = make_document_block(relevant_documents)
    base_prompt = render_rag_prompt(relevant, question)
    base_tokens = encoder.count(base_prompt)
    upper = int(target_tokens * 1.05)
    lower = int(target_tokens * 0.95)
    if base_tokens > upper:
        raise ValueError(f"Relevant document already has {base_tokens} tokens for {target_tokens} bucket")

    pool = [item for item in distractors if item.strip()]
    if not pool:
        pool = relevant_documents
    rng.shuffle(pool)
    filler_parts: list[str] = []
    cycle = 0
    while True:
        source = pool[cycle % len(pool)]
        filler_parts.append(f"[Distractor {cycle + 1}]\n{source}")
        candidate_context = relevant + "\n\n" + "\n\n".join(filler_parts)
        if encoder.count(render_rag_prompt(candidate_context, question)) >= target_tokens:
            break
        cycle += 1
        if cycle > 2000:
            raise RuntimeError("Unable to fill target context")

    filler = "\n\n".join(filler_parts)
    low, high = 0, len(filler)
    best_context, best_count = relevant, base_tokens
    while low <= high:
        middle = (low + high) // 2
        candidate_context = relevant + ("\n\n" + filler[:middle] if middle else "")
        candidate_prompt = render_rag_prompt(candidate_context, question)
        count = encoder.count(candidate_prompt)
        if count <= target_tokens:
            best_context, best_count = candidate_context, count
            low = middle + 1
        else:
            high = middle - 1
    prompt = render_rag_prompt(best_context, question)
    if not lower <= best_count <= upper:
        raise ValueError(
            f"Packed prompt has {best_count} tokens, outside {lower}..{upper} for {target_tokens}"
        )
    return prompt, best_context, best_count


def prepare_rag_dataset(
    name: str,
    examples: list[QAExample],
    source: Path,
    output_dir: Path,
    encoder: TokenEncoder,
    per_bucket: int = 250,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + (0 if name == "squad" else 1))
    shuffled = examples[:]
    rng.shuffle(shuffled)
    distractor_pool = [doc for example in shuffled for doc in example.all_documents]
    manifests: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        for example in shuffled:
            try:
                prompt, _, actual_tokens = pack_context(
                    example.relevant_documents,
                    distractor_pool,
                    example.question,
                    bucket,
                    encoder,
                    rng,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "text": prompt,
                    "session_id": f"{name}-{example.sample_id}-b{bucket}",
                    "output_length": 256,
                    "extra": {"temperature": 0, "min_tokens": 256, "ignore_eos": True},
                }
            )
            index_rows.append(
                {
                    "sample_id": example.sample_id,
                    "dataset": name,
                    "bucket_tokens": bucket,
                    "actual_input_tokens": actual_tokens,
                    "answers": example.answers,
                    "question_is_last": prompt.rstrip().endswith("Answer:"),
                }
            )
            if len(rows) == per_bucket:
                break
        if len(rows) < per_bucket:
            raise ValueError(f"{name}/{bucket}: produced {len(rows)}, need {per_bucket}")
        output = output_dir / f"{name}-{bucket}.jsonl"
        index = output.with_suffix(".index.jsonl")
        write_jsonl(output, rows)
        write_jsonl(index, index_rows)
        manifests.append(
            output_manifest(
                output,
                index,
                encoder,
                [source],
                {
                    "dataset": name,
                    "seed": seed,
                    "bucket_tokens": bucket,
                    "tolerance_pct": 5,
                    "output_tokens": 256,
                },
            )
        )
    return manifests


def prepare_prefix(
    squad_source: Path,
    output_dir: Path,
    encoder: TokenEncoder,
    groups: int = 100,
    questions_per_group: int = 4,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    examples = load_squad(squad_source)
    by_context: dict[str, QAExample] = {}
    for example in examples:
        key = example.relevant_documents[0]
        if len(example.group_questions) >= questions_per_group:
            by_context.setdefault(key, example)
    candidates = list(by_context.values())
    rng = random.Random(seed)
    rng.shuffle(candidates)
    distractor_pool = [example.relevant_documents[0] for example in examples]
    manifests: list[dict[str, Any]] = []

    for bucket in (4096, 16384):
        on_rows: list[dict[str, Any]] = []
        off_rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        for group_index, example in enumerate(candidates[:groups]):
            questions = example.group_questions[:questions_per_group]
            _, shared_context, _ = pack_context(
                example.relevant_documents,
                distractor_pool,
                questions[0],
                bucket,
                encoder,
                rng,
            )
            group_id = f"squad-prefix-{bucket}-{group_index:04d}"
            for turn_order, question in enumerate(questions):
                prompt = render_rag_prompt(shared_context, question)
                actual_tokens = encoder.count(prompt)
                if not int(bucket * 0.95) <= actual_tokens <= int(bucket * 1.05):
                    raise ValueError(f"Prefix prompt out of bucket: {actual_tokens} vs {bucket}")
                common_row = {
                    "session_id": group_id,
                    "output_length": 128,
                    "extra": {"temperature": 0, "min_tokens": 128, "ignore_eos": True},
                }
                on_rows.append(
                    {
                        **common_row,
                        "text": prompt,
                    }
                )
                off_rows.append(
                    {
                        **common_row,
                        "text": f"[cache-bust:{group_id}-q{turn_order}]\n{prompt}",
                    }
                )
                index_rows.append(
                    {
                        "sample_id": f"{group_id}-q{turn_order}",
                        "dataset": "squad-prefix",
                        "group_id": group_id,
                        "turn_order": turn_order,
                        "cache_role": "cold" if turn_order == 0 else "warm",
                        "bucket_tokens": bucket,
                        "actual_input_tokens": actual_tokens,
                    }
                )
        if len(candidates) < groups:
            raise ValueError(f"Only {len(candidates)} SQuAD contexts have {questions_per_group} questions")
        for cache_mode, rows in (("on", on_rows), ("off", off_rows)):
            output = output_dir / f"squad-prefix-{bucket}-{cache_mode}.jsonl"
            index = output.with_suffix(".index.jsonl")
            write_jsonl(output, rows)
            write_jsonl(
                index,
                (
                    {
                        **metadata,
                        "actual_input_tokens": encoder.count(dataset_row["text"]),
                        "cache_mode": cache_mode,
                    }
                    for metadata, dataset_row in zip(index_rows, rows)
                ),
            )
            manifests.append(
                output_manifest(
                    output,
                    index,
                    encoder,
                    [squad_source],
                    {
                        "dataset": "squad-prefix",
                        "seed": seed,
                        "bucket_tokens": bucket,
                        "groups": groups,
                        "questions_per_group": questions_per_group,
                        "prompt_order": "system-context-question",
                        "cache_mode": cache_mode,
                        "cache_off_method": "unique-token-zero-marker" if cache_mode == "off" else None,
                    },
                )
            )
    return manifests


def infer_ruler_length(path: Path, record: dict[str, Any]) -> int | None:
    for key in ("length", "max_seq_length", "context_length"):
        value = record.get(key)
        if isinstance(value, int) and value in {4096, 8192, 16384, 32768, 61440}:
            return value
    text = str(path).lower()
    for value in (61440, 32768, 16384, 8192, 4096):
        if str(value) in text or f"{value // 1024}k" in text:
            return value
    return None


def prepare_ruler(
    source_dir: Path,
    output_dir: Path,
    encoder: TokenEncoder,
    default_length: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[dict[str, Any], Path]]] = {}
    for path in sorted(source_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
                prompt = (
                    record.get("question")
                    or record.get("input")
                    or record.get("prompt")
                    or record.get("text")
                )
                if not isinstance(prompt, str) or not prompt:
                    continue
                length = infer_ruler_length(path, record) or default_length
                if length:
                    grouped.setdefault(length, []).append((record, path))

    manifests: list[dict[str, Any]] = []
    for length in (4096, 8192, 16384, 32768, 61440):
        entries = grouped.get(length, [])
        if not entries:
            continue
        rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        source_paths: set[Path] = set()
        for position, (record, source_path) in enumerate(entries):
            prompt = str(
                record.get("question")
                or record.get("input")
                or record.get("prompt")
                or record.get("text")
            )
            actual_tokens = encoder.count(prompt)
            source_paths.add(source_path)
            task = str(record.get("task") or source_path.parent.name)
            source_id = record.get("index")
            if source_id is None:
                source_id = record.get("id")
            if source_id is None:
                source_id = position
            sample_id = f"{task}-{source_id}"
            rows.append(
                {
                    "text": prompt,
                    "session_id": f"ruler-{length}-{sample_id}",
                    "output_length": 128,
                    "extra": {"temperature": 0, "min_tokens": 128, "ignore_eos": True},
                }
            )
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": "ruler",
                    "task": task,
                    "bucket_tokens": length,
                    "actual_input_tokens": actual_tokens,
                    "expected": (
                        record.get("expected_answer")
                        or record.get("outputs")
                        or record.get("output")
                        or record.get("answer")
                    ),
                    "source_file": str(source_path),
                }
            )
        output = output_dir / f"ruler-{length}.jsonl"
        index = output.with_suffix(".index.jsonl")
        write_jsonl(output, rows)
        write_jsonl(index, index_rows)
        manifests.append(
            output_manifest(
                output,
                index,
                encoder,
                sorted(source_paths),
                {"dataset": "ruler", "bucket_tokens": length, "mode": "performance-import"},
            )
        )
    if not manifests:
        raise ValueError(f"No RULER JSONL prompts found under {source_dir}")
    return manifests


def add_common_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--approximate-tokenizer",
        action="store_true",
        help="Use whitespace counts for unit tests only; never for formal results",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and hash public sources")
    download.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    download.add_argument("--hotpot-revision", default="main")

    sharegpt = subparsers.add_parser("sharegpt", help="Prepare fixed and recorded-OSL ShareGPT")
    sharegpt.add_argument("--source", type=Path, required=True)
    sharegpt.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED / "sharegpt")
    sharegpt.add_argument("--count", type=int, default=5000)
    sharegpt.add_argument("--max-turns", type=int, default=4)
    add_common_tokenizer_args(sharegpt)

    rag = subparsers.add_parser("rag", help="Prepare SQuAD and HotpotQA RAG buckets")
    rag.add_argument("--squad-source", type=Path, required=True)
    rag.add_argument("--hotpot-source", type=Path, required=True)
    rag.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED / "rag")
    rag.add_argument("--per-bucket", type=int, default=250)
    rag.add_argument(
        "--dataset",
        choices=["all", "squad", "hotpotqa"],
        default="all",
        help="Prepare both datasets or resume only one of them (default: all)",
    )
    add_common_tokenizer_args(rag)

    prefix = subparsers.add_parser("prefix", help="Prepare grouped SQuAD cache prompts")
    prefix.add_argument("--squad-source", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED / "prefix")
    prefix.add_argument("--groups", type=int, default=100)
    prefix.add_argument("--questions-per-group", type=int, default=4)
    add_common_tokenizer_args(prefix)

    ruler = subparsers.add_parser("ruler", help="Import official prepared RULER JSONL")
    ruler.add_argument("--source-dir", type=Path, required=True)
    ruler.add_argument("--output-dir", type=Path, default=DEFAULT_PREPARED / "ruler")
    ruler.add_argument("--default-length", type=int, choices=[4096, 8192, 16384, 32768, 61440])
    add_common_tokenizer_args(ruler)

    args = parser.parse_args()
    if args.command == "download":
        download_sources(args.raw_dir, args.hotpot_revision)
        return

    encoder = make_encoder(args.tokenizer, args.approximate_tokenizer)
    if args.command == "sharegpt":
        manifests = prepare_sharegpt(
            args.source, args.output_dir, encoder, args.count, args.max_turns
        )
    elif args.command == "rag":
        manifests = []
        if args.dataset in {"all", "squad"}:
            manifests += prepare_rag_dataset(
                "squad",
                load_squad(args.squad_source),
                args.squad_source,
                args.output_dir,
                encoder,
                args.per_bucket,
            )
        if args.dataset in {"all", "hotpotqa"}:
            manifests += prepare_rag_dataset(
                "hotpotqa",
                load_hotpot(args.hotpot_source),
                args.hotpot_source,
                args.output_dir,
                encoder,
                args.per_bucket,
            )
    elif args.command == "prefix":
        manifests = prepare_prefix(
            args.squad_source,
            args.output_dir,
            encoder,
            args.groups,
            args.questions_per_group,
        )
    elif args.command == "ruler":
        manifests = prepare_ruler(
            args.source_dir, args.output_dir, encoder, args.default_length
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
