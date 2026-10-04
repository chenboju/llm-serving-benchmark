#!/usr/bin/env python3
"""Capture the immutable model, software, container and GPU environment (DGX Spark).

Ported from the RTX 4090 version. The vLLM container and HF snapshot sections are
kept because both backends run here. Added: CPU architecture and CUDA compute
capability, because this host is aarch64 Grace Blackwell rather than x86_64 Ada,
and the llama.cpp build identity, because an aarch64 CUDA build is not
interchangeable with the 4090's x86_64 build even at the same tag.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gpu_telemetry
from benchmark_matrix import DEFAULT_CONFIG, load_config
from validate_prepared import sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "datasets" / "manifests" / "environment.json"


def capture(command: list[str], timeout: float = 60) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        return f"unavailable: {error}"
    return result.stdout.strip()


def git_revision(path: Path) -> str:
    if not path.exists():
        return "missing"
    output = capture(["git", "-C", str(path), "rev-parse", "HEAD"])
    return output.splitlines()[-1] if output else "unknown"


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_identity(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "status": "missing", "files": []}
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        identity = file_identity(item)
        identity["relative_path"] = item.relative_to(path).as_posix()
        files.append(identity)
    return {
        "path": str(path),
        "revision": path.name,
        "file_count": len(files),
        "bytes": sum(int(item.get("bytes", 0)) for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vllm-container", default="gpt_oss_20b_vllm")
    parser.add_argument("--llama-model", type=Path, default=None)
    parser.add_argument("--vllm-snapshot", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    home = Path.home()
    workspace = Path(os.environ.get("WORKSPACE_ROOT", home / "LLM_infra"))
    llama_model = args.llama_model or (
        workspace / "models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf"
    )
    vllm_snapshot = args.vllm_snapshot or Path(config["backends"]["vllm"]["tokenizer"])
    aiperf_root = Path(os.environ.get("AIPERF_ROOT", workspace / "aiperf"))
    llama_cpp_root = Path(os.environ.get("LLAMA_CPP_ROOT", workspace / "llama.cpp"))
    llama_benchy_root = Path(os.environ.get("LLAMA_BENCHY_ROOT", workspace / "llama-benchy"))
    llama_server = Path(os.environ.get("LLAMA_SERVER", llama_cpp_root / "build/bin/llama-server"))
    telemetry_backend = gpu_telemetry.detect_backend()

    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os": capture(["uname", "-a"]),
            "telemetry_backend": telemetry_backend,
        },
        "gpu": gpu_telemetry.describe(telemetry_backend),
        "gpu_compute_capability": capture(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"]
        ),
        "software": {
            "aiperf_version": capture([str(aiperf_root / ".venv/bin/aiperf"), "--version"]),
            "aiperf_commit": git_revision(aiperf_root),
            "llama_cpp_commit": git_revision(llama_cpp_root),
            "llama_server_version": capture([str(llama_server), "--version"]),
            "llama_server_devices": capture([str(llama_server), "--list-devices"]),
            "llama_benchy_commit": git_revision(llama_benchy_root),
            "llama_benchy_version": capture(
                [str(llama_benchy_root / ".venv/bin/llama-benchy"), "--version"]
            ),
            "nvcc": capture(["nvcc", "--version"]),
            "python": capture([str(aiperf_root / ".venv/bin/python"), "--version"]),
            "pip_freeze": capture([str(aiperf_root / ".venv/bin/python"), "-m", "pip", "freeze"]),
        },
        "models": {
            "llama_cpp_gguf": file_identity(llama_model),
            "vllm_hf_snapshot": tree_identity(vllm_snapshot),
        },
        "vllm_container": {
            "name": args.vllm_container,
            "image": capture(
                ["docker", "inspect", "--format", "{{.Config.Image}}", args.vllm_container]
            ),
            "image_id": capture(
                ["docker", "inspect", "--format", "{{.Image}}", args.vllm_container]
            ),
            "architecture": capture(
                ["docker", "inspect", "--format", "{{.Architecture}}", args.vllm_container]
            ),
            "command": capture(
                ["docker", "inspect", "--format", "{{json .Config.Cmd}}", args.vllm_container]
            ),
        },
        "llama_server_config": config.get("llama_server", {}),
        "pinned_revisions": config["matrix"]["ruler_accuracy"],
        "benchmark_config_sha256": sha256_file(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Environment manifest: {args.output}")
    if platform.machine() not in {"aarch64", "arm64"}:
        print(f"Warning: expected an aarch64 host, got {platform.machine()!r}.")
    if telemetry_backend != "nvidia-smi":
        print(f"Warning: telemetry backend is {telemetry_backend!r}, expected 'nvidia-smi'.")


if __name__ == "__main__":
    main()
