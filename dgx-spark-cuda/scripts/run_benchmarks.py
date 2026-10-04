#!/usr/bin/env python3
"""Run reproducible AIPerf dataset cells against llama.cpp or vLLM on DGX Spark.

Rebased on the RTX 4090 harness (canonical copy, 2026-08-14). DGX Spark is
Linux + CUDA + Docker, so the vLLM backend, the container prefix-cache checks,
the llama.cpp RAM-prompt-cache guard and the POSIX process handling all carry
over unchanged. Three things differ:

  * GPU counters go through gpu_telemetry, which on this host detects nvidia-smi
    and reproduces the 4090 behaviour exactly, including --gpu-telemetry pynvml.
    The indirection exists so all machines share one harness shape.
  * llama-server flags are config-driven rather than hardcoded, because the
    aarch64 CUDA build must be verified on the machine (probe_llama_server.sh)
    and because the values actually used belong in the run manifest.
  * run_capture() tolerates a missing executable instead of raising.

Matrix, seed, durations, repeat/stability rules and artifact layout are
unchanged, so results pair up with the 4090 campaign cell by cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import gpu_telemetry
from benchmark_matrix import (
    DEFAULT_CONFIG,
    BenchmarkCell,
    backend_family,
    expand_matrix,
    load_config,
    select_backend_cells,
    select_cells,
)
from validate_prepared import sha256_file, validate_dataset


ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_capture(command: list[str], timeout: float = 30) -> str:
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
    value = run_capture(["git", "-C", str(path), "rev-parse", "HEAD"])
    return value.splitlines()[-1] if value else "unknown"


def get_json(url: str, api_key: str, timeout: float = 10) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float = 60) -> dict[str, Any]:
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


def endpoint_preflight(backend: dict[str, Any], smoke_requests: int = 10) -> dict[str, Any]:
    url = backend["url"].rstrip("/")
    api_key = backend["api_key"]
    health_request = urllib.request.Request(f"{url}/health")
    with urllib.request.urlopen(health_request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Health check returned HTTP {response.status}")
    models = get_json(f"{url}/v1/models", api_key)
    model_ids = [item.get("id") for item in models.get("data", [])]
    if backend["model"] not in model_ids:
        raise RuntimeError(f"Expected model {backend['model']!r}; endpoint serves {model_ids}")
    for index in range(smoke_requests):
        response = post_json(
            f"{url}/v1/chat/completions",
            {
                "model": backend["model"],
                "messages": [{"role": "user", "content": f"Reply OK. smoke={index}"}],
                "max_completion_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            api_key,
        )
        if not response.get("choices"):
            raise RuntimeError(f"Smoke request {index + 1} returned no choices")
    return {"smoke_requests": smoke_requests, "model_ids": model_ids, "completed_at": utc_now()}


def wait_for_gpu_idle(config: dict[str, Any], telemetry_backend: str) -> dict[str, Any]:
    """Gate the next repeat on an idle GPU.

    On DGX Spark this uses nvidia-smi and matches the 4090 behaviour exactly.
    The fixed-cooldown branch only fires on a host with no telemetry at all; if
    it ever fires here the driver stack is not visible and the run should be
    stopped rather than silently weakened.
    """
    if telemetry_backend == "none":
        cooldown = float(config.get("gpu_idle_fallback_cooldown_seconds", 60))
        time.sleep(cooldown)
        return {"gate": "fixed-cooldown", "cooldown_seconds": cooldown}

    threshold = float(config["gpu_idle_utilization_pct"])
    stable_seconds = float(config["gpu_idle_stable_seconds"])
    timeout = float(config["gpu_idle_timeout_seconds"])
    started = time.monotonic()
    stable_started: float | None = None
    latest: dict[str, float] = {}
    while time.monotonic() - started < timeout:
        latest = gpu_telemetry.query_gpu(telemetry_backend)
        if latest["utilization_pct"] < threshold:
            stable_started = stable_started or time.monotonic()
            if time.monotonic() - stable_started >= stable_seconds:
                return {"gate": telemetry_backend, **latest}
        else:
            stable_started = None
        time.sleep(1)
    raise TimeoutError(f"GPU did not remain below {threshold}% for {stable_seconds}s; latest={latest}")


def detect_vllm_prefix_mode(container: str) -> str:
    command = run_capture(["docker", "inspect", "--format", "{{json .Config.Cmd}}", container])
    if "--enable-prefix-caching" in command:
        return "on"
    if "--no-enable-prefix-caching" in command:
        return "off"
    return "unknown"


def required_prefix_cache_mode(cells: list[BenchmarkCell]) -> str:
    """Return the server cache mode required by a homogeneous invocation."""
    has_prefix = any(cell.suite == "prefix" for cell in cells)
    has_other = any(cell.suite != "prefix" for cell in cells)
    if has_prefix and has_other:
        raise RuntimeError(
            "Run prefix and non-prefix suites separately because they require different "
            "server prompt-cache modes."
        )
    modes = {str(cell.cache_mode) for cell in cells if cell.suite == "prefix"}
    if len(modes) > 1:
        raise RuntimeError(
            "Run prefix cache-off and cache-on cells separately; use "
            "--suite prefix --cache-mode off/on."
        )
    return next(iter(modes)) if modes else "off"


def validate_backend_mode(
    backend_name: str, cells: list[BenchmarkCell], container: str
) -> str:
    expected = required_prefix_cache_mode(cells)
    if backend_name != "vllm":
        return expected
    mode = detect_vllm_prefix_mode(container)
    if mode != expected:
        flag = "--enable-prefix-caching" if expected == "on" else "--no-enable-prefix-caching"
        raise RuntimeError(
            f"vLLM prefix-cache mode is {mode!r}; selected suite requires {expected!r}. "
            f"Recreate {container} with {flag}."
        )
    return expected


def llama_cache_args(prefix_cache_mode: str) -> list[str]:
    """Return flags for both llama.cpp slot reuse and RAM prompt caching."""
    if prefix_cache_mode == "on":
        return ["--cache-prompt"]
    if prefix_cache_mode == "off":
        return ["--no-cache-prompt", "--cache-ram", "0"]
    raise ValueError(f"Unsupported llama.cpp prefix cache mode: {prefix_cache_mode}")


def validate_llama_cache_log(log_text: str, expected_mode: str) -> None:
    cache_enabled = "prompt cache is enabled" in log_text
    if expected_mode == "off" and cache_enabled:
        raise RuntimeError(
            "llama.cpp RAM prompt cache is enabled during a cache-off run; "
            "the server must be started with --cache-ram 0"
        )
    if expected_mode == "on" and not cache_enabled:
        raise RuntimeError(
            "llama.cpp RAM prompt cache is not enabled during a cache-on run"
        )


def resolve_llama_server() -> Path:
    override = os.environ.get("LLAMA_SERVER")
    if override:
        return Path(override)
    workspace = Path(os.environ.get("WORKSPACE_ROOT", Path.home() / "LLM_infra"))
    return workspace / "llama.cpp/build/bin/llama-server"


def build_llama_server_command(
    backend: dict[str, Any], server: dict[str, Any], prefix_cache_mode: str
) -> list[str]:
    """Assemble the llama-server command from config rather than hardcoded flags."""
    executable = resolve_llama_server()
    workspace = Path(os.environ.get("WORKSPACE_ROOT", Path.home() / "LLM_infra"))
    model = Path(
        os.environ.get("MODEL_FILE", workspace / "models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf")
    )
    if not executable.is_file() or not model.is_file():
        raise FileNotFoundError(f"llama-server/model missing: {executable}, {model}")
    command = [
        str(executable),
        "--model",
        str(model),
        "--alias",
        backend["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(urllib.parse.urlparse(backend["url"]).port or 8080),
        "--n-gpu-layers",
        str(server.get("n_gpu_layers", 99)),
    ]
    flash_attn = str(server.get("flash_attn", "on"))
    if flash_attn != "unset":
        command += ["--flash-attn", flash_attn]
    command += [
        "--ctx-size",
        str(server.get("ctx_size", 65536)),
        "--parallel",
        str(server.get("parallel", 16)),
    ]
    if server.get("kv_unified", True):
        command.append("--kv-unified")
    command += llama_cache_args(prefix_cache_mode)
    command += [str(argument) for argument in server.get("extra_args", [])]
    return command


@contextmanager
def managed_llama_server(
    enabled: bool,
    backend: dict[str, Any],
    server: dict[str, Any],
    log_path: Path,
    prefix_cache_mode: str = "off",
) -> Iterator[list[str] | None]:
    if not enabled:
        yield None
        return
    command = build_llama_server_command(backend, server, prefix_cache_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + float(server.get("startup_timeout_seconds", 900))
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with {process.returncode}; see {log_path}. "
                        "An immediate exit usually means the aarch64 CUDA build rejected a "
                        "flag (check llama_server.flash_attn) or ran out of memory."
                    )
                try:
                    with urllib.request.urlopen(backend["url"].rstrip("/") + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("llama-server readiness timeout")
            validate_llama_cache_log(
                log_path.read_text(encoding="utf-8", errors="replace"),
                prefix_cache_mode,
            )
            yield command
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def find_metric_csv(artifact_dir: Path) -> Path:
    candidates = sorted(artifact_dir.rglob("profile_export_aiperf.csv"))
    candidates = [path for path in candidates if "warmup" not in path.parts]
    if not candidates:
        raise FileNotFoundError(f"No profile_export_aiperf.csv under {artifact_dir}")
    return candidates[-1]


def metric_value(rows: list[dict[str, str]], prefix: str, statistic: str = "avg") -> float:
    for row in rows:
        if row.get("Metric", "").startswith(prefix):
            value = row.get(statistic, "")
            return float(value) if value not in {"", "N/A", None} else math.nan
    return math.nan


def extract_repeat_metrics(artifact_dir: Path) -> dict[str, float]:
    csv_path = find_metric_csv(artifact_dir)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "output_token_throughput": metric_value(rows, "Output Token Throughput ("),
        "request_throughput": metric_value(rows, "Request Throughput ("),
        "request_count": metric_value(rows, "Request Count"),
        "request_error_rate_pct": metric_value(rows, "Request Error Rate"),
        "ttft_p50_ms": metric_value(rows, "Time to First Token", "p50"),
        "ttft_p95_ms": metric_value(rows, "Time to First Token", "p95"),
        "ttft_p99_ms": metric_value(rows, "Time to First Token", "p99"),
        "itl_p50_ms": metric_value(rows, "Inter Token Latency", "p50"),
        "itl_p95_ms": metric_value(rows, "Inter Token Latency", "p95"),
        "request_latency_p95_ms": metric_value(rows, "Request Latency", "p95"),
        "input_tokens_avg": metric_value(rows, "Input Sequence Length", "avg"),
        "output_tokens_avg": metric_value(rows, "Output Sequence Length", "avg"),
        "good_request_count": metric_value(rows, "Good Request Count", "avg"),
    }


def coefficient_of_variation(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 2 or statistics.mean(values) == 0:
        return math.inf
    return statistics.stdev(values) / abs(statistics.mean(values))


def build_aiperf_command(
    aiperf: Path,
    backend: dict[str, Any],
    cell: BenchmarkCell,
    config: dict[str, Any],
    artifact_dir: Path,
    telemetry_backend: str,
) -> list[str]:
    campaign = config["campaign"]
    slo = config["slo"][cell.slo_profile]
    command = [
        str(aiperf),
        "profile",
        "--model",
        backend["model"],
        "--url",
        backend["url"],
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-server-token-count",
        "--tokenizer",
        backend["tokenizer"],
        "--input-file",
        str(ROOT / cell.dataset),
        "--custom-dataset-type",
        cell.dataset_type,
        "--concurrency",
        str(cell.concurrency),
        "--benchmark-duration",
        str(campaign["duration_seconds"]),
        "--warmup-duration",
        str(campaign["warmup_seconds"]),
        "--random-seed",
        str(campaign["seed"]),
        "--goodput",
        f"time_to_first_token:{slo['ttft_ms']} inter_token_latency:{slo['itl_ms']}",
        "--extra-inputs",
        '{"temperature":0}',
        "--api-key",
        backend["api_key"],
        "--artifact-dir",
        str(artifact_dir),
        "--export-level",
        "records",
        "--export-outputs-json",
        # Same slot the RTX 4090 harness used, so command.txt diffs between the
        # two campaigns show only paths rather than a reordered argument list.
        *gpu_telemetry.aiperf_flags(telemetry_backend),
        "--failed-request-threshold",
        "0.01",
        "--ui",
        "none",
    ]
    if cell.request_rate is not None:
        command += ["--request-rate", str(cell.request_rate)]
    if cell.arrival_pattern:
        command += ["--arrival-pattern", cell.arrival_pattern]
    if cell.suite == "prefix":
        command += ["--cache-bust", "warmup_isolation_first_turn"]
    return command


def run_repeat(
    command: list[str], artifact_dir: Path, log_path: Path, dry_run: bool
) -> tuple[int, dict[str, float] | None]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if dry_run:
        print(shlex.join(command))
        return 0, None
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        return result.returncode, None
    return 0, extract_repeat_metrics(artifact_dir)


OOM_PATTERN = re.compile(
    r"(?:\b(?:CUDA\s+)?OOM\b|out\s+of\s+(?:device\s+)?memory|"
    r"cudaErrorMemoryAllocation|failed\s+to\s+allocate\s+.*(?:VRAM|GPU))",
    re.IGNORECASE,
)


def log_has_oom(path: Path) -> bool:
    """Return True when a client-visible error reports an unexpected GPU OOM."""
    if not path.is_file():
        return False
    return bool(OOM_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")))


def write_repeat_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return
    fields = list(dict.fromkeys(key for row in metrics for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def summarize_cell(
    cell: BenchmarkCell,
    repeats: list[dict[str, Any]],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    valid = [row for row in repeats if row.get("valid")]
    throughput_cv = coefficient_of_variation(
        [float(row["output_token_throughput"]) for row in valid]
    )
    ttft_cv = coefficient_of_variation([float(row["ttft_p95_ms"]) for row in valid])
    request_count = sum(float(row["request_count"]) for row in valid)
    unstable = len(valid) >= campaign["profile_runs"] and (
        throughput_cv > campaign["throughput_cv_limit"]
        or ttft_cv > campaign["p95_ttft_cv_limit"]
    )
    if len(valid) >= campaign["profile_runs"]:
        status = "valid" if len(valid) == len(repeats) else "valid-with-retries"
    else:
        status = "invalid"
    throughput_cv_value = throughput_cv if math.isfinite(throughput_cv) else None
    ttft_cv_value = ttft_cv if math.isfinite(ttft_cv) else None
    return {
        "cell": asdict(cell),
        "valid_repeats": len(valid),
        "invalid_repeats": len(repeats) - len(valid),
        "oom_repeats": sum(bool(row.get("oom_detected")) for row in repeats),
        "output_throughput_cv": throughput_cv_value,
        "p95_ttft_cv": ttft_cv_value,
        "unstable": unstable,
        "request_count": request_count,
        "p99_eligible": request_count >= campaign["p99_min_samples"],
        "status": status,
    }


def run_cell(
    cell: BenchmarkCell,
    backend: dict[str, Any],
    config: dict[str, Any],
    aiperf: Path,
    run_dir: Path,
    dry_run: bool,
    telemetry_backend: str,
) -> dict[str, Any]:
    campaign = config["campaign"]
    cell_dir = run_dir / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "cell.json").write_text(
        json.dumps(asdict(cell), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    repeats: list[dict[str, Any]] = []
    target_repeats = campaign["profile_runs"]
    repeat_number = 1
    while repeat_number <= target_repeats:
        if not dry_run:
            idle = wait_for_gpu_idle(campaign, telemetry_backend)
        else:
            idle = {}
        artifact_dir = cell_dir / f"repeat-{repeat_number:02d}"
        command = build_aiperf_command(
            aiperf, backend, cell, config, artifact_dir, telemetry_backend
        )
        console_log = artifact_dir / "aiperf-console.log"
        exit_code, metrics = run_repeat(
            command, artifact_dir, console_log, dry_run
        )
        oom_detected = False if dry_run else log_has_oom(console_log)
        row: dict[str, Any] = {
            "repeat": repeat_number,
            "exit_code": exit_code,
            "oom_detected": oom_detected,
            **idle,
        }
        if metrics:
            row.update(metrics)
            row["valid"] = (
                exit_code == 0
                and not oom_detected
                and metrics["request_error_rate_pct"] <= campaign["max_error_rate_pct"]
            )
        else:
            row["valid"] = dry_run
        repeats.append(row)

        if repeat_number == campaign["profile_runs"] and not dry_run:
            interim = summarize_cell(cell, repeats, campaign)
            if interim["unstable"]:
                throughput_label = (
                    "N/A"
                    if interim["output_throughput_cv"] is None
                    else f"{interim['output_throughput_cv']:.1%}"
                )
                ttft_label = (
                    "N/A"
                    if interim["p95_ttft_cv"] is None
                    else f"{interim['p95_ttft_cv']:.1%}"
                )
                print(
                    f"{cell.cell_id}: unstable after 3 repeats; "
                    f"throughput CV={throughput_label}, "
                    f"P95 TTFT CV={ttft_label}; adding two repeats."
                )
                target_repeats = campaign["unstable_profile_runs"]
            elif interim["status"] == "invalid":
                print(
                    f"{cell.cell_id}: invalid after {campaign['profile_runs']} repeats; "
                    "not adding variability retries."
                )
        repeat_number += 1

    write_repeat_csv(cell_dir / "repeat-metrics.csv", repeats)
    summary = summarize_cell(cell, repeats, campaign) if not dry_run else {
        "cell": asdict(cell), "status": "dry-run", "planned_repeats": target_repeats
    }
    (cell_dir / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        help="Deployment name from config backends (llama-cpp, vllm, vllm-120b, ...)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", action="append")
    parser.add_argument("--cell", action="append")
    parser.add_argument("--scope", action="append", choices=["comparison", "vllm_stress"])
    parser.add_argument("--cache-mode", choices=["off", "on"])
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--manage-llama-server", action="store_true")
    parser.add_argument("--vllm-container", default="gpt_oss_20b_vllm")
    parser.add_argument("--aiperf", type=Path, default=None)
    parser.add_argument(
        "--allow-unverified-server",
        action="store_true",
        help="Run even though llama_server.build is still UNVERIFIED in the config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    server = config.get("llama_server", {})
    # Deployment name (results directory, endpoint) vs backend family (which cells
    # apply, which server checks run). vllm-120b is a distinct deployment of the
    # vllm family, so it selects the vllm cells but writes to results/vllm-120b/.
    family = backend_family(config, args.backend)
    cells = select_cells(expand_matrix(config), args.suite, args.cell)
    cells = select_backend_cells(cells, family, args.scope)
    if args.cache_mode:
        cells = [cell for cell in cells if cell.cache_mode == args.cache_mode]
    if not cells:
        raise SystemExit("No benchmark cells selected")
    required_cache_mode = required_prefix_cache_mode(cells)
    backend = config["backends"][args.backend]

    if "REPLACE_WITH_LOCAL_HF_SNAPSHOT" in str(backend.get("tokenizer", "")):
        raise SystemExit(
            "benchmark-config.json still has the placeholder tokenizer path. Point "
            f"backends.{args.backend}.tokenizer at the local openai/gpt-oss-20b snapshot."
        )
    if (
        family == "llama-cpp"
        and str(server.get("build", "UNVERIFIED")) == "UNVERIFIED"
        and not args.allow_unverified_server
    ):
        raise SystemExit(
            "llama_server.build is still UNVERIFIED. Run probe_llama_server.sh, record the "
            "aarch64 CUDA build identity and flash_attn setting in benchmark-config.json, "
            "or pass --allow-unverified-server for a throwaway run."
        )

    home = Path.home()
    aiperf_root = Path(os.environ.get("AIPERF_ROOT", home / "LLM_infra/aiperf"))
    aiperf = args.aiperf or Path(os.environ.get("AIPERF", aiperf_root / ".venv/bin/aiperf"))
    if not aiperf.is_file() and not args.dry_run:
        raise SystemExit(f"AIPerf executable not found: {aiperf}")

    telemetry_backend = gpu_telemetry.detect_backend()
    if telemetry_backend != "nvidia-smi" and not args.dry_run:
        print(
            f"WARNING: telemetry backend is {telemetry_backend!r}, not 'nvidia-smi'. On DGX "
            "Spark that means the driver stack is not visible; GPU counters and the idle "
            "gate will not match the RTX 4090 campaign.",
            file=sys.stderr,
        )

    if not args.dry_run:
        version = run_capture([str(aiperf), "--version"])
        revision = git_revision(aiperf_root)
        expected_version = config["campaign"]["aiperf_version"]
        expected_revision = config["campaign"]["aiperf_commit"]
        if expected_version not in version or revision != expected_revision:
            raise SystemExit(
                f"AIPerf pin mismatch: version={version!r}, revision={revision!r}; "
                f"expected {expected_version}/{expected_revision}"
            )
        required_cache_mode = validate_backend_mode(family, cells, args.vllm_container)
        unique_datasets = {(cell.dataset, cell.dataset_type) for cell in cells}
        for relative, dataset_type in unique_datasets:
            validate_dataset(ROOT / relative, dataset_type, formal=True)

    environment_manifest = ROOT / "datasets" / "manifests" / "environment.json"
    if not args.dry_run and not environment_manifest.is_file():
        raise SystemExit(
            "Missing datasets/manifests/environment.json; run "
            "bash scripts/dataset-tests/00-preflight.sh first."
        )

    campaign_id = args.campaign_id or f"{config['campaign']['name']}-{timestamp()}"
    run_dir = ROOT / "results" / args.backend / "dataset" / campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "started_at": utc_now(),
        "backend": args.backend,
        "backend_config": backend,
        "llama_server_config": server if family == "llama-cpp" else None,
        "campaign_config": config["campaign"],
        "required_prefix_cache_mode": required_cache_mode,
        "host": {
            "platform": sys.platform,
            "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
            "telemetry_backend": telemetry_backend,
        },
        "environment_manifest": (
            {
                "path": str(environment_manifest),
                "sha256": sha256_file(environment_manifest),
            }
            if environment_manifest.exists()
            else {"status": "dry-run/missing"}
        ),
        "selected_cells": [cell.cell_id for cell in cells],
        "selected_scopes": sorted({cell.scope for cell in cells}),
        "dataset_sha256": {
            cell.dataset: sha256_file(ROOT / cell.dataset)
            for cell in cells
            if (ROOT / cell.dataset).exists()
        },
        "gpu": gpu_telemetry.describe(telemetry_backend) if not args.dry_run else "dry-run",
        "aiperf_revision": git_revision(aiperf_root) if not args.dry_run else "dry-run",
        "invocations": [
            {
                "started_at": utc_now(),
                "selected_cells": [cell.cell_id for cell in cells],
                "selected_scopes": sorted({cell.scope for cell in cells}),
                "dry_run": args.dry_run,
                "required_prefix_cache_mode": required_cache_mode,
                "managed_llama_server": bool(args.manage_llama_server),
                "llama_server_command": None,
                "vllm_container_command": (
                    run_capture(
                        ["docker", "inspect", "--format", "{{json .Config.Cmd}}", args.vllm_container]
                    )
                    if family == "vllm" and not args.dry_run
                    else None
                ),
            }
        ],
    }
    campaign_path = run_dir / "campaign.json"
    if campaign_path.exists():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if existing.get("backend") != args.backend:
            raise SystemExit(f"Campaign ID already belongs to {existing.get('backend')}: {campaign_id}")
        metadata["started_at"] = existing.get("started_at", metadata["started_at"])
        metadata["selected_cells"] = list(
            dict.fromkeys(existing.get("selected_cells", []) + metadata["selected_cells"])
        )
        metadata["selected_scopes"] = sorted(
            set(existing.get("selected_scopes", [])) | set(metadata["selected_scopes"])
        )
        metadata["dataset_sha256"] = {
            **existing.get("dataset_sha256", {}),
            **metadata["dataset_sha256"],
        }
        metadata["invocations"] = existing.get("invocations", []) + metadata["invocations"]
    campaign_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    server_log = run_dir / f"server-{timestamp()}.log"
    with managed_llama_server(
        args.manage_llama_server and family == "llama-cpp",
        backend,
        server,
        server_log,
        required_cache_mode,
    ) as server_command:
        if server_command is not None:
            metadata["invocations"][-1]["llama_server_command"] = shlex.join(server_command)
            campaign_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if not args.dry_run and not args.skip_smoke:
            metadata["preflight"] = endpoint_preflight(backend)
            campaign_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        summaries = []
        for index, cell in enumerate(cells, 1):
            print(f"[{index}/{len(cells)}] {args.backend} {cell.cell_id}")
            summaries.append(
                run_cell(cell, backend, config, aiperf, run_dir, args.dry_run, telemetry_backend)
            )

    validation_path = run_dir / "campaign-validation.json"
    prior_cells: dict[str, dict[str, Any]] = {}
    if validation_path.exists():
        prior = json.loads(validation_path.read_text(encoding="utf-8"))
        prior_cells = {
            str(summary.get("cell", {}).get("cell_id")): summary
            for summary in prior.get("cells", [])
        }
    for summary in summaries:
        prior_cells[str(summary.get("cell", {}).get("cell_id"))] = summary
    all_summaries = list(prior_cells.values())
    completed = {
        "completed_at": utc_now(),
        "cells": all_summaries,
        "valid_cells": sum(str(summary.get("status", "")).startswith("valid") for summary in all_summaries),
        "total_cells": len(all_summaries),
    }
    validation_path.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
