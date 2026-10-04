#!/usr/bin/env python3
"""GPU telemetry abstraction shared by every machine in this benchmark project.

The original RTX 4090 harness called nvidia-smi inline and passed
`--gpu-telemetry pynvml` to AIPerf. That works on NVIDIA and nowhere else, so
telemetry moved behind this probe: NVIDIA hosts keep the exact same behaviour,
AMD hosts fall back to amd-smi/rocm-smi, and a host with none of them degrades
to a fixed cooldown instead of crashing.

Probe order: nvidia-smi -> amd-smi -> rocm-smi -> none.
Override with GPU_TELEMETRY=nvidia-smi|amd-smi|rocm-smi|none.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
from typing import Any


TELEMETRY_ENV = "GPU_TELEMETRY"
SUPPORTED = ("nvidia-smi", "amd-smi", "rocm-smi", "none")


def _run(command: list[str], timeout: float = 20) -> str:
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


def detect_backend() -> str:
    """Return the telemetry backend to use for this host."""
    override = os.environ.get(TELEMETRY_ENV, "").strip()
    if override:
        if override not in SUPPORTED:
            raise ValueError(f"{TELEMETRY_ENV} must be one of {SUPPORTED}; got {override!r}")
        return override
    for candidate in ("nvidia-smi", "amd-smi", "rocm-smi"):
        if shutil.which(candidate):
            return candidate
    return "none"


def _first_number(value: Any) -> float | None:
    """Pull a number out of values like 12, "12", "12 %", {"value": 12}."""
    if isinstance(value, dict):
        for key in ("value", "gfx", "used", "total"):
            if key in value:
                return _first_number(value[key])
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().split()[0] if value.strip() else ""
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _dig(payload: Any, *keys: str) -> float | None:
    """Find the first numeric value under any of `keys`, at any depth."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in keys:
                number = _first_number(value)
                if number is not None:
                    return number
        for value in payload.values():
            found = _dig(value, *keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _dig(item, *keys)
            if found is not None:
                return found
    return None


def _query_nvidia_smi() -> dict[str, float]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Unexpected nvidia-smi output: {output}")
    values = [part.strip() for part in lines[0].split(",")]
    if len(values) < 4:
        raise RuntimeError(f"Unexpected nvidia-smi output: {output}")

    def number(text: str) -> float:
        # Grace Blackwell reports [N/A] for some counters on unified memory parts.
        try:
            return float(text)
        except ValueError:
            return float("nan")

    utilization = number(values[0])
    if utilization != utilization:  # NaN
        raise RuntimeError(f"nvidia-smi reported no GPU utilization: {output}")
    return {
        "utilization_pct": utilization,
        "memory_used_mib": number(values[1]),
        "temperature_c": number(values[2]),
        "power_w": number(values[3]),
    }


def _query_amd_smi() -> dict[str, float]:
    output = _run(["amd-smi", "metric", "--usage", "--mem", "--power", "--temperature", "--json"])
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Unexpected amd-smi output: {output[:400]}") from error
    utilization = _dig(payload, "gfx_activity", "gfx_usage", "gpu_activity", "gfx")
    memory = _dig(payload, "used_vram", "vram_used", "used")
    temperature = _dig(payload, "edge", "hotspot", "junction", "temperature")
    power = _dig(payload, "socket_power", "average_socket_power", "power")
    if utilization is None:
        raise RuntimeError(f"amd-smi reported no GPU activity field: {output[:400]}")
    return {
        "utilization_pct": float(utilization),
        "memory_used_mib": float(memory) if memory is not None else float("nan"),
        "temperature_c": float(temperature) if temperature is not None else float("nan"),
        "power_w": float(power) if power is not None else float("nan"),
    }


def _query_rocm_smi() -> dict[str, float]:
    output = _run(["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showpower", "--csv"])
    rows = list(csv.DictReader(io.StringIO(output)))
    if not rows:
        raise RuntimeError(f"Unexpected rocm-smi output: {output[:400]}")
    row = rows[0]

    def field(*needles: str) -> float:
        for key, value in row.items():
            lowered = (key or "").lower()
            if any(needle in lowered for needle in needles):
                number = _first_number(value)
                if number is not None:
                    return number
        return float("nan")

    utilization = field("gpu use", "gpu_use", "gpu util")
    if utilization != utilization:  # NaN
        raise RuntimeError(f"rocm-smi reported no GPU use field: {list(row)}")
    return {
        "utilization_pct": utilization,
        "memory_used_mib": field("memory use", "vram"),
        "temperature_c": field("temp"),
        "power_w": field("power", "watt"),
    }


def query_gpu(backend: str) -> dict[str, float]:
    """Return current GPU counters. Raises when the backend is unusable."""
    if backend == "nvidia-smi":
        return _query_nvidia_smi()
    if backend == "amd-smi":
        return _query_amd_smi()
    if backend == "rocm-smi":
        return _query_rocm_smi()
    raise RuntimeError("No GPU telemetry backend available")


def describe(backend: str) -> str:
    """Human-readable identity of the accelerator, for the run manifest."""
    if backend == "nvidia-smi":
        return _run(["nvidia-smi"], timeout=20)
    if backend == "amd-smi":
        return _run(["amd-smi", "static", "--asic", "--driver", "--json"])
    if backend == "rocm-smi":
        return _run(["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"])
    return "unavailable: no GPU telemetry backend (GPU_TELEMETRY=none)"


def aiperf_flags(backend: str) -> list[str]:
    """AIPerf GPU-telemetry flags. pynvml is NVIDIA-only."""
    if backend == "nvidia-smi":
        return ["--gpu-telemetry", "pynvml"]
    return []


if __name__ == "__main__":
    detected = detect_backend()
    print(f"backend: {detected}")
    print(describe(detected))
    if detected != "none":
        print(json.dumps(query_gpu(detected), indent=2))
