#!/usr/bin/env bash
# DGX Spark preflight. Same as the RTX 4090 version plus an architecture check
# and a reminder that prepared datasets must be copied, not regenerated.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
AIPERF_ROOT="${AIPERF_ROOT:-$WORKSPACE_ROOT/aiperf}"
AIPERF="${AIPERF:-$AIPERF_ROOT/.venv/bin/aiperf}"
PYTHON="${PYTHON:-$AIPERF_ROOT/.venv/bin/python}"
EXPECTED_AIPERF_COMMIT="03c9c6ddc5e6227782e53ded177f1227d332af48"

for command_name in curl git docker nvidia-smi; do
    command -v "$command_name" >/dev/null || {
        echo "Error: missing command: $command_name" >&2
        exit 1
    }
done

architecture="$(uname -m)"
[[ "$architecture" == "aarch64" || "$architecture" == "arm64" ]] || {
    echo "Warning: expected aarch64, got $architecture. This folder targets DGX Spark;" >&2
    echo "         on x86_64 use the RTX 4090 copy instead." >&2
}

[[ -x "$AIPERF" ]] || { echo "Error: missing AIPerf: $AIPERF" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Error: missing Python: $PYTHON" >&2; exit 1; }

version="$($AIPERF --version)"
commit="$(git -C "$AIPERF_ROOT" rev-parse HEAD)"
[[ "$version" == *"0.12.0"* ]] || { echo "Error: expected AIPerf 0.12.0, got $version" >&2; exit 1; }
[[ "$commit" == "$EXPECTED_AIPERF_COMMIT" ]] || {
    echo "Error: expected AIPerf $EXPECTED_AIPERF_COMMIT, got $commit" >&2
    exit 1
}

"$PYTHON" "$SCRIPT_DIR/benchmark_matrix.py" >/dev/null
"$PYTHON" "$SCRIPT_DIR/validate_prepared.py"
"$PYTHON" "$SCRIPT_DIR/capture_environment.py"

echo
echo "AIPerf: $version ($commit)"
echo "Architecture: $architecture"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
echo "--- GPU telemetry ---"
"$PYTHON" "$SCRIPT_DIR/gpu_telemetry.py" | head -3
echo
echo "Preflight OK: llama.cpp=45 comparison cells; vLLM=45 comparison + 34 stress cells."
echo "Datasets must have been copied from the reference machine, not regenerated:"
echo "  a locally regenerated datasets/prepared/ breaks cross-hardware comparability"
echo "  even when validate_prepared.py passes against locally written manifests."
