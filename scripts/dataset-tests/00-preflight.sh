#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
AIPERF_ROOT="${AIPERF_ROOT:-$WORKSPACE_ROOT/aiperf}"
AIPERF="${AIPERF:-$AIPERF_ROOT/.venv/bin/aiperf}"
PYTHON="${PYTHON:-$AIPERF_ROOT/.venv/bin/python}"
EXPECTED_AIPERF_COMMIT="03c9c6ddc5e6227782e53ded177f1227d332af48"

for command_name in curl git nvidia-smi; do
    command -v "$command_name" >/dev/null || {
        echo "Error: missing command: $command_name" >&2
        exit 1
    }
done

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

echo "AIPerf: $version ($commit)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Preflight OK: llama.cpp=45 comparison cells; vLLM=45 comparison + 34 stress cells; environment manifest captured."
