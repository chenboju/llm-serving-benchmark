#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
AIPERF_PYTHON="${AIPERF_PYTHON:-$WORKSPACE_ROOT/aiperf/.venv/bin/python}"
VLLM_CONTAINER="${VLLM_CONTAINER:-gpt_oss_20b_vllm}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:18000/health}"
CAMPAIGN_ID="${CAMPAIGN_ID:-gpt-oss-20b-vllm-stress-v1-$(date +%Y%m%d-%H%M%S)}"
SUITES=(sharegpt_closed sharegpt_open sharegpt_natural rag ruler_performance)

for command_name in curl docker; do
    command -v "$command_name" >/dev/null || {
        echo "Error: missing command: $command_name" >&2
        exit 1
    }
done
[[ -x "$AIPERF_PYTHON" ]] || { echo "Error: missing Python: $AIPERF_PYTHON" >&2; exit 1; }

container_command="$(docker inspect --format '{{json .Config.Cmd}}' "$VLLM_CONTAINER" 2>/dev/null || true)"
if [[ "$container_command" != *"--no-enable-prefix-caching"* ]]; then
    echo "Error: vLLM stress deployment must use --no-enable-prefix-caching." >&2
    exit 1
fi

if [[ "$(docker inspect --format '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null || true)" != "true" ]]; then
    docker start "$VLLM_CONTAINER" >/dev/null
fi

for ((check = 1; check <= 1200; check++)); do
    status="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 2 "$VLLM_HEALTH_URL" || true)"
    [[ "$status" == "200" ]] && break
    if [[ "$check" -eq 1200 ]]; then
        echo "Error: vLLM readiness timeout: $VLLM_HEALTH_URL" >&2
        exit 1
    fi
    sleep 0.5
done

for suite in "${SUITES[@]}"; do
    "$AIPERF_PYTHON" "$SCRIPT_DIR/run_benchmarks.py" \
        --backend vllm \
        --scope vllm_stress \
        --suite "$suite" \
        --campaign-id "$CAMPAIGN_ID" \
        --vllm-container "$VLLM_CONTAINER"
done

echo
echo "vLLM stress campaign completed: $CAMPAIGN_ID"
echo "Results: $PROJECT_ROOT/results/vllm/dataset/$CAMPAIGN_ID"
