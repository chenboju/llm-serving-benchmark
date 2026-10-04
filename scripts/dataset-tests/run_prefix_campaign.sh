#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
AIPERF_PYTHON="${AIPERF_PYTHON:-$WORKSPACE_ROOT/aiperf/.venv/bin/python}"
VLLM_CONTAINER="${VLLM_CONTAINER:-gpt_oss_20b_vllm}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:18000/health}"
CAMPAIGN_ID="${CAMPAIGN_ID:-gpt-oss-20b-dataset-prefix-v1-$(date +%Y%m%d-%H%M%S)}"

for command_name in curl docker; do
    command -v "$command_name" >/dev/null || {
        echo "Error: missing command: $command_name" >&2
        exit 1
    }
done
[[ -x "$AIPERF_PYTHON" ]] || { echo "Error: missing Python: $AIPERF_PYTHON" >&2; exit 1; }

container_cache_mode() {
    local command
    command="$(docker inspect --format '{{json .Config.Cmd}}' "$VLLM_CONTAINER" 2>/dev/null || true)"
    if [[ "$command" == *"--no-enable-prefix-caching"* ]]; then
        printf 'off\n'
    elif [[ "$command" == *"--enable-prefix-caching"* ]]; then
        printf 'on\n'
    else
        printf 'unknown\n'
    fi
}

wait_for_vllm() {
    for ((check = 1; check <= 1200; check++)); do
        status="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 2 "$VLLM_HEALTH_URL" || true)"
        [[ "$status" == "200" ]] && return
        sleep 0.5
    done
    echo "Error: vLLM readiness timeout: $VLLM_HEALTH_URL" >&2
    exit 1
}

start_vllm() {
    if [[ "$(docker inspect --format '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null || true)" != "true" ]]; then
        docker start "$VLLM_CONTAINER" >/dev/null
    fi
    wait_for_vllm
}

stop_vllm() {
    if [[ "$(docker inspect --format '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null || true)" == "true" ]]; then
        docker stop "$VLLM_CONTAINER" >/dev/null
    fi
}

require_vllm_mode() {
    local expected="$1"
    local flag
    flag="--no-enable-prefix-caching"
    [[ "$expected" == "on" ]] && flag="--enable-prefix-caching"
    while [[ "$(container_cache_mode)" != "$expected" ]]; do
        echo
        echo "vLLM prefix cache must be $expected ($flag)."
        echo "Recreate $VLLM_CONTAINER with that flag, then press Enter to check again."
        read -r
    done
    start_vllm
}

run_llama() {
    local mode="$1"
    stop_vllm
    "$AIPERF_PYTHON" "$SCRIPT_DIR/run_benchmarks.py" \
        --backend llama-cpp --suite prefix --cache-mode "$mode" \
        --campaign-id "$CAMPAIGN_ID" --manage-llama-server
}

run_vllm() {
    local mode="$1"
    require_vllm_mode "$mode"
    "$AIPERF_PYTHON" "$SCRIPT_DIR/run_benchmarks.py" \
        --backend vllm --suite prefix --cache-mode "$mode" \
        --campaign-id "$CAMPAIGN_ID" --vllm-container "$VLLM_CONTAINER"
}

# Alternate order across the two cache modes.
run_llama off
run_vllm off
run_vllm on
run_llama on
start_vllm

echo
echo "Prefix A/B campaign completed: $CAMPAIGN_ID"
echo "llama.cpp: $PROJECT_ROOT/results/llama-cpp/dataset/$CAMPAIGN_ID"
echo "vLLM:      $PROJECT_ROOT/results/vllm/dataset/$CAMPAIGN_ID"
