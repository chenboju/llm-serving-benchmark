#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
AIPERF_PYTHON="${AIPERF_PYTHON:-$WORKSPACE_ROOT/aiperf/.venv/bin/python}"
VLLM_CONTAINER="${VLLM_CONTAINER:-gpt_oss_20b_vllm}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:18000/health}"
CAMPAIGN_ID="${CAMPAIGN_ID:-gpt-oss-20b-dataset-v1-$(date +%Y%m%d-%H%M%S)}"
SUITES=(${DATASET_SUITES:-sharegpt_closed sharegpt_open sharegpt_natural rag ruler_performance})

for suite in "${SUITES[@]}"; do
    if [[ "$suite" == "prefix" ]]; then
        echo "Error: prefix requires server cache-mode switching; use run_prefix_campaign.sh." >&2
        exit 1
    fi
done

for command_name in curl docker; do
    command -v "$command_name" >/dev/null || {
        echo "Error: missing command: $command_name" >&2
        exit 1
    }
done
[[ -x "$AIPERF_PYTHON" ]] || { echo "Error: missing Python: $AIPERF_PYTHON" >&2; exit 1; }

wait_for_vllm() {
    local max_checks=1200
    for ((check = 1; check <= max_checks; check++)); do
        status="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 2 "$VLLM_HEALTH_URL" || true)"
        if [[ "$status" == "200" ]]; then
            return
        fi
        sleep 0.5
    done
    echo "Error: vLLM readiness timeout: $VLLM_HEALTH_URL" >&2
    docker logs --tail 100 "$VLLM_CONTAINER" >&2 || true
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

run_backend_suite() {
    local backend="$1"
    local suite="$2"
    echo
    echo "================================================================"
    echo "campaign=$CAMPAIGN_ID backend=$backend suite=$suite"
    echo "================================================================"
    if [[ "$backend" == "llama-cpp" ]]; then
        stop_vllm
        "$AIPERF_PYTHON" "$SCRIPT_DIR/run_benchmarks.py" \
            --backend llama-cpp \
            --suite "$suite" \
            --scope comparison \
            --campaign-id "$CAMPAIGN_ID" \
            --manage-llama-server
    else
        start_vllm
        "$AIPERF_PYTHON" "$SCRIPT_DIR/run_benchmarks.py" \
            --backend vllm \
            --suite "$suite" \
            --scope comparison \
            --campaign-id "$CAMPAIGN_ID" \
            --vllm-container "$VLLM_CONTAINER"
    fi
}

for index in "${!SUITES[@]}"; do
    suite="${SUITES[$index]}"
    if ((index % 2 == 0)); then
        run_backend_suite llama-cpp "$suite"
        run_backend_suite vllm "$suite"
    else
        run_backend_suite vllm "$suite"
        run_backend_suite llama-cpp "$suite"
    fi
done

start_vllm

echo
echo "A/B campaign completed: $CAMPAIGN_ID"
echo "llama.cpp: $PROJECT_ROOT/results/llama-cpp/dataset/$CAMPAIGN_ID"
echo "vLLM:      $PROJECT_ROOT/results/vllm/dataset/$CAMPAIGN_ID"
