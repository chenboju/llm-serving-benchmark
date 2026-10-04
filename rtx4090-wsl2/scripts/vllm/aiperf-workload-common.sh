#!/usr/bin/env bash
set -Eeuo pipefail

: "${WORKLOAD_ID:?WORKLOAD_ID must be set}"
: "${RESULT_DIR_NAME:?RESULT_DIR_NAME must be set}"
: "${ISL_TOKENS:?ISL_TOKENS must be set}"
: "${OSL_TOKENS:?OSL_TOKENS must be set}"
: "${SERVER_MAX_NUM_SEQS:?SERVER_MAX_NUM_SEQS must be set}"
: "${TTFT_SLO_MS:?TTFT_SLO_MS must be set}"
: "${ITL_SLO_MS:?ITL_SLO_MS must be set}"

if ! declare -p CONCURRENCIES >/dev/null 2>&1; then
    echo "Error: CONCURRENCIES must be set." >&2
    exit 1
fi

if [[ -n "${AIPERF_CONCURRENCIES:-}" ]]; then
    read -r -a CONCURRENCIES <<<"$AIPERF_CONCURRENCIES"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vllm-server-common.sh"

AIPERF_ROOT="${AIPERF_ROOT:-$WORKSPACE_ROOT/aiperf}"
AIPERF="${AIPERF:-$AIPERF_ROOT/.venv/bin/aiperf}"
AIPERF_PYTHON="${AIPERF_PYTHON:-$AIPERF_ROOT/.venv/bin/python}"

BENCHMARK_DURATION_SECONDS="${BENCHMARK_DURATION_SECONDS:-120}"
PROFILE_RUNS="${PROFILE_RUNS:-3}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CELL_COOLDOWN_SECONDS="${CELL_COOLDOWN_SECONDS:-5}"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/vllm/aiperf/$RESULT_DIR_NAME}"
AIPERF_ISL_TOKENS="${AIPERF_ISL_TOKENS:-$ISL_TOKENS}"

# 05、06、07 使用 Docker Compose 中已關閉的 prefix cache。
VLLM_PREFIX_CACHE_MODE="${VLLM_PREFIX_CACHE_MODE:-off}"

VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-gpt_oss_20b_vllm}"

EXTERNAL_SERVER_MAX_MODEL_LEN="unknown"
EXTERNAL_CONTAINER_COMMAND="unknown"

check_external_vllm_server() {
    local health_status
    local models_json
    local expected_prefix_flag

    echo "Checking external vLLM server: $SERVER_URL"

    health_status="$(
        curl \
            --silent \
            --output /dev/null \
            --write-out '%{http_code}' \
            --max-time 10 \
            "$HEALTH_URL" 2>/dev/null || true
    )"

    if [[ "$health_status" != "200" ]]; then
        echo "Error: vLLM health check failed: $HEALTH_URL" >&2
        echo "HTTP status: $health_status" >&2
        return 1
    fi

    if ! models_json="$(
        curl \
            --silent \
            --show-error \
            --fail \
            --max-time 10 \
            -H "Authorization: Bearer $API_KEY" \
            "$MODELS_URL"
    )"; then
        echo "Error: failed to query vLLM models endpoint: $MODELS_URL" >&2
        return 1
    fi

    if ! EXTERNAL_SERVER_MAX_MODEL_LEN="$(
        printf '%s' "$models_json" |
            "$AIPERF_PYTHON" -c '
import json
import sys

model_alias = sys.argv[1]
response = json.load(sys.stdin)

for model in response.get("data", []):
    if model.get("id") == model_alias:
        print(model.get("max_model_len", "unknown"))
        break
else:
    raise SystemExit(
        f"Error: model {model_alias!r} was not found in /v1/models"
    )
' "$MODEL_ALIAS"
    )"; then
        echo "Error: invalid response from $MODELS_URL" >&2
        echo "Server response: $models_json" >&2
        return 1
    fi

    if [[ "$EXTERNAL_SERVER_MAX_MODEL_LEN" =~ ^[0-9]+$ ]] &&
        ((EXTERNAL_SERVER_MAX_MODEL_LEN < SERVER_MAX_MODEL_LEN)); then
        echo "Error: Docker server max-model-len is smaller than expected." >&2
        echo "Expected: $SERVER_MAX_MODEL_LEN" >&2
        echo "Actual: $EXTERNAL_SERVER_MAX_MODEL_LEN" >&2
        return 1
    fi

    if [[ "$EXTERNAL_PREFIX_CACHE_MODE" != "$VLLM_PREFIX_CACHE_MODE" ]]; then
        echo "Error: configured prefix-cache modes do not match." >&2
        echo "EXTERNAL_PREFIX_CACHE_MODE=$EXTERNAL_PREFIX_CACHE_MODE" >&2
        echo "VLLM_PREFIX_CACHE_MODE=$VLLM_PREFIX_CACHE_MODE" >&2
        return 1
    fi

    if ! docker inspect "$VLLM_CONTAINER_NAME" >/dev/null 2>&1; then
        echo "Error: Docker container not found: $VLLM_CONTAINER_NAME" >&2
        return 1
    fi

    if [[ "$(
        docker inspect \
            --format '{{.State.Running}}' \
            "$VLLM_CONTAINER_NAME" 2>/dev/null || true
    )" != "true" ]]; then
        echo "Error: Docker container is not running: $VLLM_CONTAINER_NAME" >&2
        return 1
    fi

    EXTERNAL_CONTAINER_COMMAND="$(
        docker inspect \
            --format '{{json .Config.Cmd}}' \
            "$VLLM_CONTAINER_NAME"
    )"

    case "$VLLM_PREFIX_CACHE_MODE" in
        off)
            expected_prefix_flag="--no-enable-prefix-caching"
            ;;
        on)
            expected_prefix_flag="--enable-prefix-caching"
            ;;
        *)
            echo "Error: VLLM_PREFIX_CACHE_MODE must be 'on' or 'off'." >&2
            return 1
            ;;
    esac

    if [[ "$EXTERNAL_CONTAINER_COMMAND" != *"$expected_prefix_flag"* ]]; then
        echo "Error: Docker vLLM was not started with $expected_prefix_flag." >&2
        echo "Container command: $EXTERNAL_CONTAINER_COMMAND" >&2
        return 1
    fi

    # Server 已在 benchmark 開始前啟動，因此不計算啟動時間。
    SERVER_READY_TIME_MS=0

    echo "External vLLM server is ready."
    echo "Model: $MODEL_ALIAS"
    echo "Max model length: $EXTERNAL_SERVER_MAX_MODEL_LEN"
    echo "Max sequences: $SERVER_MAX_NUM_SEQS"
    echo "Prefix cache: $VLLM_PREFIX_CACHE_MODE"
}

write_external_vllm_metadata() {
    local container_image="unknown"
    local container_image_id="unknown"
    local vllm_version="unknown"

    if docker inspect "$VLLM_CONTAINER_NAME" >/dev/null 2>&1; then
        container_image="$(
            docker inspect \
                --format '{{.Config.Image}}' \
                "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        )"

        container_image_id="$(
            docker inspect \
                --format '{{.Image}}' \
                "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        )"

        vllm_version="$(
            docker exec "$VLLM_CONTAINER_NAME" \
                python -c 'import vllm; print(vllm.__version__)' \
                2>/dev/null || true
        )"
    fi

    echo "server_mode=external-docker"
    echo "server_reused=true"
    echo "container_name=$VLLM_CONTAINER_NAME"
    echo "container_image=${container_image:-unknown}"
    echo "container_image_id=${container_image_id:-unknown}"
    echo "container_command=$EXTERNAL_CONTAINER_COMMAND"
    echo "vllm_version=${vllm_version:-unknown}"
    echo "model_cache_root=$MODEL_CACHE_ROOT"
    echo "model_path=$MODEL_PATH"
    echo "model_revision=$MODEL_REVISION"
    echo "model_alias=$MODEL_ALIAS"
    echo "tokenizer=$TOKENIZER"
    echo "server_url=$SERVER_URL"
    echo "server_max_model_len=$EXTERNAL_SERVER_MAX_MODEL_LEN"
    echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "dtype=$VLLM_DTYPE"
    echo "generation_config=vllm"
    echo "external_prefix_cache_mode=$EXTERNAL_PREFIX_CACHE_MODE"
}

require_executable "$AIPERF"
require_executable "$AIPERF_PYTHON"

for command_name in curl docker git nvidia-smi tee; do
    require_command "$command_name"
done

if ((${#CONCURRENCIES[@]} == 0)); then
    echo "Error: CONCURRENCIES cannot be empty." >&2
    exit 1
fi

for concurrency in "${CONCURRENCIES[@]}"; do
    if ((concurrency < 1 || concurrency > SERVER_MAX_NUM_SEQS)); then
        echo "Error: concurrency $concurrency exceeds max-num-seqs=$SERVER_MAX_NUM_SEQS." >&2
        exit 1
    fi
done

if ((ISL_TOKENS + OSL_TOKENS > SERVER_MAX_MODEL_LEN)); then
    echo "Error: request length exceeds max-model-len=$SERVER_MAX_MODEL_LEN." >&2
    exit 1
fi

case "$VLLM_MANAGE_SERVER" in
    1)
        # 保留原本 WSL native vLLM 模式。
        preflight_vllm
        check_no_existing_service
        ;;
    0)
        # 使用已經運行的 Docker vLLM。
        check_external_vllm_server
        ;;
    *)
        echo "Error: VLLM_MANAGE_SERVER must be 0 or 1." >&2
        exit 1
        ;;
esac

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
COMMANDS_FILE="$RUN_DIR/commands.txt"

mkdir -p "$RAW_DIR"

if [[ "$VLLM_MANAGE_SERVER" == "1" ]]; then
    trap stop_vllm_server EXIT
fi

trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

if [[ "$VLLM_MANAGE_SERVER" == "1" ]]; then
    start_vllm_server \
        "$SERVER_LOG" \
        "$VLLM_PREFIX_CACHE_MODE" \
        truncate
else
    # Docker server 已經啟動，只重新確認連線與模型。
    check_external_vllm_server

    {
        echo "External Docker vLLM server"
        echo "container=$VLLM_CONTAINER_NAME"
        echo "server_url=$SERVER_URL"
        echo "model=$MODEL_ALIAS"
        echo "prefix_cache=$EXTERNAL_PREFIX_CACHE_MODE"
        echo
        echo "[container-command]"
        echo "$EXTERNAL_CONTAINER_COMMAND"
    } >"$SERVER_LOG"
fi

nvidia-smi >"$RAW_DIR/nvidia-smi-ready.txt"

AIPERF_COMMIT="$(
    git -C "$AIPERF_ROOT" rev-parse HEAD 2>/dev/null || echo unknown
)"

{
    echo "test=$WORKLOAD_ID"
    echo "timestamp=$TIMESTAMP"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "aiperf_commit=$AIPERF_COMMIT"
    echo "aiperf_version=$($AIPERF --version)"
    echo "isl_tokens=$ISL_TOKENS"
    echo "aiperf_synthetic_isl_tokens=$AIPERF_ISL_TOKENS"
    echo "osl_tokens=$OSL_TOKENS"
    echo "concurrency_levels=${CONCURRENCIES[*]}"
    echo "benchmark_duration_seconds=$BENCHMARK_DURATION_SECONDS"
    echo "profile_runs=$PROFILE_RUNS"
    echo "confidence_level=$CONFIDENCE_LEVEL"
    echo "random_seed=$RANDOM_SEED"
    echo "ttft_slo_ms=$TTFT_SLO_MS"
    echo "itl_slo_ms=$ITL_SLO_MS"
    echo "fixed_output_tokens=true"
    echo "vllm_prefix_cache_mode=$VLLM_PREFIX_CACHE_MODE"
    echo "server_ready_time_ms=$SERVER_READY_TIME_MS"

    if [[ "$VLLM_MANAGE_SERVER" == "1" ]]; then
        write_vllm_metadata
    else
        write_external_vllm_metadata
    fi

    echo
    echo "[gpu]"

    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.used \
        --format=csv,noheader
} >"$METADATA_FILE"

EXTRA_INPUTS="$(
    printf \
        '{"temperature":0,"min_tokens":%s,"ignore_eos":true}' \
        "$OSL_TOKENS"
)"

for index in "${!CONCURRENCIES[@]}"; do
    concurrency="${CONCURRENCIES[$index]}"
    cell_dir="$RUN_DIR/concurrency-$concurrency"
    cell_log="$RAW_DIR/concurrency-$concurrency.log"

    AIPERF_COMMAND=(
        "$AIPERF" profile
        --model "$MODEL_ALIAS"
        --url "$SERVER_URL"
        --endpoint-type chat
        --streaming
        --use-server-token-count
        --cache-bust first_turn_prefix
        --tokenizer "$TOKENIZER"
        --apply-chat-template
        --prompt-input-tokens-mean "$AIPERF_ISL_TOKENS"
        --prompt-input-tokens-stddev 0
        --output-tokens-mean "$OSL_TOKENS"
        --output-tokens-stddev 0
        --concurrency "$concurrency"
        --benchmark-duration "$BENCHMARK_DURATION_SECONDS"
        --warmup-request-count "$concurrency"
        --num-profile-runs "$PROFILE_RUNS"
        --confidence-level "$CONFIDENCE_LEVEL"
        --random-seed "$RANDOM_SEED"
        --goodput
        "time_to_first_token:$TTFT_SLO_MS inter_token_latency:$ITL_SLO_MS"
        --extra-inputs "$EXTRA_INPUTS"
        --api-key "$API_KEY"
        --artifact-dir "$cell_dir"
        --export-level records
        --ui none
    )

    echo
    echo "Running $WORKLOAD_ID at concurrency=$concurrency..."

    # commands.txt 不寫入實際 API key。
    {
        printf '%q ' \
            "${AIPERF_COMMAND[@]:0:${#AIPERF_COMMAND[@]}-6}"

        printf '%q ' \
            --api-key '<redacted>' \
            --artifact-dir "$cell_dir" \
            --export-level records \
            --ui none

        echo
    } >>"$COMMANDS_FILE"

    "${AIPERF_COMMAND[@]}" 2>&1 | tee "$cell_log"

    nvidia-smi \
        >"$RAW_DIR/nvidia-smi-concurrency-$concurrency-after.txt"

    if ((index + 1 < ${#CONCURRENCIES[@]} &&
        CELL_COOLDOWN_SECONDS > 0)); then
        sleep "$CELL_COOLDOWN_SECONDS"
    fi
done

nvidia-smi >"$RAW_DIR/nvidia-smi-after.txt"

if [[ "$VLLM_MANAGE_SERVER" == "0" ]]; then
    {
        echo
        echo "[container-log-tail]"
        docker logs --tail 200 "$VLLM_CONTAINER_NAME" 2>&1 || true
    } >>"$SERVER_LOG"
fi

echo
echo "AIPerf workload completed: $RUN_DIR"