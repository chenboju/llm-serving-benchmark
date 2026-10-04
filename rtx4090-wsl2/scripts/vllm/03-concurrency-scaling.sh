#!/usr/bin/env bash
set -Eeuo pipefail

PP_TOKENS="${PP_TOKENS:-2048}"
TG_TOKENS="${TG_TOKENS:-128}"
DEPTH="${DEPTH:-0}"
CONCURRENCIES=(1 2 4 8 16)
RUNS="${RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"

# 必須與 Docker Compose 的 --max-num-seqs 一致。
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vllm-server-common.sh"

LLAMA_BENCHY="${LLAMA_BENCHY:-$WORKSPACE_ROOT/llama-benchy/.venv/bin/llama-benchy}"
PLOT_PYTHON="${PLOT_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/vllm/concurrency}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-gpt_oss_20b_vllm}"

check_external_vllm_server() {
    local health_status
    local models_json

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

    models_json="$(
        curl \
            --silent \
            --show-error \
            --fail \
            --max-time 10 \
            -H "Authorization: Bearer $API_KEY" \
            "$MODELS_URL"
    )"

    if ! printf '%s\n' "$models_json" |
        grep -Eq "\"id\"[[:space:]]*:[[:space:]]*\"${MODEL_ALIAS}\""; then
        echo "Error: model '$MODEL_ALIAS' was not found at $MODELS_URL." >&2
        echo "Server response: $models_json" >&2
        return 1
    fi

    if [[ "$EXTERNAL_PREFIX_CACHE_MODE" != "off" ]]; then
        echo "Error: concurrency-scaling requires prefix caching to be disabled." >&2
        echo "Current EXTERNAL_PREFIX_CACHE_MODE=$EXTERNAL_PREFIX_CACHE_MODE" >&2
        return 1
    fi

    SERVER_READY_TIME_MS=0

    echo "External vLLM server is ready."
    echo "Model: $MODEL_ALIAS"
    echo "Max model length: $SERVER_MAX_MODEL_LEN"
    echo "Max sequences: $SERVER_MAX_NUM_SEQS"
}

write_external_vllm_metadata() {
    local container_image="unknown"
    local container_image_id="unknown"
    local vllm_version="unknown"

    if command -v docker >/dev/null 2>&1 &&
        docker inspect "$VLLM_CONTAINER_NAME" >/dev/null 2>&1; then
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

    echo "server_mode=external"
    echo "container_name=$VLLM_CONTAINER_NAME"
    echo "container_image=${container_image:-unknown}"
    echo "container_image_id=${container_image_id:-unknown}"
    echo "vllm_version=${vllm_version:-unknown}"
    echo "model_cache_root=$MODEL_CACHE_ROOT"
    echo "model_path=$MODEL_PATH"
    echo "model_revision=$MODEL_REVISION"
    echo "model_alias=$MODEL_ALIAS"
    echo "tokenizer=$TOKENIZER"
    echo "server_url=$SERVER_URL"
    echo "server_max_model_len=$SERVER_MAX_MODEL_LEN"
    echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "dtype=$VLLM_DTYPE"
    echo "generation_config=vllm"
    echo "external_prefix_cache_mode=$EXTERNAL_PREFIX_CACHE_MODE"
}

require_executable "$LLAMA_BENCHY"
require_executable "$PLOT_PYTHON"

for command_name in curl grep nvidia-smi tee; do
    require_command "$command_name"
done

case "$VLLM_MANAGE_SERVER" in
    1)
        # 保留原本的 WSL native vLLM 模式。
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

for concurrency in "${CONCURRENCIES[@]}"; do
    if ((concurrency > SERVER_MAX_NUM_SEQS)); then
        echo "Error: concurrency $concurrency exceeds max-num-seqs=$SERVER_MAX_NUM_SEQS." >&2
        exit 1
    fi
done

if ((DEPTH + PP_TOKENS + TG_TOKENS > SERVER_MAX_MODEL_LEN)); then
    echo "Error: requested context exceeds max-model-len=$SERVER_MAX_MODEL_LEN." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
RAW_CSV="$RAW_DIR/llama-benchy.csv"
BENCHY_LOG="$RAW_DIR/llama-benchy.log"

mkdir -p "$RAW_DIR"

if [[ "$VLLM_MANAGE_SERVER" == "1" ]]; then
    trap stop_vllm_server EXIT
fi

trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

if [[ "$VLLM_MANAGE_SERVER" == "1" ]]; then
    start_vllm_server "$SERVER_LOG" off truncate
else
    # Docker server 已經啟動，因此只再次確認連線。
    check_external_vllm_server
fi

nvidia-smi >"$RAW_DIR/nvidia-smi-ready.txt"

{
    echo "test=concurrency-scaling"
    echo "timestamp=$TIMESTAMP"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "pp_tokens=$PP_TOKENS"
    echo "tg_tokens=$TG_TOKENS"
    echo "depth=$DEPTH"
    echo "concurrency_levels=${CONCURRENCIES[*]}"
    echo "runs=$RUNS"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "prefix_caching=false"
    echo "client_cache_avoidance=true"
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

"$LLAMA_BENCHY" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL_ALIAS" \
    --served-model-name "$MODEL_ALIAS" \
    --tokenizer "$TOKENIZER" \
    --pp "$PP_TOKENS" \
    --tg "$TG_TOKENS" \
    --exact-tg \
    --depth "$DEPTH" \
    --concurrency "${CONCURRENCIES[@]}" \
    --runs "$RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --no-cache \
    --latency-mode generation \
    --save-result "$RAW_CSV" \
    --format csv \
    --exit-on-first-fail \
    2>&1 | tee "$BENCHY_LOG"

nvidia-smi >"$RAW_DIR/nvidia-smi-after.txt"

"$PLOT_PYTHON" "$SCRIPT_DIR/plot-concurrency-scaling.py" \
    --run-dir "$RUN_DIR" \
    --pp "$PP_TOKENS" \
    --tg "$TG_TOKENS" \
    --concurrencies "${CONCURRENCIES[@]}"

echo
echo "Concurrency-scaling benchmark completed: $RUN_DIR"