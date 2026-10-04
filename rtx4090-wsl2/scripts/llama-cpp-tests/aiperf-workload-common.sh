#!/usr/bin/env bash
set -Eeuo pipefail

: "${WORKLOAD_ID:?WORKLOAD_ID must be set}"
: "${RESULT_DIR_NAME:?RESULT_DIR_NAME must be set}"
: "${ISL_TOKENS:?ISL_TOKENS must be set}"
: "${OSL_TOKENS:?OSL_TOKENS must be set}"
: "${SERVER_PARALLEL:?SERVER_PARALLEL must be set}"
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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$WORKSPACE_ROOT/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
AIPERF_ROOT="${AIPERF_ROOT:-$WORKSPACE_ROOT/aiperf}"
AIPERF="${AIPERF:-$AIPERF_ROOT/.venv/bin/aiperf}"
MODEL_FILE="${MODEL_FILE:-$WORKSPACE_ROOT/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf}"

MODEL_ALIAS="${MODEL_ALIAS:-openai/gpt-oss-20b}"
TOKENIZER="${TOKENIZER:-openai/gpt-oss-20b}"
API_KEY="${API_KEY:-EMPTY}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8080}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
HEALTH_URL="$SERVER_URL/health"
MODELS_URL="$SERVER_URL/v1/models"

SERVER_CTX_SIZE="${SERVER_CTX_SIZE:-65536}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-300}"
BENCHMARK_DURATION_SECONDS="${BENCHMARK_DURATION_SECONDS:-120}"
PROFILE_RUNS="${PROFILE_RUNS:-3}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CELL_COOLDOWN_SECONDS="${CELL_COOLDOWN_SECONDS:-5}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/llama-cpp/aiperf/$RESULT_DIR_NAME}"
AIPERF_ISL_TOKENS="${AIPERF_ISL_TOKENS:-$ISL_TOKENS}"

for executable in "$LLAMA_SERVER" "$AIPERF"; do
    if [[ ! -x "$executable" ]]; then
        echo "Error: executable not found: $executable" >&2
        exit 1
    fi
done

if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Error: model not found: $MODEL_FILE" >&2
    exit 1
fi

for command_name in curl git nvidia-smi tee; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: command not found: $command_name" >&2
        exit 1
    fi
done

if (( ${#CONCURRENCIES[@]} == 0 )); then
    echo "Error: CONCURRENCIES cannot be empty." >&2
    exit 1
fi

for concurrency in "${CONCURRENCIES[@]}"; do
    if (( concurrency < 1 || concurrency > SERVER_PARALLEL )); then
        echo "Error: concurrency $concurrency exceeds server parallel $SERVER_PARALLEL." >&2
        exit 1
    fi
done

REQUIRED_CONTEXT_TOKENS=$(((ISL_TOKENS + OSL_TOKENS) * SERVER_PARALLEL))
if (( REQUIRED_CONTEXT_TOKENS > SERVER_CTX_SIZE )); then
    echo "Error: server context $SERVER_CTX_SIZE is below the minimum $REQUIRED_CONTEXT_TOKENS." >&2
    exit 1
fi

EXISTING_HTTP_STATUS="$(
    curl --silent --output /dev/null --write-out '%{http_code}' \
        --max-time 2 "$HEALTH_URL" 2>/dev/null || true
)"

if [[ "$EXISTING_HTTP_STATUS" != "000" ]]; then
    echo "Error: an HTTP service is already responding on port $SERVER_PORT." >&2
    echo "Stop it before running this benchmark." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
COMMANDS_FILE="$RUN_DIR/commands.txt"

mkdir -p "$RAW_DIR"
echo "Results directory: $RUN_DIR"

SERVER_ARGS=(
    --model "$MODEL_FILE"
    --alias "$MODEL_ALIAS"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --n-gpu-layers 99
    --flash-attn on
    --ctx-size "$SERVER_CTX_SIZE"
    --parallel "$SERVER_PARALLEL"
    --kv-unified
)

SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping llama-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        for ((wait_count = 0; wait_count < 50; wait_count++)); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "llama-server did not stop in time; sending SIGKILL." >&2
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

echo "Starting llama-server with parallel=$SERVER_PARALLEL..."
SERVER_START_MS="$(date +%s%3N)"
"$LLAMA_SERVER" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
MAX_READY_CHECKS=$((SERVER_READY_TIMEOUT_SECONDS * 2))
for ((check = 1; check <= MAX_READY_CHECKS; check++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: llama-server exited before becoming ready." >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi

    HEALTH_STATUS="$(
        curl --silent --output /dev/null --write-out '%{http_code}' \
            --max-time 2 "$HEALTH_URL" 2>/dev/null || true
    )"
    MODELS_STATUS="$(
        curl --silent --output /dev/null --write-out '%{http_code}' \
            --max-time 2 -H "Authorization: Bearer $API_KEY" \
            "$MODELS_URL" 2>/dev/null || true
    )"

    if [[ "$HEALTH_STATUS" == "200" && "$MODELS_STATUS" == "200" ]]; then
        READY=1
        break
    fi
    sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
    echo "Error: llama-server was not ready within ${SERVER_READY_TIMEOUT_SECONDS}s." >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
fi

SERVER_READY_MS="$(date +%s%3N)"
SERVER_READY_TIME_MS=$((SERVER_READY_MS - SERVER_START_MS))
echo "llama-server is ready after ${SERVER_READY_TIME_MS} ms."
nvidia-smi >"$RAW_DIR/nvidia-smi-ready.txt"

LLAMA_CPP_COMMIT="$(git -C "$LLAMA_CPP_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
AIPERF_COMMIT="$(git -C "$AIPERF_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
AIPERF_VERSION="$("$AIPERF" --version)"

{
    echo "test=$WORKLOAD_ID"
    echo "timestamp=$TIMESTAMP"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "model_file=$MODEL_FILE"
    echo "model_alias=$MODEL_ALIAS"
    echo "model_size_bytes=$(stat --format='%s' "$MODEL_FILE")"
    echo "tokenizer=$TOKENIZER"
    echo "llama_cpp_commit=$LLAMA_CPP_COMMIT"
    echo "aiperf_commit=$AIPERF_COMMIT"
    echo "aiperf_version=$AIPERF_VERSION"
    echo "server_url=$SERVER_URL"
    echo "server_parallel=$SERVER_PARALLEL"
    echo "server_ctx_size=$SERVER_CTX_SIZE"
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
    echo "server_ready_time_ms=$SERVER_READY_TIME_MS"
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
        --format=csv,noheader
    echo
    echo "[server_command]"
    printf '%q ' "$LLAMA_SERVER" "${SERVER_ARGS[@]}"
    echo
} >"$METADATA_FILE"

EXTRA_INPUTS="{\"temperature\":0,\"min_tokens\":$OSL_TOKENS,\"ignore_eos\":true}"

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
        --goodput "time_to_first_token:$TTFT_SLO_MS inter_token_latency:$ITL_SLO_MS"
        --extra-inputs "$EXTRA_INPUTS"
        --api-key "$API_KEY"
        --artifact-dir "$cell_dir"
        --export-level records
        --ui none
    )

    echo
    echo "Running $WORKLOAD_ID at concurrency=$concurrency..."
    printf '%q ' "${AIPERF_COMMAND[@]}" | tee -a "$COMMANDS_FILE"
    echo | tee -a "$COMMANDS_FILE"
    "${AIPERF_COMMAND[@]}" 2>&1 | tee "$cell_log"
    nvidia-smi >"$RAW_DIR/nvidia-smi-concurrency-$concurrency-after.txt"

    if (( index + 1 < ${#CONCURRENCIES[@]} && CELL_COOLDOWN_SECONDS > 0 )); then
        echo "Cooling down for ${CELL_COOLDOWN_SECONDS}s..."
        sleep "$CELL_COOLDOWN_SECONDS"
    fi
done

nvidia-smi >"$RAW_DIR/nvidia-smi-after.txt"

echo
echo "AIPerf workload completed."
echo "Results:  $RUN_DIR"
echo "Metadata: $METADATA_FILE"
echo "Server:   $SERVER_LOG"
echo "Commands: $COMMANDS_FILE"
