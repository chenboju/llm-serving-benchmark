#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$HOME/LLM_infra/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
LLAMA_BENCHY_ROOT="${LLAMA_BENCHY_ROOT:-$HOME/LLM_infra/llama-benchy}"
LLAMA_BENCHY="${LLAMA_BENCHY:-$LLAMA_BENCHY_ROOT/.venv/bin/llama-benchy}"
MODEL_FILE="${MODEL_FILE:-$HOME/LLM_infra/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf}"

MODEL_ALIAS="openai/gpt-oss-20b"
SERVER_HOST="127.0.0.1"
SERVER_PORT=8080
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"

RESULTS_ROOT="$PROJECT_ROOT/results/llama-cpp/concurrency"

PP_TOKENS=2048
TG_TOKENS=128
DEPTH=0
CONCURRENCIES=(1 2 4 8 16)
RUNS=3
WARMUP_RUNS=1
SERVER_PARALLEL=16
SERVER_READY_TIMEOUT_SECONDS=300

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

for executable in "$LLAMA_SERVER" "$LLAMA_BENCHY"; do
    if [[ ! -x "$executable" ]]; then
        echo "Error: executable not found: $executable" >&2
        exit 1
    fi
done

if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Error: model not found: $MODEL_FILE" >&2
    exit 1
fi

for command_name in curl nvidia-smi; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: command not found: $command_name" >&2
        exit 1
    fi
done

EXISTING_HTTP_STATUS="$(
    curl \
        --silent \
        --output /dev/null \
        --write-out '%{http_code}' \
        --max-time 2 \
        "$HEALTH_URL" 2>/dev/null || true
)"

if [[ "$EXISTING_HTTP_STATUS" != "000" ]]; then
    echo "Error: an HTTP service is already responding on port $SERVER_PORT." >&2
    echo "Stop it before running this benchmark." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Result paths
# ---------------------------------------------------------------------------

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
RAW_CSV="$RAW_DIR/llama-benchy.csv"
BENCHY_LOG="$RAW_DIR/llama-benchy.log"

mkdir -p "$RAW_DIR"

echo "Results directory: $RUN_DIR"

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

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

SERVER_ARGS=(
    --model "$MODEL_FILE"
    --alias "$MODEL_ALIAS"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --n-gpu-layers 99
    --flash-attn on
    --ctx-size 65536
    --parallel "$SERVER_PARALLEL"
    --kv-unified
)

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

echo "Starting llama-server..."
SERVER_START_MS="$(date +%s%3N)"

"$LLAMA_SERVER" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "llama-server PID: $SERVER_PID"
echo "Waiting for server readiness..."

READY=0
MAX_READY_CHECKS=$((SERVER_READY_TIMEOUT_SECONDS * 2))

for ((check = 1; check <= MAX_READY_CHECKS; check++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: llama-server exited before becoming ready." >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi

    HEALTH_STATUS="$(
        curl \
            --silent \
            --output /dev/null \
            --write-out '%{http_code}' \
            --max-time 2 \
            "$HEALTH_URL" 2>/dev/null || true
    )"

    if [[ "$HEALTH_STATUS" == "200" ]]; then
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

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

LLAMA_CPP_COMMIT="$(
    git -C "$LLAMA_CPP_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"
)"

LLAMA_CPP_TAG="$(
    git -C "$LLAMA_CPP_ROOT" describe --tags --exact-match HEAD 2>/dev/null ||
        echo "untagged"
)"

{
    echo "test=concurrency-scaling"
    echo "timestamp=$TIMESTAMP"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "model_file=$MODEL_FILE"
    echo "model_alias=$MODEL_ALIAS"
    echo "model_size_bytes=$(stat --format='%s' "$MODEL_FILE")"
    echo "llama_cpp_commit=$LLAMA_CPP_COMMIT"
    echo "llama_cpp_tag=$LLAMA_CPP_TAG"
    echo "llama_benchy_version=$("$LLAMA_BENCHY" --version)"
    echo "base_url=$BASE_URL"
    echo "server_parallel=$SERVER_PARALLEL"
    echo "server_ctx_size=65536"
    echo "n_gpu_layers=99"
    echo "flash_attn=on"
    echo "kv_unified=true"
    echo "pp_tokens=$PP_TOKENS"
    echo "tg_tokens=$TG_TOKENS"
    echo "depth=$DEPTH"
    echo "concurrency_levels=${CONCURRENCIES[*]}"
    echo "runs=$RUNS"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "prefix_cache_avoidance=true"
    echo "latency_mode=generation"
    echo "exact_tg=true"
    echo "server_ready_time_ms=$SERVER_READY_TIME_MS"
    echo
    echo "[gpu]"
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.used \
        --format=csv,noheader
    echo
    echo "[server_command]"
    printf '%q ' "$LLAMA_SERVER" "${SERVER_ARGS[@]}"
    echo
} >"$METADATA_FILE"

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

echo
echo "Starting concurrency-scaling benchmark..."
echo "Prompt tokens: $PP_TOKENS"
echo "Generation tokens: $TG_TOKENS"
echo "Concurrency levels: ${CONCURRENCIES[*]}"
echo "Runs per shape: $RUNS"
echo

"$LLAMA_BENCHY" \
    --base-url "$BASE_URL" \
    --model "$MODEL_ALIAS" \
    --served-model-name "$MODEL_ALIAS" \
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

echo
echo "Benchmark completed."
echo "Raw CSV:  $RAW_CSV"
echo "Metadata: $METADATA_FILE"
echo "Server:   $SERVER_LOG"
echo "Raw log:  $BENCHY_LOG"
