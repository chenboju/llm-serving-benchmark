#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$HOME/LLM_infra/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
LLAMA_BENCHY="${LLAMA_BENCHY:-$HOME/LLM_infra/llama-benchy/.venv/bin/llama-benchy}"
MODEL_FILE="${MODEL_FILE:-$HOME/LLM_infra/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf}"

MODEL_ALIAS="openai/gpt-oss-20b"
SERVER_HOST="127.0.0.1"
SERVER_PORT=8080
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"

RESULTS_ROOT="$PROJECT_ROOT/results/llama-cpp/prefix-cache"

PP_TOKENS=512
TG_TOKENS=64
DEPTHS=(4096 16384)
RUNS=3
WARMUP_RUNS=1
SERVER_PARALLEL=1
SERVER_READY_TIMEOUT_SECONDS=300

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
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
READY_TIMES_CSV="$RAW_DIR/server-ready-times.csv"

mkdir -p "$RAW_DIR"
echo "case,ready_time_ms" >"$READY_TIMES_CSV"

echo "Results directory: $RUN_DIR"

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

SERVER_PID=""

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping llama-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true

        for ((wait_count = 0; wait_count < 50; wait_count++)); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""
}

cleanup() {
    stop_server
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
    local case_name="$1"
    local server_start_ms
    local server_ready_ms
    local ready_time_ms
    local health_status
    local ready=0
    local max_ready_checks=$((SERVER_READY_TIMEOUT_SECONDS * 2))

    {
        echo
        echo "================================================================"
        echo "case=$case_name"
        echo "started_at=$(date --iso-8601=seconds)"
        echo "================================================================"
    } >>"$SERVER_LOG"

    echo "Starting llama-server for $case_name..."
    server_start_ms="$(date +%s%3N)"

    "$LLAMA_SERVER" "${SERVER_ARGS[@]}" >>"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    for ((check = 1; check <= max_ready_checks; check++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: llama-server exited during $case_name." >&2
            tail -n 80 "$SERVER_LOG" >&2 || true
            exit 1
        fi

        health_status="$(
            curl \
                --silent \
                --output /dev/null \
                --write-out '%{http_code}' \
                --max-time 2 \
                "$HEALTH_URL" 2>/dev/null || true
        )"

        if [[ "$health_status" == "200" ]]; then
            ready=1
            break
        fi

        sleep 0.5
    done

    if [[ "$ready" -ne 1 ]]; then
        echo "Error: server readiness timeout during $case_name." >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi

    server_ready_ms="$(date +%s%3N)"
    ready_time_ms=$((server_ready_ms - server_start_ms))

    echo "$case_name,$ready_time_ms" >>"$READY_TIMES_CSV"
    echo "llama-server ready for $case_name after ${ready_time_ms} ms."
}

run_benchmark() {
    local cache_mode="$1"
    local depth="$2"
    local case_name="${cache_mode}-d${depth}"
    local result_csv="$RAW_DIR/${case_name}.csv"
    local bench_log="$RAW_DIR/${case_name}.log"
    local cache_args=()

    if [[ "$cache_mode" == "cache-off" ]]; then
        cache_args=(--no-cache)
    elif [[ "$cache_mode" == "cache-on" ]]; then
        cache_args=(--enable-prefix-caching)
    else
        echo "Error: unknown cache mode: $cache_mode" >&2
        exit 1
    fi

    start_server "$case_name"

    echo
    echo "Running $case_name..."
    echo "PP=$PP_TOKENS TG=$TG_TOKENS depth=$depth"

    "$LLAMA_BENCHY" \
        --base-url "$BASE_URL" \
        --model "$MODEL_ALIAS" \
        --served-model-name "$MODEL_ALIAS" \
        --pp "$PP_TOKENS" \
        --tg "$TG_TOKENS" \
        --exact-tg \
        --depth "$depth" \
        --concurrency 1 \
        --runs "$RUNS" \
        --warmup-runs "$WARMUP_RUNS" \
        --latency-mode generation \
        --save-result "$result_csv" \
        --format csv \
        --exit-on-first-fail \
        "${cache_args[@]}" \
        2>&1 | tee "$bench_log"

    stop_server
}

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

LLAMA_CPP_COMMIT="$(
    git -C "$LLAMA_CPP_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"
)"

LLAMA_CPP_TAG="$(
    git -C "$LLAMA_CPP_ROOT" describe --tags --exact-match HEAD 2>/dev/null ||
        echo "untagged"
)"

{
    echo "test=prefix-cache"
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
    echo "depths=${DEPTHS[*]}"
    echo "cache_modes=cache-off cache-on"
    echo "concurrency=1"
    echo "runs=$RUNS"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "latency_mode=generation"
    echo "exact_tg=true"
    echo "server_restart_policy=fresh server per depth and cache mode"
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

for depth in "${DEPTHS[@]}"; do
    run_benchmark "cache-off" "$depth"
    run_benchmark "cache-on" "$depth"
done

nvidia-smi >"$RAW_DIR/nvidia-smi-after.txt"

echo
echo "Prefix-cache benchmark completed."
echo "Results:  $RUN_DIR"
echo "Metadata: $METADATA_FILE"
echo "Server:   $SERVER_LOG"
echo "Raw data: $RAW_DIR"
