#!/usr/bin/env bash
set -Eeuo pipefail

PP_TOKENS="${PP_TOKENS:-512}"
TG_TOKENS="${TG_TOKENS:-64}"
DEPTHS=(4096 16384)
RUNS="${RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vllm-server-common.sh"

LLAMA_BENCHY="${LLAMA_BENCHY:-$WORKSPACE_ROOT/llama-benchy/.venv/bin/llama-benchy}"
PLOT_PYTHON="${PLOT_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/vllm/prefix-cache}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-gpt_oss_20b_vllm}"

require_executable "$LLAMA_BENCHY"
require_executable "$PLOT_PYTHON"

for command_name in curl docker grep nvidia-smi tee; do
    require_command "$command_name"
done

max_depth="$(printf '%s\n' "${DEPTHS[@]}" | sort -n | tail -n 1)"

if ((max_depth + PP_TOKENS + TG_TOKENS > SERVER_MAX_MODEL_LEN)); then
    echo "Error: requested context exceeds max-model-len=$SERVER_MAX_MODEL_LEN." >&2
    exit 1
fi

container_cache_mode() {
    local container_command

    container_command="$(
        docker inspect \
            --format '{{json .Config.Cmd}}' \
            "$VLLM_CONTAINER_NAME" 2>/dev/null
    )"

    if [[ "$container_command" == *"--no-enable-prefix-caching"* ]]; then
        printf 'off\n'
    elif [[ "$container_command" == *"--enable-prefix-caching"* ]]; then
        printf 'on\n'
    else
        printf 'unknown\n'
    fi
}

wait_for_vllm() {
    local start_ms="$1"
    local health_status
    local models_json
    local now_ms
    local max_checks=$((SERVER_READY_TIMEOUT_SECONDS * 2))

    for ((check = 1; check <= max_checks; check++)); do
        if [[ "$(
            docker inspect \
                --format '{{.State.Running}}' \
                "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        )" != "true" ]]; then
            sleep 0.5
            continue
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
            models_json="$(
                curl \
                    --silent \
                    --show-error \
                    --fail \
                    --max-time 5 \
                    -H "Authorization: Bearer $API_KEY" \
                    "$MODELS_URL" 2>/dev/null || true
            )"

            if printf '%s\n' "$models_json" |
                grep -Eq "\"id\"[[:space:]]*:[[:space:]]*\"${MODEL_ALIAS}\""; then
                now_ms="$(date +%s%3N)"
                SERVER_READY_TIME_MS=$((now_ms - start_ms))

                echo "vLLM is ready after ${SERVER_READY_TIME_MS} ms."
                return
            fi
        fi

        sleep 0.5
    done

    echo "Error: vLLM did not become ready within ${SERVER_READY_TIMEOUT_SECONDS}s." >&2
    docker logs --tail 100 "$VLLM_CONTAINER_NAME" >&2 || true
    return 1
}

restart_current_container() {
    local expected_mode="$1"
    local actual_mode
    local start_ms

    actual_mode="$(container_cache_mode)"

    if [[ "$actual_mode" != "$expected_mode" ]]; then
        echo "Error: expected prefix-cache=$expected_mode, actual=$actual_mode." >&2
        return 1
    fi

    echo
    echo "Restarting Docker container with prefix-cache=$expected_mode..."

    start_ms="$(date +%s%3N)"

    docker restart "$VLLM_CONTAINER_NAME" >/dev/null

    wait_for_vllm "$start_ms"
}

wait_for_cache_mode() {
    local expected_mode="$1"
    local actual_mode

    while true; do
        actual_mode="$(container_cache_mode)"

        if [[ "$actual_mode" == "$expected_mode" ]]; then
            echo "Confirmed prefix-cache=$expected_mode."
            return
        fi

        echo
        echo "Current prefix-cache mode: $actual_mode"
        echo "Required prefix-cache mode: $expected_mode"
        echo
        echo "Please update Docker to prefix-cache=$expected_mode and recreate it."

        if [[ "$expected_mode" == "on" ]]; then
            echo "Required flag: --enable-prefix-caching"
        else
            echo "Required flag: --no-enable-prefix-caching"
        fi

        echo
        read -r -p "After Docker is healthy, press Enter to check again..."
    done
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULTS_ROOT/$TIMESTAMP"
RAW_DIR="$RUN_DIR/raw"
SERVER_LOG="$RUN_DIR/server.log"
METADATA_FILE="$RUN_DIR/metadata.txt"
READY_TIMES_CSV="$RAW_DIR/server-ready-times.csv"

mkdir -p "$RAW_DIR"

printf 'case,ready_time_ms\n' >"$READY_TIMES_CSV"
: >"$SERVER_LOG"

trap 'exit 130' INT
trap 'exit 143' TERM

run_benchmark() {
    local cache_mode="$1"
    local depth="$2"
    local case_name="cache-${cache_mode}-d${depth}"
    local result_csv="$RAW_DIR/${case_name}.csv"
    local bench_log="$RAW_DIR/${case_name}.log"
    local client_cache_args=()

    case "$cache_mode" in
        off)
            client_cache_args=(--no-cache)
            ;;
        on)
            client_cache_args=(--enable-prefix-caching)
            ;;
        *)
            echo "Error: unknown cache mode: $cache_mode" >&2
            return 1
            ;;
    esac

    restart_current_container "$cache_mode"

    printf '%s,%s\n' \
        "$case_name" \
        "$SERVER_READY_TIME_MS" \
        >>"$READY_TIMES_CSV"

    nvidia-smi >"$RAW_DIR/nvidia-smi-${case_name}-ready.txt"

    "$LLAMA_BENCHY" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --model "$MODEL_ALIAS" \
        --served-model-name "$MODEL_ALIAS" \
        --tokenizer "$TOKENIZER" \
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
        "${client_cache_args[@]}" \
        2>&1 | tee "$bench_log"

    nvidia-smi >"$RAW_DIR/nvidia-smi-${case_name}-after.txt"

    {
        echo
        echo "===== $case_name ====="
        docker logs "$VLLM_CONTAINER_NAME" 2>&1 || true
    } >>"$SERVER_LOG"
}

nvidia-smi >"$RAW_DIR/nvidia-smi-before.txt"

{
    echo "test=prefix-cache"
    echo "timestamp=$TIMESTAMP"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "pp_tokens=$PP_TOKENS"
    echo "tg_tokens=$TG_TOKENS"
    echo "depths=${DEPTHS[*]}"
    echo "cache_modes=on off"
    echo "concurrency=1"
    echo "runs=$RUNS"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "server_mode=external-docker"
    echo "server_restart_policy=docker restart before every case"
    echo "server_cache_mode_switch=manual"
    echo "container_name=$VLLM_CONTAINER_NAME"
    echo "model_alias=$MODEL_ALIAS"
    echo "tokenizer=$TOKENIZER"
    echo "server_url=$SERVER_URL"
    echo "server_max_model_len=$SERVER_MAX_MODEL_LEN"
    echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS"
} >"$METADATA_FILE"

# 目前 Docker 已經是 prefix cache on，因此先測 on。
wait_for_cache_mode on

for depth in "${DEPTHS[@]}"; do
    run_benchmark on "$depth"
done

echo
echo "Prefix-cache ON cases completed."
echo
echo "In another terminal:"
echo "1. Change the Docker flag to --no-enable-prefix-caching"
echo "2. Recreate the Docker container"
echo "3. Wait until /health returns HTTP 200"
echo
read -r -p "After switching Docker to prefix-cache=off, press Enter..."

wait_for_cache_mode off

for depth in "${DEPTHS[@]}"; do
    run_benchmark off "$depth"
done

nvidia-smi >"$RAW_DIR/nvidia-smi-after.txt"

"$PLOT_PYTHON" "$SCRIPT_DIR/plot-prefix-cache.py" \
    --run-dir "$RUN_DIR" \
    --pp "$PP_TOKENS" \
    --tg "$TG_TOKENS" \
    --depths "${DEPTHS[@]}"

echo
echo "Prefix-cache benchmark completed: $RUN_DIR"
echo "Docker remains in prefix-cache=off mode."