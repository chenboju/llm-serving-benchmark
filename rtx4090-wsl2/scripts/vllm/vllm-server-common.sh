#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"

# 僅在 VLLM_MANAGE_SERVER=1，由 WSL 啟動 native vLLM 時使用。
VLLM_ROOT="${VLLM_ROOT:-$WORKSPACE_ROOT/vllm-gpt-oss}"
VLLM="${VLLM:-$VLLM_ROOT/.venv/bin/vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-$VLLM_ROOT/.venv/bin/python}"

# 使用與 Docker volume 相同的 WSL Hugging Face cache。
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/huggingface/hub/models--openai--gpt-oss-20b}"
MODEL_ALIAS="${MODEL_ALIAS:-gpt-oss-20b}"
API_KEY="${API_KEY:-EMPTY}"

# 連接已經運行的 Docker vLLM。
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18000}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
BASE_URL="$SERVER_URL/v1"
HEALTH_URL="$SERVER_URL/health"
MODELS_URL="$SERVER_URL/v1/models"

# 必須和 Docker Compose 的 server 參數一致。
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-65536}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-600}"

# 0：連接既有 Docker server；1：由 Bash 啟停 WSL native vLLM。
VLLM_MANAGE_SERVER="${VLLM_MANAGE_SERVER:-0}"

# Docker Compose 已使用 --no-enable-prefix-caching。
EXTERNAL_PREFIX_CACHE_MODE="${EXTERNAL_PREFIX_CACHE_MODE:-off}"

normalize_wsl_path() {
    local candidate="$1"

    case "$candidate" in
        [A-Za-z]:\\*)
            if ! command -v wslpath >/dev/null 2>&1; then
                echo "Error: wslpath is required for Windows model paths." >&2
                return 1
            fi
            candidate="$(wslpath -u "$candidate")"
            ;;
    esac

    printf '%s\n' "$candidate"
}

resolve_model_path() {
    local candidate
    local revision

    candidate="$(normalize_wsl_path "${MODEL_PATH:-$MODEL_CACHE_ROOT}")"

    if [[ -f "$candidate/config.json" ]]; then
        printf '%s\n' "$candidate"
        return
    fi

    if [[ -f "$candidate/refs/main" ]]; then
        revision="$(tr -d '\r\n' <"$candidate/refs/main")"
        candidate="$candidate/snapshots/$revision"
    fi

    if [[ ! -f "$candidate/config.json" ]]; then
        echo "Error: no Hugging Face model snapshot found at: $candidate" >&2
        return 1
    fi

    printf '%s\n' "$candidate"
}

MODEL_PATH="$(resolve_model_path)"
MODEL_REVISION="$(basename "$MODEL_PATH")"
TOKENIZER="${TOKENIZER:-$MODEL_PATH}"

require_executable() {
    local executable="$1"
    if [[ ! -x "$executable" ]]; then
        echo "Error: executable not found: $executable" >&2
        return 1
    fi
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: command not found: $command_name" >&2
        return 1
    fi
}

validate_model_snapshot() {
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "Error: model config not found: $MODEL_PATH/config.json" >&2
        return 1
    fi

    if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
        echo "Error: model weight index not found in: $MODEL_PATH" >&2
        return 1
    fi

    "$VLLM_PYTHON" - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

model_path = pathlib.Path(sys.argv[1])
index_path = model_path / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(f"Error: no weight shards are referenced by {index_path}")
missing = [name for name in shards if not (model_path / name).is_file()]
if missing:
    raise SystemExit("Error: missing model shards: " + ", ".join(missing))
PY
}

model_size_bytes() {
    "$VLLM_PYTHON" - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

model_path = pathlib.Path(sys.argv[1])
index = json.loads(
    (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
)
print(sum((model_path / name).stat().st_size for name in set(index["weight_map"].values())))
PY
}

preflight_vllm() {
    require_executable "$VLLM"
    require_executable "$VLLM_PYTHON"
    for command_name in curl nvidia-smi setsid; do
        require_command "$command_name"
    done
    validate_model_snapshot
}

check_no_existing_service() {
    local http_status
    http_status="$(
        curl --silent --output /dev/null --write-out '%{http_code}' \
            --max-time 2 "$HEALTH_URL" 2>/dev/null || true
    )"

    if [[ "$http_status" != "000" ]]; then
        echo "Error: an HTTP service is already responding on port $SERVER_PORT." >&2
        echo "Stop it before running this benchmark." >&2
        return 1
    fi
}

VLLM_SERVER_ARGS=()
build_vllm_server_args() {
    local prefix_cache_mode="${1:-off}"

    VLLM_SERVER_ARGS=(
        "$MODEL_PATH"
        --served-model-name "$MODEL_ALIAS"
        --host "$SERVER_HOST"
        --port "$SERVER_PORT"
        --api-key "$API_KEY"
        --dtype "$VLLM_DTYPE"
        --max-model-len "$SERVER_MAX_MODEL_LEN"
        --max-num-seqs "$SERVER_MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --generation-config vllm
    )

    case "$prefix_cache_mode" in
        on)
            VLLM_SERVER_ARGS+=(--enable-prefix-caching)
            ;;
        off)
            VLLM_SERVER_ARGS+=(--no-enable-prefix-caching)
            ;;
        *)
            echo "Error: prefix cache mode must be 'on' or 'off': $prefix_cache_mode" >&2
            return 1
            ;;
    esac
}

SERVER_PID=""
SERVER_READY_TIME_MS=""
SERVER_ENGINE_READY_TIME_MS=""

server_group_alive() {
    [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null
}

stop_vllm_server() {
    if ! server_group_alive; then
        SERVER_PID=""
        return
    fi

    echo "Stopping vLLM server process group $SERVER_PID..."
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true

    for ((wait_count = 0; wait_count < 300; wait_count++)); do
        if ! server_group_alive; then
            break
        fi
        sleep 0.1
    done

    if server_group_alive; then
        echo "vLLM did not stop in time; sending SIGKILL." >&2
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi

    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

start_vllm_server() {
    local server_log="$1"
    local prefix_cache_mode="${2:-off}"
    local log_mode="${3:-truncate}"
    local start_ms
    local now_ms
    local health_status
    local models_status
    local max_ready_checks=$((SERVER_READY_TIMEOUT_SECONDS * 2))

    build_vllm_server_args "$prefix_cache_mode"
    mkdir -p "$(dirname "$server_log")"
    if [[ "$log_mode" == "truncate" ]]; then
        : >"$server_log"
    fi

    SERVER_READY_TIME_MS=""
    SERVER_ENGINE_READY_TIME_MS=""

    echo "Starting vLLM with max-num-seqs=$SERVER_MAX_NUM_SEQS, prefix-cache=$prefix_cache_mode..."
    start_ms="$(date +%s%3N)"
    setsid "$VLLM" serve "${VLLM_SERVER_ARGS[@]}" >>"$server_log" 2>&1 &
    SERVER_PID=$!

    for ((check = 1; check <= max_ready_checks; check++)); do
        if ! server_group_alive; then
            echo "Error: vLLM exited before becoming ready." >&2
            tail -n 100 "$server_log" >&2 || true
            return 1
        fi

        now_ms="$(date +%s%3N)"
        if [[ -z "$SERVER_ENGINE_READY_TIME_MS" ]] && \
            grep -Eqi 'engine.*ready|application startup complete|starting vllm api server' "$server_log"; then
            SERVER_ENGINE_READY_TIME_MS=$((now_ms - start_ms))
        fi

        health_status="$(
            curl --silent --output /dev/null --write-out '%{http_code}' \
                --max-time 2 "$HEALTH_URL" 2>/dev/null || true
        )"
        models_status="$(
            curl --silent --output /dev/null --write-out '%{http_code}' \
                --max-time 2 -H "Authorization: Bearer $API_KEY" \
                "$MODELS_URL" 2>/dev/null || true
        )"

        if [[ "$health_status" == "200" && "$models_status" == "200" ]]; then
            SERVER_READY_TIME_MS=$((now_ms - start_ms))
            if [[ -z "$SERVER_ENGINE_READY_TIME_MS" ]]; then
                SERVER_ENGINE_READY_TIME_MS="$SERVER_READY_TIME_MS"
            fi
            echo "vLLM is ready after ${SERVER_READY_TIME_MS} ms."
            return
        fi

        sleep 0.5
    done

    echo "Error: vLLM was not ready within ${SERVER_READY_TIMEOUT_SECONDS}s." >&2
    tail -n 100 "$server_log" >&2 || true
    return 1
}

write_vllm_metadata() {
    echo "model_cache_root=$MODEL_CACHE_ROOT"
    echo "model_path=$MODEL_PATH"
    echo "model_revision=$MODEL_REVISION"
    echo "model_alias=$MODEL_ALIAS"
    echo "model_size_bytes=$(model_size_bytes)"
    echo "tokenizer=$TOKENIZER"
    echo "vllm_version=$($VLLM --version)"
    echo "server_url=$SERVER_URL"
    echo "server_max_model_len=$SERVER_MAX_MODEL_LEN"
    echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "dtype=$VLLM_DTYPE"
    echo "generation_config=vllm"
    echo
    echo "[server_command]"
    printf '%q ' "$VLLM" serve "${VLLM_SERVER_ARGS[@]}"
    echo
}
