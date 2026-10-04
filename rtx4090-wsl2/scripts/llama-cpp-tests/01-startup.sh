#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/LLM_infra/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/LLM_infra/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/results/llama-cpp/startup/$RUN_ID"
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "找不到 llama-server：$LLAMA_SERVER"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "找不到模型：$MODEL"
    exit 1
fi

if curl -fsS http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "Port 8080 已有服務，請先停止目前的 llama-server。"
    exit 1
fi

SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT INT TERM

printf 'attempt,load_time_ms,ready_time_ms,baseline_vram_mib,ready_vram_mib,idle_vram_mib,status\n' \
    > "$SUMMARY_CSV"

for ATTEMPT in 1 2 3; do
    SERVER_LOG="$OUTPUT_DIR/startup-$ATTEMPT.log"

    echo "開始第 $ATTEMPT 次啟動"

    BASELINE_VRAM_MIB="$(
        nvidia-smi \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits |
        head -n 1 |
        tr -d ' '
    )"

    LOAD_TIME_MS=""
    START_NS="$(date +%s%N)"

    "$LLAMA_SERVER" \
        --model "$MODEL" \
        --alias openai/gpt-oss-20b \
        --host 127.0.0.1 \
        --port 8080 \
        --n-gpu-layers 99 \
        --flash-attn on \
        --ctx-size 65536 \
        --parallel 1 \
        --kv-unified \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    READY_TIME_MS=""

    for _ in $(seq 1 1800); do
        NOW_NS="$(date +%s%N)"

        if [[ -z "$LOAD_TIME_MS" ]] &&
           grep -qi "model loaded" "$SERVER_LOG"; then
            LOAD_TIME_MS=$(( (NOW_NS - START_NS) / 1000000 ))
        fi

        if curl -fsS http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
            READY_TIME_MS=$(( (NOW_NS - START_NS) / 1000000 ))
            break
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "llama-server 啟動失敗，請查看：$SERVER_LOG"
            exit 1
        fi

        sleep 0.2
    done

    if [[ -z "$READY_TIME_MS" ]]; then
        echo "等待 llama-server ready 超時"
        exit 1
    fi

    # 等待 GPU 記憶體用量穩定，這時 server 尚未收到測試請求。
    sleep 1

    READY_VRAM_MIB="$(
        nvidia-smi \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits |
        head -n 1 |
        tr -d ' '
    )"

    IDLE_VRAM_MIB=$((READY_VRAM_MIB - BASELINE_VRAM_MIB))

    printf '%s,%s,%s,%s,%s,%s,success\n' \
        "$ATTEMPT" \
        "$LOAD_TIME_MS" \
        "$READY_TIME_MS" \
        "$BASELINE_VRAM_MIB" \
        "$READY_VRAM_MIB" \
        "$IDLE_VRAM_MIB" \
        >> "$SUMMARY_CSV"

    echo "第 $ATTEMPT 次完成：ready=${READY_TIME_MS} ms，VRAM=${IDLE_VRAM_MIB} MiB"

    stop_server
    sleep 2
done

trap - EXIT INT TERM

echo
echo "啟動基準完成：$OUTPUT_DIR"
cat "$SUMMARY_CSV"
