#!/usr/bin/env bash
# Probe what this llama.cpp build actually supports on Ryzen AI Max+ 395, then
# print the llama_server block to paste into benchmark-config.json.
#
# Nothing here is assumed: every flag is tested by booting a real server and
# waiting for /health. Run this before the first campaign, and again after any
# llama.cpp rebuild or driver update.
#
# Usage: bash probe_llama_server.sh [--full]
#   --full  also probe the 65536-context / parallel-16 configuration the
#           campaign uses (slow: loads the whole KV cache)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/LLM_infra}"
MODEL_FILE="${MODEL_FILE:-$WORKSPACE_ROOT/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf}"
PROBE_PORT="${PROBE_PORT:-8099}"
PROBE_URL="http://127.0.0.1:${PROBE_PORT}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/results/probe}"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

resolve_server() {
    if [[ -n "${LLAMA_SERVER:-}" ]]; then
        printf '%s\n' "$LLAMA_SERVER"
        return
    fi
    for candidate in \
        "$WORKSPACE_ROOT/llama.cpp/build/bin/llama-server.exe" \
        "$WORKSPACE_ROOT/llama.cpp/build/bin/llama-server" \
        "$WORKSPACE_ROOT/llama.cpp/build/bin/Release/llama-server.exe"; do
        [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
    done
    echo "Error: llama-server not found; set LLAMA_SERVER" >&2
    exit 1
}

LLAMA_SERVER_BIN="$(resolve_server)"
[[ -f "$MODEL_FILE" ]] || { echo "Error: model not found: $MODEL_FILE" >&2; exit 1; }
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "llama-server: $LLAMA_SERVER_BIN"
echo "model:        $MODEL_FILE"
echo "logs:         $LOG_DIR"
echo "================================================================"
echo
echo "--- build identity ---"
"$LLAMA_SERVER_BIN" --version 2>&1 | head -20 || true
echo
echo "--- devices reported by this build ---"
"$LLAMA_SERVER_BIN" --list-devices 2>&1 | head -30 || echo "(--list-devices not supported by this build)"
echo

# Phase A: does the binary even accept the flag?
echo "--- flag acceptance (--help) ---"
help_text="$("$LLAMA_SERVER_BIN" --help 2>&1 || true)"
declare -A FLAG_IN_HELP=()
for flag in --flash-attn --kv-unified --cache-prompt --no-cache-prompt --parallel --n-gpu-layers; do
    if grep -q -- "$flag" <<<"$help_text"; then
        FLAG_IN_HELP["$flag"]=yes
        printf '  %-18s present\n' "$flag"
    else
        FLAG_IN_HELP["$flag"]=no
        printf '  %-18s ABSENT\n' "$flag"
    fi
done
echo

# Phase B: boot a real server and wait for /health.
wait_for_health() {
    local pid="$1" deadline=$((SECONDS + ${2:-300}))
    while ((SECONDS < deadline)); do
        kill -0 "$pid" 2>/dev/null || return 1
        if [[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 2 "$PROBE_URL/health" || true)" == "200" ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

try_launch() {
    local label="$1"; shift
    local log="$LOG_DIR/probe-${label}.log"
    printf '  %-28s ' "$label"
    "$LLAMA_SERVER_BIN" \
        --model "$MODEL_FILE" \
        --host 127.0.0.1 --port "$PROBE_PORT" \
        --n-gpu-layers 99 \
        "$@" >"$log" 2>&1 &
    local pid=$!
    if wait_for_health "$pid" 420; then
        local reply
        reply="$(curl --silent --max-time 60 "$PROBE_URL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"probe","messages":[{"role":"user","content":"Reply OK."}],"max_completion_tokens":8,"temperature":0,"stream":false}' || true)"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        if grep -q '"choices"' <<<"$reply"; then
            echo "OK"
            return 0
        fi
        echo "BOOTED but completion failed (see $log)"
        return 1
    fi
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "FAILED (see $log)"
    return 1
}

echo "--- boot tests, ctx=4096 ---"
FLASH_ATTN_RESULT="unset"
try_launch baseline --ctx-size 4096 --parallel 1 --no-cache-prompt || true
if [[ "${FLAG_IN_HELP[--flash-attn]}" == "yes" ]]; then
    if try_launch flash-attn-on --ctx-size 4096 --parallel 1 --no-cache-prompt --flash-attn on; then
        FLASH_ATTN_RESULT="on"
    elif try_launch flash-attn-off --ctx-size 4096 --parallel 1 --no-cache-prompt --flash-attn off; then
        FLASH_ATTN_RESULT="off"
    fi
fi
KV_UNIFIED_RESULT="false"
if [[ "${FLAG_IN_HELP[--kv-unified]}" == "yes" ]]; then
    try_launch kv-unified --ctx-size 4096 --parallel 1 --no-cache-prompt --kv-unified \
        && KV_UNIFIED_RESULT="true"
fi
if [[ "${FLAG_IN_HELP[--cache-prompt]}" == "yes" ]]; then
    try_launch cache-prompt-on --ctx-size 4096 --parallel 1 --cache-prompt || true
fi

CAMPAIGN_CTX_OK="untested"
if ((FULL)); then
    echo
    echo "--- boot test, campaign configuration (ctx=65536, parallel=16) ---"
    campaign_args=(--ctx-size 65536 --parallel 16 --no-cache-prompt)
    [[ "$FLASH_ATTN_RESULT" != "unset" ]] && campaign_args+=(--flash-attn "$FLASH_ATTN_RESULT")
    [[ "$KV_UNIFIED_RESULT" == "true" ]] && campaign_args+=(--kv-unified)
    if try_launch campaign-65k "${campaign_args[@]}"; then
        CAMPAIGN_CTX_OK="ok"
    else
        CAMPAIGN_CTX_OK="FAILED"
    fi
fi

echo
echo "================================================================"
echo "Paste into benchmark-config.json as llama_server, after setting"
echo "\"backend\" to what --list-devices actually showed (rocm/vulkan/cpu):"
echo "================================================================"
cat <<JSON
  "llama_server": {
    "backend": "SET_ME_FROM_LIST_DEVICES_ABOVE",
    "n_gpu_layers": 99,
    "ctx_size": 65536,
    "parallel": 16,
    "kv_unified": $KV_UNIFIED_RESULT,
    "flash_attn": "$FLASH_ATTN_RESULT",
    "extra_args": [],
    "startup_timeout_seconds": 900
  }
JSON
echo
echo "campaign ctx=65536/parallel=16 boot: $CAMPAIGN_CTX_OK"
if [[ "$CAMPAIGN_CTX_OK" == "untested" ]]; then
    echo "Re-run with --full before the first real campaign; a 65536-token KV cache"
    echo "at parallel=16 is the configuration that actually has to fit."
fi
echo
echo "If --list-devices showed no GPU, STOP. llama.cpp will run on CPU and the"
echo "numbers will not mean anything for a GPU comparison."
