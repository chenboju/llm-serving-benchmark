#!/usr/bin/env bash
set -Eeuo pipefail

WORKLOAD_ID="long-context-aiperf"
RESULT_DIR_NAME="long-context"

# Long-context workload：
# 16384 input tokens、128 output tokens。
ISL_TOKENS=16384
OSL_TOKENS=128

# Long-context 測試最高 concurrency 為 2；
# server 實際的 --max-num-seqs 為 16。
CONCURRENCIES=(1 2)
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-16}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-65536}"

# Goodput SLO。
TTFT_SLO_MS=5000
ITL_SLO_MS=75

# 使用已經運行的 Docker vLLM。
VLLM_MANAGE_SERVER="${VLLM_MANAGE_SERVER:-0}"
EXTERNAL_PREFIX_CACHE_MODE="${EXTERNAL_PREFIX_CACHE_MODE:-off}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18000}"
MODEL_ALIAS="${MODEL_ALIAS:-gpt-oss-20b}"
API_KEY="${API_KEY:-EMPTY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aiperf-workload-common.sh"