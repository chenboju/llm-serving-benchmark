#!/usr/bin/env bash
set -Eeuo pipefail

WORKLOAD_ID="interactive-aiperf"
RESULT_DIR_NAME="interactive"

# Interactive workload：
# 512 input tokens、128 output tokens。
ISL_TOKENS=512
OSL_TOKENS=128

# 必須在 Docker 的 --max-num-seqs=16 範圍內。
CONCURRENCIES=(1 2 4 8 16)
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-16}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-65536}"

# Goodput SLO。
TTFT_SLO_MS=1000
ITL_SLO_MS=50

# 使用已經運行的 Docker vLLM。
VLLM_MANAGE_SERVER="${VLLM_MANAGE_SERVER:-0}"
EXTERNAL_PREFIX_CACHE_MODE="${EXTERNAL_PREFIX_CACHE_MODE:-off}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18000}"
MODEL_ALIAS="${MODEL_ALIAS:-gpt-oss-20b}"
API_KEY="${API_KEY:-EMPTY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aiperf-workload-common.sh"