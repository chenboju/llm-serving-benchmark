#!/usr/bin/env bash
set -Eeuo pipefail

WORKLOAD_ID="rag-aiperf"
RESULT_DIR_NAME="rag"
ISL_TOKENS=4096
OSL_TOKENS=256
CONCURRENCIES=(1 2 4 8)
SERVER_PARALLEL=8
TTFT_SLO_MS=2000
ITL_SLO_MS=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aiperf-workload-common.sh"
