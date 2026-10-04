#!/usr/bin/env bash
set -Eeuo pipefail

WORKLOAD_ID="long-context-aiperf"
RESULT_DIR_NAME="long-context"
ISL_TOKENS=16384
OSL_TOKENS=128
CONCURRENCIES=(1 2)
SERVER_PARALLEL=2
TTFT_SLO_MS=5000
ITL_SLO_MS=75

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aiperf-workload-common.sh"
