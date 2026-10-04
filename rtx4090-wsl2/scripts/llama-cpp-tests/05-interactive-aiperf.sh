#!/usr/bin/env bash
set -Eeuo pipefail

WORKLOAD_ID="interactive-aiperf"
RESULT_DIR_NAME="interactive"
ISL_TOKENS=512
OSL_TOKENS=128
CONCURRENCIES=(1 2 4 8 16)
SERVER_PARALLEL=16
TTFT_SLO_MS=1000
ITL_SLO_MS=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aiperf-workload-common.sh"
