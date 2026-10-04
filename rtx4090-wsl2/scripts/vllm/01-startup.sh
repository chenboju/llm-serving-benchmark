#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vllm-server-common.sh"

preflight_vllm
check_no_existing_service

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/results/vllm/startup/$RUN_ID"
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
METADATA_FILE="$OUTPUT_DIR/metadata.txt"
mkdir -p "$OUTPUT_DIR"

trap stop_vllm_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'attempt,engine_ready_time_ms,api_ready_time_ms,baseline_vram_mib,ready_vram_mib,idle_vram_mib,status\n' \
    >"$SUMMARY_CSV"

for attempt in 1 2 3; do
    server_log="$OUTPUT_DIR/startup-$attempt.log"
    baseline_vram_mib="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            head -n 1 | tr -d ' '
    )"

    echo "Startup attempt $attempt..."
    start_vllm_server "$server_log" off truncate
    sleep 1

    ready_vram_mib="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            head -n 1 | tr -d ' '
    )"
    idle_vram_mib=$((ready_vram_mib - baseline_vram_mib))

    printf '%s,%s,%s,%s,%s,%s,success\n' \
        "$attempt" "$SERVER_ENGINE_READY_TIME_MS" "$SERVER_READY_TIME_MS" \
        "$baseline_vram_mib" "$ready_vram_mib" "$idle_vram_mib" \
        >>"$SUMMARY_CSV"

    stop_vllm_server
    sleep 2
done

build_vllm_server_args off
{
    echo "test=startup"
    echo "timestamp=$RUN_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "attempts=3"
    echo "prefix_caching=false"
    write_vllm_metadata
} >"$METADATA_FILE"

trap - EXIT INT TERM

echo
echo "Startup benchmark completed: $OUTPUT_DIR"
cat "$SUMMARY_CSV"
