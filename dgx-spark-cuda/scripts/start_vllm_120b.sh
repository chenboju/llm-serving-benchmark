#!/bin/bash

# ==========================================
# vLLM 伺服器啟動腳本：GPT-OSS-120B baseline（DGX Spark / ARM64）
# API: http://127.0.0.1:18001/v1
#
# 這是 DFlash A/B 的對照組。跟 start_vllm_120b_dflash.sh 只差
# --speculative-config 一項，其餘旗標必須逐字相同，否則兩邊不可比。
# ==========================================

set -Eeuo pipefail

export TARGET_MODEL="openai/gpt-oss-120b"
export SERVED_MODEL_NAME="gpt-oss-120b"
export HOST_PORT="18001"

# max-num-seqs 是這台機器的自變數，刻意不給預設值。
# 設錯不會讓任何東西壞掉 —— 它會安靜地產出看起來正常、實際上在量客戶端佇列的
# 資料。20B 那 30 格 stress 就是這麼來的：Running 16 / Waiting 48 / KV 使用率
# 0.3%，68 GiB 的池子有 91% 從沒被碰過。
#   comparison scope（c<=16）：16，與 20B campaign 對齊
#   stress scope：>= 該格的 concurrency，否則量到的不是這台機器
export MAX_NUM_SEQS="${MAX_NUM_SEQS:?請顯式指定，例如 MAX_NUM_SEQS=64 bash $0}"
case "$MAX_NUM_SEQS" in
  ''|*[!0-9]*) echo "錯誤：MAX_NUM_SEQS 必須是正整數，收到 '$MAX_NUM_SEQS'" >&2; exit 1 ;;
esac

# 容器名帶組態，避免不同 max-num-seqs 的 campaign 誤用同一個容器 ——
# 那種錯誤在結果資料上完全看不出來。
export CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_120b_vllm_s$(printf '%03d' "$MAX_NUM_SEQS")}"

# 與 20B campaign 同一個 digest，確保跨模型比較時 vLLM 版本是同一個。
# 若這個 build 不支援 dflash，dflash 那支腳本會需要換映像 —— 那時兩支都要換，
# 並在報告偏離事項表記一筆。
export DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai@sha256:928dde7d6c19478c9c6ab975fed12e6466f6d9b7cb3695d218e287c0c88ad608}"
export TIKTOKEN_CACHE_DIR="$HOME/tiktoken_encodings"

# 權重 61 GiB。0.70（20B 用的值）只會剩約 21 GiB 給 KV，KV 池會變成瓶頸而汙染
# 「20B vs 120B」的解讀；0.85 約留 39 GiB。實際值以啟動日誌的
# "GPU KV cache size: N tokens" 為準，那個數字要抄進報告。
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# harness 用 docker inspect 讀 .Config.Cmd 比對，旗標必須顯式存在，
# 缺了會判為 unknown 並中止（run_benchmarks.py 的 detect_vllm_prefix_mode）。
export PREFIX_CACHING="${PREFIX_CACHING:-off}"
if [ "$PREFIX_CACHING" = "on" ]; then
  CACHE_FLAG="--enable-prefix-caching"
else
  CACHE_FLAG="--no-enable-prefix-caching"
fi

# 61 GiB 權重塞不進 20B 容器留下的空間。先擋掉，不要等 30 分鐘後才 OOM。
RUNNING="$(docker ps --format '{{.Names}}' --filter ancestor=vllm/vllm-openai || true)"
if [ -n "$RUNNING" ]; then
  echo "錯誤：還有 vLLM 容器在跑，先停掉再啟動 120B：" >&2
  echo "$RUNNING" | sed 's/^/  docker stop /' >&2
  exit 1
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "錯誤：容器 $CONTAINER_NAME 已存在。先留存證再刪除：" >&2
  echo "  docker inspect --format '{{json .Config.Cmd}}' $CONTAINER_NAME | tee ~/${CONTAINER_NAME}-cmd-\$(date +%Y%m%d).json" >&2
  echo "  docker rm $CONTAINER_NAME" >&2
  exit 1
fi

# openai-harmony 透過 TIKTOKEN_ENCODINGS_BASE 讀 vocab 時，找的是原始檔名
# "$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"，不是 tiktoken 的 SHA1 雜湊檔名。
# 少了這個，/v1/models 正常但 /v1/chat/completions 會回 500 vocab file 錯誤。
mkdir -p "$TIKTOKEN_CACHE_DIR"
TIKTOKEN_FILE="$TIKTOKEN_CACHE_DIR/o200k_base.tiktoken"
if [ ! -f "$TIKTOKEN_FILE" ]; then
  echo "下載 tiktoken vocab..."
  curl -L "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" -o "$TIKTOKEN_FILE"
fi

echo "正在啟動 vLLM 伺服器：GPT-OSS-120B（baseline，無投機解碼）"
echo "Model: $TARGET_MODEL"
echo "Container: $CONTAINER_NAME  Port: 127.0.0.1:$HOST_PORT"
echo "Docker Image: $DOCKER_IMAGE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Prefix Caching: $PREFIX_CACHING ($CACHE_FLAG)"
echo "=========================================="

# -d 而非 --rm：harness 會 docker start/stop/inspect 這個容器，--rm 一停就沒了。
docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=64gb \
  -p 127.0.0.1:$HOST_PORT:8000 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TIKTOKEN_RS_CACHE_DIR="/tiktoken_encodings" \
  -e TIKTOKEN_ENCODINGS_BASE="/tiktoken_encodings" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  -v "$TIKTOKEN_CACHE_DIR:/tiktoken_encodings" \
  $DOCKER_IMAGE \
    $TARGET_MODEL \
    --served-model-name $SERVED_MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    $CACHE_FLAG

echo "等待就緒（120B 權重 61 GiB，首次載入含 torch.compile 可能要數分鐘）..."
for i in $(seq 1 1800); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$HOST_PORT/health || true)" = "200" ]; then
    echo "就緒：http://127.0.0.1:$HOST_PORT/v1"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'KV cache size|Available KV cache|attention backend|MoE backend' || true
    exit 0
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
    echo "錯誤：容器已退出。docker logs $CONTAINER_NAME" >&2
    exit 1
  fi
  sleep 1
done
echo "錯誤：就緒逾時。docker logs -f $CONTAINER_NAME" >&2
exit 1
