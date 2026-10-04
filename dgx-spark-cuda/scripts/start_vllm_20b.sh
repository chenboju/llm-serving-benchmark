#!/bin/bash

# ==========================================
# vLLM 伺服器啟動腳本：GPT-OSS-20B（DGX Spark / ARM64）
# API: http://127.0.0.1:18000/v1
# ==========================================

export TARGET_MODEL="openai/gpt-oss-20b"
export SERVED_MODEL_NAME="gpt-oss-20b"
# vllm/vllm-openai:nightly 於 2026-08-07 拉取，vLLM build commit c810e5ee9976ad86b81d1277b53e76d0ee639414
# 用 digest 而非 :nightly，確保日後標籤移動也還是同一個映像（本機已有，不會觸發下載）
export DOCKER_IMAGE="vllm/vllm-openai@sha256:928dde7d6c19478c9c6ab975fed12e6466f6d9b7cb3695d218e287c0c88ad608"
export TIKTOKEN_CACHE_DIR="$HOME/tiktoken_encodings"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# capacity 掃描的自變數，刻意不給預設值。設錯不會壞掉，只會安靜地量到客戶端
# 佇列 —— formal-stress-65k-20260817-dgx 那 30 格就是在 16 之下跑的。
# 重跑 formal-split 的 prefix 格子時要沿用原條件：
#   MAX_NUM_SEQS=16 CONTAINER_NAME=gpt_oss_20b_vllm bash start_vllm_20b.sh
export MAX_NUM_SEQS="${MAX_NUM_SEQS:?請顯式指定，例如 MAX_NUM_SEQS=64 bash $0}"
case "$MAX_NUM_SEQS" in
  ''|*[!0-9]*) echo "錯誤：MAX_NUM_SEQS 必須是正整數，收到 '$MAX_NUM_SEQS'" >&2; exit 1 ;;
esac

# 容器名帶組態，避免不同 max-num-seqs 的 campaign 誤用同一個容器。
export CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_20b_vllm_s$(printf '%03d' "$MAX_NUM_SEQS")}"

# prefix suite 需要兩種快取模式，各自重建容器。PREFIX_CACHING=on 時改帶 --enable-prefix-caching。
# harness 用 docker inspect 讀 .Config.Cmd 比對，旗標必須顯式存在，缺了會判為 unknown 並中止。
export PREFIX_CACHING="${PREFIX_CACHING:-off}"
if [ "$PREFIX_CACHING" = "on" ]; then
  CACHE_FLAG="--enable-prefix-caching"
else
  CACHE_FLAG="--no-enable-prefix-caching"
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

echo "正在啟動 vLLM 伺服器：GPT-OSS-20B"
echo "Model: $TARGET_MODEL"
echo "Container: $CONTAINER_NAME  Port: 127.0.0.1:18000"
echo "Docker Image: $DOCKER_IMAGE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Prefix Caching: $PREFIX_CACHING ($CACHE_FLAG)"
echo "=========================================="

docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=64gb \
  -p 127.0.0.1:18000:8000 \
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

echo "日誌: docker logs -f $CONTAINER_NAME"
echo "harness 要帶: --vllm-container $CONTAINER_NAME"

# docker run -d 一送出就返回，但 vLLM 還要載入權重、compile、capture graph。
# 沒有這段等待，接在後面的 harness 會在 preflight 撞上 ConnectionResetError。
echo "等待就緒..."
for i in $(seq 1 1800); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:18000/health || true)" = "200" ]; then
    echo "就緒：http://127.0.0.1:18000/v1"
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
