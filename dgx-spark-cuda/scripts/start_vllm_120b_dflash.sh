#!/bin/bash

# ==========================================
# vLLM 伺服器啟動腳本：GPT-OSS-120B + DFlash 投機解碼（DGX Spark / ARM64）
# API: http://127.0.0.1:18002/v1
#
# Target : openai/gpt-oss-120b
# Drafter: z-lab/gpt-oss-120b-DFlash
#
# 這是 DFlash A/B 的實驗組。跟 start_vllm_120b.sh 只差 --speculative-config，
# 其餘旗標必須逐字相同，否則量到的差異不能歸因給投機解碼。
# ==========================================

set -Eeuo pipefail

export TARGET_MODEL="openai/gpt-oss-120b"
export SERVED_MODEL_NAME="gpt-oss-120b-dflash"
export HOST_PORT="18002"

# 與 start_vllm_120b.sh 相同：自變數，不給預設值。設錯會安靜地量到客戶端佇列。
# DFlash 這邊還多一層 —— 投機解碼的收益本來就跟 batch 大小強相關（batch 越大、
# 每個 slot 分到的算力越少，draft 的相對成本越高），所以 max-num-seqs 選什麼
# 會直接改變 A/B 的結論。baseline 與 DFlash 兩邊必須用同一個值。
export MAX_NUM_SEQS="${MAX_NUM_SEQS:?請顯式指定，且必須與 baseline 那次相同，例如 MAX_NUM_SEQS=64 bash $0}"
case "$MAX_NUM_SEQS" in
  ''|*[!0-9]*) echo "錯誤：MAX_NUM_SEQS 必須是正整數，收到 '$MAX_NUM_SEQS'" >&2; exit 1 ;;
esac

export CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_120b_dflash_vllm_s$(printf '%03d' "$MAX_NUM_SEQS")}"

# 本機 cache 的 main ref 是 1278df34，該 snapshot 只有 config.json + model.safetensors，
# 沒有 config 的 auto_map 指向的 dflash.py。若 vLLM 走 remote-code 路徑載入失敗，
# 改用下面那行直接指到含 dflash.py 的舊 snapshot（同一份權重 blob）：
export DRAFTER_MODEL="${DRAFTER_MODEL:-z-lab/gpt-oss-120b-DFlash}"
# export DRAFTER_MODEL="/root/.cache/huggingface/hub/models--z-lab--gpt-oss-120b-DFlash/snapshots/b9332d872c72f5cf891409fc3750ef2d21138347"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"

# 與 baseline 及 20B campaign 同一個 digest。若這個 build 不支援 method=dflash，
# 換映像時 start_vllm_120b.sh 必須一起換，並在報告偏離事項表記一筆。
export DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai@sha256:928dde7d6c19478c9c6ab975fed12e6466f6d9b7cb3695d218e287c0c88ad608}"
export TIKTOKEN_CACHE_DIR="$HOME/tiktoken_encodings"

# 必須與 start_vllm_120b.sh 相同。drafter 另外吃約 1.5 GiB（bf16）加自己的 KV，
# 所以 0.85 之下 KV 池會比 baseline 略小 —— 這是 DFlash 的真實成本，不要為了
# 補齊 KV 而在這裡調高，否則 A/B 就多了第二個變因。
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

export PREFIX_CACHING="${PREFIX_CACHING:-off}"
if [ "$PREFIX_CACHING" = "on" ]; then
  CACHE_FLAG="--enable-prefix-caching"
else
  CACHE_FLAG="--no-enable-prefix-caching"
fi

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

mkdir -p "$TIKTOKEN_CACHE_DIR"
TIKTOKEN_FILE="$TIKTOKEN_CACHE_DIR/o200k_base.tiktoken"
if [ ! -f "$TIKTOKEN_FILE" ]; then
  echo "下載 tiktoken vocab..."
  curl -L "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" -o "$TIKTOKEN_FILE"
fi

SPECULATIVE_CONFIG="{\"method\":\"dflash\",\"model\":\"$DRAFTER_MODEL\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"

echo "正在啟動 vLLM 伺服器：GPT-OSS-120B + DFlash"
echo "Model: $TARGET_MODEL"
echo "Drafter: $DRAFTER_MODEL (num_speculative_tokens=$NUM_SPECULATIVE_TOKENS)"
echo "Container: $CONTAINER_NAME  Port: 127.0.0.1:$HOST_PORT"
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
    $CACHE_FLAG \
    --speculative-config "$SPECULATIVE_CONFIG"

echo "等待就緒（120B 權重 61 GiB + drafter，首次載入含 torch.compile 可能要數分鐘）..."
for i in $(seq 1 1800); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$HOST_PORT/health || true)" = "200" ]; then
    echo "就緒：http://127.0.0.1:$HOST_PORT/v1"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'KV cache size|Available KV cache|attention backend|MoE backend|[Ss]peculative|[Dd]raft' || true
    exit 0
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
    echo "錯誤：容器已退出。常見原因：這個映像不支援 method=dflash，或 drafter" >&2
    echo "      snapshot 缺 dflash.py（改用腳本裡註解掉的本機 snapshot 路徑）。" >&2
    echo "      docker logs $CONTAINER_NAME" >&2
    exit 1
  fi
  sleep 1
done
echo "錯誤：就緒逾時。docker logs -f $CONTAINER_NAME" >&2
exit 1
