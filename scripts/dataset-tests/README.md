# Dataset benchmark suite

這套流程把原本的 synthetic 微基準擴充成「45 格公平比較＋34 格 vLLM-only 壓力測試」。llama.cpp 最高維持 concurrency 16；vLLM 的 client load 最高提高到 concurrency 64。正式執行會保留原始 AIPerf records、GPU telemetry、資料 revision、SHA256、三次重複與 95% CI；不會修改既有 `scripts/llama-cpp-tests/`、`scripts/vllm/` 或舊結果。

## 測試矩陣

| Suite | 公平比較／引擎 | vLLM 額外壓力格 | 主要目的 |
|---|---:|---:|---|
| `sharegpt_closed` | 5：C=1/2/4/8/16 | 2：C=32/64 | closed-loop 聊天容量與排隊上限 |
| `sharegpt_open` | 5：0.5/1/2/4/8 req/s，C=16 | 2：16/32 req/s，C=64 | Poisson 到達率與飽和點 |
| `sharegpt_natural` | 1：4 req/s，C=16 | 2：8/16 req/s，C=64 | 真實輸出長度高壓負載 |
| `rag` | 16：兩資料集 × 四長度 × C=1/8 | 16：相同資料 × C=16/32 | 高並發 RAG context 成本 |
| `prefix` | 8：兩長度 × off/on × C=1/8 | 4：相同條件 × C=16 | 高並發共用前綴效益 |
| `ruler_performance` | 10：五長度 × C=1/2 | 8：依長度提高至 C=4/8/16/32 | 長上下文 batching、排隊與 OOM 邊界 |
| 合計 | llama.cpp 45；vLLM 共同 45 | vLLM 額外 34，總計 79 | 基礎 22.5 小時＋壓力 8.5 小時 |

公平比較區只使用兩端都有的 45 格。vLLM-only 格用來找容量上限，不會進入 `engine-comparisons.csv` 的兩引擎勝負計算。vLLM server 仍維持實際部署的 `max-num-seqs=16`；C32/C64 是同時掛入的 client requests，藉此測量 vLLM queueing 與 continuous batching，而不是改變 server 配置。

以每格三次、每次五分鐘計算，公平比較約 22.5 GPU 小時，vLLM 壓力區再增加約 8.5 小時；加上暖機、微基準與 RULER evaluator，整套預估約 45～55 小時。若壓力格變異過大而追加到五次，時間會再增加。

RULER accuracy 另跑官方 13 任務 × 4K/8K/16K/32K，每任務 100 筆，自然 EOS，不併入 AIPerf 效能格。

## 既有固定 token 微基準

資料集測試不取代原本的 llama-benchy 微基準。正式資料集 campaign 前，先依序執行既有腳本的啟動、context depth、concurrency 與 prefix cache 測試：

```bash
# llama.cpp：01 啟動、02 基準、03 concurrency、04 prefix cache
bash scripts/llama-cpp-tests/01-startup.sh
bash scripts/llama-cpp-tests/02-context-scaling.sh
bash scripts/llama-cpp-tests/03-concurrency-scaling.sh
bash scripts/llama-cpp-tests/04-prefix-cache.sh

# vLLM：01 啟動、02 基準、03 concurrency、04 prefix cache
bash scripts/vllm/01-startup.sh
bash scripts/vllm/02-context-scaling.sh
bash scripts/vllm/03-concurrency-scaling.sh
bash scripts/vllm/04-prefix-cache.sh
```

這一階段輸出 Prefill／Decode token/s、context depth 與 concurrency scaling；ShareGPT、RAG、RULER 則負責端到端資料集負載，兩類結果分開判讀。

## 0. WSL 環境

建議在現有 AIPerf virtualenv 中安裝資料準備依賴：

```bash
cd /path/to/aiperf
.venv/bin/pip install -r /path/to/llm-serving-benchmark/scripts/dataset-tests/requirements.txt
```

正式 runner 會拒絕非 AIPerf `0.12.0`／commit `03c9c6ddc5e6227782e53ded177f1227d332af48`。

## 1. 準備資料集

先下載原始資料：

```bash
cd /path/to/llm-serving-benchmark
~/LLM_infra/aiperf/.venv/bin/python scripts/dataset-tests/prepare_datasets.py download
```

使用 GPT-OSS-20B 官方 tokenizer 固定 ShareGPT、RAG 和 prefix 資料：

```bash
PY=~/LLM_infra/aiperf/.venv/bin/python
TOKENIZER=~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee

$PY scripts/dataset-tests/prepare_datasets.py sharegpt \
  --source datasets/raw/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer "$TOKENIZER"

$PY scripts/dataset-tests/prepare_datasets.py rag \
  --squad-source datasets/raw/squad-train-v1.1.json \
  --hotpot-source datasets/raw/hotpotqa-distractor-train.jsonl \
  --tokenizer "$TOKENIZER"

$PY scripts/dataset-tests/prepare_datasets.py prefix \
  --squad-source datasets/raw/squad-train-v1.1.json \
  --tokenizer "$TOKENIZER"
```

正式資料列只保留 AIPerf 0.12.0 支援的欄位。`sample_id`、bucket、答案、cold/warm 等 metadata 存在同名 `.index.jsonl`；每個輸出另有 SHA256 manifest。

Prefix cache 測試有兩份 prompt：`on` 讓同組 request 共享 token-0 prefix；`off` 在 token 0 加入每請求唯一 marker。兩者使用相同文件和問題。相同 `session_id` 只用來保證 cold → warm 的因果順序；AIPerf single-turn loader 會把每輪完整 prompt 原樣送出，不會把前一輪的 live response 疊加到下一輪。

## 2. 準備官方 RULERv1

資料產生器使用 NVIDIA/RULER main commit `c3f5e3b4...`；評分定義固定 RULER `rulerv1-ns` commit `e8bbff67...` 與 NeMo-Skills `chsieh/ruler-remove-prefix` commit `f4a3fd8e...`。

```bash
cd ~/LLM_infra
git clone https://github.com/NVIDIA/RULER.git
git -C RULER checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a

cd /path/to/llm-serving-benchmark
bash scripts/dataset-tests/prepare_ruler_official.sh
```

腳本會為 4K/8K/16K/32K 建立 13 任務 × 每任務 100 筆 official JSONL，另為 60K 建立每任務 20 筆的效能壓力資料，再匯出 AIPerf JSONL。60K 會由 sampler 循環至 300 秒結束，只作效能與 OOM 邊界；不列入官方 accuracy 曲線。

## 3. 驗證、環境鎖定與列出矩陣

```bash
python scripts/dataset-tests/benchmark_matrix.py
python scripts/dataset-tests/validate_prepared.py
bash scripts/dataset-tests/00-preflight.sh
```

`validate_prepared.py` 會檢查：AIPerf schema、SHA256、tokenizer、每桶 ±5%、prefix group order、cold/warm 數量。用 `--allow-approximate` 只可驗證 unit-test fixture，不能產生正式結果。

`00-preflight.sh` 另會產生 `datasets/manifests/environment.json`，記錄 GGUF 與 HF snapshot 每個檔案的 SHA256、tokenizer、GPU/driver、AIPerf、llama.cpp、llama-benchy、Python 套件、vLLM image/command，以及 RULER revision。首次雜湊大型模型會花一些時間；正式 runner 若找不到這份 manifest 會拒絕執行。

## 4. 執行 AIPerf

### 建議：A/B suite block 交替

在 vLLM container 使用 `--no-enable-prefix-caching` 時，直接執行：

```bash
bash scripts/dataset-tests/run_ab_campaign.sh
```

它只跑 45 格公平比較，按 suite 交替 `llama.cpp → vLLM`、下一 suite `vLLM → llama.cpp`，自動停止／啟動 vLLM container，並讓 llama.cpp runner 管理 llama-server。兩端使用相同 campaign ID。非 prefix 測試要求兩端關閉 prompt/prefix cache，避免有限資料集循環時產生非預期跨請求快取。

公平比較完成後，在相同 vLLM container 設定下執行額外高壓格：

```bash
bash scripts/dataset-tests/run_vllm_stress_campaign.sh
```

這會跑除 prefix 外的 30 個 vLLM-only 壓力格。Prefix 額外四格會由下方的 prefix campaign 一起完成。

Prefix 使用專用腳本，分別驗證 cache off/on 的實際 server mode：llama.cpp 由 runner 加入 `--no-cache-prompt`／`--cache-prompt`；vLLM 則要求 container 分別以 `--no-enable-prefix-caching`／`--enable-prefix-caching` 建立。腳本會在需要切換 vLLM mode 時停下並提示：

```bash
bash scripts/dataset-tests/run_prefix_campaign.sh
```

以下是需要單獨控制 backend 或 cell 時的命令。

### llama.cpp

可讓 runner 以既定參數管理 llama-server：

```bash
~/LLM_infra/aiperf/.venv/bin/python scripts/dataset-tests/run_benchmarks.py \
  --backend llama-cpp \
  --suite sharegpt_closed --suite sharegpt_open --suite sharegpt_natural \
  --suite rag --suite ruler_performance \
  --manage-llama-server
```

若 server 已自行啟動在 `127.0.0.1:8080`，省略 `--manage-llama-server`。

### vLLM

非 prefix suite 要求 Docker 使用 `--no-enable-prefix-caching`：

```bash
~/LLM_infra/aiperf/.venv/bin/python scripts/dataset-tests/run_benchmarks.py \
  --backend vllm \
  --scope comparison \
  --suite sharegpt_closed --suite sharegpt_open --suite sharegpt_natural \
  --suite rag --suite ruler_performance
```

Prefix cache-on 要先用 `--enable-prefix-caching` 重建同一個 vLLM container，再獨立執行；cache-off 則改成 `--cache-mode off` 並要求 `--no-enable-prefix-caching`：

```bash
~/LLM_infra/aiperf/.venv/bin/python scripts/dataset-tests/run_benchmarks.py \
  --backend vllm --suite prefix --cache-mode on
```

llama.cpp 也用 `--suite prefix --cache-mode off/on --manage-llama-server` 分開跑。`--cell <cell-id>` 可單跑一格。正式測試採每個 repeat 一次獨立 AIPerf invocation，以便檢查變異後自動追加第 4、5 次。

Runner 行為：

- 開始 campaign 前做 10 次 smoke requests。
- 每個 repeat 前等待 GPU utilization < 5% 持續 30 秒。
- 每次 30 秒暖機＋300 秒 profiling。
- error rate > 1% 或 AIPerf log 偵測到非預期 GPU OOM 的 repeat 無效。
- AIPerf 0.12 sampler 會循環資料集直到時間條件結束：sequential 回到第一筆，shuffle 每輪重洗；因此每桶 250 筆不會使 300 秒 profiling 提前結束。
- output throughput CV > 10% 或 P95 TTFT CV > 15% 時自動改成 5 次。
- 每格 pooled sample < 1,000 時，報告中的 P99 為 `N/A`。

## 5. RULER 正確率

兩引擎分別產生 natural-EOS prediction：

```bash
python scripts/dataset-tests/run_ruler_accuracy.py --backend llama-cpp --resume
python scripts/dataset-tests/run_ruler_accuracy.py --backend vllm --resume
```

準備 NeMo-Skills evaluator：

```bash
cd ~/LLM_infra
git clone --branch chsieh/ruler-remove-prefix https://github.com/NVIDIA-NeMo/Skills.git
git -C Skills checkout f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3
python -m pip install -e Skills
```

使用官方 `eval_ruler` 評分，NIAH／VT／CWE／FWE 採 `all`，QA 採 `part`：

```bash
python scripts/dataset-tests/evaluate_ruler_official.py \
  --backend llama-cpp --nemo-skills-root ~/LLM_infra/Skills
python scripts/dataset-tests/evaluate_ruler_official.py \
  --backend vllm --nemo-skills-root ~/LLM_infra/Skills
```

## 6. 彙整與圖表

Windows Python 已有 pandas、Matplotlib、Seaborn，可以直接執行：

```powershell
python scripts\dataset-tests\analyze_dataset_results.py
python scripts\dataset-tests\plot_dataset_results.py
```

分析器預設會尋找最新的「公平比較 campaign」、「vLLM stress campaign」與「prefix-only campaign」後合併；也可重複指定 `--llama-campaign <path>`／`--vllm-campaign <path>`。同一 cell 出現多次時保留修改時間最新的結果。

輸出位置：

- `analysis/dataset-output/dataset-cell-summary.csv`：每格 pooled P50/P95/P99、三至五次中位數與固定 seed bootstrap 95% CI。
- `analysis/dataset-output/engine-comparisons.csv`：vLLM/llama.cpp 比值。
- `analysis/dataset-output/vllm-stress-summary.csv`：只含 C32/C64 等 vLLM-only 高壓格。
- `analysis/dataset-output/concurrency-scaling.csv`：相同條件下 Cn/C1 的 request 與 output throughput 倍數。
- `analysis/dataset-output/dataset-benchmark-report.md`：中文摘要、有效性與單位。
- `analysis/dataset-charts/`：ShareGPT、RAG、prefix、長上下文、RULER accuracy、SLO heatmap；C16 以上會標示為 vLLM-only stress 區。

RAG 數據只代表已組裝 context 的模型推論，不包含 retrieval、reranker、資料庫或網路延遲。
