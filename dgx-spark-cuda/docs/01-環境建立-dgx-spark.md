# 01 — 環境建立：NVIDIA DGX Spark（GB10, ARM64）

> 照順序做完，最後一步會產出可執行的雙 backend campaign。每步都有完成判準。

這台是 Linux + CUDA + Docker，harness 幾乎原樣可用。真正的工作在
**aarch64 的建置**與 **ARM64 vLLM 容器**，不在程式碼移植。

## 步驟 0 — 確認機器身分

```bash
uname -m
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
free -g
```

| 檢查 | 預期 |
|---|---|
| 架構 | `aarch64` |
| GPU | GB10 Grace Blackwell |
| Compute capability | 記錄實際值，建置時要用 |
| 記憶體 | 約 128 GB（統一記憶體，`free` 看到的就是 GPU 也在用的那塊） |

**不要憑印象假設 compute capability**，用上面的指令查出來再往下走。

## 步驟 1 — 目錄配置

```
$WORKSPACE_ROOT/                 # 預設 ~/LLM_infra
├── dgx-spark-cuda/              # 本資料夾
├── aiperf/                      # clone，pin 到指定 commit
├── llama.cpp/                   # clone，checkout b8407，aarch64 CUDA build
└── models/gpt-oss-20b-gguf/     # 從參考機複製
```

環境變數與參考機相同：`WORKSPACE_ROOT`、`LLAMA_SERVER`、`MODEL_FILE`、
`AIPERF_ROOT`、`AIPERF`、`LLAMA_CPP_ROOT`、`GPU_TELEMETRY`。

## 步驟 2 — 檔案來源與完整性

**資料集已經包含在這個資料夾裡**，不需要另外複製。它們是從參考機的正式產出複製過來的：

```
來源（正式產出，Windows）:
C:\Users\<user>\Desktop\LLM_infra\wsl_result\datasets
從 WSL 存取: /mnt/c/Users/<user>/Desktop/LLM_infra/wsl_result/datasets
```

已內含（873 MB）：

| 目錄 | 內容 | 檔數 |
|---|---|---:|
| `datasets/prepared/sharegpt/` | `sharegpt-fixed-128`、`sharegpt-natural-osl`，各 5,000 sessions | 2 |
| `datasets/prepared/rag/` | `squad-*`、`hotpotqa-*`，1K/4K/8K/16K，各 250 筆 | 8 |
| `datasets/prepared/prefix/` | `squad-prefix-{4096,16384}-{off,on}`，各 400 筆 | 4 |
| `datasets/prepared/ruler/` | `ruler-{4096,8192,16384,32768}` 各 1,300 筆、`ruler-61440` 260 筆 | 5 |
| `datasets/manifests/` | 每檔的 SHA-256、tokenizer、bucket 設定 | 19 + sources.json |
| `datasets/raw/ruler-official/` | RULER accuracy 官方題目，5 個長度 × 13 個任務 | 65 個 `test.jsonl` |

`datasets/manifests/environment.json` **刻意沒有帶過來**——那記錄的是參考機的硬體身分，
本機的要由 `00-preflight.sh` 產生。

仍然需要自己準備的：

| 項目 | 來源 | 大小 | 必要性 |
|---|---|---:|---|
| GGUF 模型 | 參考機 `models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf` | 12.1 GB | **必要** |
| HF tokenizer snapshot | `openai/gpt-oss-20b` @ `6cee5e81` | — | **必要**（兩個 backend 共用同一份） |
| `datasets/raw/` 其餘原始檔（ShareGPT / SQuAD / HotpotQA 原始下載） | — | 1.5 GB | 不需要，除非要重新產生資料集（正式流程不該這麼做） |

驗證整包（比對 `PACKAGE-MANIFEST.sha256` 內每個檔案的 SHA-256，再驗資料集）：

```bash
bash verify_package.sh
```

模型複製完成後兩台都算一次雜湊並比對：

```bash
sha256sum "$WORKSPACE_ROOT/models/gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf"
```

**完成判準**：`verify_package.sh` 三段全部 OK，且模型 SHA-256 與參考機一致。

**不要在本機重新產生資料集**。理由見 [00-測試目標與範圍.md](00-測試目標與範圍.md)：
本機重跑 `prepare_datasets.py` 會產生自洽但內容不同的資料集，驗證照樣通過，但跨硬體
比較的前提就消失了。

## 步驟 3 — 建立工具鏈

### 3.1 AIPerf（版本硬性鎖定）

```bash
git clone https://github.com/ai-dynamo/aiperf.git "$WORKSPACE_ROOT/aiperf"
git -C "$WORKSPACE_ROOT/aiperf" checkout 03c9c6ddc5e6227782e53ded177f1227d332af48
```

依 aiperf 自己的 README 在 `.venv` 建環境。**aarch64 上部分相依可能沒有預編 wheel，
需要從原始碼建置**，先確保有 build 工具鏈（`build-essential`、`cmake`、Python headers）。

```bash
"$AIPERF_ROOT/.venv/bin/python" -m pip install -r scripts/dataset-tests/requirements.txt
```

**完成判準**：`aiperf --version` 含 `0.12.0`，且 repo HEAD 等於上面的 commit。
兩者任一不符，harness 會 `SystemExit`——這是刻意的，AIPerf 的 metric 定義跨版本會變。

### 3.2 llama.cpp（b8407，aarch64 CUDA build）

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$WORKSPACE_ROOT/llama.cpp"
git -C "$WORKSPACE_ROOT/llama.cpp" checkout b8407
```

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

`CMAKE_CUDA_ARCHITECTURES=native` 讓 nvcc 依實機決定 target，避免手寫錯 GB10 的
compute capability。若要固定成明確值，用步驟 0 查到的 `compute_cap`（例如 `12.1` 寫成 `121`）。

需要的 CUDA toolkit 版本以 DGX OS 內建者為準；`nvcc --version` 與驅動版本都會被
`capture_environment.py` 記錄下來。

**完成判準**：`llama-server --version` 印出 b8407 對應 build，`--list-devices` 列出 GB10。

### 3.3 llama-benchy

```bash
git clone https://github.com/eugr/llama-benchy.git "$WORKSPACE_ROOT/llama-benchy"
git -C "$WORKSPACE_ROOT/llama-benchy" checkout e9be344
```

在 `.venv` 建環境（`uv sync` 或 `uv pip install -e .`）。

llama-benchy 是純 Python 的 OpenAI 相容端點 client，量不同 context depth 下的
prompt processing (pp) 與 token generation (tg) 速度。它只透過 HTTP 說話，**aarch64
不需要任何移植**；只有相依套件可能需要從原始碼建置。

| 它在這裡的角色 | 說明 |
|---|---|
| 環境指紋 | `capture_environment.py` 會記錄版本與 commit。沒安裝則 manifest 寫 `missing` |
| 微基準 suite | 參考機報告的 context scaling / concurrency scaling 表格由它產出。**那些 suite 尚未移植到本資料夾**，見 [00-測試目標與範圍.md](00-測試目標與範圍.md#九尚未移植的部分) |

dataset campaign 本身**不會呼叫** llama-benchy，它不是執行 campaign 的必要條件，
但環境指紋要完整就得裝。

**完成判準**：`llama-benchy --version` 有輸出（參考機為 0.4.1.dev1，commit `e9be3445`）。

### 3.4 vLLM 容器（ARM64）

vLLM 在這台跑得起來，但**映像必須是 ARM64 的**，x86_64 映像不能用。

容器建立時必須帶 `--no-enable-prefix-caching`：

| 為什麼 | 說明 |
|---|---|
| harness 會檢查 | `validate_backend_mode()` 用 `docker inspect` 讀容器命令列，模式不符就中止 |
| `unknown` 也擋 | 只是「沒寫 `--enable-prefix-caching`」不算通過，旗標必須顯式存在以便存證 |
| prefix suite 才切換 | 跑 prefix 時另建一個 `--enable-prefix-caching` 的容器 |

其他必要對齊項：

| 項目 | 值 | 說明 |
|---|---|---|
| `--max-model-len` | 65536 | 與參考機相同 |
| `--max-num-seqs` | 16 | 與參考機相同（client 端可壓到 64） |
| Host port | 18000 | 或改 `benchmark-config.json` 的 `backends.vllm.url` 對應實際 publish 的埠 |
| `--gpu-memory-utilization` | **需要判斷** | 見下 |

**統一記憶體的陷阱**：`--gpu-memory-utilization 0.90` 在 24 GB 離散顯卡上是留 2.4 GB，
在 128 GB 統一記憶體上是留 12.8 GB 給**整個系統**（含 OS、benchmark client、docker）。
同一個數字在這裡的絕對量完全不同。刻意選一個值，並把它記進報告。

**完成判準**：容器啟動、`curl http://127.0.0.1:18000/health` 回 200、
`docker inspect` 看得到 `--no-enable-prefix-caching`。

## 步驟 4 — 探測 llama.cpp build 能力

```bash
bash scripts/dataset-tests/probe_llama_server.sh --full
```

CUDA build 通常支援 `--flash-attn on`（參考機就是這樣跑的），但 **aarch64 build 沒有
被同樣驗證過**，所以還是實測一次。probe 會實際啟動 server、測每個旗標、並用
ctx=65536 parallel=16 試跑，最後印出可貼進 config 的 `llama_server` 區塊。

**完成判準**：`--list-devices` 有 GB10，且 `campaign ctx=65536/parallel=16 boot: ok`。

## 步驟 5 — 填設定

編輯 `scripts/dataset-tests/benchmark-config.json`：

| 欄位 | 改成 |
|---|---|
| `backends.llama-cpp.tokenizer` | 本機 HF snapshot 路徑 |
| `backends.vllm.tokenizer` | **同一個路徑**（兩個 backend 必須用同一個 tokenizer） |
| `backends.vllm.url` | 對應容器實際 publish 的 host port |
| `backends.vllm.deployment` | 填入實際容器映像與 tag |
| `llama_server` | 貼上 probe 輸出 |
| `llama_server.build` | 例如 `b8407 aarch64 CUDA sm_121`（取代 `UNVERIFIED`） |

**完成判準**：檔案裡不再有 `REPLACE_WITH_LOCAL_HF_SNAPSHOT`、`REPLACE_WITH_CONTAINER_IMAGE`、
`UNVERIFIED`。留著任一個，`run_benchmarks.py` 會拒跑。

## 步驟 6 — Preflight

```bash
bash scripts/dataset-tests/00-preflight.sh
```

會檢查 aarch64、AIPerf pin、matrix（45 + 34）、資料集完整性，並產生
`datasets/manifests/environment.json`。

**完成判準**：印出 `Preflight OK`。

## 步驟 7 — 試跑

```bash
python scripts/dataset-tests/run_benchmarks.py --backend llama-cpp --suite sharegpt_closed --scope comparison --dry-run
```

兩個 backend 各跑一格最短的（各約 20 分鐘）：

```bash
python scripts/dataset-tests/run_benchmarks.py --backend llama-cpp \
  --cell sharegpt-closed-c01 --campaign-id smoke-$(date +%Y%m%d) --manage-llama-server
```

```bash
docker start gpt_oss_20b_vllm
python scripts/dataset-tests/run_benchmarks.py --backend vllm \
  --cell sharegpt-closed-c01 --campaign-id smoke-$(date +%Y%m%d)
```

**完成判準**：兩邊的 `validation.json` 都是 `valid`、`oom_repeats` 為 0。

## 步驟 8 — 正式執行

用參考機同一支 A/B 腳本，它會**逐 suite 交錯**執行兩個 backend（ABBA），把時間漂移
攤平到兩邊，並在切換時自動啟停 vLLM 容器：

```bash
CAMPAIGN_ID="formal-split-65k-20260814-dgx" bash scripts/dataset-tests/run_ab_campaign.sh
```

prefix suite（需要中途換 cache 模式，會提示重建容器）：

```bash
CAMPAIGN_ID="formal-split-65k-20260814-dgx" bash scripts/dataset-tests/run_prefix_campaign.sh
```

vLLM 專屬壓力測試：

```bash
CAMPAIGN_ID="formal-split-65k-20260814-dgx" bash scripts/dataset-tests/run_vllm_stress_campaign.sh
```

執行期間不要在這台做其他 GPU 或大量記憶體工作。統一記憶體代表**主機端的記憶體壓力
會直接影響 GPU 可用量**，這點與離散顯卡不同。

## 步驟 9 — 收尾

| 動作 | 說明 |
|---|---|
| 比對兩個 backend 的 `campaign-validation.json` | 兩邊 `total_cells` 應皆為 45 |
| 保留 `server-*.log` 與 `docker logs` | invalid cell 的根因來源 |
| 確認 `campaign.json` 的 `vllm_container_command` | 存證容器實際參數 |
| 填寫報告 | 依 [03-報告範本.md](03-報告範本.md) |

## 常見狀況

| 症狀 | 原因與處理 |
|---|---|
| `vLLM prefix-cache mode is 'unknown'` | 容器建立時沒顯式帶 `--no-enable-prefix-caching`。重建容器 |
| vLLM 容器起不來 | 映像不是 ARM64。確認 `docker inspect --format '{{.Architecture}}'` |
| llama-server 啟動後立刻退出 | 看 `server-*.log`；aarch64 build 若不接受某旗標，把 `llama_server.flash_attn` 改 `unset` |
| 系統整體變慢或 OOM kill | `--gpu-memory-utilization` 在統一記憶體上留太少給 OS。調低後重跑該 suite |
| AIPerf 安裝失敗 | aarch64 缺預編 wheel，補齊 build 工具鏈後重試 |
| `AIPerf pin mismatch` | 硬性擋，不要繞過 |
| llama.cpp 吞吐遠低於預期 | 確認 `--list-devices` 真的用到 GPU，且 `-DGGML_CUDA=ON` 有生效 |
