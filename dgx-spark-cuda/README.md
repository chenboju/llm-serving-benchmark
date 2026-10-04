# DGX Spark（GB10 / ARM64）測試

本目錄是 GPT-OSS 系列在 **NVIDIA DGX Spark** 上的測試紀錄，與 repository 根層的
RTX 4090 / WSL2 測試共用同一套 harness、同一份資料集與同一組有效性規則。

> 與 4090 的關係：`formal-split-65k` 這組 campaign 在兩個平台上執行的是**完全相同的
> 37 個 cells**，兩邊皆 37/37 valid，因此可直接對比。其餘 campaign 為平台專屬，
> 對比邊界見下方[可比性](#可比性)。

> 本目錄的 vLLM server 組態出自姊妹專案 [**dgx-spark-serving**](https://github.com/chenboju/dgx-spark-serving)，
> 那裡有這套 ARM64 環境的完整部署紀錄：aarch64 相依處理、雙節點 Ray 叢集、
> 推測解碼組態，以及 QSFP 連線的除錯過程。本目錄只負責量測。

## 執行環境

| 項目 | 值 |
|---|---|
| 硬體 | NVIDIA DGX Spark，GB10 Grace-Blackwell |
| 架構 | ARM64 / aarch64（unified memory） |
| Driver / CUDA | 580.142 / 13.0 |
| vLLM | `vllm/vllm-openai@sha256:928dde7d…`（linux/arm64，2026-08-07 拉取的 nightly，vLLM 0.26.1rc1.dev457+gc810e5ee9） |
| llama.cpp | 見各 campaign 的 `llama_server_config` |
| AIPerf | commit `03c9c6ddc5e6227782e53ded177f1227d332af48` |

`vllm/vllm-openai` 以 **image digest** 而非 `:nightly` 標籤釘選，確保日後標籤移動後仍指向同一個映像。

GPT-OSS-20B 使用 HF revision `6cee5e81`，量化 `gpt_oss_mxfp4`，走 MARLIN MoE backend
與 `TRITON_ATTN`；`max_model_len=65536`、`server_max_num_seqs=16`、`max_concurrency=64`。

## Campaign 清單

| Campaign | Backend | 模型 | Cells | 狀態 |
|---|---|---|---:|---|
| `formal-split-65k-20260816-dgx` | vLLM | GPT-OSS-20B | **37 / 37** | valid — 對應 4090 的 `formal-split-65k-20260814` |
| `formal-stress-65k-20260817-dgx` | vLLM | GPT-OSS-20B | **30 / 30** | valid — 壓力與容量邊界 |
| `formal-20b-s064-20260818-dgx` | vLLM | GPT-OSS-20B | **7 / 7** | valid — ShareGPT closed-loop C1–C64 |
| `formal-120b-s064-20260818-dgx` | vLLM | GPT-OSS-120B | **7 / 7** | valid — 120B closed-loop C1–C64，SLO 上限 C2 |
| `smoke-vllm-20260816` | vLLM | GPT-OSS-20B | 1 / 1 | valid — 冒煙測試 |
| `gpt-oss-20b-dataset-v1-dgx-20260816-100653` | llama.cpp | GPT-OSS-20B | 0 / 5 | **dry-run**，不可作為結果 |
| `gpt-oss-20b-dataset-v1-dgx-20260818-135825` | vLLM | GPT-OSS-120B | 0 / 1 | **dry-run**，不可作為結果 |

合計 **82 個 valid cells**。兩個 dry-run campaign 刻意保留——它們是矩陣展開的驗證步驟，
不是失敗的測試，但也**不得**併入任何彙總。

有效性判準（≥3 valid repeats、error rate ≤1%、CV 門檻、P99 需 pooled requests ≥1,000）
與根層一致，定義見 [`../docs/RESULTS.md`](../docs/RESULTS.md)。

## 可比性

| 對比 | 可否直接比較 | 說明 |
|---|---|---|
| 4090 vs DGX，`formal-split-65k` | **可以** | 同一 harness、同一 37 cells、同一資料集 SHA256、皆 vLLM |
| 4090 vs DGX，其他 campaign | 不可直接 | cell 集合不同，需逐格對照 |
| DGX 20B vs DGX 120B | 可以，但須解讀為容量而非引擎效能 | 同平台同 backend，模型規模不同 |
| DGX vLLM vs DGX llama.cpp | **目前不可** | llama.cpp 側只有 dry-run，無 valid 資料 |

DGX Spark 是 unified memory 架構，「GPU 記憶體」與主機 RAM 是同一個池；
與 4090 的獨立 24 GB VRAM 在記憶體壓力下的行為不同，解讀容量類指標時需納入考量。

## 目錄內容

```text
dgx-spark-cuda/
├── README.md          本文件
├── docs/              DGX 專屬文件（4 份）
│   ├── 00-測試目標與範圍.md
│   ├── 01-環境建立-dgx-spark.md      ARM64 環境建立，每步附完成判準
│   ├── 02-驗證清單.md                 八道驗收關卡
│   └── 03-報告範本.md
├── scripts/           平台專屬腳本（10 支，其餘與根層 scripts/ 共用）
├── campaigns/         各 campaign 的 campaign.json 與 campaign-validation.json
└── analysis/          逐 cell 彙總、跨平台對比、圖表與重建腳本
    └── README.md        結果與解讀邊界的完整說明
```

## 主要結果

| 主題 | 結論 | 詳見 |
|---|---|---|
| 對比 RTX 4090 | closed-loop 吞吐約 **0.28×**，P95 TTFT 中位數約 **3.0 倍** | [analysis/](analysis/README.md#吞吐dgx-spark-約為-rtx-4090-的-028) |
| open-loop 的陷阱 | 低到達率下看似平手，實為未飽和；SLO 達成率已崩 | [analysis/](analysis/README.md#open-loop-的接近平手是負載未飽和) |
| **120B 容量** | 95% SLO 門檻下併發上限 **C2**（20B 為 C8） | [analysis/](analysis/README.md#20b-vs-120b單台-dgx-spark-的容量上限) |
| GPU 記憶體遙測 | GB10 上**不可得**，非測試失誤 | [analysis/](analysis/README.md#gpu-記憶體遙測在-gb10-上不可得) |
| 功耗 | 兩平台電源域不同，**不可跨平台比較** | [analysis/](analysis/README.md#功耗數字不可跨平台比較) |

### vLLM server 啟動腳本

上表各 campaign 的 server 端由這三支拉起。它們刻意不給 `MAX_NUM_SEQS` 預設值——
設錯不會讓服務壞掉，只會安靜地量到客戶端佇列而非伺服器容量，因此強制顯式指定：

| 腳本 | 對應 campaign | 說明 |
|---|---|---|
| `start_vllm_20b.sh` | `formal-split-65k`／`formal-stress-65k`／`formal-20b-s064` | 容器名帶組態（`gpt_oss_20b_vllm_s016`），避免不同 `max-num-seqs` 的 campaign 誤用同一容器 |
| `start_vllm_120b.sh` | `formal-120b-s064` | 同時是 DFlash A/B 的**對照組** |
| `start_vllm_120b_dflash.sh` | DFlash A/B 實驗組 | 與對照組**只差 `--speculative-config`**，其餘旗標逐字相同 |

三支使用同一個 vLLM image digest，確保跨模型比較時引擎版本一致。

### 其餘 7 支：與根層的差異

DGX 的 harness 是根層 [`../scripts/dataset-tests/`](../scripts/dataset-tests/) 的分支。
此處只保留**與根層不同**的部分：

| 檔案 | 差異來源 |
|---|---|
| `00-preflight.sh` | ARM64 環境檢查、雙 backend 驗證 |
| `benchmark-config.json` | DGX 的路徑、容器與 backend 設定 |
| `benchmark_matrix.py` | 平台專屬的 cell 展開 |
| `capture_environment.py` | aarch64 與 nvidia-smi telemetry 差異 |
| `run_benchmarks.py` | 雙 backend 執行流程 |
| `gpu_telemetry.py` | **DGX 專屬**——unified memory 下的 GPU 遙測 |
| `probe_llama_server.sh` | **DGX 專屬**——區分「仍在載入」與「已崩潰」 |

其餘 14 支 harness 腳本（`prepare_datasets.py`、`validate_prepared.py`、`analyze_dataset_results.py` 等）
與根層相同，不重複收錄。

## 未包含的內容

依 repository 的發布策略，本目錄**只提供衍生資料與後設資料**：

- **原始執行產物**（約 5.9 GB）未包含。ShareGPT workload 的 `inputs.json` 含第三方
  可識別個資，不對外散布；詳見 [`../docs/DATASETS.md`](../docs/DATASETS.md)。
- **準備後資料集**（約 872 MB）未包含。請依
  [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) 自行產生，並以
  `campaign.json` 內的 `dataset_sha256`（17 筆）比對一致性。

`campaigns/` 底下的 `campaign.json` 已內含完整溯源資訊——`dataset_sha256`、
`environment_manifest`（含 SHA256）、`aiperf_revision`、`gpu`、`backend_config`
與完整 server 啟動命令——可在沒有原始檔的情況下驗證測試條件。

檔案內的使用者路徑已置換為 `/home/<user>`；重現時請改為自己的實際路徑。
