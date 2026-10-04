[English](README.md) | **繁體中文**

# GPT-OSS 本機推論基準測試

以共用測試流程量測 GPT-OSS 在 **RTX 4090／WSL2 與 DGX Spark** 上的吞吐、延遲與 SLO 達成率，公開逐測試格（cell）CSV、分析程式、圖表與繁體中文報告。

本專案回答兩個問題：相同負載下兩平台的服務表現如何，以及單台 DGX Spark 在指定延遲要求下能承受多少併發。量測已完成，資料快照截至 **2026-08-18**。

| 平台 | 模型與量測範圍 | 公開逐 cell 彙總 |
|---|---|---|
| RTX 4090 24 GB／WSL2 | GPT-OSS-20B；llama.cpp 與 vLLM | 124 cells：116 valid、8 invalid |
| DGX Spark（GB10／ARM64） | GPT-OSS-20B、120B；有效量測為 vLLM | 83 cells：81 valid、2 invalid |

## 主要結果

| 發現 | 比較條件與解讀 |
|---|---|
| DGX 吞吐約為 4090 的 **0.28 倍**，P95 TTFT 約 **3.01 倍** | GPT-OSS-20B、兩端皆 vLLM；37 個共同 cell 中排除 5 個 open-loop 後，32 格的比值中位數。吞吐比範圍 0.221–0.323 |
| 低到達率下吞吐接近，無法代表飽和容量相同 | ShareGPT open-loop：0.5 req/s 時吞吐比約 0.99；8 req/s 時約 0.33，DGX SLO 達成率降至 9.5% |
| DGX 上 20B 最高達標併發為 **C8**，120B 為 **C2** | ShareGPT closed-loop、固定輸出 128 tokens；TTFT ≤1000 ms、ITL ≤50 ms，且 ≥95% 請求同時達標；指已測試併發中的最高達標值 |

![RTX 4090 與 DGX Spark：吞吐、open-loop 負載與各 suite 比值](dgx-spark-cuda/analysis/figures/01-cross-platform-throughput.png)

跨平台比較使用相同的 37 個測試格、相同 vLLM image digest 與資料集。15 個共用 prepared 檔案的 SHA256 一致。上述結果適用於本次組態與工作負載，不是所有模型或部署方式的硬體排名。詳見 [跨平台與模型容量分析](dgx-spark-cuda/analysis/README.md) 與 [資料集 manifest](docs/DATASET_MANIFEST.md)。

## 閱讀現有結果

不需 GPU 或原始對話資料即可閱讀：

| 想了解的內容 | 入口 |
|---|---|
| 跨平台差異、20B／120B 容量 | [DGX 分析與圖表](dgx-spark-cuda/analysis/README.md) |
| 4090 上的完整測試方法與結果 | [正式報告](rtx4090-wsl2/reports/dataset_benchmark_complete_report_zh-TW.md) |
| 逐格數字與有效性 | [4090 CSV](rtx4090-wsl2/analysis/output/dataset-cell-summary-rtx4090.csv)、[DGX CSV](dgx-spark-cuda/analysis/dataset-cell-summary.csv)、[37 格對比 CSV](dgx-spark-cuda/analysis/cross-platform-comparison.csv) |
| 指標、有效性規則與比較邊界 | [RESULTS](docs/RESULTS.md) |

兩份平台 cell 彙總均為 109 欄，保留 invalid 格、有效重複次數、失敗原因與統計區間。`valid` 表示通過量測有效性規則，不表示符合延遲 SLO。

## 重新產生分析

以下命令從 repository 根目錄執行。只用公開 CSV 重畫兩張 DGX 分析圖，不需原始 results 或 GPU：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas matplotlib
python dgx-spark-cuda/analysis/plot_cross_platform.py
python dgx-spark-cuda/analysis/plot_model_capacity.py
```

若要由原始 results 重建 CSV，需自行備妥本機原始資料；本 repository 不散布該資料：

```bash
python -m pip install -r scripts/dataset-tests/requirements.txt
python dgx-spark-cuda/analysis/build_cross_platform.py   --dgx-results /path/to/dgx-spark-cuda/results   --rtx4090-results /path/to/llm-serving-benchmark/results
```

其餘分析入口與輸出說明見 [RESULTS](docs/RESULTS.md)。

## 重新執行量測

使用 Python 3.12 與 WSL2／Linux，另備模型、GPU serving 環境、AIPerf 與 llama-benchy。先依 [重現指南](docs/REPRODUCIBILITY.md) 與 [外部工具版本](docs/EXTERNAL_TOOLS.md) 安裝依賴，再檢查共用 harness：

```bash
python -m pip install -r scripts/dataset-tests/requirements.txt
python -m unittest discover -s scripts/dataset-tests -p 'test_*.py'
python scripts/dataset-tests/benchmark_matrix.py
```

接著依 [dataset campaign 流程](scripts/dataset-tests/README.md) 準備資料、執行 preflight 與正式量測；DGX 的設定另見 [平台指南](dgx-spark-cuda/README.md)。完整 campaign 需要數十 GPU 小時。可選 benchlab wrapper 另需外部專案，見 [安裝說明](scripts/README.md#可選的-benchlab-cli)。

量測使用第三方 AIPerf 與 llama-benchy；本專案提供測試組態、執行流程、有效性檢查與分析。工作負載涵蓋啟動、context／concurrency scaling、prefix cache、ShareGPT、SQuAD／HotpotQA RAG 與 RULER。

## Repository 結構

```text
.
├── README.md / README.zh-TW.md
├── .github/                 # 語法 CI、issue 與 PR 模板
├── docs/                    # 資料來源、有效性、版本、重現與發布
├── scripts/
│   ├── dataset-tests/       # 兩平台共用 campaign harness
│   └── benchlab.sh / .ps1   # 可選外部 CLI wrapper
├── rtx4090-wsl2/
│   ├── analysis/            # CSV、分析程式與圖表
│   ├── reports/             # 三份繁體中文正式報告
│   └── scripts/             # llama.cpp／vLLM 微基準與環境檢查
├── dgx-spark-cuda/
│   ├── analysis/            # 平台彙總、跨平台與模型容量分析
│   ├── campaigns/           # Campaign 後設資料與 validation
│   ├── docs/                # DGX 平台文件
│   └── scripts/             # 平台腳本與 vLLM 啟動器
├── results/                 # 公開說明；原始 results 僅本機保留
│   ├── README.md
│   └── generated-summary.md # 舊流程摘要
├── CONTRIBUTING.md
└── LICENSE
```

頂層 README 為雙語，其餘文件主要為繁體中文。完整索引見 [Repository 結構](docs/REPOSITORY_STRUCTURE.md)。姊妹專案 [**dgx-spark-serving**](https://github.com/chenboju/dgx-spark-serving) 保存 DGX 的 vLLM／TensorRT-LLM 單機與雙節點部署紀錄。

## 比較邊界與資料發布

- 跨平台結果為單節點 vLLM、負載產生器與 server 同機；不代表遠端網路服務延遲、雙節點 scaling 或 TensorRT-LLM 效能。
- 4090 的 llama.cpp／vLLM 比較包含 GGUF 與 Hugging Face 模型封裝差異；不是只控制引擎的單變量實驗。DGX 沒有有效 llama.cpp 量測。
- RAG 只量測已組裝 context 的推論，不包含 retrieval 或 reranker；舊 synthetic 基線與正式資料集 campaign 不可合併平均。
- 本次 GB10 遙測未取得 GPU 記憶體用量，空值不代表零；兩平台功耗涵蓋的電源域不同，不提出跨平台能源效率結論。
- 原始 results 含 ShareGPT 對話與第三方個資，保留於本機，不進 Git 或 release。公開 CSV、圖表與報告保留彙總結果與有效性後設資料；從原始結果重算仍需原始資料。

資料來源與發布理由見 [DATASETS](docs/DATASETS.md)、[PUBLISHING](docs/PUBLISHING.md)。自有內容採 [MIT 授權](LICENSE)；AIPerf 為 Apache-2.0、llama-benchy 為 MIT，模型與資料集適用各自授權。
