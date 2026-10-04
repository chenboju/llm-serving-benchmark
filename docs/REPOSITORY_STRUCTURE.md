# Repository 結構

本 repository 分成**共用層**與**兩個平台目錄**。共用層放兩平台都用的 harness 與文件，
平台目錄各自放該硬體的腳本、分析與結果。

```text
.
├── docs/                    共用文件
├── scripts/                 共用 harness
├── rtx4090-wsl2/            平台：RTX 4090 / WSL2
├── dgx-spark-cuda/          平台：DGX Spark（GB10 / ARM64）
└── results/                 公開說明與舊摘要；原始產物不散布
    ├── README.md
    └── generated-summary.md
```

兩平台共用同一份資料集與同一組有效性規則；`formal-split-65k` 這組 campaign
在兩邊執行的是完全相同的 37 個 cells，可直接對比。

## `scripts/` — 共用 harness

- `dataset-tests/`：資料準備、matrix 展開、preflight、campaign runner、RULER 評分、
  分析與離線 regression tests。**兩個平台共用**；DGX 側對其中 5 支做了平台專屬分支，
  收在 `dgx-spark-cuda/scripts/`。
- `benchlab.sh`、`benchlab.ps1`：benchlab CLI wrapper；需設定 `BENCHLAB_PROJECT` 指向外部專案，安裝方式見 [腳本索引](../scripts/README.md#可選的-benchlab-cli)。

## `rtx4090-wsl2/` — RTX 4090 / WSL2

| 目錄 | 內容 |
|---|---|
| `analysis/` | `analyze_benchmarks.py`、`plot_benchmarks.py`：舊 baseline 結果的摘要與圖表<br>`plot_dataset_figures.py`：正式資料集圖表產生器<br>`output/`：核心 CSV、Markdown 分析、逐 cell 彙總<br>`charts/`、`dataset-figures/`：PNG/SVG 圖表及其表格孿生檔 |
| `reports/` | `dataset_benchmark_complete_report_zh-TW.md`：最完整的繁體中文研究報告<br>`dataset_benchmark_table_report_zh-TW.md`：表格導向版本<br>`dataset_benchmark_visual_guide_zh-TW.md`：圖表閱讀指南 |
| `scripts/` | `llama-cpp-tests/`：llama.cpp 啟動、context、concurrency、prefix cache 與三種 AIPerf workload<br>`vllm/`：對應的 vLLM 測試與共用 server lifecycle helper<br>`verify-rtx4090-wsl2-env.sh`：檢查 WSL2、GPU 與相關工具環境 |

詳見 [`../rtx4090-wsl2/README.md`](../rtx4090-wsl2/README.md)。

## `dgx-spark-cuda/` — DGX Spark（GB10 / ARM64）

| 目錄 | 內容 |
|---|---|
| `docs/` | 平台專屬文件：測試範圍、ARM64 環境建立、驗收清單、報告範本 |
| `scripts/` | 10 支平台專屬腳本（含 3 支 vLLM server 啟動器），其餘與共用 harness 相同 |
| `campaigns/` | 各 campaign 的 `campaign.json` 與 `campaign-validation.json`，內含溯源資訊 |
| `analysis/` | 逐 cell 彙總、跨平台對比 CSV、圖表與重建腳本 |

詳見 [`../dgx-spark-cuda/README.md`](../dgx-spark-cuda/README.md)。

## `results/` — 公開說明與本機原始產物

公開版本包含 `README.md` 與 `generated-summary.md`（舊流程摘要）；以下原始資料目錄僅保留於本機：

- `llama-cpp/`、`vllm/`：依 backend、suite、run/campaign、cell、repeat 分層的原始產物。
- `interrupted/`：未完整結束的執行，僅供診斷。

**不隨 repository 散布**，也不作為 release asset 發布——ShareGPT workload 的輸入檔
含第三方可識別個資。理由與替代方案見
[`DATASETS.md`](DATASETS.md#為什麼不散布-sharegpt-衍生資料) 與
[`PUBLISHING.md`](PUBLISHING.md)。

## `docs/` — 共用文件

- [`DATASETS.md`](DATASETS.md)：資料來源、用途、準備方式與授權注意事項。
- [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md)：逐檔 SHA256 與兩平台一致性比對。
- [`RESULTS.md`](RESULTS.md)：結果層級、常見檔案、有效性規則與判讀邊界。
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)：重建環境與重跑測試的順序。
- [`PUBLISHING.md`](PUBLISHING.md)：只發布衍生資料的理由與做法。
- [`EXTERNAL_TOOLS.md`](EXTERNAL_TOOLS.md)：AIPerf 與 llama-benchy 的來源、授權、版本與分工。
- [`COPY_MANIFEST.md`](COPY_MANIFEST.md)：建立 repository 時的來源與容量紀錄。

## 平台目錄的路徑慣例

兩個平台的分析腳本都以兩個常數定位檔案，彼此對稱：

```python
PLATFORM_ROOT = Path(__file__).resolve().parents[1]   # rtx4090-wsl2/ 或 dgx-spark-cuda/
REPO_ROOT     = PLATFORM_ROOT.parent                   # repository 根層
```

輸出寫回 `PLATFORM_ROOT`，共用 harness 與 `results/` 則從 `REPO_ROOT` 取得。

## 文件語言

本 repository 有 31 份 Markdown，頂層 README 提供英文與繁體中文版本，其餘文件主要為繁體中文。檔名是否含 `zh-TW` 不代表內容語言，不以檔名計算英／中文份數。
