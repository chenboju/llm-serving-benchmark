# RTX 4090 / WSL2 測試

GPT-OSS-20B 在單張 **NVIDIA GeForce RTX 4090 24 GB**（Windows + WSL2）上，
以 llama.cpp 與 vLLM 提供 OpenAI-compatible API 的測試內容。

與 [`../dgx-spark-cuda/`](../dgx-spark-cuda/) 共用 [`../scripts/dataset-tests/`](../scripts/dataset-tests/)
的 harness、同一份資料集與同一組有效性規則。`formal-split-65k` 這組 campaign
在兩平台執行的是**完全相同的 37 個 cells**，可直接對比。

## 執行環境

| 項目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| 平台 | Windows + WSL2 |
| 模型 | GPT-OSS-20B（MXFP4 GGUF／HF revision `6cee5e81`）|
| Backend | llama.cpp commit `312cf0332`、vLLM |
| AIPerf | commit `03c9c6ddc` |

完整版本鎖定見 [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)。

## 內容

```text
rtx4090-wsl2/
├── analysis/            分析程式、圖表與表格孿生檔
│   ├── plot_dataset_figures.py   8 張正式 dataset 圖
│   ├── plot_benchmarks.py        微基準圖表
│   ├── analyze_benchmarks.py     CSV 彙整
│   ├── dataset-figures/          8 張圖（PNG + SVG）與 tables/
│   ├── charts/                   6 張微基準圖
│   └── output/                   彙總 CSV 與逐 cell 記錄
├── reports/             三份繁體中文正式報告
└── scripts/             平台專屬腳本
    ├── llama-cpp-tests/   llama.cpp 微基準與 AIPerf（11 支）
    ├── vllm/              vLLM 微基準與 AIPerf（13 支）
    └── verify-rtx4090-wsl2-env.sh
```

正式 dataset campaign 的執行腳本在共用的
[`../scripts/dataset-tests/`](../scripts/dataset-tests/)，不在此處。

## 測試結果

**124 個 cell，116 valid、8 invalid**，涵蓋五個正式 campaign：

| Campaign | Backend | Cells |
|---|---|---|
| `formal-split-65k-20260814` | vLLM | 37 valid |
| `formal-split-65k-20260814` | llama.cpp | 31 valid / 6 invalid |
| `formal-prefix-65k-20260816` | vLLM | 12 valid |
| `formal-prefix-65k-20260816` | llama.cpp | 6 valid / 2 invalid |
| `formal-vllm-stress-65k-20260816` | vLLM | 30 valid |

llama.cpp 側的 8 個 invalid 格子（RAG 8K/16K C8、ShareGPT natural、RULER 60K）
`valid_repeats` 皆為 0，保留在
[`analysis/output/dataset-cell-summary-rtx4090.csv`](analysis/output/dataset-cell-summary-rtx4090.csv)
內未刪除——它們記錄的是這張卡在該組態下的實際邊界。

## 可讀的成果

- [`reports/dataset_benchmark_complete_report_zh-TW.md`](reports/dataset_benchmark_complete_report_zh-TW.md)——完整報告
- [`reports/dataset_benchmark_table_report_zh-TW.md`](reports/dataset_benchmark_table_report_zh-TW.md)——表格版
- [`reports/dataset_benchmark_visual_guide_zh-TW.md`](reports/dataset_benchmark_visual_guide_zh-TW.md)——視覺導讀
- [`analysis/dataset-figures/`](analysis/dataset-figures/)——8 張正式圖與對應 CSV
- [`analysis/output/benchmark_analysis.md`](analysis/output/benchmark_analysis.md)——分析程式產生的核心比較

與 DGX Spark 的跨平台對比在
[`../dgx-spark-cuda/analysis/`](../dgx-spark-cuda/analysis/)。

## 重建分析

原始 `results/` 不隨 repository 散布（見
[`../docs/DATASETS.md`](../docs/DATASETS.md)），需在本機備妥後執行：

```bash
python analysis/analyze_benchmarks.py
python analysis/plot_benchmarks.py
python analysis/plot_dataset_figures.py
```

三支腳本以 `PLATFORM_ROOT`（本目錄）與 `REPO_ROOT`（repository 根層）分別定位
輸出與 `results/`，與 [`../dgx-spark-cuda/analysis/`](../dgx-spark-cuda/analysis/)
的寫法一致。

## 比較邊界

- 公平比較只使用兩個 backend 共有的 45 個 cells；vLLM-only 的 C32/C64 壓力格
  不得直接對照 llama.cpp 的 C16。
- llama.cpp 的 GGUF 路徑與 vLLM 的 Hugging Face 路徑在模型封裝與 kernel 上不同，
  結論代表**部署方案**比較，不是引擎的普遍結論。
- 全部為 co-located benchmark，不代表網路服務延遲。

完整規則見 [`../docs/RESULTS.md`](../docs/RESULTS.md)。
