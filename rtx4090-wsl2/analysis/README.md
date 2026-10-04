# RTX 4090 分析輸出

由 `results/` 的原始執行產物產生的分析程式與衍生資料。原始產物不隨 repository
散布（見 [`../../docs/DATASETS.md`](../../docs/DATASETS.md)），本目錄是可公開的替代品。

## 程式

| 檔案 | 作用 |
|---|---|
| `analyze_benchmarks.py` | 解析微基準與 AIPerf aggregate，產生 `output/` 的 CSV 與 Markdown |
| `plot_benchmarks.py` | 產生 `charts/` 的 6 張核心比較圖 |
| `plot_dataset_figures.py` | 產生 `dataset-figures/` 的 8 張正式圖與 `tables/` 表格孿生檔 |

```bash
python analysis/analyze_benchmarks.py
python analysis/plot_benchmarks.py
python analysis/plot_dataset_figures.py
```

三支都以 `PLATFORM_ROOT`（本平台目錄）與 `REPO_ROOT`（repository 根層）分別定位
輸出與 `results/`，與 [`../../dgx-spark-cuda/analysis/`](../../dgx-spark-cuda/analysis/)
的寫法一致。

## 資料檔

| 檔案 | 內容 |
|---|---|
| `output/dataset-cell-summary-rtx4090.csv` | **權威記錄**：五個正式 campaign 的逐 cell 彙總，124 列 × 109 欄 |
| `output/core_benchmark_metrics.csv` | 微基準核心指標（context / concurrency / prefix cache） |
| `output/aiperf_key_metrics.csv` | AIPerf workload 的 latency、throughput 與 SLO 指標 |
| `output/benchmark_analysis.md` | 由 `analyze_benchmarks.py` 產生的中文分析 |
| `dataset-figures/*.png`／`.svg` | 8 張正式圖 |
| `dataset-figures/tables/*.csv` | 每張圖的表格孿生檔，只含該圖繪出的數值 |
| `charts/*.png` | 6 張微基準圖 |

### 哪一份才是判定有效性的依據

`dataset-cell-summary-rtx4090.csv` 每列都帶 `status`、`failure_reason`、
`valid_repeats`、`invalid_repeats`、`oom_repeats`、`request_count`、`p99_eligible`，
以及每個指標的 median／mean／CI95 上下界。**要判斷某個數字能不能用，看這一份。**

`dataset-figures/tables/*.csv` 是各張圖的表格孿生檔，只保存該圖實際繪出的數值，
不含完整有效性後設資料——它們的用途是讓讀者核對圖上的點，不是拿來做判定。

### 124 個 cell 的狀態分布

| Campaign | Backend | valid | invalid |
|---|---|---:|---:|
| `formal-split-65k-20260814` | vLLM | 37 | 0 |
| `formal-split-65k-20260814` | llama.cpp | 31 | 6 |
| `formal-prefix-65k-20260816` | vLLM | 12 | 0 |
| `formal-prefix-65k-20260816` | llama.cpp | 6 | 2 |
| `formal-vllm-stress-65k-20260816` | vLLM | 30 | 0 |

8 個 invalid 全部在 llama.cpp 側，`valid_repeats` 皆為 0——RAG 8K/16K C8、
ShareGPT natural r4-c16、RULER 60K C2。它們保留在 CSV 內未刪除，
記錄的是這張卡在該組態下的實際邊界，不是遺漏。

## 與 DGX Spark 的對比

`formal-split-65k` 的 vLLM 分支（37 cells）與 DGX Spark 是**完全相同的 cell 集合**，
兩邊皆全數 valid，可直接對比。對比結果、圖表與比值在
[`../../dgx-spark-cuda/analysis/`](../../dgx-spark-cuda/analysis/)。

## 解讀邊界

- 公平比較只使用兩個 backend 共有的 45 個 cells；vLLM-only 的 C32/C64 壓力格
  不得直接對照 llama.cpp 的 C16。
- 舊 synthetic/AIPerf baseline 與新版真實資料集結果是不同批次，不可合併平均。
- llama.cpp 的 GGUF 路徑與 vLLM 的 Hugging Face 路徑在模型封裝與 kernel 上不同，
  結論代表部署方案比較。
- 全部為 co-located benchmark，不代表網路服務延遲。

完整規則見 [`../../docs/RESULTS.md`](../../docs/RESULTS.md)。
