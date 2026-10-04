# 測試結果說明

## 目錄層級

微基準通常採以下結構：

```text
results/<backend>/<suite>/<timestamp>/
├── metadata.txt
├── server.log
├── summary.csv
├── raw/
└── <chart>.png
```

正式 dataset campaign 通常採以下結構：

```text
results/<backend>/dataset/<campaign>/
├── campaign.json
├── campaign-validation.json
├── server-*.log
└── <cell-id>/
    ├── cell.json
    └── repeat-01/
        ├── command.txt
        ├── inputs.json
        ├── outputs.json
        ├── profile_export_aiperf.csv
        ├── profile_export_aiperf.json
        ├── profile_export.jsonl
        ├── gpu_telemetry_export.jsonl
        ├── phase_manifest.json
        └── logs/
```

實際檔案依 AIPerf 版本與測試是否完整結束而略有不同。

## 常見指標

| 指標 | 意義 | 判讀重點 |
|---|---|---|
| TTFT | Time to First Token | Prompt processing、排隊與 prefill 延遲 |
| ITL | Inter-token Latency | 串流生成時相鄰 token 的延遲 |
| Request latency | 完整請求時間 | 同時受輸入、輸出長度與排隊影響 |
| Output tokens/s | 總輸出吞吐 | 高併發服務的主要容量指標之一 |
| Requests/s | 完成請求率 | 必須搭配實際輸出長度判讀 |
| Good Request Fraction | 同時通過 TTFT 與 ITL SLO 的比例 | 建議容量判定門檻為 ≥95% |
| Error rate | HTTP、timeout 或其他請求錯誤比例 | 與 SLO miss、OOM 分開統計 |

## 有效性規則

- Repeat 必須 exit code 為 0、未偵測到 GPU OOM，且 request error rate ≤1%。
- Cell 至少需要 3 個 valid repeats 才能形成正式彙總。
- Output throughput CV >10% 或 P95 TTFT CV >15% 時，正式流程會增加到 5 repeats。
- P99 僅在 pooled valid requests ≥1,000 時解讀。
- `valid` 不等於通過 SLO；請求可以無錯誤完成但延遲不合格。
- `results/interrupted/` 不屬於有效結果，只可用於失敗原因分析。

## 比較邊界

- 公平比較只使用兩個 backend 共有的 45 個 cells；vLLM-only 的 C32/C64 壓力格不得直接對照 llama.cpp C16 宣稱引擎倍數。
- 舊 synthetic/AIPerf baseline 與新版真實資料集結果是不同批次，不可合併平均。
- llama.cpp 的 GGUF 路徑與 vLLM 的 Hugging Face 路徑包含模型封裝與 kernel 差異；結論代表部署方案比較。
- 先讀逐 cell 彙總的 `status` / `valid_repeats` / `failure_reason` 欄位，再使用吞吐或 latency 欄位。

## 公開發布的衍生資料

原始 `results/` 不對外散布（見 [DATASETS.md](DATASETS.md)），因此有效性後設資料已
提升到彙總 CSV 內，讀者不需原始檔即可自行套用上節的規則：

| 檔案 | 平台 | 內容 |
|---|---|---|
| [`../rtx4090-wsl2/analysis/output/dataset-cell-summary-rtx4090.csv`](../rtx4090-wsl2/analysis/output/dataset-cell-summary-rtx4090.csv) | RTX 4090 | 124 cells（116 valid / 8 invalid） |
| [`../dgx-spark-cuda/analysis/dataset-cell-summary.csv`](../dgx-spark-cuda/analysis/dataset-cell-summary.csv) | DGX Spark | 83 cells（81 valid / 2 invalid） |
| [`../dgx-spark-cuda/analysis/cross-platform-comparison.csv`](../dgx-spark-cuda/analysis/cross-platform-comparison.csv) | 兩者 | 37 個共同 cell 的對照與比值 |

兩份 cell 彙總皆為 109 欄，每列帶 `status`、`failure_reason`、`valid_repeats`、
`invalid_repeats`、`oom_repeats`、`request_count`、`p99_eligible`，以及每個指標的
median／mean／CI95 上下界。invalid 的格子保留在檔案內，不刪除。

`rtx4090-wsl2/analysis/dataset-figures/tables/*.csv` 是各張圖的表格孿生檔，只含該圖繪出的數值；
需要有效性判定時請以上述 cell 彙總為準。

`rtx4090-wsl2/analysis/output/aiperf_key_metrics.csv` 的 `source` 欄是**執行識別碼**，用來對照
campaign 與 cell，不是可解析的檔案路徑——它指向未公開的 `results/`。

## 由原始結果重新產生分析

```bash
python rtx4090-wsl2/analysis/analyze_benchmarks.py
python rtx4090-wsl2/analysis/plot_benchmarks.py
python rtx4090-wsl2/analysis/plot_dataset_figures.py

python scripts/dataset-tests/analyze_dataset_results.py
python scripts/dataset-tests/plot_dataset_results.py
```

輸出檔不可取代原始 result record；兩者應使用相同 release/tag 一起保存。

