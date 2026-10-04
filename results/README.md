# 完整測試結果（本機保留，不對外散布）

此目錄在執行環境中存放測試的原始執行產物，依 backend 與 suite 分類：

- `llama-cpp/`、`vllm/`：各 suite 與 dataset campaign 的逐 repeat 產物
- `interrupted/`：中斷執行留下的診斷檔，不計入正式有效結果

**本 repository 不包含這些檔案，也不會以 release asset 形式發布。**

完整快照約 12.26 GB／9,397 個檔案。不散布的原因不只是體積：ShareGPT workload 的
`inputs.json` 含第三方可識別個資，本專案沒有再散布的立場。詳見
[`../docs/DATASETS.md`](../docs/DATASETS.md#為什麼不散布-sharegpt-衍生資料)。

## 改用什麼

對外提供的是彙總後的衍生資料，其中已內含有效性後設資料：

| 內容 | 位置 |
|---|---|
| RTX 4090 逐 cell 彙總 | [`../analysis/output/dataset-cell-summary-rtx4090.csv`](../rtx4090-wsl2/analysis/output/dataset-cell-summary-rtx4090.csv) |
| DGX Spark 逐 cell 彙總 | [`../dgx-spark-cuda/analysis/dataset-cell-summary.csv`](../dgx-spark-cuda/analysis/dataset-cell-summary.csv) |
| 跨平台對比 | [`../dgx-spark-cuda/analysis/cross-platform-comparison.csv`](../dgx-spark-cuda/analysis/cross-platform-comparison.csv) |
| 圖表與表格孿生檔 | [`../analysis/`](../rtx4090-wsl2/analysis/) |
| 正式報告 | [`../reports/`](../rtx4090-wsl2/reports/) |

欄位定義、有效性規則與比較邊界見 [`../docs/RESULTS.md`](../docs/RESULTS.md)。

## 自行重現

依 [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) 準備環境與資料，
並以 [`../docs/DATASET_MANIFEST.md`](../docs/DATASET_MANIFEST.md) 的 SHA256
確認資料與本專案一致後再執行。完整 campaign 約需 45–55 GPU 小時。
