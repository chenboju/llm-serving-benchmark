# 發布方式

## 只發布衍生資料

本 repository 對外提供**彙總後的衍生資料**：CSV、圖表、報告與腳本。
原始執行產物（`results/`，約 12.26 GB／9,397 檔）既不進 Git，也不作為
release asset 發布。

決定的理由有兩層：

1. **第三方個資。** ShareGPT workload 的 `inputs.json` 含真實使用者對話，
   抽樣單一 11 MB 檔案即有 66 個不重複 email 與 1 段 private key 區塊。
   本專案合法取得並用作 workload，但沒有再散布的立場。
   詳見 [DATASETS.md](DATASETS.md#為什麼不散布-sharegpt-衍生資料)。
2. **體積與可用性。** 多個 `inputs.json` 超過 100 MB，超過 GitHub 單檔上限；
   即使改用 release asset 或 LFS，clone 與 review 也會變得笨重。

遮蔽個資後再發布不是可行折衷——改動測試輸入等於改變 token 數與 bucket 落點，
結果就不再與本專案的數字可比。

## 衍生資料如何維持可驗證

不發原始檔的代價是讀者無法自行重算，因此溯源資訊必須提升到衍生檔內：

- **有效性後設資料進 CSV**：逐 cell 彙總每列都帶 `status`、`failure_reason`、
  `valid_repeats`、`invalid_repeats`、`oom_repeats`、`request_count`、
  `p99_eligible`，以及各指標的 median／mean／CI95。invalid 的格子保留不刪。
- **資料一致性可比對**：[DATASET_MANIFEST.md](DATASET_MANIFEST.md) 列出每個
  prepared 資料檔的 SHA256，重現者可確認自己準備的資料與本專案相同。
- **環境與組態可查**：DGX 側的 `campaign.json` 收錄於
  [`../dgx-spark-cuda/campaigns/`](../dgx-spark-cuda/campaigns/)，內含
  `dataset_sha256`、`environment_manifest` SHA256、`aiperf_revision`、
  `backend_config` 與完整 server 啟動命令。

## 發布前檢查

- 搜尋 access token、API key、email、主機名稱與不應公開的絕對路徑。
  使用者路徑一律置換為 `<user>`。
- 確認公開的是 valid 結果，並完整保留 invalid/OOM 狀態而非靜默刪除。
- 從乾淨 clone 執行語法檢查（`python -m compileall`、`bash -n`）與分析重建。
- 確認模型與資料集授權允許目前的散布形式。

## Git LFS

本專案不使用 LFS。結果主要是大量 JSON/JSONL/CSV/LOG，且已決定不散布；
若日後需要發布大型二進位 artifact，應只追蹤明確選定的檔案，
不要把整個 `results/` 無條件加入 LFS。
