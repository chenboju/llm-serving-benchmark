# 資料集說明

本 repository 複製測試腳本、執行結果、分析與報告，**不重新散布原始或準備後的 dataset 檔案**。資料應以固定 revision、tokenizer 與 SHA256 重新準備；正式測試不可使用 approximate tokenizer。

## 使用的資料

| 資料集 | 用途 | 規模／長度 | 授權或散布注意事項 |
|---|---|---|---|
| ShareGPT | closed/open-loop chat、自然輸出長度 | 最多 4 turns，5,000 sessions | **不散布**，見下方說明 |
| SQuAD v1.1 | RAG、prefix cache | 1K/4K/8K/16K | CC BY-SA 4.0 |
| HotpotQA distractor/train | 多跳 RAG | 1K/4K/8K/16K | CC BY-SA 4.0 |
| NVIDIA RULER | 長上下文效能與正確率 | 4K/8K/16K/32K；60K 效能壓力 | Apache-2.0；固定 generator/evaluator revisions |

完整 revision 與資料欄位記錄於 [`../reports/dataset_benchmark_complete_report_zh-TW.md`](../rtx4090-wsl2/reports/dataset_benchmark_complete_report_zh-TW.md)。
逐檔 SHA256 與兩平台一致性見 [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md)。

## 為什麼不散布 ShareGPT 衍生資料

ShareGPT 是真實使用者自願分享的對話，內容含可識別個資。本專案抽樣檢查單一 11 MB 的
執行輸入檔即出現 66 個不重複 email 位址與 1 段 `BEGIN ... PRIVATE KEY`；共 31 個結果
目錄使用這批資料。

這些資料本專案合法取得並作為 workload 使用，但**本專案不是這些個資的持有人，沒有
再散布的立場**。因此：

- 原始執行產物（`results/`，約 12.26 GB）不進 Git，也不作為 release asset 發布。
- 對外只提供彙總後的衍生資料——CSV、圖表與報告，其中不含對話原文。
- 需要重現者請依下方流程自 upstream 自行準備，並以
  [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md) 的 SHA256 比對一致性。

遮蔽個資不是可行的替代方案：改動測試輸入就改變了 token 數與 bucket 落點，
結果將不再與本專案的數字可比。

## 準備後資料的約束

- 每個正式 JSONL 應有同名 `.index.jsonl`，記錄 sample ID、token 數、bucket、prefix group 與 cold/warm role。
- Manifest 應保存來源 revision、tokenizer revision、row count、建立時間與 SHA256。
- RAG/prefix 的實際輸入長度應落在目標 bucket 的 ±5%。
- Prefix 每組 4 個問題，順序為 0–3，並包含 1 個 cold 與 3 個 warm request。
- ShareGPT 每筆須有 `session_id`，並包含 1–4 turns。
- RULER 60K 僅作效能與資源邊界測試，不納入官方 accuracy 曲線。

## 準備流程

```bash
python scripts/dataset-tests/prepare_datasets.py download

python scripts/dataset-tests/prepare_datasets.py sharegpt \
  --source datasets/raw/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer /path/to/gpt-oss-20b-tokenizer

python scripts/dataset-tests/prepare_datasets.py rag \
  --squad-source datasets/raw/squad-train-v1.1.json \
  --hotpot-source datasets/raw/hotpotqa-distractor-train.jsonl \
  --tokenizer /path/to/gpt-oss-20b-tokenizer

python scripts/dataset-tests/prepare_datasets.py prefix \
  --squad-source datasets/raw/squad-train-v1.1.json \
  --tokenizer /path/to/gpt-oss-20b-tokenizer

python scripts/dataset-tests/validate_prepared.py
```

RULER 的固定 commit 與 NeMo-Skills evaluator 操作請依 [`../scripts/dataset-tests/README.md`](../scripts/dataset-tests/README.md) 執行。

