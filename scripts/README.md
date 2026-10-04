# 測試腳本索引

| 目錄／檔案 | 用途 |
|---|---|
| `../rtx4090-wsl2/scripts/llama-cpp-tests/` | llama.cpp 啟動、context、concurrency、prefix cache 與 AIPerf workload |
| `../rtx4090-wsl2/scripts/vllm/` | vLLM server lifecycle、對應微基準與 AIPerf workload |
| `dataset-tests/` | 資料準備、正式矩陣、campaign、RULER、分析與測試 |
| `../rtx4090-wsl2/scripts/verify-rtx4090-wsl2-env.sh` | WSL2/GPU/依賴環境檢查 |
| `benchlab.sh`, `benchlab.ps1` | 外部 benchlab 專案的 CLI wrapper |

所有 `.sh` 預期在 WSL2/Linux 執行。請從 repository 根目錄啟動，以確保輸出落在正確的 `results/`。執行前先閱讀 [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)。

完整 dataset suite 的入口與參數請見 [`dataset-tests/README.md`](dataset-tests/README.md)。


## 可選的 benchlab CLI

這兩支 wrapper 需要另外取得 `local-llm-serving-lab` 原始專案；本 repository 不包含其實作，也未提供公開下載網址。若沒有該專案，直接使用 `dataset-tests/` 的正式流程即可。

外部專案需包含 `pyproject.toml` 與 `src/benchlab/`，並宣告 `benchlab = "benchlab.cli:main"` 入口。該專案要求 Python ≥3.11；請先安裝 uv，再由 uv 依外部專案的依賴宣告建立環境。`BENCHLAB_PROJECT` 請設為絕對路徑。

Linux / WSL2（從本 repository 根目錄執行）：

```bash
export BENCHLAB_PROJECT="/path/to/local-llm-serving-lab"
uv sync --project "$BENCHLAB_PROJECT"
bash scripts/benchlab.sh --help
```

PowerShell：

```powershell
$env:BENCHLAB_PROJECT = "C:\path\to\local-llm-serving-lab"
uv sync --project $env:BENCHLAB_PROJECT
./scripts/benchlab.ps1 --help
```

wrapper 會將參數原樣傳給外部 CLI 並回傳其結束碼。未設定專案路徑、缺少 `pyproject.toml` 或找不到 uv 時會中止並提示；執行工作目錄仍是呼叫時的位置。
