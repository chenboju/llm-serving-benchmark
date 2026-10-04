# Contributing

感謝協助改善測試方法、可重現性或文件。由於 benchmark 很容易因環境差異產生不可比結果，提交變更時請遵守以下原則。

## 提交前

1. 不要提交模型權重、token、憑證、個人路徑或未確認授權的原始資料集。
2. 不要直接手動修改由程式產生的圖表與彙整表；應修改分析程式後重新產生。
3. 若修改 workload、SLO、模型、cache、sampling 或 server 參數，請在 PR 中說明可比性影響。
4. 新結果須包含 backend 版本、模型 revision、GPU/driver、完整命令、seed、warm-up、profiling 時間與重複次數。
5. 中斷、OOM、HTTP error 與 SLO miss 必須分開標記，不得只保留成功樣本。

## 基本檢查

```bash
python -m compileall -q scripts analysis
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m unittest discover -s scripts/dataset-tests -p 'test_*.py'
```

Dataset tests 需要先安裝 `scripts/dataset-tests/requirements.txt`。需要 GPU 的完整 benchmark 不在一般 CI 中執行。

## Pull request 說明

PR 至少應包含：變更目的、影響的 suite/cell、驗證方法、是否破壞與既有結果的可比性，以及是否產生新的大型 artifact。

