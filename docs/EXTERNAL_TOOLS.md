# 外部工具

本專案**不包含**負載產生器本身，只包含呼叫它們的腳本與分析程式。所有量測數字都由
以下兩個第三方工具產生；它們**不是本專案作者的作品**。

| 工具 | 上游 | 授權 | 本專案使用的版本 |
|---|---|---|---|
| **AIPerf** | [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) | Apache-2.0 | 0.12.0，commit [`03c9c6ddc`](https://github.com/ai-dynamo/aiperf/commit/03c9c6ddc5e6227782e53ded177f1227d332af48) |
| **llama-benchy** | [eugr/llama-benchy](https://github.com/eugr/llama-benchy) | MIT © Eugene Rakhmatulin | 0.4.1.dev1，commit [`e9be344`](https://github.com/eugr/llama-benchy/commit/e9be344578cec17745066b220798b80a0d2686d3) |

兩者皆以固定 commit 使用。commit 記錄在每個 campaign 的 `campaign.json`
（`aiperf_revision` 欄位），與量測結果一併保存——換版本就換了量測工具，
結果不應跨版本並列比較。

## 各自負責什麼

| | AIPerf | llama-benchy |
|---|---|---|
| 量測範圍 | TTFT、ITL、request latency、output throughput、error rate、SLO 達成率、GPU telemetry | prefill / decode 吞吐 |
| 負載型態 | closed-loop 與 open-loop（可指定到達率） | 固定 prompt / 輸出長度的微基準 |
| 用在哪 | 正式 dataset campaign（[`../scripts/dataset-tests/`](../scripts/dataset-tests/)） | 微基準（[`../rtx4090-wsl2/scripts/`](../rtx4090-wsl2/scripts/)） |
| 產出 | `profile_export_aiperf.csv`、`profile_export.jsonl`、`gpu_telemetry_export.jsonl` | `summary.csv` |

正式的跨平台對比數字全部來自 AIPerf；llama-benchy 用於 context / concurrency
scaling 的微基準，兩者的數字不混用、不平均。

## 取得方式

兩個專案都需與本 repository 放在同一層：

```text
workspace/
├── aiperf/                  # commit 03c9c6ddc
├── llama-benchy/            # commit e9be344
└── llm-serving-benchmark/   # 本 repository
```

```bash
git clone https://github.com/ai-dynamo/aiperf.git
git -C aiperf checkout 03c9c6ddc5e6227782e53ded177f1227d332af48

git clone https://github.com/eugr/llama-benchy.git
git -C llama-benchy checkout e9be344578cec17745066b220798b80a0d2686d3
```

安裝步驟與環境驗證見 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)；
DGX Spark 平台的完整安裝流程見
[`../dgx-spark-cuda/docs/01-環境建立-dgx-spark.md`](../dgx-spark-cuda/docs/01-環境建立-dgx-spark.md)。

`00-preflight.sh` 會在每次 campaign 前檢查兩者是否存在且 revision 正確，
不符即中止。

## 本專案沒有修改它們

本 repository 不含對 AIPerf 或 llama-benchy 的任何修改。

需要修改第三方專案時，應以 patch 形式獨立保存並標明來源，而不是把改動混進
本專案的腳本裡。姊妹專案 [**dgx-spark-serving**](https://github.com/chenboju/dgx-spark-serving) 的 [`vllm/upstream/`](https://github.com/chenboju/dgx-spark-serving/tree/main/vllm/upstream) 就是這樣處理
對 `eugr/spark-vllm-docker` 的修改。

## 其他相依

模型權重、llama.cpp 與 vLLM 同樣不包含在本 repository 內。版本與 revision
記錄於 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) 與各 campaign 的
`campaign.json`。
