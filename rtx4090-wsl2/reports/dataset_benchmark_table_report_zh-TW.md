# GPT-OSS-20B 推論基準表格版報告

## 一、報告概覽

| 項目 | 內容 |
|---|---|
| 報告日期 | 2026-08-14 |
| 比較對象 | llama.cpp、vLLM |
| 模型 | GPT-OSS-20B |
| 硬體 | NVIDIA GeForce RTX 4090，24564 MiB |
| 正式計畫 | `gpt-oss-20b-dataset-v1` |
| 新版測試狀態 | 資料與框架驗證完成；少數 pilot 完成；完整矩陣尚未完成 |
| 舊版測試狀態 | synthetic/AIPerf 基線已完成，但不得與新版結果混合平均 |

## 二、執行環境與公平性

| 項目 | llama.cpp | vLLM | 公平性說明 |
|---|---|---|---|
| 模型格式 | GPT-OSS-20B MXFP4 GGUF | GPT-OSS-20B Hugging Face snapshot | 同模型家族，但格式與 kernel 路徑不同 |
| Model ID | `openai/gpt-oss-20b` | `gpt-oss-20b` | API 名稱不同，不影響資料集 |
| Context 上限 | 65,536 | 65,536 | 相同 |
| 最大 client concurrency | 16 | 64 | 公平區只比較共同負載；C32/C64 另列壓力測試 |
| Server 排程 | parallel slots 16 | `max-num-seqs=16` | vLLM C32/C64 包含 server queueing |
| 一般測試 prefix cache | 關閉 | 關閉 | 相同 |
| Prefix 專項 | off/on 分開 | off/on 分開 | 必須以相同 cache mode 比較 |
| Tokenizer | GPT-OSS-20B snapshot `6cee5e81...` | 相同 | 相同 |
| Random seed | 42 | 42 | 相同 |
| Temperature | 0 | 0 | 相同 |
| API | Chat、streaming、server token count | 相同 | 相同 |

| 軟體／硬體項目 | 版本或識別 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24564 MiB |
| NVIDIA Driver | 595.79 |
| CUDA（環境紀錄） | 13.2 |
| AIPerf | 0.12.0，commit `03c9c6dd...` |
| llama.cpp | commit `312cf033...` |
| llama-benchy | 0.4.1.dev1，commit `e9be3445...` |

## 三、資料集

### 3.1 資料集用途與規模

| 資料集 | 使用測試 | 工作負載特性 | Context／模式 | 每個檔案 | 輸出 |
|---|---|---|---|---:|---|
| ShareGPT fixed | Closed-loop、Open-loop | 真實多輪聊天，最多 4 turns | 自然輸入分布 | 5,000 sessions | 固定 128 tokens |
| ShareGPT natural | Natural output | 真實多輪聊天與輸出長度分布 | 自然輸入分布 | 5,000 sessions | 記錄 OSL，不強制最少 128 |
| SQuAD | RAG | 單段落、單跳問答 | 1K/4K/8K/16K | 每長度 250 | 固定 256 tokens |
| HotpotQA | RAG | 多文件、多跳問答 | 1K/4K/8K/16K | 每長度 250 | 固定 256 tokens |
| SQuAD prefix | Prefix cache | 共享 context、cold/warm request | 4K/16K、off/on | 每檔 400 | 固定 128 tokens |
| RULER performance | 長上下文效能 | Prefill、TTFT、吞吐、錯誤與 OOM | 4K/8K/16K/32K | 每長度 1,300 | 固定 128 tokens |
| RULER performance 60K | 長上下文邊界 | 接近 65K context 上限 | 60K | 260 | 固定 128 tokens |
| RULER accuracy | 長上下文正確率 | 官方 13 種任務 | 4K/8K/16K/32K | 13 × 100／長度 | Natural EOS |

### 3.2 HotpotQA 的位置

| 項目 | HotpotQA 設定 |
|---|---|
| 使用位置 | RAG suite |
| 用途 | 模擬需要整合多段文件資訊的多跳 RAG |
| Context buckets | 1K、4K、8K、16K |
| 每個 bucket | 250 筆 |
| 固定輸出 | 256 tokens |
| 公平比較 | llama.cpp、vLLM：C1、C8 |
| vLLM 壓力 | C16、C32 |
| 與 SQuAD 差異 | SQuAD 偏單跳；HotpotQA 偏多文件、多跳推理 |
| 已有 pilot | llama.cpp HotpotQA 16K C8：5/5 repeats 無效，error rate 62.5% |

### 3.3 來源與授權

| 資料集 | 來源／revision | 授權註記 |
|---|---|---|
| ShareGPT | `ShareGPT_V3_unfiltered_cleaned_split.json`；下載後以 SHA256 固定 | 重新散布前確認 upstream terms |
| SQuAD | train-v1.1 | CC BY-SA 4.0 |
| HotpotQA | `hotpotqa/hotpot_qa`，distractor/train | CC BY-SA 4.0 |
| RULER | generator `c3f5e3b4...`；rulerv1-ns `e8bbff67...`；NeMo-Skills `f4a3fd8e...` | Apache-2.0 |

### 3.4 資料品質規範

| 驗證項目 | 規範 | 目前結果 |
|---|---|---|
| JSONL schema | 欄位必須符合 AIPerf 0.12.0 | 通過 |
| Sidecar | 每檔必須有 `.index.jsonl` | 通過 |
| 完整性 | 資料檔與 sidecar SHA256 必須符合 manifest | 通過 |
| Tokenizer | 正式 GPT-OSS-20B tokenizer，不接受 approximate | 通過 |
| RAG/prefix bucket | 實際 tokens 位於目標 ±5% | 通過 |
| Prefix group | 每組 4 題，1 cold + 3 warm，順序 0–3 | 通過 |
| ShareGPT turns | 每 session 1–4 turns | 通過 |
| 總驗證 | 19 個資料檔、79 個 conditions | 全部通過 |
| 離線框架測試 | 14 tests | 14/14 通過 |

## 四、測試矩陣

| 測試 | 測量目的 | llama.cpp 公平比較 | vLLM 公平比較 | vLLM 額外壓力 |
|---|---|---|---|---|
| ShareGPT closed-loop | 固定進行中請求下的吞吐與延遲擴展 | C1/2/4/8/16 | 相同 | C32/64 |
| ShareGPT open-loop | 固定到達率、排隊與最大可持續流量 | 0.5/1/2/4/8 req/s，最大 C16 | 相同 | 16/32 req/s，最大 C64 |
| ShareGPT natural | 自然輸出長度下的服務效能 | 4 req/s，C16 | 相同 | 8/16 req/s，C64 |
| RAG | 長 prompt、多文件與多跳負載 | SQuAD/HotpotQA × 1K/4K/8K/16K × C1/8 | 相同 | 每組 C16/32 |
| Prefix cache | 共享 prefix 的 TTFT 與吞吐改善 | 4K/16K × off/on × C1/8 | 相同 | 每組 C16 |
| RULER performance | 長上下文 prefill、延遲、錯誤與 OOM | 4K/8K/16K/32K/60K × C1/2 | 相同 | 4K:C8/16/32；8K:C8/16；16K:C4/8；32K:C4 |

| 矩陣範圍 | Conditions | Backend-cells |
|---|---:|---:|
| 公平比較 | 45 | 90（兩個後端各 45） |
| vLLM-only 壓力 | 34 | 34 |
| AIPerf 合計 | 79 | 124 |
| RULER accuracy | 另行執行 | 每後端 5,200 samples |

## 五、執行與判定規範

### 5.1 每個 cell 的執行流程

| 順序 | 階段 | 規範 |
|---:|---|---|
| 1 | Smoke test | 10 requests，確認 endpoint 與 model ID |
| 2 | GPU idle | Utilization < 5%，穩定 30 秒；最多等待 900 秒 |
| 3 | Warm-up | 30 秒 |
| 4 | Profiling | 300 秒 |
| 5 | Repeats | 預設 3 次 |
| 6 | Cooldown | Repeat 間 30 秒 |
| 7 | 不穩定重跑 | Output throughput CV > 10% 或 P95 TTFT CV > 15% 時增至 5 次 |
| 8 | 保存產物 | Command、logs、records、outputs、GPU telemetry、metrics、validation JSON |

| 執行範圍 | 純 profiling 時數 |
|---|---:|
| 兩後端公平矩陣 | 22.5 GPU 小時 |
| vLLM stress | 8.5 GPU 小時 |
| 合計下限 | 31 GPU 小時；未含 warm-up、cooldown、restart、idle wait 與重跑 |

### 5.2 有效性、穩定性與 SLO

| 層級 | 通過條件 | 說明 |
|---|---|---|
| Repeat valid | Exit code 0、無 OOM、error rate ≤ 1% | 只代表執行有效 |
| Cell valid | 至少 3 個 valid repeats | 可形成正式彙總 |
| Cell stable | Throughput CV ≤ 10%、P95 TTFT CV ≤ 15% | 超標則增加到 5 repeats |
| SLO pass | Good Request Fraction ≥ 95% | Valid 不等於 SLO pass |
| P99 eligible | Pooled valid requests ≥ 1,000 | 不足時 P99 標記 N/A |

| SLO Profile | TTFT 上限 | ITL 上限 | 適用測試 |
|---|---:|---:|---|
| Chat | 1,000 ms | 50 ms | ShareGPT |
| RAG | 2,000 ms | 50 ms | RAG、Prefix |
| Long-context | 5,000 ms | 75 ms | RULER performance |

### 5.3 報告指標

| 類別 | 必報指標 |
|---|---|
| 延遲 | TTFT P50/P95、ITL P50/P95、Request latency P95 |
| 吞吐 | Output tokens/s、Requests/s、Per-user tokens/s |
| 負載 | Concurrency、目標與實際 request rate、實際 input/output tokens |
| 品質 | Good Request Fraction、SLO attainment |
| 可靠性 | Error rate、timeout、OOM、invalid repeats |
| 統計 | Repeats、request count、95% CI、throughput CV、P95 TTFT CV |
| 資源 | GPU memory、utilization、power、energy |

## 六、新版真實資料集結果

### 6.1 完成狀態

| Backend | Cell | 狀態 | Repeats | 是否納入正式比較 |
|---|---|---|---:|---|
| llama.cpp | ShareGPT closed C1 | Valid、stable | 3 | 是 |
| llama.cpp | ShareGPT closed C16 | Valid、stable | 3 | 是 |
| llama.cpp | RULER 60K C1 | Valid、stable，但 SLO fail | 3 | 是 |
| vLLM | RULER 60K C1 | Valid、stable，但 SLO fail | 3 | 是 |
| llama.cpp | HotpotQA 16K C8 | Invalid | 5 | 否 |
| llama.cpp | RULER 60K C2 | Invalid | 5 | 否 |
| 兩個後端 | 其餘 comparison cells | 尚未完成 | — | 否 |
| vLLM | Stress cells | 尚未完成 | — | 否 |

### 6.2 有效 pilot 結果

| Backend | Cell | Repeats | Requests | Output tok/s | Req/s | P95 TTFT | P95 ITL | Error | Good % | SLO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama.cpp | ShareGPT closed C1 | 3 | 1,246 | 172.97 | 1.38 | 189.95 ms | 5.31 ms | 0% | 100% | Pass |
| llama.cpp | ShareGPT closed C16 | 3 | 2,646 | 365.17 | 2.92 | 1,223.39 ms | 47.42 ms | 0% | 88.78% | Fail |
| llama.cpp | RULER 60K C1 | 3 | 72 | 9.88 | 0.08 | 12,485.06 ms | 6.79 ms | 0% | 0% | Fail |
| vLLM | RULER 60K C1 | 3 | 100 | 14.06 | 0.11 | 8,644.14 ms | 7.35 ms | 0% | 0% | Fail |

### 6.3 60K C1 公平比較

| 指標 | llama.cpp | vLLM | 差異 | 判讀 |
|---|---:|---:|---:|---|
| Valid repeats | 3 | 3 | 相同 | 可進行配對層級比較 |
| Requests | 72 | 100 | vLLM +38.9% | 固定時間內完成更多 requests |
| Output tok/s | 9.88 | 14.06 | vLLM 1.42× | vLLM 較高 |
| Req/s | 0.08 | 0.11 | vLLM 約 1.38× | vLLM 較高 |
| P95 TTFT | 12,485.06 ms | 8,644.14 ms | vLLM 低 30.8% | vLLM 較快，但仍超過 SLO |
| P95 ITL | 6.79 ms | 7.35 ms | llama.cpp 低 7.6% | 兩者均遠低於 75 ms |
| Error rate | 0% | 0% | 相同 | 請求可靠完成 |
| Good % | 0% | 0% | 相同 | 兩者均未通過 TTFT ≤ 5 秒 |
| P99 eligibility | 否 | 否 | 相同 | Requests 均不足 1,000 |

### 6.4 無效邊界

| Backend | Cell | Repeats | 觀察 | 結論 |
|---|---|---:|---|---|
| llama.cpp | HotpotQA 16K C8 | 5/5 invalid | 每次約 90 requests；error rate 62.5%；核心指標無法形成有效彙總 | 不可用邊界，需查 server error 類型 |
| llama.cpp | RULER 60K C2 | 5/5 invalid | 部分 exit code 1；其餘 error rate 約 98.39% | 不可用邊界，不得計算性能倍數 |

## 七、舊版 synthetic/AIPerf 基線

### 7.1 Concurrency scaling（2,048 input / 128 output）

| C | llama decode tok/s | vLLM decode tok/s | vLLM/llama | llama TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 207.91 | 189.53 | 0.91× | 274.98 ms | 135.51 ms |
| 2 | 260.32 | 291.98 | 1.12× | 630.75 ms | 250.61 ms |
| 4 | 342.53 | 470.86 | 1.37× | 1,988.46 ms | 478.44 ms |
| 8 | 309.30 | 573.25 | 1.85× | 4,262.51 ms | 826.17 ms |
| 16 | 279.51 | 694.23 | 2.48× | 7,766.79 ms | 1,356.00 ms |

### 7.2 Context scaling（C1）

| Context | llama prefill tok/s | vLLM prefill tok/s | llama TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|
| 0 | 7,459.66 | 19,898.65 | 300.92 ms | 134.92 ms |
| 4K | 8,173.13 | 17,264.33 | 779.09 ms | 390.06 ms |
| 16K | 7,836.46 | 13,084.50 | 2,380.28 ms | 1,441.02 ms |
| 32K | 7,202.86 | 9,950.90 | 4,863.98 ms | 3,527.64 ms |

### 7.3 Prefix cache

| Context | llama off → on TTFT | llama 降幅 | vLLM off → on TTFT | vLLM 降幅 |
|---:|---:|---:|---:|---:|
| 4K | 601.73 → 101.16 ms | 83.19% | 296.84 → 69.22 ms | 76.68% |
| 16K | 2,156.43 → 136.70 ms | 93.66% | 1,297.74 → 99.22 ms | 92.35% |

### 7.4 SLO attainment

| Workload | llama.cpp | vLLM | 舊版判讀 |
|---|---|---|---|
| Interactive C1–16 | 100% / 1% / 0% / 0% / 0% | 全部 100% | vLLM 高併發明顯較穩定 |
| RAG C1/2/4/8 | 100% / 82% / 43% / 21% | 100% / 100% / 100% / 99% | vLLM 維持較高 SLO |
| Long-context C1/2 | 100% / 90% | 100% / 100% | vLLM 較穩定 |
| Request errors | 全部 0% | 全部 0% | 差異主要來自延遲，而非 HTTP failure |

## 八、結論與待辦

| 類別 | 結論／待辦 |
|---|---|
| 已確認 | 資料集與框架具可重現性；19 檔與 14/14 tests 通過 |
| ShareGPT | llama.cpp C1 通過；C16 throughput 增加，但 Good % 降至 88.78% |
| HotpotQA | 用於多跳 RAG；llama.cpp 16K C8 pilot 已進入高錯誤率邊界 |
| 60K context | vLLM throughput 約為 llama.cpp 1.42×，TTFT 低 30.8%；兩者仍皆未通過 5 秒 SLO |
| 舊版趨勢 | vLLM 適合高併發；llama.cpp 單流 decode 有競爭力 |
| 尚缺 | 兩後端各 45 個 comparison cells |
| 尚缺 | 34 個 vLLM stress cells |
| 尚缺 | Prefix cache off/on 正式矩陣與 cached-token reporting |
| 尚缺 | 每後端 5,200 筆 RULER accuracy |
| 尚缺 | Invalid cells 的 server error 根因分類 |
| 最終發布門檻 | ≥3 valid repeats、error ≤1%、Good Request Fraction ≥95% |

## 九、主要檔案

| 檔案／目錄 | 用途 |
|---|---|
| `scripts/dataset-tests/benchmark-config.json` | 正式參數來源 |
| `scripts/dataset-tests/benchmark_matrix.py` | 展開 79 個 conditions |
| `scripts/dataset-tests/run_benchmarks.py` | 執行與有效性判定 |
| `scripts/dataset-tests/validate_prepared.py` | Schema、token、sidecar、SHA256 驗證 |
| `datasets/manifests/` | 資料來源、revision、rows、hash |
| `results/*/dataset/` | 新版真實資料集結果 |
| `analysis/output/benchmark_analysis.md` | 舊版 synthetic/AIPerf 分析 |

> 報告時必須分開「資料已驗證」、「cell 有效」、「cell 通過 SLO」與「完整公平矩陣已完成」四種狀態。
