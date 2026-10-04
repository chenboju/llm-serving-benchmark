# GPT-OSS-20B 推論基準完整報告

報告日期：2026-08-14  
比較對象：llama.cpp 與 vLLM  
硬體：NVIDIA GeForce RTX 4090 24 GB（24564 MiB）  
正式資料集計畫：`gpt-oss-20b-dataset-v1`

## 1. 報告目的與結果狀態

本報告整理四個層面：資料集、測試矩陣、執行與判定規範、現有測試結果。結果分成兩代，不能混為同一批公平比較：

1. **新版真實資料集套件**：使用 ShareGPT、SQuAD、HotpotQA、RULER，為目前正式規格。資料與測試框架已驗證完成，但完整矩陣尚未跑完；目前只有少數 llama.cpp pilot 與 vLLM 60K pilot。
2. **舊版 synthetic/AIPerf 基線**：2026-08-10 至 2026-08-11 已完成，可用來理解系統特性，但不是新版資料集矩陣的正式結果。

因此，目前可以確認趨勢與少數邊界條件，但**還不能用新版資料集宣告 llama.cpp 與 vLLM 的最終勝負**。

## 2. 執行環境

| 項目 | llama.cpp | vLLM |
|---|---|---|
| 模型 | GPT-OSS-20B MXFP4 GGUF | GPT-OSS-20B Hugging Face snapshot |
| 模型識別 | `openai/gpt-oss-20b` | `gpt-oss-20b` |
| Context 上限 | 65,536 tokens | 65,536 tokens |
| 最大測試 client concurrency | 16 | 64 |
| Server 排程上限 | parallel slots 16 | `max-num-seqs=16` |
| Prefix cache（一般測試） | 關閉 | 關閉 |
| Prefix cache（專項測試） | 依 cell 明確切換 | 依 cell 明確切換 |

共用環境如下：

- GPU：NVIDIA GeForce RTX 4090，24564 MiB。
- Driver：595.79；環境紀錄顯示 CUDA 13.2。
- AIPerf：0.12.0，commit `03c9c6ddc5e6227782e53ded177f1227d332af48`。
- llama.cpp commit：`312cf03328496df9c56fe843952cf02a8964c59a`。
- llama-benchy：0.4.1.dev1，commit `e9be344578cec17745066b220798b80a0d2686d3`。
- Tokenizer：同一個 GPT-OSS-20B snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`。
- 固定 random seed：42。
- 生成溫度：0。
- API：OpenAI-compatible chat endpoint、streaming、使用 server 回報 token count。

### 2.1 公平性的邊界

這是「實際部署方案比較」，不是只隔離 serving engine 的純微基準。llama.cpp 使用 MXFP4 GGUF，vLLM 使用 Hugging Face 模型封裝，kernel、量化路徑與服務器實作不同。報告結論應寫成「這兩個部署方案在同一硬體與工作負載下的差異」，不能寫成所有 llama.cpp 或所有 vLLM 的普遍結論。

另外，vLLM 的 client concurrency 可到 64，但 server `max-num-seqs=16`。C32/C64 主要測量排隊、continuous batching 與壓力承受能力，不代表 32 或 64 條序列同時在 GPU 執行。

## 3. 資料集

### 3.1 資料集總表

| 資料集 | 用途 | 長度／模式 | 每檔樣本 | 輸出設定 |
|---|---|---|---:|---|
| ShareGPT fixed | closed/open-loop chat | 多輪，最多 4 turns | 5,000 sessions | 固定 128 tokens |
| ShareGPT natural | 真實聊天輸出分布 | 多輪，最多 4 turns | 5,000 sessions | 使用資料中記錄的 OSL，不強制最少 128 |
| SQuAD | RAG | 1K/4K/8K/16K | 每長度 250 | 固定 256 tokens |
| HotpotQA | 多跳 RAG | 1K/4K/8K/16K | 每長度 250 | 固定 256 tokens |
| SQuAD prefix | Prefix cache | 4K/16K、cache off/on | 每檔 400 | 固定 128 tokens |
| RULER performance | 長上下文效能 | 4K/8K/16K/32K | 每長度 1,300 | 固定 128 tokens |
| RULER performance 60K | 長上下文邊界 | 60K | 260 | 固定 128 tokens |
| RULER accuracy | 長上下文正確率 | 4K/8K/16K/32K | 13 tasks × 100 | natural EOS |

正式效能測試採「固定時間」，不是每格完整跑完資料檔。資料檔是可重現的取樣池；300 秒內完成多少 request 取決於後端速度，必要時 sampler 會循環使用資料。

### 3.2 來源與授權紀錄

| 資料集 | 來源／revision | 授權註記 |
|---|---|---|
| ShareGPT | `ShareGPT_V3_unfiltered_cleaned_split.json`，下載後以 SHA256 固定 | 重新散布前需再確認 upstream terms |
| SQuAD | train-v1.1 | CC BY-SA 4.0 |
| HotpotQA | `hotpotqa/hotpot_qa`，distractor/train | CC BY-SA 4.0 |
| RULER | generator `c3f5e3b4...`；rulerv1-ns `e8bbff67...`；NeMo-Skills `f4a3fd8e...` | Apache-2.0 |

### 3.3 資料準備與完整性規範

每個正式 JSONL 都有：

- 對應 `.index.jsonl` sidecar，保存 sample ID、實際 token 數、bucket、prefix group 與 cold/warm role。
- manifest 中的資料來源、tokenizer、row count、建立時間與 SHA256。
- 使用 GPT-OSS-20B 正式 tokenizer 建立，不接受 approximate tokenizer 作為正式測試。
- RAG/prefix 的實際輸入 token 必須在目標 bucket 的 ±5% 內。
- Prefix 每組固定 4 個問題，順序為 0–3，必須正好包含 1 個 cold 與 3 個 warm request。
- 多輪 ShareGPT 每筆必須有 `session_id` 且包含 1–4 turns。

2026-08-14 的本地驗證結果：**19 個資料檔全部通過** schema、sidecar、SHA256、token bucket 與 prefix group 驗證；共覆蓋 79 個測試條件。離線框架測試 **14/14 通過**。

## 4. 測試矩陣

### 4.1 公平比較與額外壓力

| 測試 | llama.cpp 公平比較 | vLLM 公平比較 | vLLM 額外壓力 |
|---|---|---|---|
| ShareGPT closed-loop | C1/2/4/8/16 | C1/2/4/8/16 | C32/64 |
| ShareGPT open-loop | 0.5/1/2/4/8 req/s，最大 C16 | 相同 | 16/32 req/s，最大 C64 |
| ShareGPT natural | 4 req/s，C16 | 相同 | 8/16 req/s，C64 |
| RAG | SQuAD/HotpotQA × 1K/4K/8K/16K × C1/8 | 相同 | 每組 C16/32 |
| Prefix cache | 4K/16K × off/on × C1/8 | 相同 | 每組 C16 |
| RULER performance | 4K/8K/16K/32K/60K × C1/2 | 相同 | 4K:C8/16/32；8K:C8/16；16K:C4/8；32K:C4 |

矩陣共有：

- 公平條件 45 個；每個後端各跑一次完整矩陣，合計 90 backend-cells。
- vLLM-only 壓力條件 34 個。
- 全部合計 124 backend-cells。
- RULER accuracy 另行執行，不計入上述 AIPerf 79 conditions。

### 4.2 各測試要回答的問題

#### ShareGPT closed-loop

維持固定數量的進行中 session；一個完成就補下一個。用來測量飽和前後的吞吐擴展、TTFT、ITL、每個使用者的速度與排隊效應。

#### ShareGPT open-loop

請求以 Poisson arrival 依固定 req/s 進入，不等待上一個完成。用來找最大可持續 arrival rate、排隊開始惡化的位置與 P95/P99 尾延遲。

#### ShareGPT natural

保留 ShareGPT 記錄的輸出長度分布，不使用固定 128-token 強制輸出。此結果更接近自然聊天，但比較時必須同時查看 request throughput 與 output-token throughput，避免短回答讓 req/s 看起來虛高。

#### RAG

使用 SQuAD 與 HotpotQA，將 context 打包到 1K、4K、8K、16K。主要觀察長 prompt 的 prefill、TTFT、顯存壓力、單跳與多跳問答負載差異。

#### Prefix cache

以 100 組共享 context、每組 4 個問題形成 cold/warm 序列。比較 cache off/on 的 TTFT、prompt processing、throughput 與實際 cached-token 指標。

#### RULER performance 與 accuracy

Performance 測試固定輸出 128 tokens，觀察 4K 到 60K 的 prefill、TTFT、吞吐、錯誤與 OOM。Accuracy 測試則使用 natural EOS，以官方 13 個任務、每任務 100 筆評估長上下文能力。兩者用途不同，效能結果不能代替正確率。

## 5. 執行規範

### 5.1 每個 cell 的流程

1. 執行 10 個 smoke requests，確認 endpoint 與 model ID。
2. 等待 GPU utilization 低於 5%，並穩定 30 秒；最多等 900 秒。
3. AIPerf warm-up 30 秒。
4. 正式 profiling 300 秒。
5. 每個 cell 執行 3 repeats；repeat 間 cooldown 30 秒。
6. 若 output throughput CV > 10% 或 P95 TTFT CV > 15%，增加到 5 repeats。
7. 保存 command、console/log、records、outputs、GPU telemetry、repeat metrics 與 validation JSON。

純 profiling 時數約為：公平比較 `45 × 2 × 3 × 300 秒 = 22.5 GPU 小時`；vLLM 壓力測試 `34 × 3 × 300 秒 = 8.5 GPU 小時`，合計至少 31 GPU 小時。加入 warm-up、cooldown、server restart、GPU idle wait 與不穩定重跑後，實際時間會更長。

### 5.2 Cell 有效性

單次 repeat 同時符合下列條件才算 valid：

- 程式 exit code 為 0。
- 未偵測到 GPU OOM。
- request error rate ≤ 1%。

至少 3 個 valid repeats，cell 才能形成正式彙總。注意：**valid 不等於通過 SLO**。例如沒有錯誤但 TTFT 超過門檻，cell 仍可 valid，但 Good Request Fraction 可能是 0%。

### 5.3 SLO

| Profile | TTFT 上限 | ITL 上限 | 適用測試 |
|---|---:|---:|---|
| Chat | 1,000 ms | 50 ms | ShareGPT |
| RAG | 2,000 ms | 50 ms | RAG、Prefix |
| Long-context | 5,000 ms | 75 ms | RULER performance |

一個 good request 必須同時通過 TTFT 與 ITL。建議容量判定使用 `Good Request Fraction ≥ 95%`，並同時要求 error rate ≤ 1%。

### 5.4 統計與報告規範

- 主要報告 P50/P95；P99 只在 pooled valid requests ≥ 1,000 時正式解讀。
- 每個 cell 報告 repeats 數、總 request count、CV、error rate、Good Request Fraction。
- open-loop 另外報告目標 arrival rate、實際送出率、完成率與排隊延遲。
- throughput 至少同時報告 output tokens/s、requests/s、per-user tokens/s。
- RAG/長上下文必須報告實際 input tokens，而非只寫目標 bucket。
- OOM、HTTP error、timeout、SLO miss 分開統計；不得把延遲超標寫成請求失敗。
- 公平比較與 vLLM-only 壓力結果分開，不以 vLLM C32/C64 對照 llama.cpp C16 宣稱引擎倍數。

## 6. 新版真實資料集：目前 pilot 結果

### 6.1 完成狀態

截至本報告建立時：

- llama.cpp 有 3 個有效正式 pilot cells：ShareGPT closed C1、C16、RULER 60K C1。
- llama.cpp 有 2 個無效邊界 cells：HotpotQA 16K C8、RULER 60K C2。
- vLLM RULER 60K C1 已完成 3 個有效且穩定的 repeats。
- dry-run 只驗證命令，不計入效能結果。
- 完整 45 × 2 公平矩陣及 34 個 vLLM stress cells 尚未完成。

### 6.2 llama.cpp 有效 pilot

以下數字為各 valid repeats 的平均；Good % 以所有 valid repeats 的 good requests／requests 計算。

| Cell | Repeats | Requests | Output tok/s | Req/s | P95 TTFT | P95 ITL | Error | Good % | 判讀 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ShareGPT closed C1 | 3 | 1,246 | 172.97 | 1.38 | 189.95 ms | 5.31 ms | 0% | 100% | 通過 Chat SLO |
| ShareGPT closed C16 | 3 | 2,646 | 365.17 | 2.92 | 1,223.39 ms | 47.42 ms | 0% | 88.78% | 未達 95% SLO；主要受 TTFT 影響 |
| RULER 60K C1 | 3 | 72 | 9.88 | 0.08 | 12,485.06 ms | 6.79 ms | 0% | 0% | repeat 有效，但完全未通過 5 秒 TTFT SLO |

解讀：

- ShareGPT 從 C1 增至 C16，總 output throughput 約增加 2.11 倍，但 P95 TTFT 升至約 1.22 秒，Good Request Fraction 降到 88.78%。
- RULER 60K C1 的 decode/ITL 並不差，瓶頸是約 60K prompt 的 prefill/TTFT；平均實際輸入約 59,959 tokens。
- RULER 60K 三次只累積 72 requests，因此不符合 P99 最少 1,000 samples 的門檻。

### 6.3 llama.cpp 無效邊界 pilot

| Cell | 結果 | 原因 |
|---|---|---|
| HotpotQA 16K C8 | 5/5 repeats 無效 | 每次約 90 requests，request error rate 62.5%，核心延遲／吞吐欄位無法形成有效彙總 |
| RULER 60K C2 | 5/5 repeats 無效 | 部分 repeat exit code 1；其餘幾乎所有 request 失敗，約 98.39% error rate |

這兩格沒有偵測到明確 OOM 字樣，但仍屬服務不可用邊界。正式報告應列為 invalid/error boundary，不能用 NaN 吞吐或單一成功 request 計算性能倍數。

### 6.4 vLLM RULER 60K C1

三次有效 repeat 如下：

| Repeat | Requests | Output tok/s | Req/s | P95 TTFT | P95 ITL | Error | Good requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 14.09 | 0.11 | 8,606.59 ms | 7.34 ms | 0% | 0 |
| 2 | 33 | 14.06 | 0.11 | 8,653.69 ms | 7.35 ms | 0% | 0 |
| 3 | 33 | 14.04 | 0.11 | 8,672.15 ms | 7.36 ms | 0% | 0 |

三次平均顯示 vLLM 在 60K C1 約為 llama.cpp 的 1.42 倍 output throughput，P95 TTFT 約低 30.8%；但兩者都超過 5 秒 long-context TTFT SLO，因此兩者 Good Request Fraction 都是 0%。兩邊 pooled requests 都少於 1,000，P99 不具正式解讀資格。

## 7. 舊版 synthetic/AIPerf 基線結果

本節來自 2026-08-10/11 已完成的舊批次。用途是補充系統特性，不可與第 6 節直接合併平均。

### 7.1 Concurrency scaling（2,048 input / 128 output）

| C | llama total decode tok/s | vLLM total decode tok/s | vLLM/llama | llama TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 207.91 | 189.53 | 0.91× | 274.98 ms | 135.51 ms |
| 2 | 260.32 | 291.98 | 1.12× | 630.75 ms | 250.61 ms |
| 4 | 342.53 | 470.86 | 1.37× | 1,988.46 ms | 478.44 ms |
| 8 | 309.30 | 573.25 | 1.85× | 4,262.51 ms | 826.17 ms |
| 16 | 279.51 | 694.23 | 2.48× | 7,766.79 ms | 1,356.00 ms |

舊基線顯示 llama.cpp 單流 decode 略快，但總吞吐在 C4 後下降；vLLM 可持續擴展到 C16。

### 7.2 Context scaling（C1，2,048 新輸入 / 128 輸出）

| 既有 context | llama prefill tok/s | vLLM prefill tok/s | llama TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|
| 0 | 7,459.66 | 19,898.65 | 300.92 ms | 134.92 ms |
| 4K | 8,173.13 | 17,264.33 | 779.09 ms | 390.06 ms |
| 16K | 7,836.46 | 13,084.50 | 2,380.28 ms | 1,441.02 ms |
| 32K | 7,202.86 | 9,950.90 | 4,863.98 ms | 3,527.64 ms |

### 7.3 Prefix cache 基線

| Context | llama off → on TTFT | llama 降幅 | vLLM off → on TTFT | vLLM 降幅 |
|---:|---:|---:|---:|---:|
| 4K | 601.73 → 101.16 ms | 83.19% | 296.84 → 69.22 ms | 76.68% |
| 16K | 2,156.43 → 136.70 ms | 93.66% | 1,297.74 → 99.22 ms | 92.35% |

兩個後端都能從共享 prefix 獲得顯著 TTFT 改善；新版 prefix suite 仍需用真實分組資料重新確認。

### 7.4 舊版 AIPerf SLO 摘要

| Workload | llama.cpp | vLLM |
|---|---|---|
| Interactive C1–16 | C1 100%；C2 約 1%；C4–16 約 0% | C1–16 均 100% |
| RAG C1/2/4/8 | 100% / 82% / 43% / 21% | 100% / 100% / 100% / 99% |
| Long-context C1/2 | 100% / 90% | 100% / 100% |
| Request error rate | 所有正式 cells 0% | 所有正式 cells 0% |

舊版的高併發差距主要是 TTFT/SLO，而不是 HTTP 錯誤。但舊版 interactive 的實際 prompt tokens、cache-bust 路徑與兩個後端並不完全一致，因此只能作基線，不能取代新版公平矩陣。

## 8. 目前可下的結論

1. **新版測試框架與資料已具可重現性。** 19 個資料檔和 14 個離線測試均通過，矩陣可穩定展開為 79 conditions。
2. **llama.cpp 適合低併發，但邊界在真實資料上已出現。** ShareGPT C1 完整通過 SLO；C16 throughput 提高但 Good % 降到 88.78%。
3. **60K context 的主要問題是 TTFT。** llama.cpp 與 vLLM 都有低 ITL、零錯誤，但因 prefill 太久而 Good % 為 0%；vLLM 的 throughput 約為 1.42 倍，P95 TTFT 約低 30.8%。
4. **llama.cpp 的 HotpotQA 16K C8 與 RULER 60K C2 目前不可用。** 這是錯誤率邊界，不只是延遲惡化。
5. **舊基線持續支持 vLLM 的高併發優勢。** 但正式部署建議仍應等新版 45 個公平 cells 在兩個後端都完成後再定案。

## 9. 尚缺結果與完成條件

要把本報告升級成最終報告，必須完成：

- llama.cpp 與 vLLM 各 45 個 comparison cells。
- vLLM 34 個 stress cells。
- Prefix cache off/on 的獨立 server 模式與 cached-token reporting 驗證。
- RULER 4K/8K/16K/32K accuracy：每後端 5,200 samples。
- 對 invalid cells 保存錯誤類型分布與 server log 根因，不只記錄 error rate。
- 使用同一份 analyzer 產生 paired comparison、95% CI、SLO heatmap、capacity frontier 與能耗比較。

最終結論應以三個門檻共同判定：

1. cell 至少 3 個 valid repeats；
2. request error rate ≤ 1%；
3. Good Request Fraction ≥ 95%。

若 pooled requests < 1,000，P99 應標記為 `N/A`，不能用少量樣本宣稱尾延遲。

## 10. 可重現命令與主要檔案

```bash
# 展開 79 個測試條件
python scripts/dataset-tests/benchmark_matrix.py

# 驗證所有準備後資料
python scripts/dataset-tests/validate_prepared.py

# 離線 regression tests
cd scripts/dataset-tests
python -m unittest test_dataset_suite.py

# llama.cpp 公平矩陣
python scripts/dataset-tests/run_benchmarks.py \
  --backend llama-cpp --scope comparison --manage-llama-server

# vLLM 公平矩陣
python scripts/dataset-tests/run_benchmarks.py \
  --backend vllm --scope comparison

# vLLM 額外壓力矩陣
python scripts/dataset-tests/run_benchmarks.py \
  --backend vllm --scope vllm_stress
```

主要依據：

- `scripts/dataset-tests/benchmark-config.json`：唯一正式測試參數來源。
- `scripts/dataset-tests/benchmark_matrix.py`：矩陣展開與 scope 驗證。
- `scripts/dataset-tests/run_benchmarks.py`：執行、重跑、錯誤與穩定性判定。
- `scripts/dataset-tests/validate_prepared.py`：資料 schema、token、sidecar 與 SHA256 驗證。
- `datasets/manifests/`：資料來源、revision、row count 與 hash。
- `results/*/dataset/`：新版 pilot 與未來正式結果。
- `analysis/output/benchmark_analysis.md`：舊版 synthetic/AIPerf 分析。

---

本報告的核心原則是：把「資料已準備」、「cell 有效」、「通過 SLO」與「完整公平比較已完成」視為四個不同狀態。現況是資料與框架已就緒、少數 pilot 已完成，但完整公平比較仍在進行中。
