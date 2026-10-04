# LLM 推論測試 Python 分析

本報告由 `analysis/analyze_benchmarks.py` 直接解析 `results/` 產生。AIPerf 主分析採用每個工作負載最新且具 3 次 profile runs 的 aggregate 批次；較早的單次批次不混入平均。

## 結論摘要

- **整體併發與長提示吞吐量：vLLM 明顯領先。** 固定 2,048-token prompt 的 concurrency 測試中，並發 16 的總 decode throughput 為 694.2 vs 279.5 tokens/s（2.48×）；TTFT 為 1,356.0 vs 7,766.8 ms。
- **單使用者 decode：llama.cpp 略快或相近。** concurrency=1 時 llama.cpp 為 207.9 tokens/s，vLLM 為 189.5 tokens/s；但 vLLM 的 prefill 與 TTFT 快很多。
- **AIPerf interactive 是差距最大的項目。** llama.cpp 只有 concurrency=1 達成 100% SLO；concurrency=2 已降至約 1%，更高併發接近 0%。vLLM 在 concurrency=1–16 全部為 100%。
- **RAG：vLLM 在所有測試併發維持約 99–100% SLO。** llama.cpp 由 concurrency=1 的 100% 降至 concurrency=8 的約 21%。
- **Long-context：兩者 concurrency=1 都穩定，vLLM 延遲約為 llama.cpp 的 57%。** concurrency=2 時 vLLM 仍為 100% SLO，llama.cpp 約 90%。
- **所有正式 AIPerf cell 的 request error rate 都是 0%。** llama.cpp 高併發的問題是延遲超過 SLO，不是 HTTP／請求失敗。
- **llama.cpp 啟動資料只有一組有效結果。** 冷啟動 12,585 ms，後兩次暖啟動平均 3,478 ms（縮短 72.4%）；vLLM 沒有可用的同類 startup summary。

## 1. Startup

| attempt | load_time_ms | ready_time_ms | idle_vram_mib | status |
| --- | --- | --- | --- | --- |
| 1 | 12585 | 12585 | 0 | success |
| 2 | 3564 | 3564 | 0 | success |
| 3 | 3393 | 3393 | 0 | success |

`idle_vram_mib` 全為 0，較可能是遙測未正確取得，而非模型完全不占 VRAM。另一個 llama.cpp startup 批次只有 CSV 標頭；vLLM shared-suite 則有空白 status 與一次 FlashInfer/NVCC 啟動失敗，因此無法做公平的啟動時間比較。

## 2. Context scaling（單請求，2,048 input / 128 output tokens）

| context | llama prefill tok/s | vLLM prefill tok/s | vLLM/llama prefill | llama decode tok/s | vLLM decode tok/s | llama TTFT ms | vLLM TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7459.66 | 19898.65 | 2.67 | 212.60 | 189.49 | 300.92 | 134.92 |
| 4096 | 8173.13 | 17264.33 | 2.11 | 204.14 | 183.05 | 779.09 | 390.06 |
| 16384 | 7836.46 | 13084.50 | 1.67 | 189.55 | 168.03 | 2380.28 | 1441.02 |
| 32768 | 7202.86 | 9950.90 | 1.38 | 171.86 | 152.25 | 4863.98 | 3527.64 |

在 32K 既有 context 下，vLLM prefill 為 llama.cpp 的 1.38×，TTFT 低 27.5%。llama.cpp 的 decode token rate 全程稍高，但其 TTFT 與 prefill 較弱。

## 3. Concurrency scaling（2,048 input / 128 output tokens）

| concurrency | llama total decode | vLLM total decode | vLLM/llama decode | llama per-request | vLLM per-request | llama TTFT ms | vLLM TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 207.91 | 189.53 | 0.91 | 207.91 | 189.53 | 274.98 | 135.51 |
| 2 | 260.32 | 291.98 | 1.12 | 131.14 | 147.97 | 630.75 | 250.61 |
| 4 | 342.53 | 470.86 | 1.37 | 87.64 | 121.12 | 1988.46 | 478.44 |
| 8 | 309.30 | 573.25 | 1.85 | 50.14 | 93.28 | 4262.51 | 826.17 |
| 16 | 279.51 | 694.23 | 2.48 | 29.95 | 67.57 | 7766.79 | 1356.00 |

llama.cpp 的總 decode throughput 在 concurrency=4 達峰值，之後下降；vLLM 則持續擴展到 concurrency=16。這代表 vLLM 更適合多使用者服務，而 llama.cpp 的優勢偏向低併發單流。

## 4. Prefix cache

| context | llama cache-off TTFT | llama cache-on TTFT | llama reduction % | vLLM cache-off TTFT | vLLM cache-on TTFT | vLLM reduction % |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 601.73 | 101.16 | 83.19 | 296.84 | 69.22 | 76.68 |
| 16384 | 2156.43 | 136.70 | 93.66 | 1297.74 | 99.22 | 92.35 |

兩個後端的 prefix cache 都非常有效：4K context 可省約 77–83% TTFT，16K 可省約 92–94%。vLLM 的絕對 TTFT 較低；llama.cpp 的相對減幅略高。

## 5. AIPerf 正式三次重跑

SLO 判定來自各 run config：interactive 為 TTFT ≤ 1,000 ms 且 ITL ≤ 50 ms；RAG 為 TTFT ≤ 2,000 ms 且 ITL ≤ 50 ms；long-context 為 TTFT ≤ 5,000 ms 且 ITL ≤ 75 ms。`GoodRequestFraction` 是同時通過條件的請求比例。

### interactive

| backend | concurrency | actual prompt tokens | P95 TTFT ms | P95 ITL ms | P95 request ms | output tok/s | req/s | SLO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-cpp | 1 | 512.00 | 69.96 | 4.92 | 692.51 | 195.06 | 1.52 | 100.00 |
| llama-cpp | 2 | 512.00 | 2338.87 | 7.85 | 3326.32 | 88.32 | 0.69 | 1.00 |
| llama-cpp | 4 | 512.00 | 4472.89 | 11.28 | 5889.88 | 94.22 | 0.74 | 0.00 |
| llama-cpp | 8 | 512.00 | 11948.09 | 17.65 | 14151.98 | 90.04 | 0.70 | 0.00 |
| llama-cpp | 16 | 512.00 | 23923.43 | 25.57 | 27095.43 | 81.15 | 0.63 | 0.00 |
| vllm | 1 | 579.85 | 64.43 | 5.02 | 701.22 | 182.29 | 1.42 | 100.00 |
| vllm | 2 | 579.89 | 105.05 | 6.42 | 919.47 | 281.79 | 2.20 | 100.00 |
| vllm | 4 | 579.92 | 164.77 | 7.70 | 1138.54 | 457.17 | 3.57 | 100.00 |
| vllm | 8 | 579.85 | 281.05 | 9.34 | 1436.07 | 723.28 | 5.65 | 100.00 |
| vllm | 16 | 579.90 | 489.00 | 12.17 | 1847.55 | 1119.24 | 8.74 | 100.00 |

### rag

| backend | concurrency | actual prompt tokens | P95 TTFT ms | P95 ITL ms | P95 request ms | output tok/s | req/s | SLO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-cpp | 1 | 4096.88 | 766.38 | 6.15 | 2346.63 | 113.14 | 0.44 | 100.00 |
| llama-cpp | 2 | 4096.88 | 2072.30 | 12.07 | 4692.97 | 117.26 | 0.46 | 82.00 |
| llama-cpp | 4 | 4096.81 | 4193.20 | 24.32 | 8107.19 | 139.29 | 0.55 | 43.00 |
| llama-cpp | 8 | 4096.98 | 8240.26 | 47.55 | 16345.70 | 153.18 | 0.60 | 21.00 |
| vllm | 1 | 4163.83 | 276.13 | 5.26 | 1616.84 | 158.20 | 0.62 | 100.00 |
| vllm | 2 | 4163.98 | 509.59 | 6.73 | 2229.12 | 231.99 | 0.91 | 100.00 |
| vllm | 4 | 4163.92 | 929.82 | 9.87 | 3080.87 | 338.00 | 1.32 | 100.00 |
| vllm | 8 | 4163.84 | 950.28 | 15.14 | 4496.91 | 460.42 | 1.80 | 99.00 |

### long-context

| backend | concurrency | actual prompt tokens | P95 TTFT ms | P95 ITL ms | P95 request ms | output tok/s | req/s | SLO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-cpp | 1 | 16385.04 | 2786.79 | 6.16 | 3556.85 | 36.59 | 0.29 | 100.00 |
| llama-cpp | 2 | 16384.89 | 5112.16 | 15.83 | 7044.55 | 39.14 | 0.31 | 90.00 |
| vllm | 1 | 16452.05 | 1288.13 | 5.69 | 2010.50 | 63.68 | 0.50 | 100.00 |
| vllm | 2 | 16451.97 | 2009.33 | 12.86 | 3430.49 | 74.60 | 0.58 | 100.00 |

## 6. 可比性與資料品質限制

- llama.cpp 使用 `gpt-oss-20b-MXFP4.gguf`，vLLM 使用 Hugging Face snapshot；雖為同一模型家族，但模型封裝、kernel、tokenizer/chat template 與服務器參數並非完全相同，因此結果代表這兩套實際部署，而不是隔離其他變因的純引擎比較。
- AIPerf 實際 prompt token 數不同：interactive 約為 llama.cpp 512、vLLM 580；RAG 與 long-context 的差距較小。vLLM interactive/RAG 另啟用了 `first_turn_prefix` cache bust，而 llama.cpp interactive 沒有，因此不是完全相同的輸入路徑；報表已列出實際 token 數。
- AIPerf aggregate 為 3 次 profile run 平均，可信度高於早期單次批次；startup、context、concurrency、prefix-cache 的重複次數較少，解讀細小差異時應保守。
- vLLM metadata 無法取得版本（容器內沒有 `python` 可執行檔），所以版本只能從腳本說明推測，不應當作已驗證事實。
- `results/generated-summary.md` 的 4 successful cells 與 `??` 是舊發佈摘要，未涵蓋這次直接解析的全部測試。

## 7. 部署建議

- **多使用者、RAG 或高併發 API：選 vLLM。** 其 throughput 隨併發增加，且 SLO 保持穩定。
- **單使用者、低併發、本機簡化部署：llama.cpp 仍合理。** 單流 decode 稍快，且 prefix cache 表現很好。
- llama.cpp 若要提供互動式多人服務，應先重新檢查 `--parallel 16`、slot/KV cache、continuous batching 與批次參數，再用完全一致的 cache-bust 與實際 prompt 長度重跑 AIPerf；目前 concurrency≥2 的 TTFT 排隊效應過重。
- 修正 vLLM 映像中的版本擷取命令（可改用存在的 Python 路徑或套件 metadata），並保留有效 startup/status 資料，以補齊可重現性。
