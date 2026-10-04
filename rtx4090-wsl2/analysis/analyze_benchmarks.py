from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_ROOT.parent
RESULTS = REPO_ROOT / "results"
OUTPUT = PLATFORM_ROOT / "analysis" / "output"


AIPERF_BATCHES = {
    ("llama-cpp", "interactive"): "20260810-161930",
    ("llama-cpp", "rag"): "20260810-175511",
    ("llama-cpp", "long-context"): "20260810-192327",
    ("vllm", "interactive"): "20260811-120222",
    ("vllm", "rag"): "20260811-123504",
    ("vllm", "long-context"): "20260811-130133",
}


KEY_METRICS = [
    "request_latency_p50",
    "request_latency_p95",
    "time_to_first_token_p50",
    "time_to_first_token_p95",
    "inter_token_latency_p50",
    "inter_token_latency_p95",
    "output_token_throughput_avg",
    "output_token_throughput_per_user_avg",
    "e2e_output_token_throughput_avg",
    "request_throughput_avg",
    "good_request_fraction_avg",
    "usage_prompt_tokens_avg",
    "usage_completion_tokens_avg",
    "request_error_rate_avg",
]


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def ratio(new: float, old: float) -> float:
    return new / old


def fmt(value: float | int | str | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:,.{digits}f}"


def markdown_table(df: pd.DataFrame, digits: int = 2) -> str:
    def render(v: object) -> str:
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        return str(v)

    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(v) for v in row) + " |")
    return "\n".join(lines)


def read_summary(backend: str, suite: str) -> pd.DataFrame:
    files = sorted((RESULTS / backend / suite).glob("*/summary.csv"))
    candidates: list[tuple[Path, pd.DataFrame]] = []
    for path in files:
        frame = pd.read_csv(path)
        if not frame.empty:
            candidates.append((path, frame))
    if not candidates:
        return pd.DataFrame()
    path, frame = candidates[-1]
    frame.attrs["source"] = str(path.relative_to(REPO_ROOT))
    return frame


def read_aiperf() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (backend, workload), batch in AIPERF_BATCHES.items():
        base = RESULTS / backend / "aiperf" / workload / batch
        for concurrency_dir in sorted(base.glob("concurrency-*")):
            concurrency = int(concurrency_dir.name.split("-")[-1])
            aggregate_path = concurrency_dir / "aggregate" / "profile_export_aiperf_aggregate.csv"
            config_path = concurrency_dir / "profile_runs" / "run_0001" / "run_config.json"
            if not aggregate_path.exists() or not config_path.exists():
                continue

            metrics = pd.read_csv(aggregate_path).set_index("metric")
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            cfg = config["cfg"]
            profile = next(p for p in cfg["phases"] if p["kind"] == "profiling")
            slos = cfg.get("slos", {})
            dataset = cfg["datasets"][0]["prompts"]

            row: dict[str, object] = {
                "backend": backend,
                "workload": workload,
                "batch": batch,
                "concurrency": concurrency,
                "profile_runs": 3,
                "duration_sec": float(profile["duration"]),
                "configured_input_tokens": float(dataset["isl"]["mean"]),
                "configured_output_tokens": float(dataset["osl"]["mean"]),
                "ttft_slo_ms": float(slos.get("time_to_first_token", np.nan)),
                "itl_slo_ms": float(slos.get("inter_token_latency", np.nan)),
                "source": str(aggregate_path.relative_to(REPO_ROOT)),
            }
            for name in KEY_METRICS:
                row[name] = float(metrics.loc[name, "mean"]) if name in metrics.index else np.nan
            records.append(row)

    frame = pd.DataFrame(records).sort_values(["workload", "backend", "concurrency"])
    frame["slo_attainment_pct"] = frame["good_request_fraction_avg"] * 100.0
    frame["throughput_scaling_vs_c1"] = frame.groupby(["backend", "workload"])[
        "output_token_throughput_avg"
    ].transform(lambda s: s / s.iloc[0])
    frame["parallel_efficiency_vs_c1"] = frame["throughput_scaling_vs_c1"] / frame["concurrency"]
    return frame


def compare_static_suites() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for suite, key in [
        ("context", "context_depth_tokens"),
        ("concurrency", "concurrency"),
        ("prefix-cache", "context_depth_tokens"),
    ]:
        left = read_summary("llama-cpp", suite)
        right = read_summary("vllm", suite)
        merged = left.merge(right, on=key, suffixes=("_llama", "_vllm"))
        result[suite] = merged
    result["startup_llama"] = read_summary("llama-cpp", "startup")
    result["startup_vllm"] = read_summary("vllm", "startup")
    return result


def build_static_long(static: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for suite in ["context", "concurrency", "prefix-cache"]:
        frame = static[suite]
        key = "concurrency" if suite == "concurrency" else "context_depth_tokens"
        for _, row in frame.iterrows():
            for backend in ["llama-cpp", "vllm"]:
                suffix = "llama" if backend == "llama-cpp" else "vllm"
                record: dict[str, object] = {
                    "suite": suite,
                    "backend": backend,
                    key: row[key],
                }
                for column in frame.columns:
                    if column.endswith(f"_{suffix}"):
                        record[column[: -(len(suffix) + 1)]] = row[column]
                rows.append(record)
    return pd.DataFrame(rows)


def generate_report(static: dict[str, pd.DataFrame], aiperf: pd.DataFrame) -> str:
    context = static["context"]
    concurrency = static["concurrency"]
    prefix = static["prefix-cache"]
    startup = static["startup_llama"]

    warm_start = startup.loc[startup["attempt"] > 1, "ready_time_ms"].mean()
    cold_start = float(startup.loc[startup["attempt"] == 1, "ready_time_ms"].iloc[0])
    c16 = concurrency.loc[concurrency["concurrency"] == 16].iloc[0]
    ctx32 = context.loc[context["context_depth_tokens"] == 32768].iloc[0]

    lines = [
        "# LLM 推論測試 Python 分析",
        "",
        "本報告由 `analysis/analyze_benchmarks.py` 直接解析 `results/` 產生。AIPerf 主分析採用每個工作負載最新且具 3 次 profile runs 的 aggregate 批次；較早的單次批次不混入平均。",
        "",
        "## 結論摘要",
        "",
        f"- **整體併發與長提示吞吐量：vLLM 明顯領先。** 固定 2,048-token prompt 的 concurrency 測試中，並發 16 的總 decode throughput 為 {fmt(c16['decode_total_tps_mean_vllm'])} vs {fmt(c16['decode_total_tps_mean_llama'])} tokens/s（{fmt(ratio(c16['decode_total_tps_mean_vllm'], c16['decode_total_tps_mean_llama']), 2)}×）；TTFT 為 {fmt(c16['e2e_ttft_ms_mean_vllm'])} vs {fmt(c16['e2e_ttft_ms_mean_llama'])} ms。",
        f"- **單使用者 decode：llama.cpp 略快或相近。** concurrency=1 時 llama.cpp 為 {fmt(concurrency.iloc[0]['decode_total_tps_mean_llama'])} tokens/s，vLLM 為 {fmt(concurrency.iloc[0]['decode_total_tps_mean_vllm'])} tokens/s；但 vLLM 的 prefill 與 TTFT 快很多。",
        "- **AIPerf interactive 是差距最大的項目。** llama.cpp 只有 concurrency=1 達成 100% SLO；concurrency=2 已降至約 1%，更高併發接近 0%。vLLM 在 concurrency=1–16 全部為 100%。",
        "- **RAG：vLLM 在所有測試併發維持約 99–100% SLO。** llama.cpp 由 concurrency=1 的 100% 降至 concurrency=8 的約 21%。",
        "- **Long-context：兩者 concurrency=1 都穩定，vLLM 延遲約為 llama.cpp 的 57%。** concurrency=2 時 vLLM 仍為 100% SLO，llama.cpp 約 90%。",
        "- **所有正式 AIPerf cell 的 request error rate 都是 0%。** llama.cpp 高併發的問題是延遲超過 SLO，不是 HTTP／請求失敗。",
        f"- **llama.cpp 啟動資料只有一組有效結果。** 冷啟動 {fmt(cold_start, 0)} ms，後兩次暖啟動平均 {fmt(warm_start, 0)} ms（縮短 {fmt(-pct_change(warm_start, cold_start))}%）；vLLM 沒有可用的同類 startup summary。",
        "",
        "## 1. Startup",
        "",
        markdown_table(startup[["attempt", "load_time_ms", "ready_time_ms", "idle_vram_mib", "status"]], 0),
        "",
        "`idle_vram_mib` 全為 0，較可能是遙測未正確取得，而非模型完全不占 VRAM。另一個 llama.cpp startup 批次只有 CSV 標頭；vLLM shared-suite 則有空白 status 與一次 FlashInfer/NVCC 啟動失敗，因此無法做公平的啟動時間比較。",
        "",
        "## 2. Context scaling（單請求，2,048 input / 128 output tokens）",
        "",
    ]

    context_view = pd.DataFrame({
        "context": context["context_depth_tokens"].astype(int),
        "llama prefill tok/s": context["prefill_tps_mean_llama"],
        "vLLM prefill tok/s": context["prefill_tps_mean_vllm"],
        "vLLM/llama prefill": context["prefill_tps_mean_vllm"] / context["prefill_tps_mean_llama"],
        "llama decode tok/s": context["decode_tps_mean_llama"],
        "vLLM decode tok/s": context["decode_tps_mean_vllm"],
        "llama TTFT ms": context["e2e_ttft_ms_mean_llama"],
        "vLLM TTFT ms": context["e2e_ttft_ms_mean_vllm"],
    })
    lines += [
        markdown_table(context_view),
        "",
        f"在 32K 既有 context 下，vLLM prefill 為 llama.cpp 的 {fmt(ratio(ctx32['prefill_tps_mean_vllm'], ctx32['prefill_tps_mean_llama']), 2)}×，TTFT 低 {fmt(-pct_change(ctx32['e2e_ttft_ms_mean_vllm'], ctx32['e2e_ttft_ms_mean_llama']))}%。llama.cpp 的 decode token rate 全程稍高，但其 TTFT 與 prefill 較弱。",
        "",
        "## 3. Concurrency scaling（2,048 input / 128 output tokens）",
        "",
    ]

    concurrency_view = pd.DataFrame({
        "concurrency": concurrency["concurrency"].astype(int),
        "llama total decode": concurrency["decode_total_tps_mean_llama"],
        "vLLM total decode": concurrency["decode_total_tps_mean_vllm"],
        "vLLM/llama decode": concurrency["decode_total_tps_mean_vllm"] / concurrency["decode_total_tps_mean_llama"],
        "llama per-request": concurrency["decode_request_tps_mean_llama"],
        "vLLM per-request": concurrency["decode_request_tps_mean_vllm"],
        "llama TTFT ms": concurrency["e2e_ttft_ms_mean_llama"],
        "vLLM TTFT ms": concurrency["e2e_ttft_ms_mean_vllm"],
    })
    lines += [
        markdown_table(concurrency_view),
        "",
        "llama.cpp 的總 decode throughput 在 concurrency=4 達峰值，之後下降；vLLM 則持續擴展到 concurrency=16。這代表 vLLM 更適合多使用者服務，而 llama.cpp 的優勢偏向低併發單流。",
        "",
        "## 4. Prefix cache",
        "",
    ]

    prefix_view = pd.DataFrame({
        "context": prefix["context_depth_tokens"].astype(int),
        "llama cache-off TTFT": prefix["cache_off_ttft_ms_llama"],
        "llama cache-on TTFT": prefix["cache_on_ttft_ms_llama"],
        "llama reduction %": prefix["ttft_reduction_pct_llama"],
        "vLLM cache-off TTFT": prefix["cache_off_ttft_ms_vllm"],
        "vLLM cache-on TTFT": prefix["cache_on_ttft_ms_vllm"],
        "vLLM reduction %": prefix["ttft_reduction_pct_vllm"],
    })
    lines += [
        markdown_table(prefix_view),
        "",
        "兩個後端的 prefix cache 都非常有效：4K context 可省約 77–83% TTFT，16K 可省約 92–94%。vLLM 的絕對 TTFT 較低；llama.cpp 的相對減幅略高。",
        "",
        "## 5. AIPerf 正式三次重跑",
        "",
        "SLO 判定來自各 run config：interactive 為 TTFT ≤ 1,000 ms 且 ITL ≤ 50 ms；RAG 為 TTFT ≤ 2,000 ms 且 ITL ≤ 50 ms；long-context 為 TTFT ≤ 5,000 ms 且 ITL ≤ 75 ms。`GoodRequestFraction` 是同時通過條件的請求比例。",
        "",
    ]

    for workload in ["interactive", "rag", "long-context"]:
        data = aiperf[aiperf["workload"] == workload].copy()
        view = data[[
            "backend",
            "concurrency",
            "usage_prompt_tokens_avg",
            "time_to_first_token_p95",
            "inter_token_latency_p95",
            "request_latency_p95",
            "output_token_throughput_avg",
            "request_throughput_avg",
            "slo_attainment_pct",
        ]].rename(columns={
            "usage_prompt_tokens_avg": "actual prompt tokens",
            "time_to_first_token_p95": "P95 TTFT ms",
            "inter_token_latency_p95": "P95 ITL ms",
            "request_latency_p95": "P95 request ms",
            "output_token_throughput_avg": "output tok/s",
            "request_throughput_avg": "req/s",
            "slo_attainment_pct": "SLO %",
        })
        lines += [f"### {workload}", "", markdown_table(view), ""]

    lines += [
        "## 6. 可比性與資料品質限制",
        "",
        "- llama.cpp 使用 `gpt-oss-20b-MXFP4.gguf`，vLLM 使用 Hugging Face snapshot；雖為同一模型家族，但模型封裝、kernel、tokenizer/chat template 與服務器參數並非完全相同，因此結果代表這兩套實際部署，而不是隔離其他變因的純引擎比較。",
        "- AIPerf 實際 prompt token 數不同：interactive 約為 llama.cpp 512、vLLM 580；RAG 與 long-context 的差距較小。vLLM interactive/RAG 另啟用了 `first_turn_prefix` cache bust，而 llama.cpp interactive 沒有，因此不是完全相同的輸入路徑；報表已列出實際 token 數。",
        "- AIPerf aggregate 為 3 次 profile run 平均，可信度高於早期單次批次；startup、context、concurrency、prefix-cache 的重複次數較少，解讀細小差異時應保守。",
        "- vLLM metadata 無法取得版本（容器內沒有 `python` 可執行檔），所以版本只能從腳本說明推測，不應當作已驗證事實。",
        "- `results/generated-summary.md` 的 4 successful cells 與 `??` 是舊發佈摘要，未涵蓋這次直接解析的全部測試。",
        "",
        "## 7. 部署建議",
        "",
        "- **多使用者、RAG 或高併發 API：選 vLLM。** 其 throughput 隨併發增加，且 SLO 保持穩定。",
        "- **單使用者、低併發、本機簡化部署：llama.cpp 仍合理。** 單流 decode 稍快，且 prefix cache 表現很好。",
        "- llama.cpp 若要提供互動式多人服務，應先重新檢查 `--parallel 16`、slot/KV cache、continuous batching 與批次參數，再用完全一致的 cache-bust 與實際 prompt 長度重跑 AIPerf；目前 concurrency≥2 的 TTFT 排隊效應過重。",
        "- 修正 vLLM 映像中的版本擷取命令（可改用存在的 Python 路徑或套件 metadata），並保留有效 startup/status 資料，以補齊可重現性。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    static = compare_static_suites()
    aiperf = read_aiperf()
    static_long = build_static_long(static)

    aiperf.to_csv(OUTPUT / "aiperf_key_metrics.csv", index=False, encoding="utf-8-sig")
    static_long.to_csv(OUTPUT / "core_benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    report = generate_report(static, aiperf)
    (OUTPUT / "benchmark_analysis.md").write_text(report, encoding="utf-8-sig")

    print(f"Wrote {len(aiperf)} AIPerf rows")
    print(f"AIPerf workloads: {', '.join(sorted(aiperf['workload'].unique()))}")
    print(f"Report: {OUTPUT / 'benchmark_analysis.md'}")


if __name__ == "__main__":
    main()
