"""由原始 campaign 產生 DGX 彙總與 4090 對比 CSV。

原始執行產物不隨 repository 散布（見 ../README.md「未包含的內容」），因此本腳本
需要指向本機的 results 目錄：

    python build_cross_platform.py \\
        --dgx-results  /path/to/dgx-spark-cuda/results \\
        --rtx4090-results /path/to/llm-serving-benchmark/results

輸出（本目錄）：
    dataset-cell-summary.csv      四個 DGX campaign 的逐 cell 彙總（含 120B）
    concurrency-scaling.csv       併檔
    vllm-stress-summary.csv       併檔
    cross-platform-comparison.csv 4090 vs DGX，37 個共同 cell

產生圖表請接著執行 plot_cross_platform.py（圖 01）與 plot_model_capacity.py（圖 02）。

每次呼叫底層的 scripts/dataset-tests/analyze_dataset_results.py 一次，逐 campaign
分開跑——該腳本會對 (backend, cell_id) 去重且保留 mtime 最新者，若一次傳入多個
campaign，cell_id 重疊的格子會被跨 campaign 混合。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DGX_ROOT = HERE.parent
REPO_ROOT = DGX_ROOT.parent
ANALYZER = REPO_ROOT / "scripts" / "dataset-tests" / "analyze_dataset_results.py"
CONFIG = DGX_ROOT / "scripts" / "benchmark-config.json"

# campaign 相對於各自 results 根目錄的路徑
DGX_CAMPAIGNS = {
    "split": ("gpt-oss-20b", "vllm/dataset/formal-split-65k-20260816-dgx"),
    "stress": ("gpt-oss-20b", "vllm/dataset/formal-stress-65k-20260817-dgx"),
    "s064": ("gpt-oss-20b", "vllm/dataset/formal-20b-s064-20260818-dgx"),
    "120b": ("gpt-oss-120b", "vllm-120b/dataset/formal-120b-s064-20260818-dgx"),
}
RTX4090_SPLIT = "vllm/dataset/formal-split-65k-20260814"

# 兩平台對比時取用的指標；ratio 只對「數值越大越好／越小越好」有意義的欄位計算
COMPARE_METRICS = [
    "output_token_throughput_median",
    "ttft_p95_median",
    "itl_p95_median",
    "request_latency_p95_median",
    "slo_attainment_pct_median",
    "gpu_memory_peak_gib_median",
    "gpu_power_avg_w_median",
    "energy_j_per_token_median",
    "valid_repeats",
    "request_count",
]
RATIO_METRICS = [
    "output_token_throughput_median",
    "ttft_p95_median",
    "itl_p95_median",
    "request_latency_p95_median",
    "gpu_memory_peak_gib_median",
    "energy_j_per_token_median",
]
MERGE_KEYS = [
    "cell_id", "suite", "workload", "dataset",
    "concurrency", "bucket_tokens", "request_rate", "cache_mode",
]


def run_analyzer(campaign: Path, out_dir: Path, empty: Path, config: Path | None) -> pd.DataFrame:
    """對單一 campaign 跑一次 analyze_dataset_results.py，回傳逐 cell 彙總。"""
    if not campaign.is_dir():
        raise SystemExit(f"找不到 campaign 目錄：{campaign}")
    cmd = [
        sys.executable, str(ANALYZER),
        "--vllm-campaign", str(campaign),
        "--llama-campaign", str(empty),
        "--output-dir", str(out_dir),
    ]
    if config is not None:
        cmd[2:2] = ["--config", str(config)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    return pd.read_csv(out_dir / "dataset-cell-summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dgx-results", type=Path, required=True,
                    help="DGX Spark 的 results 根目錄")
    ap.add_argument("--rtx4090-results", type=Path, required=True,
                    help="RTX 4090 的 results 根目錄")
    ap.add_argument("--output-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="cross-platform-"))
    empty = tmp / "empty-campaign"
    empty.mkdir()
    try:
        # --- DGX：逐 campaign 分開跑後併檔 ---
        frames, extra = [], {"concurrency-scaling": [], "vllm-stress-summary": []}
        for key, (model, rel) in DGX_CAMPAIGNS.items():
            out = tmp / key
            d = run_analyzer(args.dgx_results / rel, out, empty, CONFIG)
            d.insert(0, "platform", "dgx-spark")
            d.insert(1, "model", model)
            d["campaign"] = Path(rel).name
            frames.append(d)
            for name in extra:
                path = out / f"{name}.csv"
                if path.exists():
                    e = pd.read_csv(path)
                    if len(e):
                        e.insert(0, "model", model)
                        e["campaign"] = Path(rel).name
                        extra[name].append(e)

        dgx = pd.concat(frames, ignore_index=True)
        dgx.to_csv(args.output_dir / "dataset-cell-summary.csv", index=False, encoding="utf-8")
        print(f"dataset-cell-summary.csv    {len(dgx):3d} 列  "
              f"{dgx.status.value_counts().to_dict()}")
        for name, fs in extra.items():
            if fs:
                r = pd.concat(fs, ignore_index=True)
                r.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8")
                print(f"{name}.csv{' ' * max(1, 28 - len(name) - 4)}{len(r):3d} 列")

        # --- 4090 對比：只取 formal-split 的 vLLM valid cells ---
        a = run_analyzer(args.rtx4090_results / RTX4090_SPLIT, tmp / "rtx4090", empty, None)
        a = a[(a.backend == "vllm") & (a.status == "valid")]
        d = dgx[(dgx.campaign == Path(DGX_CAMPAIGNS["split"][1]).name) & (dgx.status == "valid")]

        m = a[MERGE_KEYS + COMPARE_METRICS].merge(
            d[["cell_id"] + COMPARE_METRICS], on="cell_id", suffixes=("_rtx4090", "_dgx")
        )
        for c in RATIO_METRICS:
            m[c.replace("_median", "") + "_ratio_dgx_over_4090"] = (
                m[f"{c}_dgx"] / m[f"{c}_rtx4090"]
            ).round(4)
        m = m.sort_values(["suite", "concurrency"]).reset_index(drop=True)
        m.to_csv(args.output_dir / "cross-platform-comparison.csv", index=False, encoding="utf-8")
        print(f"cross-platform-comparison.csv {len(m):3d} 列（兩平台共同的 valid cells）")
        if len(m) != 37:
            print(f"  ! 預期 37 個共同 cell，實得 {len(m)}；請確認兩邊 campaign 是否對應")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
