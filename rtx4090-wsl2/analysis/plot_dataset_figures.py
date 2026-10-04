#!/usr/bin/env python3
"""Generate the eight formal-dataset benchmark figures.

Design rules applied throughout:

* Every figure states its scope, its units, and which direction is "better".
* Every plotted number is also written to ``dataset-figures/tables/NN-*.csv`` so
  the figure always has a table-view twin.
* State (SLO pass / SLO fail / no valid result) is never carried by colour
  alone -- each state also has its own glyph, and "no valid result" is hatched.
  This matters because the status green and red are only OKLab dE 4.1 apart
  under deuteranopia.
* A cell that produced no valid result is never drawn as zero. It is drawn as
  an explicit annotation naming the backend, the condition, and the measured
  reason.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd


# 平台目錄（rtx4090-wsl2/）與 repository 根層分開，與 dgx-spark-cuda/ 對稱
PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_ROOT.parent
OUT = PLATFORM_ROOT / "analysis" / "dataset-figures"
TABLES = OUT / "tables"
CONFIG = json.loads(
    (REPO_ROOT / "scripts" / "dataset-tests" / "benchmark-config.json").read_text(
        encoding="utf-8"
    )
)
MAX_ERROR_PCT = float(CONFIG["campaign"]["max_error_rate_pct"])
GOOD_FRACTION_PCT = 95.0

FORMAL_ROOTS = [
    ("llama.cpp", REPO_ROOT / "results" / "llama-cpp" / "dataset" / "formal-split-65k-20260814"),
    ("llama.cpp", REPO_ROOT / "results" / "llama-cpp" / "dataset" / "formal-prefix-65k-20260816"),
    ("vLLM", REPO_ROOT / "results" / "vllm" / "dataset" / "formal-split-65k-20260814"),
    ("vLLM", REPO_ROOT / "results" / "vllm" / "dataset" / "formal-prefix-65k-20260816"),
    ("vLLM", REPO_ROOT / "results" / "vllm" / "dataset" / "formal-vllm-stress-65k-20260816"),
]

# Categorical slots 1 and 2 of the validated order. Checked with the palette
# validator on a white surface: worst adjacent CVD dE 24.7, normal-vision 33.6,
# both marks >= 3:1 contrast.
C = {
    "llama.cpp": "#eb6834",
    "vLLM": "#2a78d6",
    "computed": "#2a78d6",
    "cached": "#1baf7a",
    "cache_off": "#898781",
    "cache_on": "#2a78d6",
    # Status palette -- reserved, never reused as a series colour.
    "pass": "#0ca30c",
    "warn": "#fab219",
    "invalid": "#d03b3b",
    "missing": "#c3c2b7",
    # Chrome and ink.
    "surface": "#ffffff",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "band": "#f4f4f1",
}
MARKERS = {"llama.cpp": "o", "vLLM": "s"}
BACKENDS = ["llama.cpp", "vLLM"]

GLYPH = {"pass": "✓", "slo": "!", "invalid": "✗", "missing": "·"}

FOOTER = (
    "Formal runs only · GPT-OSS-20B · RTX 4090 24 GB · context limit 65,536 tokens · "
    "a cell is valid when at least 3 repeats finished with exit code 0, request errors "
    "≤ 1%, and no unexpected GPU OOM · lines stop at the last valid point; "
    "no valid result is never plotted as zero"
)


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": C["surface"],
            "axes.facecolor": C["surface"],
            "savefig.facecolor": C["surface"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            # Leaves a clear band under each panel title for the "higher/lower
            # is better" hint, which otherwise lands on the title baseline.
            "axes.titlepad": 26,
            "axes.labelsize": 11,
            "axes.labelcolor": C["ink2"],
            "axes.edgecolor": C["axis"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": C["grid"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": C["ink2"],
            "ytick.color": C["ink2"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": C["ink"],
            "legend.frameon": False,
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
        }
    )


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value: float, digits: int = 0, unit: str = "") -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}{unit}"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_formal_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], Path]]:
    cells: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    dirs: dict[tuple[str, str], Path] = {}
    for backend, campaign in FORMAL_ROOTS:
        if not campaign.exists():
            continue
        for cell_dir in sorted(campaign.iterdir()):
            cell_path = cell_dir / "cell.json"
            validation_path = cell_dir / "validation.json"
            if not cell_dir.is_dir() or not cell_path.exists():
                continue
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            validation = (
                json.loads(validation_path.read_text(encoding="utf-8"))
                if validation_path.exists()
                else {}
            )
            dirs[(backend, cell["cell_id"])] = cell_dir

            metric_path = cell_dir / "repeat-metrics.csv"
            rep = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
            if not rep.empty:
                rep["valid"] = rep["valid"].astype(str).str.lower().eq("true")
                for col in rep.columns:
                    if col not in {"valid", "oom_detected"}:
                        rep[col] = pd.to_numeric(rep[col], errors="coerce")
                for _, row in rep.iterrows():
                    item = row.to_dict()
                    item.update(
                        {
                            "backend": backend,
                            "cell_id": cell["cell_id"],
                            "suite": cell["suite"],
                            "scope": cell["scope"],
                            "concurrency": cell.get("concurrency"),
                            "bucket_tokens": cell.get("bucket_tokens"),
                            "request_rate": cell.get("request_rate"),
                            "cache_mode": cell.get("cache_mode"),
                        }
                    )
                    request_count = as_float(item.get("request_count"))
                    good_count = as_float(item.get("good_request_count"))
                    item["good_pct"] = (
                        100 * good_count / request_count
                        if request_count > 0 and math.isfinite(good_count)
                        else math.nan
                    )
                    repeats.append(item)
            valid_rep = rep[rep["valid"]] if not rep.empty else rep

            raw_errors = (
                pd.to_numeric(rep["request_error_rate_pct"], errors="coerce").dropna()
                if not rep.empty and "request_error_rate_pct" in rep
                else pd.Series(dtype=float)
            )
            exit_fail_count = (
                int((pd.to_numeric(rep["exit_code"], errors="coerce").fillna(1) != 0).sum())
                if not rep.empty and "exit_code" in rep
                else 0
            )
            oom_repeat_count = (
                int(rep["oom_detected"].astype(str).str.lower().eq("true").sum())
                if not rep.empty and "oom_detected" in rep
                else int(validation.get("oom_repeats", 0) or 0)
            )

            def mean_col(name: str) -> float:
                return (
                    float(pd.to_numeric(valid_rep[name], errors="coerce").mean())
                    if not valid_rep.empty and name in valid_rep
                    else math.nan
                )

            total_req = float(valid_rep["request_count"].sum()) if not valid_rep.empty else 0.0
            total_good = (
                float(valid_rep["good_request_count"].sum())
                if not valid_rep.empty and "good_request_count" in valid_rep
                else 0.0
            )
            good_pct = 100 * total_good / total_req if total_req > 0 else math.nan
            error_pct = mean_col("request_error_rate_pct")
            status = validation.get("status", "missing")
            status_valid = str(status).startswith("valid")
            slo_pass = (
                status_valid
                and math.isfinite(good_pct)
                and good_pct >= GOOD_FRACTION_PCT
                and math.isfinite(error_pct)
                and error_pct <= MAX_ERROR_PCT
            )
            profile = CONFIG["slo"].get(cell.get("slo_profile"), {})
            ttft_slo = as_float(profile.get("ttft_ms"))
            itl_slo = as_float(profile.get("itl_ms"))
            ttft_p95 = mean_col("ttft_p95_ms")
            itl_p95 = mean_col("itl_p95_ms")
            cells.append(
                {
                    **cell,
                    "backend": backend,
                    "status": status,
                    "valid": status_valid,
                    "valid_repeats": int(validation.get("valid_repeats", len(valid_rep)) or 0),
                    "invalid_repeats": int(validation.get("invalid_repeats", 0) or 0),
                    "oom_repeats": oom_repeat_count,
                    "exit_fail_count": exit_fail_count,
                    "raw_error_pct": float(raw_errors.mean()) if not raw_errors.empty else math.nan,
                    "request_count": total_req,
                    "output_tps": mean_col("output_token_throughput"),
                    "request_tps": mean_col("request_throughput"),
                    "ttft_p95_ms": ttft_p95,
                    "itl_p95_ms": itl_p95,
                    "ttft_slo_ms": ttft_slo,
                    "itl_slo_ms": itl_slo,
                    "ttft_ratio": ttft_p95 / ttft_slo if ttft_slo else math.nan,
                    "itl_ratio": itl_p95 / itl_slo if itl_slo else math.nan,
                    "request_latency_p95_ms": mean_col("request_latency_p95_ms"),
                    "input_tokens_avg": mean_col("input_tokens_avg"),
                    "good_pct": good_pct,
                    "error_pct": error_pct,
                    "slo_pass": slo_pass,
                }
            )
    return pd.DataFrame(cells), pd.DataFrame(repeats), dirs


def cell_state(row: pd.Series) -> str:
    if not bool(row["valid"]):
        return "invalid"
    return "pass" if bool(row["slo_pass"]) else "slo"


def invalid_reason(row: pd.Series, *, short: bool = False) -> str:
    """A measured reason a cell produced no valid result -- never a guess."""
    parts: list[str] = []
    if int(row.get("oom_repeats", 0) or 0) > 0:
        parts.append("GPU OOM" if short else "unexpected GPU OOM")
    if int(row.get("exit_fail_count", 0) or 0) > 0:
        parts.append("run failed" if short else "run failed (non-zero exit code)")
    error_pct = as_float(row.get("raw_error_pct"))
    if math.isfinite(error_pct) and error_pct > MAX_ERROR_PCT:
        digits = 1 if error_pct < 10 else 0
        parts.append(
            f"{error_pct:.{digits}f}% err" if short else f"{error_pct:.{digits}f}% request errors"
        )
    if not parts:
        parts.append(f"{int(row.get('valid_repeats', 0) or 0)}/3 valid")
    return " · ".join(parts)


def slo_reason(row: pd.Series, *, short: bool = False) -> str:
    """Which SLO term is binding for a valid-but-failing cell."""
    good = as_float(row.get("good_pct"))
    ttft_ratio = as_float(row.get("ttft_ratio"))
    itl_ratio = as_float(row.get("itl_ratio"))
    error = as_float(row.get("error_pct"))
    parts: list[str] = []
    if math.isfinite(good) and good < GOOD_FRACTION_PCT:
        parts.append(f"{good:.0f}% good" if short else f"good requests {good:.1f}% (need ≥ 95%)")
    if math.isfinite(error) and error > MAX_ERROR_PCT:
        parts.append(f"{error:.1f}% err" if short else f"request errors {error:.2f}% (need ≤ 1%)")
    if math.isfinite(ttft_ratio) and ttft_ratio > 1:
        parts.append(
            f"TTFT {ttft_ratio:.1f}×" if short else f"P95 TTFT {ttft_ratio:.1f}× the SLO"
        )
    if math.isfinite(itl_ratio) and itl_ratio > 1:
        parts.append(f"ITL {itl_ratio:.1f}×" if short else f"P95 ITL {itl_ratio:.1f}× the SLO")
    if not parts:
        return "SLO fail"
    return parts[0] if short else " · ".join(parts)


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------


def write_table(number: int, slug: str, rows: Sequence[dict[str, Any]]) -> None:
    """Every figure ships a table-view twin holding the exact plotted numbers."""
    TABLES.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    path = TABLES / f"{number:02d}-{slug}.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{value:.3f}".rstrip("0").rstrip(".")
                        if isinstance(value, float) and math.isfinite(value)
                        else ("" if isinstance(value, float) else value)
                    )
                    for key, value in row.items()
                }
            )


def save_figure(fig: plt.Figure, number: int, slug: str, title: str, subtitle: str) -> None:
    """Place title, subtitle and footer in inches so tall figures keep the spacing."""
    OUT.mkdir(parents=True, exist_ok=True)
    height = fig.get_figheight()
    fig.suptitle(title, y=1 - 0.22 / height, va="top")
    fig.text(0.5, 1 - 0.62 / height, subtitle, ha="center", va="top", fontsize=12,
             color=C["ink2"])
    fig.text(0.008, 0.10 / height, FOOTER, fontsize=8, color=C["muted"], ha="left", va="bottom")
    fig.savefig(OUT / f"{number:02d}-{slug}.png", dpi=170, facecolor=C["surface"])
    fig.savefig(OUT / f"{number:02d}-{slug}.svg", facecolor=C["surface"])
    plt.close(fig)


def label_all(
    ax: plt.Axes,
    drawn: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    digits: int,
) -> None:
    """Direct-label every plotted point. Call only after the axes limits are final.

    Vertical side: the top value at an x sits above its marker, the rest below,
    except near the axis floor where "below" would leave the frame. Horizontal
    offset: labels that end up on the same side of the same x are fanned out, so
    two series with close values never print on top of each other.
    """
    low, high = ax.get_ylim()
    floor = low + 0.08 * (high - low)
    placed: dict[tuple[float, bool], list[tuple[int, float, str]]] = {}
    peak: dict[float, float] = {}
    for _, xs, ys in drawn:
        for x, y in zip(xs, ys):
            if math.isfinite(y):
                peak[x] = max(peak.get(x, -math.inf), y)
    for index, (color, xs, ys) in enumerate(drawn):
        for x, y in zip(xs, ys):
            if not math.isfinite(y):
                continue
            above = y >= peak[x] or y <= floor
            placed.setdefault((x, above), []).append((index, y, color))

    for (x, above), items in placed.items():
        items.sort(key=lambda item: item[0])
        # Series that coincide (e.g. everything at 100%) get one shared label in
        # neutral ink rather than the same number fanned out four times.
        if len(items) > 1 and len({fmt(y, digits) for _, y, _ in items}) == 1:
            items = [(items[0][0], items[0][1], C["ink2"])]
        span = len(items)
        for slot, (_, y, color) in enumerate(items):
            dx = (slot - (span - 1) / 2) * 27.0
            ax.annotate(
                fmt(y, digits),
                xy=(x, y),
                xytext=(dx, 9 if above else -10),
                textcoords="offset points",
                ha="center",
                va="bottom" if above else "top",
                fontsize=8,
                color=color,
                zorder=6,
                bbox={"facecolor": C["surface"], "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
            )


def slo_rule(ax: plt.Axes, value: float) -> None:
    """Draw the threshold. Its numeric value is stated in the panel hint text,
    so no on-line label is needed -- those collide with the data."""
    ax.axhline(value, color=C["ink2"], linestyle=(0, (6, 4)), linewidth=1.2, zorder=1)


def note_block(
    ax: plt.Axes,
    lines: Sequence[str],
    color: str,
    avoid: Sequence[tuple[float, float]] = (),
) -> None:
    """Put 'no valid result' notes in whichever corner has the least data in it."""
    if not lines:
        return
    width = min(0.96, 0.0108 * max(len(line) for line in lines) + 0.04)
    height = min(0.5, 0.052 * len(lines) + 0.04)
    occupied: list[tuple[float, float]] = []
    for x, y in avoid:
        if math.isfinite(x) and math.isfinite(y):
            fx, fy = ax.transLimits.transform((x, y))
            occupied.append((fx, fy))
    corners = [
        (0.98, 0.97, "right", "top"),
        (0.02, 0.97, "left", "top"),
        (0.98, 0.03, "right", "bottom"),
        (0.02, 0.03, "left", "bottom"),
    ]
    best = corners[0]
    best_hits = None
    for x, y, ha, va in corners:
        x0 = x - width if ha == "right" else x
        y0 = y - height if va == "top" else y
        hits = sum(
            1 for fx, fy in occupied
            if x0 - 0.03 <= fx <= x0 + width + 0.03 and y0 - 0.03 <= fy <= y0 + height + 0.03
        )
        if best_hits is None or hits < best_hits:
            best, best_hits = (x, y, ha, va), hits
        if hits == 0:
            break
    x, y, ha, va = best
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        color=color,
        linespacing=1.45,
        bbox={
            "facecolor": C["surface"],
            "edgecolor": color,
            "alpha": 0.95,
            "pad": 4.0,
            "linewidth": 0.8,
        },
        zorder=9,
    )


# --------------------------------------------------------------------------
# shared ordering / labelling
# --------------------------------------------------------------------------

SUITE_GROUPS = [
    ("sharegpt_closed", "ShareGPT closed-loop"),
    ("sharegpt_open", "ShareGPT open-loop"),
    ("sharegpt_natural", "ShareGPT natural"),
    ("rag", "RAG · SQuAD then HotpotQA"),
    ("prefix", "Prefix cache"),
    ("ruler_performance", "RULER long context"),
]


def ordered(frame: pd.DataFrame, suite: str) -> pd.DataFrame:
    frame = frame.copy()
    if suite == "sharegpt_closed":
        return frame.sort_values("concurrency")
    if suite in {"sharegpt_open", "sharegpt_natural"}:
        return frame.sort_values(["request_rate", "concurrency"])
    if suite == "rag":
        frame["dataset_order"] = frame["cell_id"].str.contains("hotpotqa").astype(int)
        return frame.sort_values(["dataset_order", "bucket_tokens", "concurrency"])
    if suite == "prefix":
        frame["cache_order"] = frame["cache_mode"].map({"off": 0, "on": 1})
        return frame.sort_values(["bucket_tokens", "cache_order", "concurrency"])
    return frame.sort_values(["bucket_tokens", "concurrency"])


def condition_label(row: pd.Series) -> str:
    suite = row["suite"]
    c = int(row["concurrency"])
    if suite == "sharegpt_closed":
        return f"C{c}"
    if suite == "sharegpt_open":
        return f"{row['request_rate']:g} r/s"
    if suite == "sharegpt_natural":
        return f"{row['request_rate']:g} r/s C{c}"
    if suite == "rag":
        ds = "HotpotQA" if "hotpotqa" in row["cell_id"] else "SQuAD"
        return f"{ds} {int(row['bucket_tokens'])//1024}K C{c}"
    if suite == "prefix":
        return f"{int(row['bucket_tokens'])//1024}K cache {row['cache_mode']} C{c}"
    return f"{int(row['bucket_tokens'])//1024}K C{c}"


# --------------------------------------------------------------------------
# figure 1 -- coverage / SLO / failure cause
# --------------------------------------------------------------------------


def figure_01_status(cells: pd.DataFrame) -> None:
    fair = cells[cells["scope"] == "comparison"].copy()

    columns: list[dict[str, Any]] = []
    groups: list[tuple[str, int, int]] = []
    for suite, title in SUITE_GROUPS:
        part = fair[fair["suite"] == suite]
        template = ordered(part[part["backend"] == "vLLM"], suite)
        if template.empty:
            template = ordered(part.drop_duplicates("cell_id"), suite)
        start = len(columns)
        for _, row in template.iterrows():
            columns.append({"cell_id": row["cell_id"], "label": condition_label(row)})
        groups.append((title, start, len(columns)))

    n = len(columns)
    fig = plt.figure(figsize=(max(20.0, n * 0.54 + 4.5), 7.6))
    ax = fig.add_axes([0.052, 0.255, 0.835, 0.545])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(1.55, -1.05)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([0, 1], BACKENDS, fontsize=12)
    ax.set_xticks(range(n), [col["label"] for col in columns], rotation=55, ha="right", fontsize=8.5)
    ax.tick_params(length=0)

    table_rows: list[dict[str, Any]] = []
    tally = {backend: {"pass": 0, "slo": 0, "invalid": 0, "missing": 0} for backend in BACKENDS}
    for r, backend in enumerate(BACKENDS):
        by_id = fair[fair["backend"] == backend].set_index("cell_id")
        for c, col in enumerate(columns):
            cell_id = col["cell_id"]
            if cell_id not in by_id.index:
                state, detail = "missing", ""
                row = None
            else:
                row = by_id.loc[cell_id]
                state = cell_state(row)
                # Kept to a few characters so it never spills into the next cell;
                # the full reason is in the companion CSV.
                if state == "slo":
                    good = as_float(row.get("good_pct"))
                    detail = f"{good:.0f}%" if math.isfinite(good) else "SLO"
                elif state == "invalid":
                    error = as_float(row.get("raw_error_pct"))
                    if math.isfinite(error) and error > MAX_ERROR_PCT:
                        # 1.47% must not print as "1%" -- that reads as passing.
                        detail = f"{error:.1f}%" if error < 10 else f"{error:.0f}%"
                    else:
                        detail = "run fail"
                else:
                    detail = ""
            tally[backend][state] += 1
            face = {"pass": C["pass"], "slo": C["warn"], "invalid": C["invalid"],
                    "missing": C["missing"]}[state]
            ax.add_patch(
                Rectangle(
                    (c - 0.45, r - 0.34),
                    0.90,
                    0.68,
                    facecolor=face,
                    edgecolor=C["surface"],
                    linewidth=1.4,
                    # Secondary encoding: status green and red sit only dE 4.1
                    # apart under deuteranopia, so "no valid result" is hatched.
                    hatch="//" if state == "invalid" else None,
                    zorder=2,
                )
            )
            ink = "#ffffff" if state in {"pass", "invalid"} else C["ink"]
            ax.text(c, r - (0.10 if detail else 0.0), GLYPH[state], ha="center", va="center",
                    fontsize=13, fontweight="bold", color=ink, zorder=3)
            if detail:
                ax.text(c, r + 0.16, detail, ha="center", va="center",
                        fontsize=7.2, fontweight="bold", color=ink, zorder=3)
            if row is not None:
                table_rows.append(
                    {
                        "backend": backend,
                        "suite": row["suite"],
                        "cell_id": cell_id,
                        "condition": col["label"],
                        "state": {"pass": "SLO pass", "slo": "valid, SLO fail",
                                  "invalid": "no valid result"}[state],
                        "good_requests_pct": as_float(row.get("good_pct")),
                        "request_error_pct": as_float(row.get("raw_error_pct")),
                        "p95_ttft_ms": as_float(row.get("ttft_p95_ms")),
                        "p95_ttft_slo_ms": as_float(row.get("ttft_slo_ms")),
                        "p95_itl_ms": as_float(row.get("itl_p95_ms")),
                        "p95_itl_slo_ms": as_float(row.get("itl_slo_ms")),
                        "reason": (
                            invalid_reason(row) if state == "invalid"
                            else (slo_reason(row) if state == "slo" else "")
                        ),
                    }
                )

    # Group captions alternate between two heights: a one-column group such as
    # "ShareGPT natural" would otherwise overprint its neighbours.
    for level, (title, start, end) in enumerate(groups):
        if start > 0:
            ax.axvline(start - 0.5, color=C["axis"], linewidth=1.0, ymin=0.05, ymax=0.73, zorder=1)
        mid = (start + end - 1) / 2
        y_text = -0.66 if level % 2 == 0 else -0.90
        ax.text(mid, y_text, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=C["ink"])
        ax.plot([start - 0.4, end - 0.6], [y_text + 0.13, y_text + 0.13],
                color=C["axis"], linewidth=1.2)
        ax.plot([mid, mid], [y_text + 0.13, -0.46], color=C["axis"], linewidth=0.8)

    for r, backend in enumerate(BACKENDS):
        counts = tally[backend]
        total = counts["pass"] + counts["slo"] + counts["invalid"]
        ax.text(
            n - 0.35,
            r,
            f"  {counts['pass']}/{total} SLO pass\n"
            f"  {counts['slo']} valid, SLO fail\n"
            f"  {counts['invalid']} no valid result",
            ha="left",
            va="center",
            fontsize=10,
            color=C["ink2"],
            linespacing=1.5,
        )

    fig.legend(
        handles=[
            Patch(facecolor=C["pass"], label=f"{GLYPH['pass']}  valid · SLO pass"),
            Patch(facecolor=C["warn"],
                  label=f"{GLYPH['slo']}  valid · SLO fail (number = good-request fraction)"),
            Patch(facecolor=C["invalid"], hatch="//", edgecolor=C["surface"],
                  label=f"{GLYPH['invalid']}  no valid result (number = request error rate)"),
            Patch(facecolor=C["missing"], label=f"{GLYPH['missing']}  not tested"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        fontsize=11,
    )
    write_table(1, "formal-test-status", table_rows)
    save_figure(
        fig,
        1,
        "formal-test-status",
        "1 · Formal Test Coverage, SLO Outcome, and Failure Cause",
        "45 fair-comparison cells per backend. Reads coverage and status only — it carries no "
        "throughput or latency magnitudes. Exact per-cell numbers: tables/01-formal-test-status.csv",
    )


# --------------------------------------------------------------------------
# figures 2, 3, 6 -- line panels
# --------------------------------------------------------------------------


def repeat_cloud(
    ax: plt.Axes,
    repeats: pd.DataFrame,
    backend: str,
    metric: str,
    positions: dict[float, float],
    key: str,
    offset: float,
) -> None:
    part = repeats[(repeats["backend"] == backend) & repeats["valid"]]
    if part.empty:
        return
    for raw, pos in positions.items():
        vals = pd.to_numeric(part.loc[part[key] == raw, metric], errors="coerce").dropna()
        if vals.empty:
            continue
        ax.scatter(np.full(len(vals), pos + offset), vals, s=26, color=C[backend],
                   alpha=0.30, edgecolors="none", zorder=2)


def figure_02_closed(cells: pd.DataFrame, repeats: pd.DataFrame) -> None:
    c = cells[(cells["suite"] == "sharegpt_closed") & (cells["scope"] == "comparison")]
    r = repeats[(repeats["suite"] == "sharegpt_closed") & (repeats["scope"] == "comparison")]
    levels = [1, 2, 4, 8, 16]
    pos = {value: idx for idx, value in enumerate(levels)}

    panels = [
        ("output_tps", "output_token_throughput", "Output throughput", "tokens/s", None, 0,
         "higher is better"),
        ("ttft_p95_ms", "ttft_p95_ms", "P95 time to first token", "ms", 1000.0, 0,
         "lower is better · chat SLO ≤ 1,000 ms"),
        ("itl_p95_ms", "itl_p95_ms", "P95 inter-token latency", "ms per token", 50.0, 1,
         "lower is better · chat SLO ≤ 50 ms"),
        ("good_pct", "good_pct", "Good request fraction", "% of requests", 95.0, 1,
         "higher is better · SLO ≥ 95%"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.855, bottom=0.115, hspace=0.44, wspace=0.20)
    table_rows: list[dict[str, Any]] = []

    for ax, (metric, rmetric, title, unit, slo, digits, hint) in zip(axes.flat, panels):
        drawn: list[tuple[str, list[float], list[float]]] = []
        for backend in BACKENDS:
            part = c[(c["backend"] == backend) & c["valid"]].sort_values("concurrency")
            xs = [pos[int(v)] for v in part["concurrency"]]
            ys = [as_float(v) for v in part[metric]]
            ax.plot(xs, ys, color=C[backend], marker=MARKERS[backend], markersize=8,
                    linewidth=2.2, markeredgecolor=C["surface"], markeredgewidth=1.6,
                    label=backend, zorder=4)
            repeat_cloud(ax, r, backend, rmetric, pos, "concurrency",
                         -0.06 if backend == "llama.cpp" else 0.06)
            drawn.append((C[backend], xs, ys))
        if slo is not None:
            slo_rule(ax, slo)
        if metric == "good_pct":
            ax.set_ylim(-4, 108)
        ax.set_xticks(range(len(levels)), [f"C{v}" for v in levels])
        ax.set_xlim(-0.35, len(levels) - 0.65)
        ax.set_xlabel("Client concurrency (closed loop)")
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.text(0.0, 1.006, hint, transform=ax.transAxes, fontsize=9.5, color=C["muted"],
                ha="left", va="bottom")
        ax.grid(axis="x", visible=False)
        label_all(ax, drawn, digits)

    for backend in BACKENDS:
        part = c[(c["backend"] == backend) & c["valid"]].sort_values("concurrency")
        for _, row in part.iterrows():
            table_rows.append(
                {
                    "backend": backend,
                    "concurrency": int(row["concurrency"]),
                    "output_tokens_per_s": as_float(row["output_tps"]),
                    "p95_ttft_ms": as_float(row["ttft_p95_ms"]),
                    "p95_itl_ms": as_float(row["itl_p95_ms"]),
                    "good_requests_pct": as_float(row["good_pct"]),
                    "slo_pass": bool(row["slo_pass"]),
                }
            )

    fig.legend(
        handles=[
            Line2D([0], [0], color=C[b], marker=MARKERS[b], markersize=9, linewidth=2.2, label=b)
            for b in BACKENDS
        ]
        + [
            Line2D([0], [0], color=C["muted"], marker="o", linestyle="none", alpha=0.4,
                   markersize=7, label="individual valid repeat"),
            Line2D([0], [0], color=C["ink2"], linestyle=(0, (6, 4)), linewidth=1.2, label="SLO threshold"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=4,
        fontsize=11,
    )
    write_table(2, "sharegpt-closed-loop", table_rows)
    save_figure(
        fig,
        2,
        "sharegpt-closed-loop",
        "2 · ShareGPT Closed-loop Concurrency Scaling",
        "Fixed 128-token output. Each virtual user sends its next request only after the previous "
        "one completes, so client concurrency is the load knob.",
    )


def figure_03_open(cells: pd.DataFrame, repeats: pd.DataFrame) -> None:
    c = cells[(cells["suite"] == "sharegpt_open") & cells["valid"]].copy()
    rates = [0.5, 1, 2, 4, 8, 16, 32]
    panels = [
        ("request_tps", "Completed request rate", "requests/s", None, 2, "higher is better"),
        ("output_tps", "Output throughput", "tokens/s", None, 0, "higher is better"),
        ("ttft_p95_ms", "P95 time to first token", "ms", 1000.0, 0,
         "lower is better · chat SLO ≤ 1,000 ms"),
        ("good_pct", "Good request fraction", "% of requests", 95.0, 1,
         "higher is better · SLO ≥ 95%"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.855, bottom=0.115, hspace=0.44, wspace=0.20)
    table_rows: list[dict[str, Any]] = []

    for ax, (metric, title, unit, slo, digits, hint) in zip(axes.flat, panels):
        drawn: list[tuple[str, list[float], list[float]]] = []
        for backend in BACKENDS:
            fair = c[(c["backend"] == backend) & (c["scope"] == "comparison")].sort_values("request_rate")
            xs = [float(v) for v in fair["request_rate"]]
            ys = [as_float(v) for v in fair[metric]]
            ax.plot(xs, ys, color=C[backend], marker=MARKERS[backend], markersize=8, linewidth=2.2,
                    markeredgecolor=C["surface"], markeredgewidth=1.6, zorder=4)
            drawn.append((C[backend], xs, ys))
        stress = c[(c["backend"] == "vLLM") & (c["scope"] == "vllm_stress")].sort_values("request_rate")
        anchor = c[(c["backend"] == "vLLM") & (c["scope"] == "comparison") & (c["request_rate"] == 8)]
        ext = pd.concat([anchor, stress]).sort_values("request_rate")
        xs = [float(v) for v in ext["request_rate"]]
        ys = [as_float(v) for v in ext[metric]]
        ax.plot(xs, ys, color=C["vLLM"], marker="D", markersize=8, linestyle=(0, (5, 3)),
                linewidth=2.0, markeredgecolor=C["surface"], markeredgewidth=1.6, zorder=4)
        drawn.append((C["vLLM"], xs[1:], ys[1:]))

        if metric == "request_tps":
            ax.plot(rates, rates, color=C["muted"], linestyle=(0, (2, 3)), linewidth=1.4, zorder=1)
            # Cap the y-axis on the measured rates and let the ideal line run off
            # the top; scaling to the ideal would flatten the real curves.
            reached = max((y for _, _, ys in drawn for y in ys if math.isfinite(y)), default=1.0)
            ax.set_ylim(0, reached * 1.4)
            ax.annotate("ideal: completed = target", xy=(reached * 1.28, reached * 1.28),
                        xytext=(8, -2), textcoords="offset points", ha="left", va="top",
                        fontsize=9.5, color=C["muted"])
        if slo is not None:
            slo_rule(ax, slo)
        if metric == "good_pct":
            ax.set_ylim(-4, 108)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(rates))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4", "8", "16", "32"]))
        ax.set_xlim(0.4, 42)
        ax.set_xlabel("Target request rate (requests/s, Poisson arrivals)")
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.text(0.0, 1.006, hint, transform=ax.transAxes, fontsize=9.5, color=C["muted"],
                ha="left", va="bottom")
        ax.grid(axis="x", visible=False)
        label_all(ax, drawn, digits)

    for _, row in c.sort_values(["backend", "request_rate"]).iterrows():
        table_rows.append(
            {
                "backend": row["backend"],
                "scope": row["scope"],
                "target_rate_per_s": as_float(row["request_rate"]),
                "completed_rate_per_s": as_float(row["request_tps"]),
                "output_tokens_per_s": as_float(row["output_tps"]),
                "p95_ttft_ms": as_float(row["ttft_p95_ms"]),
                "good_requests_pct": as_float(row["good_pct"]),
                "slo_pass": bool(row["slo_pass"]),
            }
        )

    fig.legend(
        handles=[
            Line2D([0], [0], color=C["llama.cpp"], marker="o", markersize=9, linewidth=2.2,
                   label="llama.cpp · fair comparison"),
            Line2D([0], [0], color=C["vLLM"], marker="s", markersize=9, linewidth=2.2,
                   label="vLLM · fair comparison"),
            Line2D([0], [0], color=C["vLLM"], marker="D", markersize=9, linestyle=(0, (5, 3)),
                   linewidth=2.0, label="vLLM-only stress extension (16, 32 r/s)"),
            Line2D([0], [0], color=C["muted"], linestyle=(0, (2, 3)), linewidth=1.4,
                   label="ideal: completed = target"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=4,
        fontsize=11,
    )
    write_table(3, "sharegpt-open-loop", table_rows)
    save_figure(
        fig,
        3,
        "sharegpt-open-loop",
        "3 · ShareGPT Open-loop Capacity",
        "Requests are injected at a fixed target rate regardless of completions. Saturation shows as "
        "the completed rate leaving the ideal line while TTFT climbs.",
    )


# --------------------------------------------------------------------------
# figure 4 -- RAG
# --------------------------------------------------------------------------


def figure_04_rag(cells: pd.DataFrame) -> None:
    c = cells[(cells["suite"] == "rag") & (cells["scope"] == "comparison")].copy()
    c["dataset_name"] = np.where(c["cell_id"].str.contains("hotpotqa"), "HotpotQA", "SQuAD")
    buckets = [1024, 4096, 8192, 16384]
    pos = {value: idx for idx, value in enumerate(buckets)}
    labels = ["1K", "4K", "8K", "16K"]

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.125, hspace=0.34, wspace=0.19)
    table_rows: list[dict[str, Any]] = []

    metrics = [
        ("output_tps", "Output throughput", "tokens/s", None, 0, "higher is better"),
        ("ttft_p95_ms", "P95 time to first token", "ms", 2000.0, 0,
         "lower is better · RAG SLO ≤ 2,000 ms (dashed)"),
    ]
    pending: list[tuple[plt.Axes, list, list[str], int]] = []
    for row_idx, dataset in enumerate(["SQuAD", "HotpotQA"]):
        for col_idx, (metric, title, unit, slo, digits, hint) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            drawn: list[tuple[str, list[float], list[float]]] = []
            notes: list[str] = []
            for backend in BACKENDS:
                for concurrency, style in [(1, "-"), (8, (0, (5, 3)))]:
                    sel = c[(c["dataset_name"] == dataset) & (c["backend"] == backend)
                            & (c["concurrency"] == concurrency)]
                    ok = sel[sel["valid"]].sort_values("bucket_tokens")
                    xs = [pos[int(v)] for v in ok["bucket_tokens"]]
                    ys = [as_float(v) for v in ok[metric]]
                    ax.plot(xs, ys, color=C[backend], marker=MARKERS[backend], markersize=8,
                            linestyle=style, linewidth=2.2, markeredgecolor=C["surface"],
                            markeredgewidth=1.6, zorder=4)
                    drawn.append((C[backend], xs, ys))
                    for _, bad in sel[~sel["valid"]].sort_values("bucket_tokens").iterrows():
                        x = pos[int(bad["bucket_tokens"])]
                        ax.plot([x], [0], marker="x", markersize=11, markeredgewidth=2.4,
                                color=C["invalid"], clip_on=False,
                                transform=ax.get_xaxis_transform(), zorder=7)
                        notes.append(
                            f"{GLYPH['invalid']} {backend} C{concurrency} @ "
                            f"{labels[x]}: no valid result — {invalid_reason(bad)}"
                        )
            if slo is not None:
                slo_rule(ax, slo)
            ax.set_xticks(range(len(buckets)), labels)
            ax.set_xlim(-0.35, len(buckets) - 0.5)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(unit)
            ax.set_title(f"{dataset} · {title}")
            ax.text(0.0, 1.006, hint, transform=ax.transAxes, fontsize=9.5, color=C["muted"],
                    ha="left", va="bottom")
            ax.grid(axis="x", visible=False)
            if row_idx == 1:
                ax.set_xlabel("Target input context bucket")
            pending.append((ax, drawn, sorted(set(notes)), digits))

    # Same y-scale per metric so the two datasets are directly comparable, with
    # headroom for the labels. Only then are labels and notes placed.
    for col_idx in range(2):
        top = max(axes[r, col_idx].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r, col_idx].set_ylim(0, top * 1.06)
    for ax, drawn, notes, digits in pending:
        label_all(ax, drawn, digits)
        note_block(ax, notes, C["invalid"],
                   [(x, y) for _, xs, ys in drawn for x, y in zip(xs, ys)])

    for _, row in c.sort_values(["dataset_name", "backend", "concurrency", "bucket_tokens"]).iterrows():
        table_rows.append(
            {
                "dataset": row["dataset_name"],
                "backend": row["backend"],
                "concurrency": int(row["concurrency"]),
                "context_bucket": f"{int(row['bucket_tokens'])//1024}K",
                "valid": bool(row["valid"]),
                "output_tokens_per_s": as_float(row["output_tps"]),
                "p95_ttft_ms": as_float(row["ttft_p95_ms"]),
                "good_requests_pct": as_float(row["good_pct"]),
                "request_error_pct": as_float(row["raw_error_pct"]),
                "reason": "" if bool(row["valid"]) else invalid_reason(row),
            }
        )

    handles = []
    for backend in BACKENDS:
        for concurrency, style in [(1, "-"), (8, (0, (5, 3)))]:
            handles.append(
                Line2D([0], [0], color=C[backend], marker=MARKERS[backend], markersize=9,
                       linestyle=style, linewidth=2.2, label=f"{backend} C{concurrency}")
            )
    handles.append(
        Line2D([0], [0], color=C["invalid"], marker="x", markersize=10, markeredgewidth=2.4,
               linestyle="none", label="no valid result at this context")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.032), ncol=5, fontsize=11)
    write_table(4, "rag-context-scaling", table_rows)
    save_figure(
        fig,
        4,
        "rag-context-scaling",
        "4 · RAG Performance by Context Length",
        "Both panels in a column share one y-scale, so SQuAD and HotpotQA are directly comparable. "
        "A line stops at the last context that produced a valid result.",
    )


# --------------------------------------------------------------------------
# figure 5 -- prefix cache cold / warm
# --------------------------------------------------------------------------


def record_metric(record: dict[str, Any], name: str) -> float:
    return as_float(record.get("metrics", {}).get(name, {}).get("value"))


def prefix_role_ttft(cells: pd.DataFrame, dirs: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = cells[(cells["suite"] == "prefix") & (cells["scope"] == "comparison")]
    for _, cell in subset.iterrows():
        if not bool(cell["valid"]):
            continue
        cell_dir = dirs[(cell["backend"], cell["cell_id"])]
        valid_numbers: set[int] = set()
        repeat_csv = cell_dir / "repeat-metrics.csv"
        if repeat_csv.exists():
            rep = pd.read_csv(repeat_csv)
            valid_numbers = set(
                pd.to_numeric(
                    rep.loc[rep["valid"].astype(str).str.lower().eq("true"), "repeat"],
                    errors="coerce",
                ).dropna().astype(int)
            )
        values: dict[str, list[float]] = {"cold": [], "warm": []}
        for repeat_dir in sorted(cell_dir.glob("repeat-*")):
            try:
                number = int(repeat_dir.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if valid_numbers and number not in valid_numbers:
                continue
            path = repeat_dir / "profile_export.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    meta = record.get("metadata", {})
                    if meta.get("benchmark_phase") != "profiling":
                        continue
                    role = "cold" if meta.get("turn_index") == 0 else "warm"
                    value = record_metric(record, "time_to_first_token")
                    if math.isfinite(value):
                        values[role].append(value)
        for role, role_values in values.items():
            rows.append(
                {
                    "backend": cell["backend"],
                    "bucket_tokens": int(cell["bucket_tokens"]),
                    "concurrency": int(cell["concurrency"]),
                    "cache_mode": cell["cache_mode"],
                    "role": role,
                    "ttft_median": float(np.median(role_values)) if role_values else math.nan,
                    "samples": len(role_values),
                }
            )
    return pd.DataFrame(rows)


def figure_05_prefix(cells: pd.DataFrame, dirs: dict[tuple[str, str], Path]) -> None:
    roles = prefix_role_ttft(cells, dirs)
    categories = [("llama.cpp", 1), ("llama.cpp", 8), ("vLLM", 1), ("vLLM", 8)]
    labels = [f"{b}\nC{c}" for b, c in categories]
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.135, hspace=0.38, wspace=0.10)
    table_rows: list[dict[str, Any]] = []

    ceiling = float(np.nanmax(roles["ttft_median"])) if not roles.empty else 1.0
    for ax, (bucket, role) in zip(
        axes.flat, [(4096, "cold"), (4096, "warm"), (16384, "cold"), (16384, "warm")]
    ):
        x = np.arange(len(categories))
        by_mode: dict[str, list[float]] = {}
        for offset, mode, color in [(-width / 2 - 0.012, "off", C["cache_off"]),
                                    (width / 2 + 0.012, "on", C["cache_on"])]:
            vals: list[float] = []
            for backend, concurrency in categories:
                part = roles[(roles["backend"] == backend) & (roles["concurrency"] == concurrency)
                             & (roles["bucket_tokens"] == bucket) & (roles["role"] == role)
                             & (roles["cache_mode"] == mode)]
                vals.append(float(part["ttft_median"].iloc[0]) if not part.empty else math.nan)
            by_mode[mode] = vals
            bars = ax.bar(x + offset, np.nan_to_num(vals, nan=0.0), width, color=color, zorder=3)
            for idx, (bar, value) in enumerate(zip(bars, vals)):
                if not math.isfinite(value):
                    bar.set_visible(False)
                    continue
                text = f"{value:,.0f} ms"
                if mode == "on" and math.isfinite(by_mode["off"][idx]) and by_mode["off"][idx] > 0:
                    change = 100 * (value - by_mode["off"][idx]) / by_mode["off"][idx]
                    if abs(change) < 0.5:
                        text += "\nno change"
                    elif change < 0:
                        text += f"\n{change:.0f}% faster"
                    else:
                        text += f"\n+{change:.0f}% slower"
                ax.text(bar.get_x() + bar.get_width() / 2, value + ceiling * 0.012, text,
                        ha="center", va="bottom", fontsize=8.5, color=C["ink2"],
                        linespacing=1.35, zorder=5)
                table_rows.append(
                    {
                        "context_bucket": f"{bucket//1024}K",
                        "role": role,
                        "backend": categories[idx][0],
                        "concurrency": categories[idx][1],
                        "cache_mode": mode,
                        "median_ttft_ms": value,
                    }
                )

        notes: list[str] = []
        for idx, (backend, concurrency) in enumerate(categories):
            missing = [m for m in ("off", "on") if not math.isfinite(by_mode[m][idx])]
            for mode in missing:
                failed = cells[(cells["suite"] == "prefix") & (cells["scope"] == "comparison")
                               & (cells["backend"] == backend) & (cells["concurrency"] == concurrency)
                               & (cells["bucket_tokens"] == bucket) & (cells["cache_mode"] == mode)]
                reason = invalid_reason(failed.iloc[0]) if not failed.empty else "not tested"
                notes.append(
                    f"{GLYPH['invalid']} {backend} C{concurrency} cache {mode}: "
                    f"no valid result — {reason}"
                )
            if missing:
                ax.plot([x[idx]], [0], marker="x", markersize=11, markeredgewidth=2.4,
                        color=C["invalid"], clip_on=False,
                        transform=ax.get_xaxis_transform(), zorder=7)

        slo_rule(ax, 2000.0)
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_ylim(0, ceiling * 1.16)
        ax.set_ylabel("Median TTFT (ms)")
        role_text = "Cold — prefix seen for the first time" if role == "cold" else \
                    "Warm — prefix already in the cache"
        ax.set_title(f"{bucket//1024}K context · {role_text}")
        ax.text(0.0, 1.006, "lower is better · RAG SLO ≤ 2,000 ms (dashed)",
                transform=ax.transAxes, fontsize=9.5, color=C["muted"], ha="left", va="bottom")
        ax.grid(axis="x", visible=False)
        # Bar tops (and the threshold) are what the note must not cover.
        occupied = [(pos, value) for mode in ("off", "on")
                    for pos, value in zip(x, by_mode[mode]) if math.isfinite(value)]
        occupied += [(pos, 2000.0) for pos in x]
        note_block(ax, sorted(set(notes)), C["invalid"], occupied)

    fig.legend(
        handles=[
            Patch(facecolor=C["cache_off"], label="Prefix cache off"),
            Patch(facecolor=C["cache_on"], label="Prefix cache on"),
            Line2D([0], [0], color=C["invalid"], marker="x", markersize=10, markeredgewidth=2.4,
                   linestyle="none", label="no valid result for this bar pair"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.038),
        ncol=3,
        fontsize=11,
    )
    write_table(5, "prefix-cache-cold-warm", table_rows)
    save_figure(
        fig,
        5,
        "prefix-cache-cold-warm",
        "5 · Prefix Cache TTFT — Cache On vs Off",
        "All four panels share one y-scale. The percentage under each blue bar is cache-on measured "
        "against the cache-off bar beside it.",
    )


# --------------------------------------------------------------------------
# figure 6 -- RULER
# --------------------------------------------------------------------------


def figure_06_ruler(cells: pd.DataFrame, repeats: pd.DataFrame) -> None:
    c = cells[(cells["suite"] == "ruler_performance") & (cells["scope"] == "comparison")].copy()
    buckets = [4096, 8192, 16384, 32768, 61440]
    pos = {value: idx for idx, value in enumerate(buckets)}
    labels = ["4K", "8K", "16K", "32K", "60K"]

    panels = [
        ("output_tps", "Output throughput", "tokens/s", None, 0, "higher is better"),
        ("ttft_p95_ms", "P95 time to first token", "ms", 5000.0, 0,
         "lower is better · long-context SLO ≤ 5,000 ms"),
        ("itl_p95_ms", "P95 inter-token latency", "ms per token", 75.0, 1,
         "lower is better · long-context SLO ≤ 75 ms"),
        ("good_pct", "Good request fraction", "% of requests", 95.0, 1,
         "higher is better · SLO ≥ 95%"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.115, hspace=0.44, wspace=0.20)
    table_rows: list[dict[str, Any]] = []

    for ax, (metric, title, unit, slo, digits, hint) in zip(axes.flat, panels):
        ax.axvspan(3.5, 4.45, color=C["band"], zorder=0)
        ax.text(4.0, 1.006, "capacity boundary", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=C["muted"])
        drawn: list[tuple[str, list[float], list[float]]] = []
        notes: list[str] = []
        for backend in BACKENDS:
            for concurrency, style in [(1, "-"), (2, (0, (5, 3)))]:
                sel = c[(c["backend"] == backend) & (c["concurrency"] == concurrency)]
                ok = sel[sel["valid"]].sort_values("bucket_tokens")
                xs = [pos[int(v)] for v in ok["bucket_tokens"]]
                ys = [as_float(v) for v in ok[metric]]
                ax.plot(xs, ys, color=C[backend], marker=MARKERS[backend], markersize=8,
                        linestyle=style, linewidth=2.2, markeredgecolor=C["surface"],
                        markeredgewidth=1.6, zorder=4)
                drawn.append((C[backend], xs, ys))
                for _, bad in sel[~sel["valid"]].sort_values("bucket_tokens").iterrows():
                    x = pos[int(bad["bucket_tokens"])]
                    ax.plot([x], [0], marker="x", markersize=11, markeredgewidth=2.4,
                            color=C["invalid"], clip_on=False,
                            transform=ax.get_xaxis_transform(), zorder=7)
                    notes.append(
                        f"{GLYPH['invalid']} {backend} C{concurrency} @ {labels[x]}: "
                        f"no valid result — {invalid_reason(bad)}"
                    )
        if slo is not None:
            slo_rule(ax, slo)
        if metric == "good_pct":
            ax.set_ylim(-4, 108)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.06)
        ax.set_xticks(range(len(buckets)), labels)
        ax.set_xlim(-0.35, len(buckets) - 0.55)
        ax.set_xlabel("Target input context bucket")
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.text(0.0, 1.006, hint, transform=ax.transAxes, fontsize=9.5, color=C["muted"],
                ha="left", va="bottom")
        ax.grid(axis="x", visible=False)
        label_all(ax, drawn, digits)
        note_block(ax, sorted(set(notes)), C["invalid"],
                   [(x, y) for _, xs, ys in drawn for x, y in zip(xs, ys)])

    for _, row in c.sort_values(["backend", "concurrency", "bucket_tokens"]).iterrows():
        table_rows.append(
            {
                "backend": row["backend"],
                "concurrency": int(row["concurrency"]),
                "context_bucket": f"{int(row['bucket_tokens'])//1024}K",
                "valid": bool(row["valid"]),
                "output_tokens_per_s": as_float(row["output_tps"]),
                "p95_ttft_ms": as_float(row["ttft_p95_ms"]),
                "p95_itl_ms": as_float(row["itl_p95_ms"]),
                "good_requests_pct": as_float(row["good_pct"]),
                "reason": "" if bool(row["valid"]) else invalid_reason(row),
            }
        )

    handles = []
    for backend in BACKENDS:
        for concurrency, style in [(1, "-"), (2, (0, (5, 3)))]:
            handles.append(
                Line2D([0], [0], color=C[backend], marker=MARKERS[backend], markersize=9,
                       linestyle=style, linewidth=2.2, label=f"{backend} C{concurrency}")
            )
    handles.append(
        Line2D([0], [0], color=C["invalid"], marker="x", markersize=10, markeredgewidth=2.4,
               linestyle="none", label="no valid result at this context")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.032), ncol=5, fontsize=11)
    write_table(6, "ruler-long-context", table_rows)
    save_figure(
        fig,
        6,
        "ruler-long-context",
        "6 · RULER Long-context Performance",
        "4K to 60K synthetic long-context retrieval. The shaded 60K column is the capacity-boundary "
        "probe, not a routine operating point.",
    )


# --------------------------------------------------------------------------
# figure 7 -- vLLM stress capacity
# --------------------------------------------------------------------------


def stress_group(row: pd.Series) -> str:
    if row["suite"] == "sharegpt_closed":
        return "ShareGPT closed-loop"
    if row["suite"] == "sharegpt_open":
        return "ShareGPT open-loop"
    if row["suite"] == "sharegpt_natural":
        return "ShareGPT natural output"
    if row["suite"] == "rag":
        return "RAG · HotpotQA" if "hotpotqa" in row["cell_id"] else "RAG · SQuAD"
    return "RULER long context"


def stress_label(row: pd.Series) -> str:
    if row["suite"] == "sharegpt_closed":
        return f"C{int(row['concurrency'])}"
    if row["suite"] in {"sharegpt_open", "sharegpt_natural"}:
        return f"{row['request_rate']:g} r/s · C{int(row['concurrency'])}"
    return f"{int(row['bucket_tokens'])//1024}K · C{int(row['concurrency'])}"


def figure_07_stress(cells: pd.DataFrame) -> None:
    stress = cells[(cells["backend"] == "vLLM") & (cells["scope"] == "vllm_stress")
                   & (cells["suite"] != "prefix")].copy()
    rank = {"sharegpt_closed": 0, "sharegpt_open": 1, "sharegpt_natural": 2, "rag": 3,
            "ruler_performance": 4}
    stress["suite_rank"] = stress["suite"].map(rank)
    stress["dataset_rank"] = np.where(stress["cell_id"].str.contains("hotpotqa"), 1, 0)
    stress = stress.sort_values(
        ["suite_rank", "dataset_rank", "bucket_tokens", "request_rate", "concurrency"],
        na_position="first",
    ).reset_index(drop=True)
    stress["group"] = stress.apply(stress_group, axis=1)
    stress["label"] = stress.apply(stress_label, axis=1)

    n = len(stress)
    y = np.arange(n)
    colors = [C["pass"] if ok else C["warn"] for ok in stress["slo_pass"]]

    height = 0.42 * n + 3.9
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=True)
    fig.subplots_adjust(left=0.245, right=0.985, top=1 - 2.05 / height,
                        bottom=1.15 / height, wspace=0.09)

    def bar_label(ax: plt.Axes, x: float, idx: int, text: str) -> None:
        ax.annotate(text, xy=(x, idx), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=C["ink2"], zorder=6,
                    bbox={"facecolor": C["surface"], "edgecolor": "none", "alpha": 0.8,
                          "pad": 1.0})

    # good requests
    axes[0].barh(y, stress["good_pct"], height=0.68, color=colors, zorder=3)
    axes[0].axvline(GOOD_FRACTION_PCT, color=C["ink2"], linestyle=(0, (6, 4)), linewidth=1.2, zorder=4)
    axes[0].set_xlim(0, 122)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_xlabel("Good requests (% of requests)")
    axes[0].set_title("Good requests · SLO ≥ 95% (dashed)")
    for idx, value in enumerate(stress["good_pct"]):
        bar_label(axes[0], value, idx, f"{value:.0f}%")

    # TTFT / SLO on a log scale so 0.2x and 15x are both readable
    ratio = stress["ttft_ratio"].to_numpy(float)
    axes[1].barh(y, np.clip(ratio, 0.05, None), height=0.68, left=0.05, color=colors, zorder=3)
    axes[1].axvline(1.0, color=C["ink2"], linestyle=(0, (6, 4)), linewidth=1.2, zorder=4)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.09, 90)
    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
    axes[1].xaxis.set_major_locator(FixedLocator(ticks))
    axes[1].xaxis.set_major_formatter(FixedFormatter(["0.1×", "0.25×", "0.5×", "1×",
                                                      "2×", "5×", "10×", "20×"]))
    axes[1].set_xlabel("P95 TTFT ÷ SLO (log scale)")
    axes[1].set_title("Latency headroom · ≤ 1× passes (dashed)")
    for idx, value in enumerate(ratio):
        bar_label(axes[1], max(value, 0.05), idx, f"{value:.1f}×")

    axes[2].barh(y, stress["output_tps"], height=0.68, color=C["vLLM"], zorder=3)
    top = float(stress["output_tps"].max())
    axes[2].set_xlim(0, top * 1.2)
    axes[2].set_xlabel("Output throughput (tokens/s)")
    axes[2].set_title("Output throughput · higher is better")
    for idx, value in enumerate(stress["output_tps"]):
        bar_label(axes[2], value, idx, f"{value:,.0f}")

    axes[0].set_yticks(y, stress["label"], fontsize=9.5)
    for ax in axes:
        ax.set_ylim(n - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)

    # Group bands, separators, and headings in the reserved left margin.
    bounds = [
        (group, int(part.index.min()), int(part.index.max()))
        for group, part in stress.groupby("group", sort=False)
    ]
    for shade, (group, lo, hi) in enumerate(bounds):
        if shade % 2 == 1:
            for ax in axes:
                ax.axhspan(lo - 0.5, hi + 0.5, color=C["band"], zorder=0)
        if lo > 0:
            for ax in axes:
                ax.axhline(lo - 0.5, color=C["axis"], linewidth=1.0, zorder=5)
        axes[0].annotate(
            group,
            xy=(0, (lo + hi) / 2),
            xytext=(-95, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=C["ink"],
            annotation_clip=False,
        )

    fig.legend(
        handles=[
            Patch(facecolor=C["pass"], label=f"{GLYPH['pass']}  SLO pass"),
            Patch(facecolor=C["warn"], label=f"{GLYPH['slo']}  valid, SLO fail"),
            Patch(facecolor=C["vLLM"], label="output throughput (magnitude, not a state)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.34 / height),
        ncol=3,
        fontsize=11,
    )
    write_table(
        7,
        "vllm-stress-capacity",
        [
            {
                "group": row["group"],
                "condition": row["label"],
                "cell_id": row["cell_id"],
                "good_requests_pct": as_float(row["good_pct"]),
                "p95_ttft_ms": as_float(row["ttft_p95_ms"]),
                "p95_ttft_slo_ms": as_float(row["ttft_slo_ms"]),
                "p95_ttft_over_slo": as_float(row["ttft_ratio"]),
                "output_tokens_per_s": as_float(row["output_tps"]),
                "slo_pass": bool(row["slo_pass"]),
                "reason": "" if bool(row["slo_pass"]) else slo_reason(row),
            }
            for _, row in stress.iterrows()
        ],
    )
    save_figure(
        fig,
        7,
        "vllm-stress-capacity",
        "7 · vLLM-only Stress Capacity",
        "The server ran with --max-num-seqs 16, so C32 and C64 are client-side load: the surplus "
        "queues. These cells measure queueing and overload, not 32 or 64 concurrent sequences.",
    )


# --------------------------------------------------------------------------
# figure 8 -- vLLM prefix cache stress (server metrics)
# --------------------------------------------------------------------------


def metric_stats(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    series = metrics.get(name, {}).get("series", [])
    return series[0].get("stats", {}) if series else {}


def load_prefix_server_metrics(
    cells: pd.DataFrame, dirs: dict[tuple[str, str], Path]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = cells[(cells["backend"] == "vLLM") & (cells["suite"] == "prefix")
                   & (cells["scope"] == "vllm_stress")]
    for _, cell in subset.iterrows():
        cell_dir = dirs[("vLLM", cell["cell_id"])]
        rep = pd.read_csv(cell_dir / "repeat-metrics.csv")
        valid_numbers = set(
            pd.to_numeric(
                rep.loc[rep["valid"].astype(str).str.lower().eq("true"), "repeat"], errors="coerce"
            ).dropna().astype(int)
        )
        for repeat_dir in sorted(cell_dir.glob("repeat-*")):
            try:
                number = int(repeat_dir.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if number not in valid_numbers:
                continue
            path = repeat_dir / "server_metrics_export.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            prompt = as_float(metric_stats(metrics, "vllm:prompt_tokens").get("rate"))
            cached = as_float(metric_stats(metrics, "vllm:prompt_tokens_cached").get("rate"))
            generation = as_float(metric_stats(metrics, "vllm:generation_tokens").get("rate"))
            running = metric_stats(metrics, "vllm:num_requests_running")
            waiting = metric_stats(metrics, "vllm:num_requests_waiting")
            kv = metric_stats(metrics, "vllm:kv_cache_usage_perc")
            hits = as_float(metric_stats(metrics, "vllm:prefix_cache_hits").get("total"))
            queries = as_float(metric_stats(metrics, "vllm:prefix_cache_queries").get("total"))
            rows.append(
                {
                    "bucket_tokens": int(cell["bucket_tokens"]),
                    "cache_mode": cell["cache_mode"],
                    "repeat": number,
                    "cached_tps": cached if math.isfinite(cached) else 0.0,
                    "computed_tps": (
                        max(prompt - cached, 0.0)
                        if math.isfinite(prompt) and math.isfinite(cached)
                        else prompt
                    ),
                    "generation_tps": generation,
                    "running_avg": as_float(running.get("avg")),
                    "running_max": as_float(running.get("max")),
                    "waiting_avg": as_float(waiting.get("avg")),
                    "waiting_max": as_float(waiting.get("max")),
                    "kv_avg_pct": 100 * as_float(kv.get("avg")),
                    "kv_max_pct": 100 * as_float(kv.get("max")),
                    "hit_rate_pct": 100 * hits / queries if queries > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def figure_08_prefix_stress(cells: pd.DataFrame, dirs: dict[tuple[str, str], Path]) -> None:
    raw = load_prefix_server_metrics(cells, dirs)
    order = [(4096, "off"), (4096, "on"), (16384, "off"), (16384, "on")]
    labels = ["4K\ncache off", "4K\ncache on", "16K\ncache off", "16K\ncache on"]
    grouped: list[dict[str, Any]] = []
    for bucket, mode in order:
        group = raw[(raw["bucket_tokens"] == bucket) & (raw["cache_mode"] == mode)]
        item: dict[str, Any] = {"bucket_tokens": bucket, "cache_mode": mode}
        for column in ["cached_tps", "computed_tps", "generation_tps", "running_avg",
                       "waiting_avg", "kv_avg_pct", "hit_rate_pct"]:
            item[column] = float(group[column].mean())
        for column in ["running_max", "waiting_max", "kv_max_pct"]:
            item[column] = float(group[column].max())
        grouped.append(item)
    data = pd.DataFrame(grouped)
    x = np.arange(len(data))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.845, bottom=0.115, hspace=0.44, wspace=0.22)

    ax = axes[0, 0]
    stack_top = float(data[["computed_tps", "cached_tps"]].sum(axis=1).max())
    gap = stack_top * 0.004
    ax.bar(x, data["computed_tps"], 0.6, color=C["computed"], zorder=3)
    ax.bar(x, data["cached_tps"], 0.6, bottom=data["computed_tps"] + gap, color=C["cached"], zorder=3)
    ax.set_ylim(0, stack_top * 1.20)
    for idx, row in data.iterrows():
        total = row["computed_tps"] + row["cached_tps"]
        ax.text(idx, total + stack_top * 0.015, f"{total:,.0f} total", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C["ink"])
        if row["computed_tps"] > stack_top * 0.10:
            ax.text(idx, row["computed_tps"] / 2, f"{row['computed_tps']:,.0f}\ncomputed",
                    ha="center", va="center", fontsize=8.5, color="#ffffff", linespacing=1.3)
        if row["cached_tps"] > stack_top * 0.10:
            ax.text(idx, row["computed_tps"] + row["cached_tps"] / 2,
                    f"{row['cached_tps']:,.0f}\nreused", ha="center", va="center", fontsize=8.5,
                    color="#ffffff", linespacing=1.3)
    ax.set_title("Prompt processing · computed + cache reuse")
    ax.set_ylabel("tokens/s")
    ax.text(0.0, 1.006, "bar top is logical prompt throughput, not GPU work",
            transform=ax.transAxes, fontsize=9.5, color=C["muted"], ha="left", va="bottom")
    ax.legend(handles=[Patch(facecolor=C["computed"], label="newly computed on GPU"),
                       Patch(facecolor=C["cached"], label="served from prefix cache")],
              fontsize=9.5, loc="upper right")

    ax = axes[0, 1]
    ax.bar(x, data["generation_tps"], 0.6, color=C["vLLM"], zorder=3)
    top = float(data["generation_tps"].max())
    ax.set_ylim(0, top * 1.18)
    for idx, value in enumerate(data["generation_tps"]):
        ax.text(idx, value + top * 0.015, f"{value:,.1f}", ha="center", va="bottom", fontsize=9.5,
                color=C["ink2"])
    ax.set_title("Generation throughput")
    ax.set_ylabel("tokens/s")
    ax.text(0.0, 1.006, "real decode work · higher is better", transform=ax.transAxes,
            fontsize=9.5, color=C["muted"], ha="left", va="bottom")

    for ax, avg_col, max_col, title, unit, hint in [
        (axes[0, 2], "running_avg", "running_max", "Running requests", "requests",
         "server cap is max-num-seqs = 16"),
        (axes[1, 0], "waiting_avg", "waiting_max", "Waiting requests (queue depth)", "requests",
         "lower is better · client sends 16 at once"),
        (axes[1, 1], "kv_avg_pct", "kv_max_pct", "GPU KV cache usage", "% of KV blocks",
         "near 100% means the KV pool is the limit"),
    ]:
        avg = data[avg_col].to_numpy(float)
        peak = data[max_col].to_numpy(float)
        ax.bar(x, avg, 0.6, color=C["vLLM"], zorder=3)
        ax.errorbar(x, avg, yerr=np.vstack([np.zeros(len(avg)), np.maximum(peak - avg, 0)]),
                    fmt="none", ecolor=C["ink2"], elinewidth=1.4, capsize=7, capthick=1.4, zorder=4)
        headroom = max(float(np.nanmax(peak)), 1.0)
        ax.set_ylim(0, headroom * 1.26)
        for idx, (mean_value, max_value) in enumerate(zip(avg, peak)):
            suffix = "%" if "%" in unit else ""
            ax.text(idx, max_value + headroom * 0.03,
                    f"max {max_value:.1f}{suffix}", ha="center", va="bottom", fontsize=8.5,
                    color=C["ink2"])
            # A near-zero mean cannot hold white text inside its own bar.
            if mean_value > headroom * 0.14:
                ax.text(idx, mean_value / 2, f"mean\n{mean_value:.1f}{suffix}", ha="center",
                        va="center", fontsize=8.5, color="#ffffff", linespacing=1.3)
            else:
                ax.annotate(f"mean {mean_value:.1f}{suffix}", xy=(idx, mean_value),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8.5, color=C["ink2"])
        if "KV" in title:
            ax.axhline(100, color=C["ink2"], linestyle=(0, (6, 4)), linewidth=1.2)
        ax.set_title(f"{title} · bar = mean, whisker = max")
        ax.set_ylabel(unit)
        ax.text(0.0, 1.006, hint, transform=ax.transAxes, fontsize=9.5, color=C["muted"],
                ha="left", va="bottom")

    ax = axes[1, 2]
    hit = data["hit_rate_pct"].to_numpy(float)
    ax.bar(x, np.nan_to_num(hit, nan=0.0), 0.6,
           color=[C["missing"] if mode == "off" else C["cached"] for mode in data["cache_mode"]],
           zorder=3)
    for idx, (mode, value) in enumerate(zip(data["cache_mode"], hit)):
        if mode == "off":
            ax.text(idx, 3, "cache disabled\n— no queries", ha="center", va="bottom",
                    fontsize=9, color=C["ink2"], linespacing=1.35)
        elif math.isfinite(value) and value < 1:
            ax.text(idx, 3, f"{value:.1f}%\n— no reuse", ha="center",
                    va="bottom", fontsize=9, color=C["ink2"], linespacing=1.35)
        else:
            ax.text(idx, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9.5,
                    color=C["ink2"])
    ax.set_ylim(0, 112)
    ax.set_title("Prefix cache hit rate")
    ax.set_ylabel("% of queried prompt tokens")
    ax.text(0.0, 1.006, "higher is better", transform=ax.transAxes, fontsize=9.5,
            color=C["muted"], ha="left", va="bottom")

    for ax in axes.flat:
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.grid(axis="x", visible=False)

    write_table(
        8,
        "vllm-prefix-stress",
        [
            {
                "context_bucket": f"{int(row['bucket_tokens'])//1024}K",
                "cache_mode": row["cache_mode"],
                "prompt_computed_tokens_per_s": row["computed_tps"],
                "prompt_reused_tokens_per_s": row["cached_tps"],
                "generation_tokens_per_s": row["generation_tps"],
                "running_mean": row["running_avg"],
                "running_max": row["running_max"],
                "waiting_mean": row["waiting_avg"],
                "waiting_max": row["waiting_max"],
                "kv_usage_mean_pct": row["kv_avg_pct"],
                "kv_usage_max_pct": row["kv_max_pct"],
                "prefix_hit_rate_pct": row["hit_rate_pct"],
            }
            for _, row in data.iterrows()
        ],
    )
    save_figure(
        fig,
        8,
        "vllm-prefix-stress",
        "8 · vLLM Prefix Cache Stress — Server-side View at Client C16",
        "Read from vLLM's own metrics endpoint. The first panel's bar top includes reused tokens, so "
        "it is not the token rate the GPU actually recomputed.",
    )


# --------------------------------------------------------------------------


def main() -> None:
    setup_style()
    cells, repeats, dirs = load_formal_data()
    if cells.empty:
        raise SystemExit("No formal dataset results found under results/*/dataset/formal-*")
    figure_01_status(cells)
    figure_02_closed(cells, repeats)
    figure_03_open(cells, repeats)
    figure_04_rag(cells)
    figure_05_prefix(cells, dirs)
    figure_06_ruler(cells, repeats)
    figure_07_stress(cells)
    figure_08_prefix_stress(cells, dirs)
    print(f"Wrote 8 PNG + 8 SVG figures to {OUT}")
    print(f"Wrote 8 companion CSV tables to {TABLES}")


if __name__ == "__main__":
    main()
