#!/usr/bin/env python3
"""Evaluate a query-count-aware runtime selector for H4.8."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VARIANT_ROWS = ROOT / "data" / "h4_7_heldout_variant_rows.csv"
BASELINE_DECISIONS = ROOT / "data" / "h4_7_heldout_selector_decisions.csv"
OUT_DIR = ROOT / "experiments" / "h4-8-query-aware-selector" / "results"
SHARED_LIMIT = 101376
THRESHOLD = 0.15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[int, str], list[dict[str, str]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(as_int(row["query_count"]), row["order_mode"])].append(row)
    return grouped


def best_for_case(rows: list[dict[str, str]]) -> tuple[str, float]:
    valid = [row for row in rows if row["status"] == "ok" and row["static_feasible"] == "True"]
    winner = min(valid, key=lambda row: as_float(row["triton_grouped_ms"]))
    return winner["variant_id"], as_float(winner["triton_grouped_ms"])


def latency_lookup(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        row["variant_id"]: as_float(row["triton_grouped_ms"])
        for row in rows
        if row["status"] == "ok" and row["static_feasible"] == "True"
    }


def order_stats_from_baseline() -> dict[tuple[int, str], dict[str, float]]:
    stats = {}
    for row in read_csv(BASELINE_DECISIONS):
        key = (as_int(row["query_count"]), row["order_mode"])
        stats[key] = {
            "order_span_mean": as_float(row["order_span_mean"]),
            "order_span_p95": as_float(row["order_span_p95"]),
            "monotonic_fraction": as_float(row["monotonic_fraction"]),
        }
    return stats


def h4_6_order_rule(stats: dict[str, float]) -> str:
    span_mean = stats["order_span_mean"]
    span_p95 = stats["order_span_p95"]
    monotonic = stats["monotonic_fraction"]
    if monotonic < 0.75 and span_mean > 512:
        return "N128 D64 V64 Q4 W8"
    if monotonic < 0.75:
        return "N64 D64 V64 Q4 W4"
    if span_p95 <= 4:
        return "N128 D64 V128 Q4 W4"
    return "N128 D128 V64 Q4 W4"


def h4_8_query_aware_rule(query_count: int, stats: dict[str, float]) -> str:
    span_mean = stats["order_span_mean"]
    span_p95 = stats["order_span_p95"]
    monotonic = stats["monotonic_fraction"]
    high_span = monotonic < 0.75 and span_mean > 512
    random_segment = monotonic < 0.75 and span_mean <= 512
    clustered = monotonic >= 0.75 and span_p95 <= 4
    sorted_order = monotonic >= 0.75 and span_p95 > 4

    if query_count <= 2:
        if clustered:
            return "N128 D128 V64 Q4 W4"
        return "N128 D64 V128 Q4 W4"

    if query_count >= 8:
        if clustered or sorted_order:
            return "N128 D64 V64 Q4 W8"
        if high_span:
            return "N128 D64 V128 Q4 W4"
        if random_segment:
            return "N128 D64 V64 Q4 W4"

    return h4_6_order_rule(stats)


def evaluate() -> tuple[list[dict], list[dict], dict]:
    variant_rows = read_csv(VARIANT_ROWS)
    grouped = group_rows(variant_rows)
    stats_by_case = order_stats_from_baseline()
    decisions = []

    for key, rows in sorted(grouped.items()):
        query_count, order_mode = key
        stats = stats_by_case[key]
        best_variant, best_latency = best_for_case(rows)
        lookup = latency_lookup(rows)
        selected = h4_8_query_aware_rule(query_count, stats)
        used_fallback = False
        if selected not in lookup:
            selected = min(lookup, key=lookup.get)
            used_fallback = True
        latency = lookup[selected]
        regret = latency / best_latency - 1.0
        block_q = 4
        decisions.append(
            {
                "selector": "query_count_aware_h4_8",
                "query_count": query_count,
                "block_q": block_q,
                "num_q_blocks": math.ceil(query_count / block_q),
                "query_pressure": query_count / block_q,
                "order_mode": order_mode,
                "selected_variant": selected,
                "used_fallback": used_fallback,
                "selected_latency_ms": latency,
                "best_variant": best_variant,
                "best_latency_ms": best_latency,
                "regret_pct": regret * 100.0,
                "within_15pct": regret <= THRESHOLD,
                **stats,
            }
        )

    baseline = [
        row
        for row in read_csv(BASELINE_DECISIONS)
        if row["selector"] == "order_aware_rule_table_h4_6"
    ]
    comparison_rows = []
    baseline_by_case = {(as_int(row["query_count"]), row["order_mode"]): row for row in baseline}
    for row in decisions:
        key = (row["query_count"], row["order_mode"])
        base = baseline_by_case[key]
        comparison_rows.append(
            {
                "query_count": row["query_count"],
                "order_mode": row["order_mode"],
                "h4_6_selected_variant": base["selected_variant"],
                "h4_6_regret_pct": as_float(base["regret_pct"]),
                "h4_8_selected_variant": row["selected_variant"],
                "h4_8_regret_pct": row["regret_pct"],
                "regret_improvement_pct": as_float(base["regret_pct"]) - row["regret_pct"],
            }
        )

    summary = {
        "selector": "query_count_aware_h4_8",
        "cases": len(decisions),
        "near_best_count": sum(1 for row in decisions if row["within_15pct"]),
        "mean_regret_pct": sum(row["regret_pct"] for row in decisions) / len(decisions),
        "max_regret_pct": max(row["regret_pct"] for row in decisions),
        "fallback_count": sum(1 for row in decisions if row["used_fallback"]),
        "q8_clustered_fixed": any(
            row["query_count"] == 8
            and row["order_mode"] == "clustered_segment"
            and row["selected_variant"] == "N128 D64 V64 Q4 W8"
            and row["within_15pct"]
            for row in decisions
        ),
    }
    result = {
        "status": "ok",
        "threshold": THRESHOLD,
        "baseline_order_aware_mean_regret_pct": sum(as_float(row["regret_pct"]) for row in baseline) / len(baseline),
        "baseline_order_aware_near_best_count": sum(row["within_15pct"] == "True" for row in baseline),
        "selector_summary": summary,
    }
    return decisions, comparison_rows, result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    decisions, comparison_rows, result = evaluate()
    write_csv(
        ROOT / "data" / "h4_8_query_aware_decisions.csv",
        decisions,
        [
            "selector",
            "query_count",
            "block_q",
            "num_q_blocks",
            "query_pressure",
            "order_mode",
            "selected_variant",
            "used_fallback",
            "selected_latency_ms",
            "best_variant",
            "best_latency_ms",
            "regret_pct",
            "within_15pct",
            "order_span_mean",
            "order_span_p95",
            "monotonic_fraction",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_8_query_aware_comparison.csv",
        comparison_rows,
        [
            "query_count",
            "order_mode",
            "h4_6_selected_variant",
            "h4_6_regret_pct",
            "h4_8_selected_variant",
            "h4_8_regret_pct",
            "regret_improvement_pct",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_8_query_aware_summary.csv",
        [result["selector_summary"]],
        [
            "selector",
            "cases",
            "near_best_count",
            "mean_regret_pct",
            "max_regret_pct",
            "fallback_count",
            "q8_clustered_fixed",
        ],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

