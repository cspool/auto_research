#!/usr/bin/env python3
"""Evaluate runtime variant-table selectors for H4.6."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = ROOT / "data" / "h4_5_kv_multi_query_summary.csv"
DEFAULT_VARIANTS = ROOT / "data" / "h4_5_kv_multi_query_variants.csv"
DEFAULT_OUT_DIR = ROOT / "experiments" / "h4-6-runtime-variant-table" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--variants-csv", default=str(DEFAULT_VARIANTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--near-best-threshold", type=float, default=0.10)
    return parser.parse_args()


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


def variant_id(row: dict[str, str]) -> str:
    return (
        f"N{as_int(row['block_n'])} D{as_int(row['block_d'])} "
        f"V{as_int(row['block_v'])} Q{as_int(row['block_q'])} W{as_int(row['warps'])}"
    )


def parse_shared_limits(rows: list[dict[str, str]]) -> tuple[int | None, dict[str, int]]:
    required_by_variant: dict[str, int] = {}
    limits = []
    for row in rows:
        error = row.get("error", "")
        required_match = re.search(r"Required:\s*(\d+)", error)
        limit_match = re.search(r"Hardware limit:\s*(\d+)", error)
        if required_match:
            required_by_variant[variant_id(row)] = int(required_match.group(1))
        if limit_match:
            limits.append(int(limit_match.group(1)))
    return (min(limits) if limits else None), required_by_variant


def shared_proxy(row: dict[str, str]) -> int:
    block_n = as_int(row["block_n"])
    block_d = as_int(row["block_d"])
    block_v = as_int(row["block_v"])
    block_q = as_int(row["block_q"])
    return block_n * block_q * (block_d + block_v)


def make_variant_table(rows: list[dict[str, str]], shared_limit: int | None) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[variant_id(row)].append(row)

    table = []
    for vid, group in sorted(grouped.items()):
        first = group[0]
        latencies = [as_float(row.get("triton_grouped_ms")) for row in group if row.get("status") == "ok"]
        failures = [row for row in group if row.get("status") != "ok"]
        proxy = shared_proxy(first)
        table.append(
            {
                "variant_id": vid,
                "block_n": as_int(first["block_n"]),
                "block_d": as_int(first["block_d"]),
                "block_v": as_int(first["block_v"]),
                "block_q": as_int(first["block_q"]),
                "warps": as_int(first["warps"]),
                "shared_proxy": proxy,
                "shared_limit": shared_limit,
                "static_feasible": bool(shared_limit is None or proxy <= shared_limit),
                "observed_failures": len(failures),
                "observed_ok": len(latencies),
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else math.nan,
                "min_latency_ms": min(latencies) if latencies else math.nan,
                "max_latency_ms": max(latencies) if latencies else math.nan,
            }
        )
    return table


def summary_by_order(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result = {}
    for row in summary_rows:
        result[row["order_mode"]] = {
            "order_span_mean": as_float(row["order_span_mean"]),
            "order_span_p95": as_float(row["order_span_p95"]),
            "monotonic_fraction": as_float(row["monotonic_fraction"]),
        }
    return result


def rows_by_order(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["order_mode"]].append(row)
    return grouped


def latency_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    lookup = {}
    for row in rows:
        if row.get("status") == "ok":
            lookup[(row["order_mode"], variant_id(row))] = as_float(row["triton_grouped_ms"])
    return lookup


def best_by_order(rows_by_mode: dict[str, list[dict[str, str]]]) -> dict[str, tuple[str, float]]:
    best = {}
    for order_mode, rows in rows_by_mode.items():
        valid = [row for row in rows if row.get("status") == "ok"]
        winner = min(valid, key=lambda row: as_float(row["triton_grouped_ms"]))
        best[order_mode] = (variant_id(winner), as_float(winner["triton_grouped_ms"]))
    return best


def global_mean_selector(variant_table: list[dict]) -> str:
    feasible = [
        item
        for item in variant_table
        if item["static_feasible"] and item["observed_ok"] > 0 and math.isfinite(item["mean_latency_ms"])
    ]
    return min(feasible, key=lambda item: item["mean_latency_ms"])["variant_id"]


def order_aware_selector(stats: dict[str, float]) -> str:
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


def fallback_if_needed(
    selected: str,
    order_mode: str,
    lookup: dict[tuple[str, str], float],
    static_feasible: set[str],
    fallback: str,
) -> tuple[str, bool]:
    if selected in static_feasible and (order_mode, selected) in lookup:
        return selected, False
    if fallback in static_feasible and (order_mode, fallback) in lookup:
        return fallback, True
    candidates = [vid for (order, vid), _ in lookup.items() if order == order_mode and vid in static_feasible]
    return min(candidates, key=lambda vid: lookup[(order_mode, vid)]), True


def evaluate_selectors(
    summary_rows: list[dict[str, str]],
    variant_rows: list[dict[str, str]],
    variant_table: list[dict],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    stats_by_order = summary_by_order(summary_rows)
    grouped = rows_by_order(variant_rows)
    lookup = latency_lookup(variant_rows)
    best = best_by_order(grouped)
    static_feasible = {item["variant_id"] for item in variant_table if item["static_feasible"]}
    global_selected = global_mean_selector(variant_table)

    decisions = []
    for order_mode in sorted(grouped):
        selectors = {
            "global_mean_latency": global_selected,
            "order_aware_rule_table": order_aware_selector(stats_by_order[order_mode]),
        }
        best_variant, best_latency = best[order_mode]
        for selector_name, selected in selectors.items():
            selected, used_fallback = fallback_if_needed(selected, order_mode, lookup, static_feasible, global_selected)
            selected_latency = lookup[(order_mode, selected)]
            regret = selected_latency / best_latency - 1.0
            decisions.append(
                {
                    "selector": selector_name,
                    "order_mode": order_mode,
                    "selected_variant": selected,
                    "used_fallback": used_fallback,
                    "selected_latency_ms": selected_latency,
                    "best_variant": best_variant,
                    "best_latency_ms": best_latency,
                    "regret_pct": regret * 100.0,
                    "within_threshold": regret <= threshold,
                    **stats_by_order[order_mode],
                }
            )

    summary = []
    for selector_name in sorted({row["selector"] for row in decisions}):
        rows = [row for row in decisions if row["selector"] == selector_name]
        summary.append(
            {
                "selector": selector_name,
                "orders": len(rows),
                "near_best_count": sum(1 for row in rows if row["within_threshold"]),
                "mean_regret_pct": sum(row["regret_pct"] for row in rows) / len(rows),
                "max_regret_pct": max(row["regret_pct"] for row in rows),
                "fallback_count": sum(1 for row in rows if row["used_fallback"]),
                "invalid_selected_count": 0,
            }
        )
    return decisions, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run(args: argparse.Namespace) -> dict:
    summary_rows = read_csv(Path(args.summary_csv))
    variant_rows = read_csv(Path(args.variants_csv))
    shared_limit, required_by_variant = parse_shared_limits(variant_rows)
    variant_table = make_variant_table(variant_rows, shared_limit)
    decisions, selector_summary = evaluate_selectors(
        summary_rows,
        variant_rows,
        variant_table,
        args.near_best_threshold,
    )

    failed_rows = [row for row in variant_rows if row.get("status") != "ok"]
    rejected_failed = [
        row
        for row in failed_rows
        if shared_proxy(row) > (shared_limit if shared_limit is not None else math.inf)
    ]

    result = {
        "status": "ok",
        "summary_csv": str(Path(args.summary_csv).resolve()),
        "variants_csv": str(Path(args.variants_csv).resolve()),
        "near_best_threshold": args.near_best_threshold,
        "shared_limit": shared_limit,
        "required_shared_bytes_by_failed_variant": required_by_variant,
        "observed_failed_variants": len(failed_rows),
        "static_rejected_failed_variants": len(rejected_failed),
        "static_rejected_failed_fraction": len(rejected_failed) / len(failed_rows) if failed_rows else 1.0,
        "selector_summary": selector_summary,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        ROOT / "data" / "h4_6_runtime_variant_table.csv",
        variant_table,
        [
            "variant_id",
            "block_n",
            "block_d",
            "block_v",
            "block_q",
            "warps",
            "shared_proxy",
            "shared_limit",
            "static_feasible",
            "observed_failures",
            "observed_ok",
            "mean_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_6_runtime_variant_decisions.csv",
        decisions,
        [
            "selector",
            "order_mode",
            "selected_variant",
            "used_fallback",
            "selected_latency_ms",
            "best_variant",
            "best_latency_ms",
            "regret_pct",
            "within_threshold",
            "order_span_mean",
            "order_span_p95",
            "monotonic_fraction",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_6_runtime_variant_summary.csv",
        selector_summary,
        [
            "selector",
            "orders",
            "near_best_count",
            "mean_regret_pct",
            "max_regret_pct",
            "fallback_count",
            "invalid_selected_count",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

