#!/usr/bin/env python3
"""Evaluate H4.6 selectors on H4.7 held-out query-count results."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "experiments" / "h4-7-variant-table-heldout-shapes" / "results"
OUT_DIR = RESULT_ROOT
SHARED_LIMIT = 101376
GLOBAL_SELECTOR = "N64 D64 V64 Q4 W4"
THRESHOLD = 0.15


def variant_label(item: dict) -> str:
    return (
        f"N{item['block_n']} D{item['block_d']} V{item['block_v']} "
        f"Q{item['block_q']} W{item['warps']}"
    )


def shared_proxy(item: dict) -> int:
    return int(item["block_n"]) * int(item["block_q"]) * (int(item["block_d"]) + int(item["block_v"]))


def order_aware_selector(stats: dict) -> str:
    span_mean = float(stats["order_span_mean"])
    span_p95 = float(stats["order_span_p95"])
    monotonic = float(stats["monotonic_fraction"])
    if monotonic < 0.75 and span_mean > 512:
        return "N128 D64 V64 Q4 W8"
    if monotonic < 0.75:
        return "N64 D64 V64 Q4 W4"
    if span_p95 <= 4:
        return "N128 D64 V128 Q4 W4"
    return "N128 D128 V64 Q4 W4"


def parse_required(error: str) -> int | None:
    match = re.search(r"Required:\s*(\d+)", error or "")
    return int(match.group(1)) if match else None


def load_results() -> list[dict]:
    results = []
    for path in sorted(RESULT_ROOT.glob("q*/**/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["result_path"] = str(path)
        results.append(result)
    if not results:
        raise SystemExit(f"no result.json files found under {RESULT_ROOT}")
    return results


def best_variant(result: dict) -> tuple[str, float]:
    valid = [item for item in result["variant_results"] if item.get("status") == "ok"]
    winner = min(valid, key=lambda item: float(item["triton_grouped_ms"]))
    return variant_label(winner), float(winner["triton_grouped_ms"])


def latency_lookup(result: dict) -> dict[str, float]:
    lookup = {}
    for item in result["variant_results"]:
        if item.get("status") == "ok" and shared_proxy(item) <= SHARED_LIMIT:
            lookup[variant_label(item)] = float(item["triton_grouped_ms"])
    return lookup


def choose(selected: str, lookup: dict[str, float]) -> tuple[str, bool]:
    if selected in lookup:
        return selected, False
    if GLOBAL_SELECTOR in lookup:
        return GLOBAL_SELECTOR, True
    return min(lookup, key=lookup.get), True


def evaluate() -> tuple[list[dict], list[dict], dict]:
    results = load_results()
    decisions = []
    raw_rows = []
    observed_failed = 0
    static_rejected = 0

    for result in results:
        q = int(result["query_count"])
        order = result["order_mode"]
        best_id, best_ms = best_variant(result)
        lookup = latency_lookup(result)
        failed_items = [item for item in result["variant_results"] if item.get("status") != "ok"]
        observed_failed += len(failed_items)
        static_rejected += sum(1 for item in failed_items if shared_proxy(item) > SHARED_LIMIT)

        for item in result["variant_results"]:
            label = variant_label(item)
            raw_rows.append(
                {
                    "query_count": q,
                    "order_mode": order,
                    "variant_id": label,
                    "status": item.get("status"),
                    "shared_proxy": shared_proxy(item),
                    "static_feasible": shared_proxy(item) <= SHARED_LIMIT,
                    "triton_grouped_ms": item.get("triton_grouped_ms"),
                    "grouped_vs_flat_speedup": item.get("grouped_vs_flat_speedup"),
                    "required_shared_bytes": parse_required(item.get("error", "")),
                    "error_type": item.get("error_type"),
                }
            )

        selectors = {
            "global_mean_latency_h4_6": GLOBAL_SELECTOR,
            "order_aware_rule_table_h4_6": order_aware_selector(result["order_stats"]),
        }
        for selector_name, selected in selectors.items():
            selected, used_fallback = choose(selected, lookup)
            latency = lookup[selected]
            regret = latency / best_ms - 1.0
            decisions.append(
                {
                    "selector": selector_name,
                    "query_count": q,
                    "order_mode": order,
                    "selected_variant": selected,
                    "used_fallback": used_fallback,
                    "selected_latency_ms": latency,
                    "best_variant": best_id,
                    "best_latency_ms": best_ms,
                    "regret_pct": regret * 100.0,
                    "within_15pct": regret <= THRESHOLD,
                    "order_span_mean": result["order_stats"]["order_span_mean"],
                    "order_span_p95": result["order_stats"]["order_span_p95"],
                    "monotonic_fraction": result["order_stats"]["monotonic_fraction"],
                }
            )

    summaries = []
    for selector in sorted({row["selector"] for row in decisions}):
        rows = [row for row in decisions if row["selector"] == selector]
        summaries.append(
            {
                "selector": selector,
                "cases": len(rows),
                "near_best_count": sum(1 for row in rows if row["within_15pct"]),
                "mean_regret_pct": sum(row["regret_pct"] for row in rows) / len(rows),
                "max_regret_pct": max(row["regret_pct"] for row in rows),
                "fallback_count": sum(1 for row in rows if row["used_fallback"]),
            }
        )

    result = {
        "status": "ok",
        "cases": len(results),
        "threshold": THRESHOLD,
        "shared_limit": SHARED_LIMIT,
        "observed_failed_rows": observed_failed,
        "static_rejected_failed_rows": static_rejected,
        "static_rejected_failed_fraction": static_rejected / observed_failed if observed_failed else 1.0,
        "selector_summary": summaries,
    }
    return raw_rows, decisions, result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    raw_rows, decisions, result = evaluate()
    write_csv(
        ROOT / "data" / "h4_7_heldout_variant_rows.csv",
        raw_rows,
        [
            "query_count",
            "order_mode",
            "variant_id",
            "status",
            "shared_proxy",
            "static_feasible",
            "triton_grouped_ms",
            "grouped_vs_flat_speedup",
            "required_shared_bytes",
            "error_type",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_7_heldout_selector_decisions.csv",
        decisions,
        [
            "selector",
            "query_count",
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
        ROOT / "data" / "h4_7_heldout_selector_summary.csv",
        result["selector_summary"],
        ["selector", "cases", "near_best_count", "mean_regret_pct", "max_regret_pct", "fallback_count"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selector_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

