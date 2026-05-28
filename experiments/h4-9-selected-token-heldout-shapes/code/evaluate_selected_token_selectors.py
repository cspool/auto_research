#!/usr/bin/env python3
"""Evaluate H4.8 selectors on H4.9 held-out selected-token-count results."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "experiments" / "h4-9-selected-token-heldout-shapes" / "results"
SHARED_LIMIT = 101376
THRESHOLD = 0.15


def variant_label(item: dict) -> str:
    return (
        f"N{item['block_n']} D{item['block_d']} V{item['block_v']} "
        f"Q{item['block_q']} W{item['warps']}"
    )


def shared_proxy(item: dict) -> int:
    return int(item["block_n"]) * int(item["block_q"]) * (int(item["block_d"]) + int(item["block_v"]))


def parse_required(error: str) -> int | None:
    match = re.search(r"Required:\s*(\d+)", error or "")
    return int(match.group(1)) if match else None


def load_results() -> list[dict]:
    results = []
    for path in sorted(RESULT_ROOT.glob("s*/**/result.json")):
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


def order_query_selector(result: dict) -> str:
    stats = result["order_stats"]
    span_mean = float(stats["order_span_mean"])
    span_p95 = float(stats["order_span_p95"])
    monotonic = float(stats["monotonic_fraction"])
    query_count = int(result["query_count"])
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
    if high_span:
        return "N128 D64 V64 Q4 W8"
    if random_segment:
        return "N64 D64 V64 Q4 W4"
    if clustered:
        return "N128 D64 V128 Q4 W4"
    return "N128 D128 V64 Q4 W4"


def selected_token_oracle(result: dict) -> str:
    """Retrospective pressure-aware rule used to diagnose the missing feature."""
    selected_tokens = int(result["selected_tokens"])
    order = result["order_mode"]
    if selected_tokens == 512:
        if order == "random_sorted":
            return "N64 D64 V64 Q4 W4"
        if order in {"random_segment", "clustered_segment"}:
            return "N128 D64 V64 Q4 W8"
        return "N128 D64 V64 Q4 W4"
    if selected_tokens == 2048:
        if order in {"random_sorted", "random_shuffled"}:
            return "N64 D64 V64 Q4 W4"
        return "N128 D64 V64 Q4 W8"
    return order_query_selector(result)


def choose(selected: str, lookup: dict[str, float]) -> tuple[str, bool]:
    if selected in lookup:
        return selected, False
    return min(lookup, key=lookup.get), True


def evaluate() -> tuple[list[dict], list[dict], dict]:
    results = load_results()
    raw_rows = []
    decisions = []
    observed_failed = 0
    static_rejected = 0

    for result in results:
        selected_tokens = int(result["selected_tokens"])
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
                    "selected_tokens": selected_tokens,
                    "order_mode": order,
                    "variant_id": label,
                    "status": item.get("status"),
                    "shared_proxy": shared_proxy(item),
                    "static_feasible": shared_proxy(item) <= SHARED_LIMIT,
                    "num_n_blocks": item.get("num_n_blocks"),
                    "triton_grouped_ms": item.get("triton_grouped_ms"),
                    "grouped_vs_flat_speedup": item.get("grouped_vs_flat_speedup"),
                    "required_shared_bytes": parse_required(item.get("error", "")),
                    "error_type": item.get("error_type"),
                }
            )

        selectors = {
            "h4_8_order_query_selector": order_query_selector(result),
            "selected_token_pressure_oracle": selected_token_oracle(result),
        }
        for selector_name, selected in selectors.items():
            selected, used_fallback = choose(selected, lookup)
            latency = lookup[selected]
            regret = latency / best_ms - 1.0
            block_n = int(selected.split()[0][1:])
            decisions.append(
                {
                    "selector": selector_name,
                    "selected_tokens": selected_tokens,
                    "query_count": int(result["query_count"]),
                    "order_mode": order,
                    "selected_variant": selected,
                    "selected_block_n": block_n,
                    "selected_num_n_blocks": math.ceil(selected_tokens / block_n),
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
        ROOT / "data" / "h4_9_selected_token_variant_rows.csv",
        raw_rows,
        [
            "selected_tokens",
            "order_mode",
            "variant_id",
            "status",
            "shared_proxy",
            "static_feasible",
            "num_n_blocks",
            "triton_grouped_ms",
            "grouped_vs_flat_speedup",
            "required_shared_bytes",
            "error_type",
        ],
    )
    write_csv(
        ROOT / "data" / "h4_9_selected_token_selector_decisions.csv",
        decisions,
        [
            "selector",
            "selected_tokens",
            "query_count",
            "order_mode",
            "selected_variant",
            "selected_block_n",
            "selected_num_n_blocks",
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
        ROOT / "data" / "h4_9_selected_token_selector_summary.csv",
        result["selector_summary"],
        ["selector", "cases", "near_best_count", "mean_regret_pct", "max_regret_pct", "fallback_count"],
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "selector_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

