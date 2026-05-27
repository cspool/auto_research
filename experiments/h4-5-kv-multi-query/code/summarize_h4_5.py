#!/usr/bin/env python3
"""Summarize H4.5 result JSON files into project-level CSV artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "h4-5-kv-multi-query" / "results" / "full"
SUMMARY_CSV = ROOT / "data" / "h4_5_kv_multi_query_summary.csv"
VARIANTS_CSV = ROOT / "data" / "h4_5_kv_multi_query_variants.csv"


def load_results() -> list[dict]:
    results = []
    for path in sorted(RESULTS_DIR.glob("*/result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        raise SystemExit(f"no results found under {RESULTS_DIR}")
    return results


def variant_label(variant: dict) -> str:
    return (
        f"N{variant['block_n']} D{variant['block_d']} V{variant['block_v']} "
        f"Q{variant.get('block_q', 1)} W{variant['warps']}"
    )


def main() -> int:
    results = load_results()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_mode",
                "selected_tokens",
                "query_count",
                "key_dim",
                "value_dim",
                "torch_flat_total_ms",
                "torch_score_softmax_ms",
                "torch_value_ms",
                "best_triton_grouped_ms",
                "best_triton_grouped_per_query_ms",
                "best_triton_vs_flat_speedup",
                "best_oneq_repeated_ms",
                "best_grouped_vs_best_oneq_repeated_speedup",
                "best_variant",
                "best_oneq_variant",
                "order_span_mean",
                "order_span_p95",
                "monotonic_fraction",
                "max_relative_diff_grouped",
                "failed_variant_count",
            ]
        )
        for result in results:
            valid_variants = [item for item in result["variant_results"] if item.get("status") == "ok"]
            best_variant = result["best_variant"]
            best_result = min(valid_variants, key=lambda item: item["triton_grouped_ms"])
            writer.writerow(
                [
                    result["order_mode"],
                    result["selected_tokens"],
                    result["query_count"],
                    result["key_dim"],
                    result["value_dim"],
                    result["torch_flat_total_ms"],
                    result["torch_score_softmax_ms"],
                    result["torch_value_ms"],
                    result["best_triton_grouped_ms"],
                    result["best_triton_grouped_per_query_ms"],
                    result["best_triton_vs_flat_speedup"],
                    result["best_oneq_repeated_ms"],
                    result["best_grouped_vs_best_oneq_repeated_speedup"],
                    variant_label(best_variant),
                    variant_label(result["best_oneq_variant"]),
                    result["order_stats"]["order_span_mean"],
                    result["order_stats"]["order_span_p95"],
                    result["order_stats"]["monotonic_fraction"],
                    best_result["validation"]["relative_max_abs_diff_grouped"],
                    len(result["variant_results"]) - len(valid_variants),
                ]
            )

    with VARIANTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_mode",
                "status",
                "block_n",
                "block_d",
                "block_v",
                "block_q",
                "warps",
                "triton_grouped_ms",
                "triton_oneq_repeated_ms",
                "grouped_vs_flat_speedup",
                "grouped_vs_oneq_repeated_speedup",
                "relative_max_abs_diff_grouped",
                "error_type",
                "error",
            ]
        )
        for result in results:
            for item in result["variant_results"]:
                validation = item.get("validation", {})
                writer.writerow(
                    [
                        result["order_mode"],
                        item.get("status"),
                        item["block_n"],
                        item["block_d"],
                        item["block_v"],
                        item["block_q"],
                        item["warps"],
                        item.get("triton_grouped_ms"),
                        item.get("triton_oneq_repeated_ms"),
                        item.get("grouped_vs_flat_speedup"),
                        item.get("grouped_vs_oneq_repeated_speedup"),
                        validation.get("relative_max_abs_diff_grouped"),
                        item.get("error_type"),
                        item.get("error"),
                    ]
                )

    print(SUMMARY_CSV)
    print(VARIANTS_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
