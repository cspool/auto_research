#!/usr/bin/env python3
"""Run H4.7 held-out query-count sparse attention sweeps."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "experiments" / "h4-5-kv-multi-query" / "code" / "bench_kv_multi_query.py"
OUT_ROOT = ROOT / "experiments" / "h4-7-variant-table-heldout-shapes" / "results"
SEGMENT_COUNTS = ",".join(["32"] * 32)
ORDER_MODES = ["random_segment", "random_sorted", "random_shuffled", "clustered_segment"]
QUERY_COUNTS = [2, 8]


def run_one(query_count: int, order_mode: str) -> dict:
    out_dir = OUT_ROOT / f"q{query_count}" / order_mode
    cmd = [
        sys.executable,
        str(BENCH),
        "--out-dir",
        str(out_dir),
        "--order-mode",
        order_mode,
        "--segment-counts",
        SEGMENT_COUNTS,
        "--tokens-per-segment",
        "256",
        "--key-dim",
        "128",
        "--value-dim",
        "128",
        "--query-count",
        str(query_count),
        "--warmup",
        "10",
        "--iters",
        "50",
    ]
    completed = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    return {
        "query_count": query_count,
        "order_mode": order_mode,
        "out_dir": str(out_dir),
        "best_triton_grouped_ms": result.get("best_triton_grouped_ms"),
        "best_triton_vs_flat_speedup": result.get("best_triton_vs_flat_speedup"),
        "best_variant": result.get("best_variant"),
        "failed_variants": sum(1 for item in result.get("variant_results", []) if item.get("status") != "ok"),
    }


def main() -> int:
    summaries = []
    for query_count in QUERY_COUNTS:
        for order_mode in ORDER_MODES:
            summary = run_one(query_count, order_mode)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "sweep_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

