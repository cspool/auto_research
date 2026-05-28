#!/usr/bin/env python3
"""Run H4.9 held-out selected-token-count sparse attention sweeps."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "experiments" / "h4-5-kv-multi-query" / "code" / "bench_kv_multi_query.py"
OUT_ROOT = ROOT / "experiments" / "h4-9-selected-token-heldout-shapes" / "results"
ORDER_MODES = ["random_segment", "random_sorted", "random_shuffled", "clustered_segment"]
SELECTED_PER_SEGMENT = [16, 64]


def run_one(selected_per_segment: int, order_mode: str) -> dict:
    selected_tokens = selected_per_segment * 32
    out_dir = OUT_ROOT / f"s{selected_tokens}" / order_mode
    segment_counts = ",".join([str(selected_per_segment)] * 32)
    cmd = [
        sys.executable,
        str(BENCH),
        "--out-dir",
        str(out_dir),
        "--order-mode",
        order_mode,
        "--segment-counts",
        segment_counts,
        "--tokens-per-segment",
        "256",
        "--key-dim",
        "128",
        "--value-dim",
        "128",
        "--query-count",
        "4",
        "--warmup",
        "10",
        "--iters",
        "50",
    ]
    completed = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    return {
        "selected_tokens": selected_tokens,
        "selected_per_segment": selected_per_segment,
        "order_mode": order_mode,
        "out_dir": str(out_dir),
        "best_triton_grouped_ms": result.get("best_triton_grouped_ms"),
        "best_triton_vs_flat_speedup": result.get("best_triton_vs_flat_speedup"),
        "best_variant": result.get("best_variant"),
        "failed_variants": sum(1 for item in result.get("variant_results", []) if item.get("status") != "ok"),
    }


def main() -> int:
    summaries = []
    for selected_per_segment in SELECTED_PER_SEGMENT:
        for order_mode in ORDER_MODES:
            summary = run_one(selected_per_segment, order_mode)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "sweep_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

