#!/usr/bin/env python3
"""Join H2.1 latency measurements with Triton compiler metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2-results-dir", default="../../h2-1-triton-fused-microkernel/results")
    parser.add_argument("--metadata-csv", default="../results/unique_variants.csv")
    parser.add_argument("--out-csv", default="../results/joined_latency_metadata.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_int(value: str) -> int:
    return int(float(value))


def load_metadata(path: Path) -> dict[tuple[int, int, int], dict]:
    rows = read_csv(path)
    table = {}
    for row in rows:
        key = (
            normalize_int(row["n_elements"]),
            normalize_int(row["block_size"]),
            normalize_int(row["num_warps"]),
        )
        table[key] = row
    return table


def load_latency_rows(results_dir: Path) -> list[dict]:
    rows = []
    for result_json in sorted(results_dir.glob("*/result.json")):
        run_name = result_json.parent.name
        if run_name == "smoke":
            continue
        result = json.loads(result_json.read_text(encoding="utf-8"))
        n_elements = int(result["num_elements"])
        for variant in result.get("variants", []):
            row = {"run": run_name, "n_elements": n_elements}
            row.update(variant)
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    metadata_csv = (script_dir / args.metadata_csv).resolve()
    h2_results_dir = (script_dir / args.h2_results_dir).resolve()
    metadata = load_metadata(metadata_csv)
    latency_rows = load_latency_rows(h2_results_dir)
    joined_rows = []
    for row in latency_rows:
        key = (int(row["n_elements"]), int(row["block_size"]), int(row["warps"]))
        meta = metadata.get(key, {})
        joined = dict(row)
        joined["num_warps"] = joined.pop("warps")
        for field in [
            "reqntid",
            "num_ctas",
            "num_stages",
            "shared_bytes",
            "reg_pred_decl",
            "reg_b16_decl",
            "reg_b32_decl",
            "reg_b64_decl",
            "ptx_bytes",
            "cubin_bytes",
            "arch",
            "triton_version",
            "cache_key",
        ]:
            joined[field] = meta.get(field, "")
        joined_rows.append(joined)

    fieldnames = [
        "run",
        "n_elements",
        "block_size",
        "num_warps",
        "fused_ms",
        "serial_ms",
        "concurrent_ms",
        "stream_speedup",
        "overlap_ratio",
        "nominal_fused_gbps",
        "reqntid",
        "num_ctas",
        "num_stages",
        "shared_bytes",
        "reg_pred_decl",
        "reg_b16_decl",
        "reg_b32_decl",
        "reg_b64_decl",
        "ptx_bytes",
        "cubin_bytes",
        "arch",
        "triton_version",
        "cache_key",
    ]
    out_csv = (script_dir / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in joined_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(json.dumps({"joined_rows": len(joined_rows), "out_csv": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
