#!/usr/bin/env python3
"""Extract Triton cache metadata for fused micro-kernel variants."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REG_PATTERN = re.compile(r"\.reg\s+\.(?P<class>\w+)\s+%[\w$]+<(?P<count>\d+)>;")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="~/.triton/cache")
    parser.add_argument("--kernel-name", default="fused_gelu_residual_kernel")
    parser.add_argument("--out-dir", default="../results")
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_block_size(ttgir: str) -> int | None:
    sizes = [int(item) for item in re.findall(r"tensor<(\d+)x", ttgir)]
    return max(sizes) if sizes else None


def parse_n_elements(ttgir: str) -> int | None:
    values = [int(item) for item in re.findall(r"dense<(\d+)>", ttgir)]
    candidates = [value for value in values if value > 1024]
    return max(candidates) if candidates else None


def parse_reqntid(ptx: str) -> int | None:
    match = re.search(r"\.reqntid\s+(\d+)", ptx)
    return int(match.group(1)) if match else None


def parse_registers(ptx: str) -> dict[str, int]:
    registers: dict[str, int] = {}
    for match in REG_PATTERN.finditer(ptx):
        registers[f"reg_{match.group('class')}_decl"] = int(match.group("count"))
    return registers


def extract_entry(meta_path: Path, kernel_name: str) -> dict | None:
    directory = meta_path.parent
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    ttgir_path = directory / f"{kernel_name}.ttgir"
    ptx_path = directory / f"{kernel_name}.ptx"
    cubin_path = directory / f"{kernel_name}.cubin"
    source_path = directory / f"{kernel_name}.source"
    ttgir = read_text(ttgir_path)
    ptx = read_text(ptx_path)
    registers = parse_registers(ptx)

    row = {
        "cache_key": directory.name,
        "kernel_name": kernel_name,
        "arch": metadata.get("arch"),
        "triton_version": metadata.get("triton_version"),
        "n_elements": parse_n_elements(ttgir),
        "block_size": parse_block_size(ttgir),
        "num_warps": metadata.get("num_warps"),
        "num_ctas": metadata.get("num_ctas"),
        "num_stages": metadata.get("num_stages"),
        "reqntid": parse_reqntid(ptx),
        "shared_bytes": metadata.get("shared"),
        "global_scratch_size": metadata.get("global_scratch_size"),
        "ptx_bytes": ptx_path.stat().st_size if ptx_path.exists() else 0,
        "cubin_bytes": cubin_path.stat().st_size if cubin_path.exists() else 0,
        "ttgir_bytes": ttgir_path.stat().st_size if ttgir_path.exists() else 0,
        "source_bytes": source_path.stat().st_size if source_path.exists() else 0,
        "cache_dir": str(directory),
    }
    row.update(registers)
    return row


def deduplicate(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("n_elements"), row.get("block_size"), row.get("num_warps"))
        current = best.get(key)
        if current is None or row.get("ptx_bytes", 0) > current.get("ptx_bytes", 0):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            row.get("n_elements") or -1,
            row.get("block_size") or -1,
            row.get("num_warps") or -1,
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "n_elements",
        "block_size",
        "num_warps",
        "reqntid",
        "num_ctas",
        "num_stages",
        "shared_bytes",
        "global_scratch_size",
        "reg_pred_decl",
        "reg_b16_decl",
        "reg_b32_decl",
        "reg_b64_decl",
        "ptx_bytes",
        "cubin_bytes",
        "ttgir_bytes",
        "source_bytes",
        "arch",
        "triton_version",
        "cache_key",
        "cache_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(cache_dir.glob(f"*/{args.kernel_name}.json"))
    rows = [row for path in meta_paths if (row := extract_entry(path, args.kernel_name)) is not None]
    unique_rows = deduplicate(rows)

    write_csv(out_dir / "all_cache_entries.csv", rows)
    write_csv(out_dir / "unique_variants.csv", unique_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "kernel_name": args.kernel_name,
                "entries": len(rows),
                "unique_variants": len(unique_rows),
                "rows": unique_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"entries": len(rows), "unique_variants": len(unique_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
