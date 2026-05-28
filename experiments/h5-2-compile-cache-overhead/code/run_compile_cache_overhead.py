#!/usr/bin/env python3
"""H5.2: measure Triton compile/cache and steady-state launch overhead."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "experiments" / "h5-2-compile-cache-overhead" / "results"
DEFAULT_DATA_DIR = ROOT / "data"


@triton.jit
def fused_gelu_residual_kernel(x_ptr, bias_ptr, residual_ptr, y_ptr, n_elements: tl.constexpr, scale: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x * scale + bias
    z3 = z * z * z
    gelu_arg = 0.7978845608028654 * (z + 0.044715 * z3)
    tanh_arg = 2.0 / (1.0 + tl.exp(-2.0 * gelu_arg)) - 1.0
    gelu = 0.5 * z * (1.0 + tanh_arg)
    tl.store(y_ptr + offsets, gelu + residual, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--variants", default="512:4,1024:8,2048:8,4096:4,8192:8")
    parser.add_argument("--num-elements", type=int, default=16_777_216)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--cache-hit-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--case", default="worker")
    parser.add_argument("--cache-dir", default="")
    return parser.parse_args()


def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = torch.cuda.get_device_capability(0)
    return env


def cache_stats(cache_dir: Path) -> dict:
    if not cache_dir.exists():
        return {"cache_files": 0, "cache_bytes": 0}
    files = [path for path in cache_dir.rglob("*") if path.is_file()]
    return {"cache_files": len(files), "cache_bytes": sum(path.stat().st_size for path in files)}


def cuda_event_time(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def run_worker(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda"}
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    x = torch.randn(args.num_elements, device=device, dtype=torch.float16)
    bias = torch.randn(args.num_elements, device=device, dtype=torch.float16)
    residual = torch.randn(args.num_elements, device=device, dtype=torch.float16)
    y = torch.empty(args.num_elements, device=device, dtype=torch.float16)

    def launch() -> None:
        grid = (triton.cdiv(args.num_elements, args.block_size),)
        fused_gelu_residual_kernel[grid](
            x,
            bias,
            residual,
            y,
            args.num_elements,
            0.75,
            args.block_size,
            num_warps=args.warps,
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    launch()
    torch.cuda.synchronize()
    first_launch_wall_ms = (time.perf_counter() - start) * 1000.0
    steady_event_ms = cuda_event_time(launch, args.warmup, args.iters)
    finite = bool(torch.isfinite(y).all().item())
    stats = cache_stats(Path(args.cache_dir)) if args.cache_dir else {"cache_files": 0, "cache_bytes": 0}
    return {
        "status": "ok",
        "case": args.case,
        "variant_id": f"B{args.block_size}_W{args.warps}",
        "block_size": args.block_size,
        "warps": args.warps,
        "num_elements": args.num_elements,
        "first_launch_wall_ms": first_launch_wall_ms,
        "steady_event_ms": steady_event_ms,
        "overhead_ratio": first_launch_wall_ms / steady_event_ms,
        "finite_outputs": finite,
        **stats,
    }


def parse_variants(text: str) -> list[tuple[int, int]]:
    result = []
    for item in text.split(','):
        if not item.strip():
            continue
        block, warps = item.split(':')
        result.append((int(block), int(warps)))
    return result


def run_parent(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    cache_root = out_dir / "triton_cache"
    rows = []
    for block_size, warps in parse_variants(args.variants):
        variant_id = f"B{block_size}_W{warps}"
        cache_dir = cache_root / variant_id
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cases = [("cold", 0)] + [("cache_hit", idx) for idx in range(args.cache_hit_repeats)]
        for case, repeat in cases:
            env = os.environ.copy()
            env["TRITON_CACHE_DIR"] = str(cache_dir)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case", case,
                "--cache-dir", str(cache_dir),
                "--block-size", str(block_size),
                "--warps", str(warps),
                "--num-elements", str(args.num_elements),
                "--warmup", str(args.warmup),
                "--iters", str(args.iters),
                "--seed", str(args.seed),
            ]
            started = time.perf_counter()
            completed = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            process_wall_ms = (time.perf_counter() - started) * 1000.0
            worker = json.loads(completed.stdout)
            worker["repeat"] = repeat
            worker["process_wall_ms"] = process_wall_ms
            worker["triton_cache_dir"] = str(cache_dir)
            rows.append(worker)
    return {"status": "ok", "args": vars(args), "environment": environment(), "rows": rows}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def summarize(rows: list[dict]) -> list[dict]:
    by_variant: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant_id"], {}).setdefault(row["case"], []).append(row)
    summary = []
    for variant_id, by_case in sorted(by_variant.items()):
        cold = by_case["cold"][0]
        hits = by_case.get("cache_hit", [])
        hit_first = sum(row["first_launch_wall_ms"] for row in hits) / len(hits)
        hit_process = sum(row["process_wall_ms"] for row in hits) / len(hits)
        steady = sum(row["steady_event_ms"] for row in hits) / len(hits)
        summary.append(
            {
                "variant_id": variant_id,
                "cold_first_launch_wall_ms": cold["first_launch_wall_ms"],
                "cache_hit_first_launch_wall_ms": hit_first,
                "steady_event_ms": steady,
                "cold_overhead_ratio": cold["first_launch_wall_ms"] / steady,
                "cache_hit_overhead_ratio": hit_first / steady,
                "cold_process_wall_ms": cold["process_wall_ms"],
                "cache_hit_process_wall_ms": hit_process,
                "cache_files": cold["cache_files"],
                "cache_bytes": cold["cache_bytes"],
            }
        )
    return summary


def write_outputs(out_dir: Path, data_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result.get("rows", [])
    if not rows:
        return
    fields = [
        "case", "repeat", "variant_id", "block_size", "warps", "num_elements",
        "first_launch_wall_ms", "steady_event_ms", "overhead_ratio", "process_wall_ms",
        "cache_files", "cache_bytes", "finite_outputs", "triton_cache_dir",
    ]
    write_csv(out_dir / "measurements.csv", rows, fields)
    summary = summarize(rows)
    summary_fields = [
        "variant_id", "cold_first_launch_wall_ms", "cache_hit_first_launch_wall_ms",
        "steady_event_ms", "cold_overhead_ratio", "cache_hit_overhead_ratio",
        "cold_process_wall_ms", "cache_hit_process_wall_ms", "cache_files", "cache_bytes",
    ]
    write_csv(out_dir / "summary.csv", summary, summary_fields)
    (data_dir / "h5_2_compile_cache_measurements.csv").write_text((out_dir / "measurements.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (data_dir / "h5_2_compile_cache_summary.csv").write_text((out_dir / "summary.csv").read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.worker:
        print(json.dumps(run_worker(args)))
        return 0
    result = run_parent(args)
    write_outputs(Path(args.out_dir), Path(args.data_dir), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
