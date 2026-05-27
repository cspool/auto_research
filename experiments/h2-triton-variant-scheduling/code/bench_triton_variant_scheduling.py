#!/usr/bin/env python3
"""H2 benchmark: Triton micro-kernel variants under concurrent scheduling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, b_ptr, y_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x + b, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--num-elements", type=int, default=16_777_216)
    parser.add_argument("--block-sizes", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = torch.cuda.get_device_capability(0)
    return env


def cuda_time(fn: Callable[[], None], warmup: int, iters: int) -> float:
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


class Workload:
    def __init__(self, matrix_size: int, num_elements: int):
        device = torch.device("cuda:0")
        self.matrix_size = matrix_size
        self.num_elements = num_elements
        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.x = torch.randn(num_elements, device=device, dtype=torch.float16)
        self.bias = torch.randn(num_elements, device=device, dtype=torch.float16)
        self.y = torch.empty(num_elements, device=device, dtype=torch.float16)

    def gemm(self) -> None:
        torch.matmul(self.a, self.b, out=self.c)

    def triton_add(self, block_size: int, warps: int) -> None:
        grid = (triton.cdiv(self.num_elements, block_size),)
        add_kernel[grid](self.x, self.bias, self.y, self.num_elements, block_size, num_warps=warps)

    def serial(self, block_size: int, warps: int) -> None:
        self.gemm()
        self.triton_add(block_size, warps)

    def concurrent(self, block_size: int, warps: int, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            self.gemm()
        with torch.cuda.stream(streams[1]):
            self.triton_add(block_size, warps)
        for stream in streams:
            current.wait_stream(stream)

    def finite_outputs(self) -> bool:
        return bool(torch.isfinite(self.c).all().item()) and bool(torch.isfinite(self.y).all().item())


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "note": "CUDA is unavailable; no GPU result was produced.",
        }

    torch.cuda.set_device(0)
    workload = Workload(args.matrix_size, args.num_elements)
    streams = [torch.cuda.Stream() for _ in range(2)]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item.strip()]
    warps_list = [int(item) for item in args.warps.split(",") if item.strip()]

    gemm_ms = cuda_time(workload.gemm, args.warmup, args.iters)
    rows = []
    for block_size in block_sizes:
        for warps in warps_list:
            add_ms = cuda_time(lambda bs=block_size, w=warps: workload.triton_add(bs, w), args.warmup, args.iters)
            serial_ms = cuda_time(lambda bs=block_size, w=warps: workload.serial(bs, w), args.warmup, args.iters)
            concurrent_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.concurrent(bs, w, streams),
                args.warmup,
                args.iters,
            )
            rows.append(
                {
                    "block_size": block_size,
                    "warps": warps,
                    "add_ms": add_ms,
                    "serial_ms": serial_ms,
                    "concurrent_ms": concurrent_ms,
                    "stream_speedup": serial_ms / concurrent_ms,
                    "overlap_ratio": 1.0 - concurrent_ms / (gemm_ms + add_ms),
                }
            )

    best_isolated = min(rows, key=lambda row: row["add_ms"])
    best_concurrent = min(rows, key=lambda row: row["concurrent_ms"])
    best_speedup = max(rows, key=lambda row: row["stream_speedup"])
    return {
        "status": "ok",
        "matrix_size": args.matrix_size,
        "num_elements": args.num_elements,
        "warmup": args.warmup,
        "iters": args.iters,
        "gemm_ms": gemm_ms,
        "variants": rows,
        "best_isolated_variant": best_isolated,
        "best_concurrent_variant": best_concurrent,
        "best_stream_speedup_variant": best_speedup,
        "best_concurrent_differs_from_best_isolated": (
            best_isolated["block_size"] != best_concurrent["block_size"]
            or best_isolated["warps"] != best_concurrent["warps"]
        ),
        "finite_outputs": workload.finite_outputs(),
    }


def write_outputs(out_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "block_size",
                "warps",
                "add_ms",
                "serial_ms",
                "concurrent_ms",
                "stream_speedup",
                "overlap_ratio",
            ]
        )
        for row in result.get("variants", []):
            writer.writerow(
                [
                    row["block_size"],
                    row["warps"],
                    row["add_ms"],
                    row["serial_ms"],
                    row["concurrent_ms"],
                    row["stream_speedup"],
                    row["overlap_ratio"],
                ]
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    env = environment()
    result = run(args)
    write_outputs(out_dir, env, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

