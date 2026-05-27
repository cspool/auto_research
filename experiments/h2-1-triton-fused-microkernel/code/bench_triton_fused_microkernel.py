#!/usr/bin/env python3
"""H2.1 benchmark: fused Triton micro-kernel variants under concurrency."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gelu_residual_kernel(
    x_ptr,
    bias_ptr,
    residual_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
):
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
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--num-elements", type=int, default=16_777_216)
    parser.add_argument("--block-sizes", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
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


def nominal_fused_gbps(num_elements: int, fused_ms: float) -> float:
    bytes_touched = num_elements * 8.0
    return bytes_touched / fused_ms / 1_000_000.0


class Workload:
    def __init__(self, matrix_size: int, num_elements: int, scale: float, seed: int):
        torch.manual_seed(seed)
        device = torch.device("cuda:0")
        self.matrix_size = matrix_size
        self.num_elements = num_elements
        self.scale = float(scale)
        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.x = torch.randn(num_elements, device=device, dtype=torch.float16)
        self.bias = torch.randn(num_elements, device=device, dtype=torch.float16)
        self.residual = torch.randn(num_elements, device=device, dtype=torch.float16)
        self.y = torch.empty(num_elements, device=device, dtype=torch.float16)
        self.y_torch = torch.empty(num_elements, device=device, dtype=torch.float16)

    def gemm(self) -> None:
        torch.matmul(self.a, self.b, out=self.c)

    def triton_fused(self, block_size: int, warps: int) -> None:
        grid = (triton.cdiv(self.num_elements, block_size),)
        fused_gelu_residual_kernel[grid](
            self.x,
            self.bias,
            self.residual,
            self.y,
            self.num_elements,
            self.scale,
            block_size,
            num_warps=warps,
        )

    def torch_unfused(self) -> None:
        self.y_torch = F.gelu(self.x * self.scale + self.bias, approximate="tanh") + self.residual

    def serial_triton(self, block_size: int, warps: int) -> None:
        self.gemm()
        self.triton_fused(block_size, warps)

    def concurrent_triton(self, block_size: int, warps: int, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            self.gemm()
        with torch.cuda.stream(streams[1]):
            self.triton_fused(block_size, warps)
        for stream in streams:
            current.wait_stream(stream)

    def serial_torch(self) -> None:
        self.gemm()
        self.torch_unfused()

    def concurrent_torch(self, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            self.gemm()
        with torch.cuda.stream(streams[1]):
            self.torch_unfused()
        for stream in streams:
            current.wait_stream(stream)

    def validate(self, block_size: int, warps: int) -> dict:
        self.triton_fused(block_size, warps)
        self.torch_unfused()
        torch.cuda.synchronize()
        max_abs_diff = (self.y.float() - self.y_torch.float()).abs().max().item()
        finite = (
            bool(torch.isfinite(self.c).all().item())
            and bool(torch.isfinite(self.y).all().item())
            and bool(torch.isfinite(self.y_torch).all().item())
        )
        return {"finite_outputs": finite, "max_abs_diff_vs_torch": float(max_abs_diff)}


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "note": "CUDA is unavailable; no GPU result was produced.",
        }

    torch.cuda.set_device(0)
    workload = Workload(args.matrix_size, args.num_elements, args.scale, args.seed)
    streams = [torch.cuda.Stream() for _ in range(2)]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item.strip()]
    warps_list = [int(item) for item in args.warps.split(",") if item.strip()]

    gemm_ms = cuda_time(workload.gemm, args.warmup, args.iters)
    torch_unfused_ms = cuda_time(workload.torch_unfused, args.warmup, args.iters)
    torch_serial_ms = cuda_time(workload.serial_torch, args.warmup, args.iters)
    torch_concurrent_ms = cuda_time(lambda: workload.concurrent_torch(streams), args.warmup, args.iters)

    rows = []
    for block_size in block_sizes:
        for warps in warps_list:
            fused_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.triton_fused(bs, w),
                args.warmup,
                args.iters,
            )
            serial_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.serial_triton(bs, w),
                args.warmup,
                args.iters,
            )
            concurrent_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.concurrent_triton(bs, w, streams),
                args.warmup,
                args.iters,
            )
            rows.append(
                {
                    "block_size": block_size,
                    "warps": warps,
                    "fused_ms": fused_ms,
                    "serial_ms": serial_ms,
                    "concurrent_ms": concurrent_ms,
                    "stream_speedup": serial_ms / concurrent_ms,
                    "overlap_ratio": 1.0 - concurrent_ms / (gemm_ms + fused_ms),
                    "nominal_fused_gbps": nominal_fused_gbps(args.num_elements, fused_ms),
                }
            )

    best_isolated = min(rows, key=lambda row: row["fused_ms"])
    best_concurrent = min(rows, key=lambda row: row["concurrent_ms"])
    best_speedup = max(rows, key=lambda row: row["stream_speedup"])
    validation = workload.validate(best_isolated["block_size"], best_isolated["warps"])
    return {
        "status": "ok",
        "matrix_size": args.matrix_size,
        "num_elements": args.num_elements,
        "scale": args.scale,
        "warmup": args.warmup,
        "iters": args.iters,
        "gemm_ms": gemm_ms,
        "torch_unfused_ms": torch_unfused_ms,
        "torch_serial_ms": torch_serial_ms,
        "torch_concurrent_ms": torch_concurrent_ms,
        "torch_stream_speedup": torch_serial_ms / torch_concurrent_ms,
        "torch_overlap_ratio": 1.0 - torch_concurrent_ms / (gemm_ms + torch_unfused_ms),
        "variants": rows,
        "best_isolated_variant": best_isolated,
        "best_concurrent_variant": best_concurrent,
        "best_stream_speedup_variant": best_speedup,
        "best_concurrent_differs_from_best_isolated": (
            best_isolated["block_size"] != best_concurrent["block_size"]
            or best_isolated["warps"] != best_concurrent["warps"]
        ),
        "best_stream_speedup_differs_from_best_concurrent": (
            best_speedup["block_size"] != best_concurrent["block_size"]
            or best_speedup["warps"] != best_concurrent["warps"]
        ),
        "validation": validation,
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
                "fused_ms",
                "serial_ms",
                "concurrent_ms",
                "stream_speedup",
                "overlap_ratio",
                "nominal_fused_gbps",
            ]
        )
        for row in result.get("variants", []):
            writer.writerow(
                [
                    row["block_size"],
                    row["warps"],
                    row["fused_ms"],
                    row["serial_ms"],
                    row["concurrent_ms"],
                    row["stream_speedup"],
                    row["overlap_ratio"],
                    row["nominal_fused_gbps"],
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
