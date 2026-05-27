#!/usr/bin/env python3
"""H3.2 benchmark: Triton MoE expert epilogue micro-operators."""

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
def fused_epilogue_kernel(
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
    parser.add_argument("--token-counts", default="64,64,64,64")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--block-sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
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


class MoeEpilogueWorkload:
    def __init__(self, token_counts: list[int], hidden_size: int, scale: float, seed: int):
        torch.manual_seed(seed)
        self.token_counts = token_counts
        self.hidden_size = hidden_size
        self.scale = float(scale)
        self.lengths = [tokens * hidden_size for tokens in token_counts]
        self.offsets = []
        running = 0
        for length in self.lengths:
            self.offsets.append(running)
            running += length
        self.total_elements = running
        device = torch.device("cuda:0")
        self.x = torch.randn(self.total_elements, device=device, dtype=torch.float16)
        self.bias = torch.randn(self.total_elements, device=device, dtype=torch.float16)
        self.residual = torch.randn(self.total_elements, device=device, dtype=torch.float16)
        self.y = torch.empty(self.total_elements, device=device, dtype=torch.float16)
        self.y_torch = torch.empty(self.total_elements, device=device, dtype=torch.float16)
        self.num_experts = len(token_counts)

    def segment(self, tensor: torch.Tensor, expert_id: int) -> torch.Tensor:
        start = self.offsets[expert_id]
        end = start + self.lengths[expert_id]
        return tensor[start:end]

    def torch_expert(self, expert_id: int) -> None:
        x = self.segment(self.x, expert_id)
        bias = self.segment(self.bias, expert_id)
        residual = self.segment(self.residual, expert_id)
        y = self.segment(self.y_torch, expert_id)
        y.copy_(F.gelu(x * self.scale + bias, approximate="tanh") + residual)

    def torch_loop(self) -> None:
        for expert_id in range(self.num_experts):
            self.torch_expert(expert_id)

    def triton_segment(self, expert_id: int, block_size: int, warps: int) -> None:
        start = self.offsets[expert_id]
        n_elements = self.lengths[expert_id]
        grid = (triton.cdiv(n_elements, block_size),)
        fused_epilogue_kernel[grid](
            self.x[start:],
            self.bias[start:],
            self.residual[start:],
            self.y[start:],
            n_elements,
            self.scale,
            block_size,
            num_warps=warps,
        )

    def triton_serial(self, block_size: int, warps: int) -> None:
        for expert_id in range(self.num_experts):
            self.triton_segment(expert_id, block_size, warps)

    def triton_concurrent(self, block_size: int, warps: int, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        for expert_id, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                self.triton_segment(expert_id, block_size, warps)
        for stream in streams:
            current.wait_stream(stream)

    def triton_grouped(self, block_size: int, warps: int) -> None:
        grid = (triton.cdiv(self.total_elements, block_size),)
        fused_epilogue_kernel[grid](
            self.x,
            self.bias,
            self.residual,
            self.y,
            self.total_elements,
            self.scale,
            block_size,
            num_warps=warps,
        )

    def validate(self, block_size: int, warps: int) -> dict:
        self.torch_loop()
        self.triton_grouped(block_size, warps)
        torch.cuda.synchronize()
        max_abs_diff = (self.y.float() - self.y_torch.float()).abs().max().item()
        finite = bool(torch.isfinite(self.y).all().item()) and bool(torch.isfinite(self.y_torch).all().item())
        return {"finite_outputs": finite, "max_abs_diff_vs_torch": float(max_abs_diff)}


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    token_counts = [int(item) for item in args.token_counts.split(",") if item.strip()]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item.strip()]
    warps_list = [int(item) for item in args.warps.split(",") if item.strip()]
    workload = MoeEpilogueWorkload(token_counts, args.hidden_size, args.scale, args.seed)
    streams = [torch.cuda.Stream() for _ in token_counts]

    torch_loop_ms = cuda_time(workload.torch_loop, args.warmup, args.iters)
    rows = []
    for block_size in block_sizes:
        for warps in warps_list:
            serial_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.triton_serial(bs, w),
                args.warmup,
                args.iters,
            )
            concurrent_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.triton_concurrent(bs, w, streams),
                args.warmup,
                args.iters,
            )
            grouped_ms = cuda_time(
                lambda bs=block_size, w=warps: workload.triton_grouped(bs, w),
                args.warmup,
                args.iters,
            )
            rows.append(
                {
                    "block_size": block_size,
                    "warps": warps,
                    "serial_ms": serial_ms,
                    "concurrent_ms": concurrent_ms,
                    "grouped_ms": grouped_ms,
                    "concurrent_vs_serial_speedup": serial_ms / concurrent_ms,
                    "grouped_vs_serial_speedup": serial_ms / grouped_ms,
                    "grouped_vs_concurrent_speedup": concurrent_ms / grouped_ms,
                    "grouped_vs_torch_speedup": torch_loop_ms / grouped_ms,
                }
            )

    best_grouped = min(rows, key=lambda row: row["grouped_ms"])
    best_serial = min(rows, key=lambda row: row["serial_ms"])
    best_concurrent = min(rows, key=lambda row: row["concurrent_ms"])
    best_grouped_speedup = max(rows, key=lambda row: row["grouped_vs_concurrent_speedup"])
    validation = workload.validate(best_grouped["block_size"], best_grouped["warps"])
    return {
        "status": "ok",
        "token_counts": token_counts,
        "hidden_size": args.hidden_size,
        "total_elements": workload.total_elements,
        "scale": args.scale,
        "warmup": args.warmup,
        "iters": args.iters,
        "torch_loop_ms": torch_loop_ms,
        "variants": rows,
        "best_grouped_variant": best_grouped,
        "best_serial_variant": best_serial,
        "best_concurrent_variant": best_concurrent,
        "best_grouped_speedup_variant": best_grouped_speedup,
        "best_grouped_differs_from_best_concurrent": (
            best_grouped["block_size"] != best_concurrent["block_size"]
            or best_grouped["warps"] != best_concurrent["warps"]
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
                "serial_ms",
                "concurrent_ms",
                "grouped_ms",
                "concurrent_vs_serial_speedup",
                "grouped_vs_serial_speedup",
                "grouped_vs_concurrent_speedup",
                "grouped_vs_torch_speedup",
            ]
        )
        for row in result.get("variants", []):
            writer.writerow(
                [
                    row["block_size"],
                    row["warps"],
                    row["serial_ms"],
                    row["concurrent_ms"],
                    row["grouped_ms"],
                    row["concurrent_vs_serial_speedup"],
                    row["grouped_vs_serial_speedup"],
                    row["grouped_vs_concurrent_speedup"],
                    row["grouped_vs_torch_speedup"],
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
