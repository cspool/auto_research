#!/usr/bin/env python3
"""H3.5 benchmark: routed no-padding MoE fragment with scatter/gather."""

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
def scatter_tokens_kernel(
    x_ptr,
    x_concat_ptr,
    token_indices_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    total = n_rows * hidden_size
    mask = offsets < total
    row = offsets // hidden_size
    col = offsets - row * hidden_size
    token_idx = tl.load(token_indices_ptr + row, mask=mask, other=0)
    values = tl.load(x_ptr + token_idx * hidden_size + col, mask=mask, other=0.0)
    tl.store(x_concat_ptr + offsets, values, mask=mask)


@triton.jit
def gather_tokens_kernel(
    y_concat_ptr,
    y_ptr,
    token_indices_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    total = n_rows * hidden_size
    mask = offsets < total
    row = offsets // hidden_size
    col = offsets - row * hidden_size
    token_idx = tl.load(token_indices_ptr + row, mask=mask, other=0)
    values = tl.load(y_concat_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + token_idx * hidden_size + col, values, mask=mask)


@triton.jit
def grouped_matmul_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    block_expert_ids_ptr,
    block_row_starts_ptr,
    expert_ends_ptr,
    k_size: tl.constexpr,
    n_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    expert_id = tl.load(block_expert_ids_ptr + pid_m)
    row_start = tl.load(block_row_starts_ptr + pid_m)
    expert_end = tl.load(expert_ends_ptr + expert_id)

    offs_m = row_start + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in tl.range(0, k_size, block_k):
        k_idxs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * k_size + k_idxs[None, :],
            mask=(offs_m[:, None] < expert_end) & (k_idxs[None, :] < k_size),
            other=0.0,
        )
        w = tl.load(
            w_ptr + expert_id * k_size * n_size + k_idxs[:, None] * n_size + offs_n[None, :],
            mask=(k_idxs[:, None] < k_size) & (offs_n[None, :] < n_size),
            other=0.0,
        )
        acc += tl.dot(a, w)

    tl.store(
        c_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < expert_end) & (offs_n[None, :] < n_size),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--token-counts", default="64,64,64,64")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--move-block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2030)
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


class RoutedMoeWorkload:
    def __init__(self, token_counts: list[int], hidden_size: int, ffn_size: int, seed: int):
        torch.manual_seed(seed)
        self.token_counts = token_counts
        self.num_experts = len(token_counts)
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.total_tokens = sum(token_counts)
        device = torch.device("cuda:0")

        # Build one-request top-1 routing with exact expert loads, then shuffle
        # original token order so scatter/gather are real movement operations.
        expert_ids_cpu = []
        for expert_id, count in enumerate(token_counts):
            expert_ids_cpu.extend([expert_id] * count)
        expert_ids = torch.tensor(expert_ids_cpu, device=device, dtype=torch.int64)
        perm = torch.randperm(self.total_tokens, device=device)
        self.assignment = expert_ids[perm]

        concat_indices = []
        self.expert_indices = []
        self.expert_offsets = []
        self.expert_ends_list = []
        running = 0
        for expert_id, count in enumerate(token_counts):
            indices = torch.nonzero(self.assignment == expert_id, as_tuple=False).flatten().contiguous()
            self.expert_indices.append(indices)
            concat_indices.append(indices.to(torch.int32))
            self.expert_offsets.append(running)
            running += count
            self.expert_ends_list.append(running)
        self.token_indices_concat = torch.cat(concat_indices).contiguous()
        self.expert_ends = torch.tensor(self.expert_ends_list, device=device, dtype=torch.int32)

        self.x = torch.randn(self.total_tokens, hidden_size, device=device, dtype=torch.float16)
        self.y_torch = torch.empty(self.total_tokens, hidden_size, device=device, dtype=torch.float16)
        self.y_triton = torch.empty(self.total_tokens, hidden_size, device=device, dtype=torch.float16)
        self.x_concat = torch.empty(self.total_tokens, hidden_size, device=device, dtype=torch.float16)
        self.hidden_triton = torch.empty(self.total_tokens, ffn_size, device=device, dtype=torch.float16)
        self.y_concat = torch.empty(self.total_tokens, hidden_size, device=device, dtype=torch.float16)

        self.w1 = [
            torch.randn(hidden_size, ffn_size, device=device, dtype=torch.float16)
            for _ in token_counts
        ]
        self.w2 = [
            torch.randn(ffn_size, hidden_size, device=device, dtype=torch.float16)
            for _ in token_counts
        ]
        self.w1_stack = torch.stack(self.w1).contiguous()
        self.w2_stack = torch.stack(self.w2).contiguous()
        self.hidden_loop = [
            torch.empty(count, ffn_size, device=device, dtype=torch.float16)
            for count in token_counts
        ]
        self.output_loop = [
            torch.empty(count, hidden_size, device=device, dtype=torch.float16)
            for count in token_counts
        ]
        self._block_map_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def block_map(self, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._block_map_cache.get(block_m)
        if cached is not None:
            return cached
        expert_ids = []
        row_starts = []
        for expert_id, tokens in enumerate(self.token_counts):
            start = self.expert_offsets[expert_id]
            for block in range(math.ceil(tokens / block_m)):
                expert_ids.append(expert_id)
                row_starts.append(start + block * block_m)
        device = torch.device("cuda:0")
        expert_tensor = torch.tensor(expert_ids, device=device, dtype=torch.int32)
        row_tensor = torch.tensor(row_starts, device=device, dtype=torch.int32)
        self._block_map_cache[block_m] = (expert_tensor, row_tensor)
        return expert_tensor, row_tensor

    def torch_routed_loop(self) -> None:
        for expert_id in range(self.num_experts):
            indices = self.expert_indices[expert_id]
            x_e = torch.index_select(self.x, 0, indices)
            torch.mm(x_e, self.w1[expert_id], out=self.hidden_loop[expert_id])
            self.hidden_loop[expert_id].relu_()
            torch.mm(self.hidden_loop[expert_id], self.w2[expert_id], out=self.output_loop[expert_id])
            self.y_torch.index_copy_(0, indices, self.output_loop[expert_id])

    def scatter(self, block_size: int) -> None:
        total = self.total_tokens * self.hidden_size
        grid = (triton.cdiv(total, block_size),)
        scatter_tokens_kernel[grid](
            self.x,
            self.x_concat,
            self.token_indices_concat,
            self.total_tokens,
            self.hidden_size,
            block_size,
            num_warps=4,
        )

    def grouped_ffn(self, block_m: int, block_n: int, block_k: int, warps: int) -> None:
        block_expert_ids, block_row_starts = self.block_map(block_m)
        row_tiles = block_expert_ids.numel()
        grid1 = (row_tiles, triton.cdiv(self.ffn_size, block_n))
        grouped_matmul_kernel[grid1](
            self.x_concat,
            self.w1_stack,
            self.hidden_triton,
            block_expert_ids,
            block_row_starts,
            self.expert_ends,
            self.hidden_size,
            self.ffn_size,
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=3,
        )
        self.hidden_triton.relu_()
        grid2 = (row_tiles, triton.cdiv(self.hidden_size, block_n))
        grouped_matmul_kernel[grid2](
            self.hidden_triton,
            self.w2_stack,
            self.y_concat,
            block_expert_ids,
            block_row_starts,
            self.expert_ends,
            self.ffn_size,
            self.hidden_size,
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=3,
        )

    def gather(self, block_size: int) -> None:
        total = self.total_tokens * self.hidden_size
        grid = (triton.cdiv(total, block_size),)
        gather_tokens_kernel[grid](
            self.y_concat,
            self.y_triton,
            self.token_indices_concat,
            self.total_tokens,
            self.hidden_size,
            block_size,
            num_warps=4,
        )

    def triton_routed(self, block_m: int, block_n: int, block_k: int, warps: int, move_block_size: int) -> None:
        self.scatter(move_block_size)
        self.grouped_ffn(block_m, block_n, block_k, warps)
        self.gather(move_block_size)

    def validate(self, block_m: int, block_n: int, block_k: int, warps: int, move_block_size: int) -> dict:
        self.torch_routed_loop()
        self.triton_routed(block_m, block_n, block_k, warps, move_block_size)
        torch.cuda.synchronize()
        reference = self.y_torch.float()
        candidate = self.y_triton.float()
        max_abs_diff = (reference - candidate).abs().max().item()
        max_ref_abs = reference.abs().max().item()
        finite = bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(candidate).all().item())
        return {
            "finite_outputs": finite,
            "max_abs_diff_vs_torch": float(max_abs_diff),
            "max_ref_abs": float(max_ref_abs),
            "relative_max_abs_diff": float(max_abs_diff / max_ref_abs) if max_ref_abs else 0.0,
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    token_counts = [int(item) for item in args.token_counts.split(",") if item.strip()]
    workload = RoutedMoeWorkload(token_counts, args.hidden_size, args.ffn_size, args.seed)
    torch_ms = cuda_time(workload.torch_routed_loop, args.warmup, args.iters)
    scatter_ms = cuda_time(lambda: workload.scatter(args.move_block_size), args.warmup, args.iters)
    compute_ms = cuda_time(
        lambda: workload.grouped_ffn(args.block_m, args.block_n, args.block_k, args.warps),
        args.warmup,
        args.iters,
    )
    gather_ms = cuda_time(lambda: workload.gather(args.move_block_size), args.warmup, args.iters)
    routed_ms = cuda_time(
        lambda: workload.triton_routed(args.block_m, args.block_n, args.block_k, args.warps, args.move_block_size),
        args.warmup,
        args.iters,
    )
    movement_ms = scatter_ms + gather_ms
    validation = workload.validate(args.block_m, args.block_n, args.block_k, args.warps, args.move_block_size)
    row_tiles = workload.block_map(args.block_m)[0].numel()
    return {
        "status": "ok",
        "token_counts": token_counts,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_size,
        "total_tokens": workload.total_tokens,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "warps": args.warps,
        "move_block_size": args.move_block_size,
        "row_tiles": row_tiles,
        "torch_routed_loop_ms": torch_ms,
        "triton_scatter_ms": scatter_ms,
        "triton_compute_ms": compute_ms,
        "triton_gather_ms": gather_ms,
        "triton_movement_ms": movement_ms,
        "triton_routed_ms": routed_ms,
        "triton_routed_speedup": torch_ms / routed_ms,
        "movement_fraction_of_routed": movement_ms / routed_ms,
        "compute_fraction_of_routed": compute_ms / routed_ms,
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
                "token_counts",
                "total_tokens",
                "row_tiles",
                "torch_routed_loop_ms",
                "triton_scatter_ms",
                "triton_compute_ms",
                "triton_gather_ms",
                "triton_routed_ms",
                "triton_routed_speedup",
                "movement_fraction_of_routed",
            ]
        )
        writer.writerow(
            [
                ",".join(str(item) for item in result.get("token_counts", [])),
                result.get("total_tokens"),
                result.get("row_tiles"),
                result.get("torch_routed_loop_ms"),
                result.get("triton_scatter_ms"),
                result.get("triton_compute_ms"),
                result.get("triton_gather_ms"),
                result.get("triton_routed_ms"),
                result.get("triton_routed_speedup"),
                result.get("movement_fraction_of_routed"),
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
