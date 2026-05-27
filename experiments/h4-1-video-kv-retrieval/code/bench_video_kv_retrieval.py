#!/usr/bin/env python3
"""H4.1 benchmark: sparse video/VLM KV retrieval gather-score fragment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def fused_gather_score_kernel(
    k_ptr,
    q_ptr,
    indices_ptr,
    out_ptr,
    n_selected: tl.constexpr,
    dim_size: tl.constexpr,
    query_count: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
    block_q: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)

    rows = pid_m * block_m + tl.arange(0, block_m)
    q_cols = pid_q * block_q + tl.arange(0, block_q)
    d_offsets = tl.arange(0, block_d)

    token_indices = tl.load(indices_ptr + rows, mask=rows < n_selected, other=0)
    acc = tl.zeros((block_m, block_q), dtype=tl.float32)

    for d0 in tl.range(0, dim_size, block_d):
        d = d0 + d_offsets
        k = tl.load(
            k_ptr + token_indices[:, None] * dim_size + d[None, :],
            mask=(rows[:, None] < n_selected) & (d[None, :] < dim_size),
            other=0.0,
        )
        q = tl.load(
            q_ptr + d[:, None] * query_count + q_cols[None, :],
            mask=(d[:, None] < dim_size) & (q_cols[None, :] < query_count),
            other=0.0,
        )
        acc += tl.dot(k, q)

    tl.store(
        out_ptr + rows[:, None] * query_count + q_cols[None, :],
        acc,
        mask=(rows[:, None] < n_selected) & (q_cols[None, :] < query_count),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--segment-counts", default=",".join(["32"] * 32))
    parser.add_argument("--tokens-per-segment", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--index-mode", choices=["random", "clustered"], default="random")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-d", type=int, default=64)
    parser.add_argument("--block-q", type=int, default=8)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2040)
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


class VideoKvWorkload:
    def __init__(
        self,
        segment_counts: list[int],
        tokens_per_segment: int,
        dim: int,
        query_count: int,
        index_mode: str,
        seed: int,
    ):
        torch.manual_seed(seed)
        self.segment_counts = segment_counts
        self.segment_count = len(segment_counts)
        self.tokens_per_segment = tokens_per_segment
        self.dim = dim
        self.query_count = query_count
        self.total_cache_tokens = self.segment_count * tokens_per_segment
        self.selected_tokens = sum(segment_counts)
        device = torch.device("cuda:0")

        self.k_cache = torch.randn(self.total_cache_tokens, dim, device=device, dtype=torch.float16)
        self.q = torch.randn(dim, query_count, device=device, dtype=torch.float16)
        self.gathered = torch.empty(self.selected_tokens, dim, device=device, dtype=torch.float16)
        self.out_loop = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float16)
        self.out_flat = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float16)
        self.out_triton = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float16)

        self.segment_indices: list[torch.Tensor] = []
        concat_indices = []
        self.segment_offsets = []
        running = 0
        for segment_id, count in enumerate(segment_counts):
            if count > tokens_per_segment:
                raise ValueError("segment count cannot exceed tokens_per_segment")
            if index_mode == "random":
                local = torch.randperm(tokens_per_segment, device=device, dtype=torch.int64)[:count]
            else:
                max_start = max(tokens_per_segment - count, 0)
                start = int(torch.randint(max_start + 1, (1,), device=device).item())
                local = torch.arange(start, start + count, device=device, dtype=torch.int64)
            indices = segment_id * tokens_per_segment + local
            indices = indices.contiguous()
            self.segment_indices.append(indices)
            concat_indices.append(indices.to(torch.int32))
            self.segment_offsets.append(running)
            running += count
        self.selected_indices = torch.cat(concat_indices).contiguous()
        self.selected_indices_long = self.selected_indices.to(torch.int64)

        self.segment_gathered = [
            torch.empty(count, dim, device=device, dtype=torch.float16) for count in segment_counts
        ]
        self.segment_outputs = [
            torch.empty(count, query_count, device=device, dtype=torch.float16) for count in segment_counts
        ]

    def torch_segment_loop(self) -> None:
        for segment_id, count in enumerate(self.segment_counts):
            start = self.segment_offsets[segment_id]
            end = start + count
            torch.index_select(
                self.k_cache,
                0,
                self.segment_indices[segment_id],
                out=self.segment_gathered[segment_id],
            )
            torch.mm(self.segment_gathered[segment_id], self.q, out=self.segment_outputs[segment_id])
            self.out_loop[start:end].copy_(self.segment_outputs[segment_id])

    def torch_gather(self) -> None:
        torch.index_select(self.k_cache, 0, self.selected_indices_long, out=self.gathered)

    def torch_score(self) -> None:
        torch.mm(self.gathered, self.q, out=self.out_flat)

    def torch_flat(self) -> None:
        self.torch_gather()
        self.torch_score()

    def triton_fused(self, block_m: int, block_d: int, block_q: int, warps: int) -> None:
        grid = (triton.cdiv(self.selected_tokens, block_m), triton.cdiv(self.query_count, block_q))
        fused_gather_score_kernel[grid](
            self.k_cache,
            self.q,
            self.selected_indices,
            self.out_triton,
            self.selected_tokens,
            self.dim,
            self.query_count,
            block_m,
            block_d,
            block_q,
            num_warps=warps,
            num_stages=3,
        )

    def validate(self, block_m: int, block_d: int, block_q: int, warps: int) -> dict:
        self.torch_segment_loop()
        self.torch_flat()
        self.triton_fused(block_m, block_d, block_q, warps)
        torch.cuda.synchronize()
        loop = self.out_loop.float()
        flat = self.out_flat.float()
        triton_out = self.out_triton.float()
        loop_flat_diff = (loop - flat).abs().max().item()
        flat_triton_diff = (flat - triton_out).abs().max().item()
        ref_abs = flat.abs().max().item()
        finite = (
            bool(torch.isfinite(loop).all().item())
            and bool(torch.isfinite(flat).all().item())
            and bool(torch.isfinite(triton_out).all().item())
        )
        return {
            "finite_outputs": finite,
            "max_abs_diff_loop_vs_flat": float(loop_flat_diff),
            "max_abs_diff_triton_vs_flat": float(flat_triton_diff),
            "max_ref_abs": float(ref_abs),
            "relative_max_abs_diff_triton": float(flat_triton_diff / ref_abs) if ref_abs else 0.0,
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    segment_counts = [int(item) for item in args.segment_counts.split(",") if item.strip()]
    workload = VideoKvWorkload(
        segment_counts=segment_counts,
        tokens_per_segment=args.tokens_per_segment,
        dim=args.dim,
        query_count=args.query_count,
        index_mode=args.index_mode,
        seed=args.seed,
    )

    # Prime the gathered buffer before timing score-only.
    workload.torch_gather()
    torch.cuda.synchronize()

    segment_loop_ms = cuda_time(workload.torch_segment_loop, args.warmup, args.iters)
    flat_total_ms = cuda_time(workload.torch_flat, args.warmup, args.iters)
    gather_ms = cuda_time(workload.torch_gather, args.warmup, args.iters)
    score_ms = cuda_time(workload.torch_score, args.warmup, args.iters)
    triton_ms = cuda_time(
        lambda: workload.triton_fused(args.block_m, args.block_d, args.block_q, args.warps),
        args.warmup,
        args.iters,
    )
    validation = workload.validate(args.block_m, args.block_d, args.block_q, args.warps)

    return {
        "status": "ok",
        "segment_counts": segment_counts,
        "segment_count": workload.segment_count,
        "tokens_per_segment": args.tokens_per_segment,
        "index_mode": args.index_mode,
        "dim": args.dim,
        "query_count": args.query_count,
        "total_cache_tokens": workload.total_cache_tokens,
        "selected_tokens": workload.selected_tokens,
        "selection_fraction": workload.selected_tokens / workload.total_cache_tokens,
        "block_m": args.block_m,
        "block_d": args.block_d,
        "block_q": args.block_q,
        "warps": args.warps,
        "torch_segment_loop_ms": segment_loop_ms,
        "torch_flat_total_ms": flat_total_ms,
        "torch_gather_ms": gather_ms,
        "torch_score_ms": score_ms,
        "triton_fused_ms": triton_ms,
        "triton_vs_segment_speedup": segment_loop_ms / triton_ms,
        "triton_vs_flat_speedup": flat_total_ms / triton_ms,
        "flat_vs_segment_speedup": segment_loop_ms / flat_total_ms,
        "gather_fraction_of_flat": gather_ms / flat_total_ms,
        "score_fraction_of_flat": score_ms / flat_total_ms,
        "approx_segment_loop_launches": workload.segment_count * 3,
        "approx_flat_launches": 2,
        "approx_triton_launches": 1,
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
                "segment_counts",
                "selected_tokens",
                "selection_fraction",
                "torch_segment_loop_ms",
                "torch_flat_total_ms",
                "torch_gather_ms",
                "torch_score_ms",
                "triton_fused_ms",
                "triton_vs_segment_speedup",
                "triton_vs_flat_speedup",
                "gather_fraction_of_flat",
            ]
        )
        writer.writerow(
            [
                ",".join(str(item) for item in result.get("segment_counts", [])),
                result.get("selected_tokens"),
                result.get("selection_fraction"),
                result.get("torch_segment_loop_ms"),
                result.get("torch_flat_total_ms"),
                result.get("torch_gather_ms"),
                result.get("torch_score_ms"),
                result.get("triton_fused_ms"),
                result.get("triton_vs_segment_speedup"),
                result.get("triton_vs_flat_speedup"),
                result.get("gather_fraction_of_flat"),
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
