#!/usr/bin/env python3
"""H4.2 benchmark: locality and tile variants for sparse KV retrieval."""

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
    parser.add_argument(
        "--order-mode",
        choices=["random_segment", "random_sorted", "random_shuffled", "clustered_segment"],
        default="random_segment",
    )
    parser.add_argument("--segment-counts", default=",".join(["32"] * 32))
    parser.add_argument("--tokens-per-segment", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument(
        "--variants",
        default="8,64,8,4;16,64,8,4;32,64,8,4;16,128,8,4;32,128,8,4;16,64,8,8",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2050)
    return parser.parse_args()


def parse_variants(value: str) -> list[tuple[int, int, int, int]]:
    variants = []
    for item in value.split(";"):
        if not item.strip():
            continue
        block_m, block_d, block_q, warps = [int(part) for part in item.split(",")]
        variants.append((block_m, block_d, block_q, warps))
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


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


def make_indices(
    segment_counts: list[int],
    tokens_per_segment: int,
    order_mode: str,
    device: torch.device,
) -> torch.Tensor:
    random_segments = []
    clustered_segments = []
    for segment_id, count in enumerate(segment_counts):
        if count > tokens_per_segment:
            raise ValueError("segment count cannot exceed tokens_per_segment")
        random_local = torch.randperm(tokens_per_segment, device=device, dtype=torch.int64)[:count]
        random_segments.append(segment_id * tokens_per_segment + random_local)

        max_start = max(tokens_per_segment - count, 0)
        start = int(torch.randint(max_start + 1, (1,), device=device).item())
        clustered_local = torch.arange(start, start + count, device=device, dtype=torch.int64)
        clustered_segments.append(segment_id * tokens_per_segment + clustered_local)

    random_concat = torch.cat(random_segments).contiguous()
    if order_mode == "random_segment":
        return random_concat.to(torch.int32).contiguous()
    if order_mode == "random_sorted":
        return torch.sort(random_concat).values.to(torch.int32).contiguous()
    if order_mode == "random_shuffled":
        perm = torch.randperm(random_concat.numel(), device=device)
        return random_concat[perm].to(torch.int32).contiguous()
    if order_mode == "clustered_segment":
        return torch.cat(clustered_segments).to(torch.int32).contiguous()
    raise ValueError(f"unknown order mode: {order_mode}")


class LocalityWorkload:
    def __init__(
        self,
        segment_counts: list[int],
        tokens_per_segment: int,
        dim: int,
        query_count: int,
        order_mode: str,
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
        self.order_mode = order_mode
        device = torch.device("cuda:0")

        self.k_cache = torch.randn(self.total_cache_tokens, dim, device=device, dtype=torch.float16)
        self.q = torch.randn(dim, query_count, device=device, dtype=torch.float16)
        self.indices = make_indices(segment_counts, tokens_per_segment, order_mode, device)
        self.indices_long = self.indices.to(torch.int64)
        self.gathered = torch.empty(self.selected_tokens, dim, device=device, dtype=torch.float16)
        self.out_flat = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float16)
        self.out_triton = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float16)

    def order_stats(self) -> dict:
        idx = self.indices_long.detach().cpu()
        if idx.numel() <= 1:
            return {"order_span_mean": 0.0, "order_span_p95": 0.0, "monotonic_fraction": 1.0}
        diffs = (idx[1:] - idx[:-1]).abs().float()
        monotonic = (idx[1:] >= idx[:-1]).float().mean().item()
        return {
            "order_span_mean": float(diffs.mean().item()),
            "order_span_p95": float(torch.quantile(diffs, 0.95).item()),
            "monotonic_fraction": float(monotonic),
        }

    def torch_gather(self) -> None:
        torch.index_select(self.k_cache, 0, self.indices_long, out=self.gathered)

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
            self.indices,
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

    def validate_variant(self, block_m: int, block_d: int, block_q: int, warps: int) -> dict:
        self.torch_flat()
        self.triton_fused(block_m, block_d, block_q, warps)
        torch.cuda.synchronize()
        reference = self.out_flat.float()
        candidate = self.out_triton.float()
        max_abs_diff = (reference - candidate).abs().max().item()
        ref_abs = reference.abs().max().item()
        finite = bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(candidate).all().item())
        return {
            "finite_outputs": finite,
            "max_abs_diff_triton_vs_flat": float(max_abs_diff),
            "max_ref_abs": float(ref_abs),
            "relative_max_abs_diff_triton": float(max_abs_diff / ref_abs) if ref_abs else 0.0,
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    segment_counts = [int(item) for item in args.segment_counts.split(",") if item.strip()]
    variants = parse_variants(args.variants)
    workload = LocalityWorkload(
        segment_counts=segment_counts,
        tokens_per_segment=args.tokens_per_segment,
        dim=args.dim,
        query_count=args.query_count,
        order_mode=args.order_mode,
        seed=args.seed,
    )

    workload.torch_gather()
    torch.cuda.synchronize()

    flat_total_ms = cuda_time(workload.torch_flat, args.warmup, args.iters)
    gather_ms = cuda_time(workload.torch_gather, args.warmup, args.iters)
    score_ms = cuda_time(workload.torch_score, args.warmup, args.iters)

    variant_results = []
    for block_m, block_d, block_q, warps in variants:
        elapsed = cuda_time(
            lambda bm=block_m, bd=block_d, bq=block_q, nw=warps: workload.triton_fused(bm, bd, bq, nw),
            args.warmup,
            args.iters,
        )
        validation = workload.validate_variant(block_m, block_d, block_q, warps)
        variant_results.append(
            {
                "block_m": block_m,
                "block_d": block_d,
                "block_q": block_q,
                "warps": warps,
                "triton_fused_ms": elapsed,
                "triton_vs_flat_speedup": flat_total_ms / elapsed,
                "validation": validation,
            }
        )

    best = min(variant_results, key=lambda item: item["triton_fused_ms"])
    return {
        "status": "ok",
        "order_mode": args.order_mode,
        "segment_counts": segment_counts,
        "segment_count": workload.segment_count,
        "tokens_per_segment": args.tokens_per_segment,
        "dim": args.dim,
        "query_count": args.query_count,
        "total_cache_tokens": workload.total_cache_tokens,
        "selected_tokens": workload.selected_tokens,
        "selection_fraction": workload.selected_tokens / workload.total_cache_tokens,
        "torch_flat_total_ms": flat_total_ms,
        "torch_gather_ms": gather_ms,
        "torch_score_ms": score_ms,
        "gather_fraction_of_flat": gather_ms / flat_total_ms,
        "score_fraction_of_flat": score_ms / flat_total_ms,
        "best_triton_fused_ms": best["triton_fused_ms"],
        "best_triton_vs_flat_speedup": flat_total_ms / best["triton_fused_ms"],
        "best_variant": {
            "block_m": best["block_m"],
            "block_d": best["block_d"],
            "block_q": best["block_q"],
            "warps": best["warps"],
        },
        "order_stats": workload.order_stats(),
        "variant_results": variant_results,
    }


def write_outputs(out_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "variants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_mode",
                "block_m",
                "block_d",
                "block_q",
                "warps",
                "triton_fused_ms",
                "triton_vs_flat_speedup",
                "max_abs_diff_triton_vs_flat",
                "relative_max_abs_diff_triton",
            ]
        )
        for item in result.get("variant_results", []):
            validation = item.get("validation", {})
            writer.writerow(
                [
                    result.get("order_mode"),
                    item.get("block_m"),
                    item.get("block_d"),
                    item.get("block_q"),
                    item.get("warps"),
                    item.get("triton_fused_ms"),
                    item.get("triton_vs_flat_speedup"),
                    validation.get("max_abs_diff_triton_vs_flat"),
                    validation.get("relative_max_abs_diff_triton"),
                ]
            )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_mode",
                "selected_tokens",
                "torch_flat_total_ms",
                "torch_gather_ms",
                "torch_score_ms",
                "gather_fraction_of_flat",
                "best_triton_fused_ms",
                "best_triton_vs_flat_speedup",
                "best_variant",
                "order_span_mean",
                "order_span_p95",
                "monotonic_fraction",
            ]
        )
        order_stats = result.get("order_stats", {})
        writer.writerow(
            [
                result.get("order_mode"),
                result.get("selected_tokens"),
                result.get("torch_flat_total_ms"),
                result.get("torch_gather_ms"),
                result.get("torch_score_ms"),
                result.get("gather_fraction_of_flat"),
                result.get("best_triton_fused_ms"),
                result.get("best_triton_vs_flat_speedup"),
                json.dumps(result.get("best_variant", {}), sort_keys=True),
                order_stats.get("order_span_mean"),
                order_stats.get("order_span_p95"),
                order_stats.get("monotonic_fraction"),
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
