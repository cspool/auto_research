#!/usr/bin/env python3
"""H4.3 benchmark: sparse KV score-softmax-value fragment."""

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
def fused_sparse_ssv_kernel(
    k_ptr,
    v_ptr,
    q_ptr,
    indices_ptr,
    out_ptr,
    n_selected: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    rows = tl.arange(0, block_n)
    d_offsets = tl.arange(0, block_d)
    v_cols = pid_v * block_v + tl.arange(0, block_v)
    mask_n = rows < n_selected
    token_indices = tl.load(indices_ptr + rows, mask=mask_n, other=0)

    scores = tl.zeros((block_n,), dtype=tl.float32)
    for d0 in tl.range(0, key_dim, block_d):
        d = d0 + d_offsets
        k = tl.load(
            k_ptr + token_indices[:, None] * key_dim + d[None, :],
            mask=(mask_n[:, None]) & (d[None, :] < key_dim),
            other=0.0,
        )
        q = tl.load(q_ptr + d, mask=d < key_dim, other=0.0)
        scores += tl.sum(k * q[None, :], axis=1)

    scores = tl.where(mask_n, scores * scale, -float("inf"))
    max_score = tl.max(scores, axis=0)
    numer = tl.exp(scores - max_score)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    values = tl.load(
        v_ptr + token_indices[:, None] * value_dim + v_cols[None, :],
        mask=(mask_n[:, None]) & (v_cols[None, :] < value_dim),
        other=0.0,
    )
    acc = tl.sum(probs[:, None] * values, axis=0)
    tl.store(out_ptr + v_cols, acc, mask=v_cols < value_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument(
        "--order-mode",
        choices=["random_segment", "random_sorted", "random_shuffled", "clustered_segment"],
        default="random_segment",
    )
    parser.add_argument("--segment-counts", default=",".join(["16"] * 16))
    parser.add_argument("--tokens-per-segment", type=int, default=256)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument(
        "--variants",
        default="256,32,32,4;256,64,32,4;256,64,64,4;256,64,128,4;256,128,64,4;256,64,64,8",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2060)
    return parser.parse_args()


def parse_variants(value: str) -> list[tuple[int, int, int, int]]:
    variants = []
    for item in value.split(";"):
        if not item.strip():
            continue
        block_n, block_d, block_v, warps = [int(part) for part in item.split(",")]
        variants.append((block_n, block_d, block_v, warps))
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


class SparseSsvWorkload:
    def __init__(
        self,
        segment_counts: list[int],
        tokens_per_segment: int,
        key_dim: int,
        value_dim: int,
        order_mode: str,
        seed: int,
    ):
        torch.manual_seed(seed)
        self.segment_counts = segment_counts
        self.segment_count = len(segment_counts)
        self.tokens_per_segment = tokens_per_segment
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.total_cache_tokens = self.segment_count * tokens_per_segment
        self.selected_tokens = sum(segment_counts)
        self.order_mode = order_mode
        self.scale = 1.0 / math.sqrt(key_dim)
        device = torch.device("cuda:0")

        self.k_cache = torch.randn(self.total_cache_tokens, key_dim, device=device, dtype=torch.float16)
        self.v_cache = torch.randn(self.total_cache_tokens, value_dim, device=device, dtype=torch.float16)
        self.q = torch.randn(key_dim, device=device, dtype=torch.float16)
        self.indices = make_indices(segment_counts, tokens_per_segment, order_mode, device)
        self.indices_long = self.indices.to(torch.int64)
        self.gathered_k = torch.empty(self.selected_tokens, key_dim, device=device, dtype=torch.float16)
        self.gathered_v = torch.empty(self.selected_tokens, value_dim, device=device, dtype=torch.float16)
        self.out_flat = torch.empty(value_dim, device=device, dtype=torch.float32)
        self.out_triton = torch.empty(value_dim, device=device, dtype=torch.float32)
        self.scores = torch.empty(self.selected_tokens, device=device, dtype=torch.float32)
        self.probs = torch.empty(self.selected_tokens, device=device, dtype=torch.float32)

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

    def torch_gather_k(self) -> None:
        torch.index_select(self.k_cache, 0, self.indices_long, out=self.gathered_k)

    def torch_gather_v(self) -> None:
        torch.index_select(self.v_cache, 0, self.indices_long, out=self.gathered_v)

    def torch_score_softmax(self) -> None:
        scores = torch.mv(self.gathered_k.float(), self.q.float()) * self.scale
        self.scores = scores
        self.probs = torch.softmax(scores, dim=0)

    def torch_value(self) -> None:
        self.out_flat = torch.mv(self.gathered_v.float().t(), self.probs)

    def torch_flat(self) -> None:
        self.torch_gather_k()
        self.torch_gather_v()
        self.torch_score_softmax()
        self.torch_value()

    def triton_ssv(self, block_n: int, block_d: int, block_v: int, warps: int) -> None:
        if self.selected_tokens > block_n:
            raise ValueError("selected token count must fit within block_n for this single-tile kernel")
        grid = (triton.cdiv(self.value_dim, block_v),)
        fused_sparse_ssv_kernel[grid](
            self.k_cache,
            self.v_cache,
            self.q,
            self.indices,
            self.out_triton,
            self.selected_tokens,
            self.key_dim,
            self.value_dim,
            self.scale,
            block_n,
            block_d,
            block_v,
            num_warps=warps,
            num_stages=3,
        )

    def validate_variant(self, block_n: int, block_d: int, block_v: int, warps: int) -> dict:
        self.torch_flat()
        self.triton_ssv(block_n, block_d, block_v, warps)
        torch.cuda.synchronize()
        reference = self.out_flat.float()
        candidate = self.out_triton.float()
        max_abs_diff = (reference - candidate).abs().max().item()
        mean_abs_diff = (reference - candidate).abs().mean().item()
        ref_abs = reference.abs().max().item()
        finite = bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(candidate).all().item())
        return {
            "finite_outputs": finite,
            "max_abs_diff_triton_vs_flat": float(max_abs_diff),
            "mean_abs_diff_triton_vs_flat": float(mean_abs_diff),
            "max_ref_abs": float(ref_abs),
            "relative_max_abs_diff_triton": float(max_abs_diff / ref_abs) if ref_abs else 0.0,
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    segment_counts = [int(item) for item in args.segment_counts.split(",") if item.strip()]
    variants = parse_variants(args.variants)
    workload = SparseSsvWorkload(
        segment_counts=segment_counts,
        tokens_per_segment=args.tokens_per_segment,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        order_mode=args.order_mode,
        seed=args.seed,
    )

    workload.torch_flat()
    torch.cuda.synchronize()

    flat_total_ms = cuda_time(workload.torch_flat, args.warmup, args.iters)
    gather_k_ms = cuda_time(workload.torch_gather_k, args.warmup, args.iters)
    gather_v_ms = cuda_time(workload.torch_gather_v, args.warmup, args.iters)
    score_softmax_ms = cuda_time(workload.torch_score_softmax, args.warmup, args.iters)
    value_ms = cuda_time(workload.torch_value, args.warmup, args.iters)

    variant_results = []
    for block_n, block_d, block_v, warps in variants:
        if workload.selected_tokens > block_n:
            continue
        elapsed = cuda_time(
            lambda bn=block_n, bd=block_d, bv=block_v, nw=warps: workload.triton_ssv(bn, bd, bv, nw),
            args.warmup,
            args.iters,
        )
        validation = workload.validate_variant(block_n, block_d, block_v, warps)
        variant_results.append(
            {
                "block_n": block_n,
                "block_d": block_d,
                "block_v": block_v,
                "warps": warps,
                "triton_ssv_ms": elapsed,
                "triton_vs_flat_speedup": flat_total_ms / elapsed,
                "validation": validation,
            }
        )

    if not variant_results:
        raise ValueError("no valid variants: block_n must be >= selected token count")

    best = min(variant_results, key=lambda item: item["triton_ssv_ms"])
    return {
        "status": "ok",
        "order_mode": args.order_mode,
        "segment_counts": segment_counts,
        "segment_count": workload.segment_count,
        "tokens_per_segment": args.tokens_per_segment,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "total_cache_tokens": workload.total_cache_tokens,
        "selected_tokens": workload.selected_tokens,
        "selection_fraction": workload.selected_tokens / workload.total_cache_tokens,
        "torch_flat_total_ms": flat_total_ms,
        "torch_gather_k_ms": gather_k_ms,
        "torch_gather_v_ms": gather_v_ms,
        "torch_score_softmax_ms": score_softmax_ms,
        "torch_value_ms": value_ms,
        "torch_component_sum_ms": gather_k_ms + gather_v_ms + score_softmax_ms + value_ms,
        "best_triton_ssv_ms": best["triton_ssv_ms"],
        "best_triton_vs_flat_speedup": flat_total_ms / best["triton_ssv_ms"],
        "best_variant": {
            "block_n": best["block_n"],
            "block_d": best["block_d"],
            "block_v": best["block_v"],
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
                "block_n",
                "block_d",
                "block_v",
                "warps",
                "triton_ssv_ms",
                "triton_vs_flat_speedup",
                "max_abs_diff_triton_vs_flat",
                "mean_abs_diff_triton_vs_flat",
                "relative_max_abs_diff_triton",
            ]
        )
        for item in result.get("variant_results", []):
            validation = item.get("validation", {})
            writer.writerow(
                [
                    result.get("order_mode"),
                    item.get("block_n"),
                    item.get("block_d"),
                    item.get("block_v"),
                    item.get("warps"),
                    item.get("triton_ssv_ms"),
                    item.get("triton_vs_flat_speedup"),
                    validation.get("max_abs_diff_triton_vs_flat"),
                    validation.get("mean_abs_diff_triton_vs_flat"),
                    validation.get("relative_max_abs_diff_triton"),
                ]
            )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order_mode",
                "selected_tokens",
                "key_dim",
                "value_dim",
                "torch_flat_total_ms",
                "torch_gather_k_ms",
                "torch_gather_v_ms",
                "torch_score_softmax_ms",
                "torch_value_ms",
                "best_triton_ssv_ms",
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
                result.get("key_dim"),
                result.get("value_dim"),
                result.get("torch_flat_total_ms"),
                result.get("torch_gather_k_ms"),
                result.get("torch_gather_v_ms"),
                result.get("torch_score_softmax_ms"),
                result.get("torch_value_ms"),
                result.get("best_triton_ssv_ms"),
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
