#!/usr/bin/env python3
"""H4.5 benchmark: multi-query sparse attention with online softmax."""

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
def partial_sparse_oneq_ssv_kernel(
    k_ptr,
    v_ptr,
    q_ptr,
    indices_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_o_ptr,
    q_query: tl.constexpr,
    n_selected: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    num_v_blocks: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_v = tl.program_id(axis=1)
    rows = pid_n * block_n + tl.arange(0, block_n)
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
        q = tl.load(q_ptr + q_query * key_dim + d, mask=d < key_dim, other=0.0)
        scores += tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), axis=1)

    scores = tl.where(mask_n, scores * scale, -float("inf"))
    local_m = tl.max(scores, axis=0)
    weights = tl.exp(scores - local_m)
    local_l = tl.sum(weights, axis=0)

    values = tl.load(
        v_ptr + token_indices[:, None] * value_dim + v_cols[None, :],
        mask=(mask_n[:, None]) & (v_cols[None, :] < value_dim),
        other=0.0,
    )
    partial_o = tl.sum(weights[:, None] * values.to(tl.float32), axis=0)

    meta_offset = pid_n * num_v_blocks + pid_v
    tl.store(partial_m_ptr + meta_offset, local_m)
    tl.store(partial_l_ptr + meta_offset, local_l)
    tl.store(partial_o_ptr + pid_n * value_dim + v_cols, partial_o, mask=v_cols < value_dim)


@triton.jit
def reduce_sparse_oneq_ssv_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_o_ptr,
    out_ptr,
    q_query: tl.constexpr,
    value_dim: tl.constexpr,
    num_n_blocks: tl.constexpr,
    num_v_blocks: tl.constexpr,
    block_v: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    v_cols = pid_v * block_v + tl.arange(0, block_v)

    global_m = -float("inf")
    for block_id in tl.range(0, num_n_blocks):
        m = tl.load(partial_m_ptr + block_id * num_v_blocks + pid_v)
        global_m = tl.maximum(global_m, m)

    denom = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((block_v,), dtype=tl.float32)
    for block_id in tl.range(0, num_n_blocks):
        meta_offset = block_id * num_v_blocks + pid_v
        m = tl.load(partial_m_ptr + meta_offset)
        l = tl.load(partial_l_ptr + meta_offset)
        coeff = tl.exp(m - global_m)
        partial_o = tl.load(
            partial_o_ptr + block_id * value_dim + v_cols,
            mask=v_cols < value_dim,
            other=0.0,
        )
        denom += coeff * l
        acc += coeff * partial_o

    tl.store(out_ptr + q_query * value_dim + v_cols, acc / denom, mask=v_cols < value_dim)


@triton.jit
def partial_sparse_multiq_ssv_kernel(
    k_ptr,
    v_ptr,
    q_ptr,
    indices_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_o_ptr,
    n_selected: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    query_count: tl.constexpr,
    num_v_blocks: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    block_v: tl.constexpr,
    block_q: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_v = tl.program_id(axis=1)
    pid_q = tl.program_id(axis=2)

    rows = pid_n * block_n + tl.arange(0, block_n)
    d_offsets = tl.arange(0, block_d)
    v_cols = pid_v * block_v + tl.arange(0, block_v)
    q_cols = pid_q * block_q + tl.arange(0, block_q)
    mask_n = rows < n_selected
    mask_q = q_cols < query_count
    token_indices = tl.load(indices_ptr + rows, mask=mask_n, other=0)

    scores = tl.zeros((block_n, block_q), dtype=tl.float32)
    for d0 in tl.range(0, key_dim, block_d):
        d = d0 + d_offsets
        k = tl.load(
            k_ptr + token_indices[:, None] * key_dim + d[None, :],
            mask=(mask_n[:, None]) & (d[None, :] < key_dim),
            other=0.0,
        )
        q = tl.load(
            q_ptr + d[:, None] + q_cols[None, :] * key_dim,
            mask=(d[:, None] < key_dim) & mask_q[None, :],
            other=0.0,
        )
        scores += tl.dot(
            k.to(tl.float32),
            q.to(tl.float32),
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    scores = tl.where(mask_n[:, None] & mask_q[None, :], scores * scale, -float("inf"))
    local_m = tl.max(scores, axis=0)
    weights = tl.exp(scores - local_m[None, :])
    local_l = tl.sum(weights, axis=0)

    values = tl.load(
        v_ptr + token_indices[:, None] * value_dim + v_cols[None, :],
        mask=(mask_n[:, None]) & (v_cols[None, :] < value_dim),
        other=0.0,
    )
    partial_o = tl.dot(
        tl.trans(weights),
        values.to(tl.float32),
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    meta_offsets = (pid_n * num_v_blocks + pid_v) * query_count + q_cols
    tl.store(partial_m_ptr + meta_offsets, local_m, mask=mask_q)
    tl.store(partial_l_ptr + meta_offsets, local_l, mask=mask_q)
    tl.store(
        partial_o_ptr
        + pid_n * query_count * value_dim
        + q_cols[:, None] * value_dim
        + v_cols[None, :],
        partial_o,
        mask=mask_q[:, None] & (v_cols[None, :] < value_dim),
    )


@triton.jit
def reduce_sparse_multiq_ssv_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_o_ptr,
    out_ptr,
    value_dim: tl.constexpr,
    query_count: tl.constexpr,
    num_n_blocks: tl.constexpr,
    num_v_blocks: tl.constexpr,
    block_v: tl.constexpr,
    block_q: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)
    v_cols = pid_v * block_v + tl.arange(0, block_v)
    q_cols = pid_q * block_q + tl.arange(0, block_q)
    mask_q = q_cols < query_count

    global_m = tl.full((block_q,), -float("inf"), dtype=tl.float32)
    for block_id in tl.range(0, num_n_blocks):
        m = tl.load(
            partial_m_ptr + (block_id * num_v_blocks + pid_v) * query_count + q_cols,
            mask=mask_q,
            other=-float("inf"),
        )
        global_m = tl.maximum(global_m, m)

    denom = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, block_v), dtype=tl.float32)
    for block_id in tl.range(0, num_n_blocks):
        meta_offsets = (block_id * num_v_blocks + pid_v) * query_count + q_cols
        m = tl.load(partial_m_ptr + meta_offsets, mask=mask_q, other=-float("inf"))
        l = tl.load(partial_l_ptr + meta_offsets, mask=mask_q, other=0.0)
        coeff = tl.exp(m - global_m)
        partial_o = tl.load(
            partial_o_ptr
            + block_id * query_count * value_dim
            + q_cols[:, None] * value_dim
            + v_cols[None, :],
            mask=mask_q[:, None] & (v_cols[None, :] < value_dim),
            other=0.0,
        )
        denom += coeff * l
        acc += coeff[:, None] * partial_o

    tl.store(
        out_ptr + q_cols[:, None] * value_dim + v_cols[None, :],
        acc / denom[:, None],
        mask=mask_q[:, None] & (v_cols[None, :] < value_dim),
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
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument(
        "--variants",
        default=(
            "128,64,64,4,4;"
            "128,64,128,4,4;"
            "256,64,64,4,4;"
            "128,128,64,4,4;"
            "128,64,64,4,8;"
            "64,64,64,4,4"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2071)
    return parser.parse_args()


def parse_variants(value: str) -> list[tuple[int, int, int, int, int]]:
    variants = []
    for item in value.split(";"):
        if not item.strip():
            continue
        block_n, block_d, block_v, block_q, warps = [int(part) for part in item.split(",")]
        variants.append((block_n, block_d, block_v, block_q, warps))
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


class MultiQuerySparseWorkload:
    def __init__(
        self,
        segment_counts: list[int],
        tokens_per_segment: int,
        key_dim: int,
        value_dim: int,
        query_count: int,
        order_mode: str,
        seed: int,
    ):
        torch.manual_seed(seed)
        self.segment_counts = segment_counts
        self.segment_count = len(segment_counts)
        self.tokens_per_segment = tokens_per_segment
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_count = query_count
        self.total_cache_tokens = self.segment_count * tokens_per_segment
        self.selected_tokens = sum(segment_counts)
        self.order_mode = order_mode
        self.scale = 1.0 / math.sqrt(key_dim)
        device = torch.device("cuda:0")

        self.k_cache = torch.randn(self.total_cache_tokens, key_dim, device=device, dtype=torch.float16)
        self.v_cache = torch.randn(self.total_cache_tokens, value_dim, device=device, dtype=torch.float16)
        self.q = torch.randn(query_count, key_dim, device=device, dtype=torch.float16)
        self.indices = make_indices(segment_counts, tokens_per_segment, order_mode, device)
        self.indices_long = self.indices.to(torch.int64)
        self.gathered_k = torch.empty(self.selected_tokens, key_dim, device=device, dtype=torch.float16)
        self.gathered_v = torch.empty(self.selected_tokens, value_dim, device=device, dtype=torch.float16)
        self.scores = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float32)
        self.probs = torch.empty(self.selected_tokens, query_count, device=device, dtype=torch.float32)
        self.out_flat = torch.empty(query_count, value_dim, device=device, dtype=torch.float32)
        self.out_grouped = torch.empty(query_count, value_dim, device=device, dtype=torch.float32)
        self.out_oneq = torch.empty(query_count, value_dim, device=device, dtype=torch.float32)
        self._grouped_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._oneq_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

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

    def grouped_buffers(self, block_n: int, block_v: int, block_q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (block_n, block_v, block_q)
        cached = self._grouped_cache.get(key)
        if cached is not None:
            return cached
        device = torch.device("cuda:0")
        num_n_blocks = triton.cdiv(self.selected_tokens, block_n)
        num_v_blocks = triton.cdiv(self.value_dim, block_v)
        partial_m = torch.empty(num_n_blocks, num_v_blocks, self.query_count, device=device, dtype=torch.float32)
        partial_l = torch.empty(num_n_blocks, num_v_blocks, self.query_count, device=device, dtype=torch.float32)
        partial_o = torch.empty(num_n_blocks, self.query_count, self.value_dim, device=device, dtype=torch.float32)
        self._grouped_cache[key] = (partial_m, partial_l, partial_o)
        return partial_m, partial_l, partial_o

    def oneq_buffers(self, block_n: int, block_v: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (block_n, block_v)
        cached = self._oneq_cache.get(key)
        if cached is not None:
            return cached
        device = torch.device("cuda:0")
        num_n_blocks = triton.cdiv(self.selected_tokens, block_n)
        num_v_blocks = triton.cdiv(self.value_dim, block_v)
        partial_m = torch.empty(num_n_blocks, num_v_blocks, device=device, dtype=torch.float32)
        partial_l = torch.empty(num_n_blocks, num_v_blocks, device=device, dtype=torch.float32)
        partial_o = torch.empty(num_n_blocks, self.value_dim, device=device, dtype=torch.float32)
        self._oneq_cache[key] = (partial_m, partial_l, partial_o)
        return partial_m, partial_l, partial_o

    def torch_gather_k(self) -> None:
        torch.index_select(self.k_cache, 0, self.indices_long, out=self.gathered_k)

    def torch_gather_v(self) -> None:
        torch.index_select(self.v_cache, 0, self.indices_long, out=self.gathered_v)

    def torch_score_softmax(self) -> None:
        scores = torch.mm(self.gathered_k.float(), self.q.float().t()) * self.scale
        self.scores = scores
        self.probs = torch.softmax(scores, dim=0)

    def torch_value(self) -> None:
        self.out_flat = torch.mm(self.probs.t(), self.gathered_v.float())

    def torch_flat(self) -> None:
        self.torch_gather_k()
        self.torch_gather_v()
        self.torch_score_softmax()
        self.torch_value()

    def triton_grouped(self, block_n: int, block_d: int, block_v: int, block_q: int, warps: int) -> None:
        partial_m, partial_l, partial_o = self.grouped_buffers(block_n, block_v, block_q)
        num_n_blocks = triton.cdiv(self.selected_tokens, block_n)
        num_v_blocks = triton.cdiv(self.value_dim, block_v)
        num_q_blocks = triton.cdiv(self.query_count, block_q)
        partial_sparse_multiq_ssv_kernel[(num_n_blocks, num_v_blocks, num_q_blocks)](
            self.k_cache,
            self.v_cache,
            self.q,
            self.indices,
            partial_m,
            partial_l,
            partial_o,
            self.selected_tokens,
            self.key_dim,
            self.value_dim,
            self.query_count,
            num_v_blocks,
            self.scale,
            block_n,
            block_d,
            block_v,
            block_q,
            num_warps=warps,
            num_stages=3,
        )
        reduce_sparse_multiq_ssv_kernel[(num_v_blocks, num_q_blocks)](
            partial_m,
            partial_l,
            partial_o,
            self.out_grouped,
            self.value_dim,
            self.query_count,
            num_n_blocks,
            num_v_blocks,
            block_v,
            block_q,
            num_warps=warps,
            num_stages=3,
        )

    def triton_onequery_repeated(self, block_n: int, block_d: int, block_v: int, warps: int) -> None:
        partial_m, partial_l, partial_o = self.oneq_buffers(block_n, block_v)
        num_n_blocks = triton.cdiv(self.selected_tokens, block_n)
        num_v_blocks = triton.cdiv(self.value_dim, block_v)
        for q_query in range(self.query_count):
            partial_sparse_oneq_ssv_kernel[(num_n_blocks, num_v_blocks)](
                self.k_cache,
                self.v_cache,
                self.q,
                self.indices,
                partial_m,
                partial_l,
                partial_o,
                q_query,
                self.selected_tokens,
                self.key_dim,
                self.value_dim,
                num_v_blocks,
                self.scale,
                block_n,
                block_d,
                block_v,
                num_warps=warps,
                num_stages=3,
            )
            reduce_sparse_oneq_ssv_kernel[(num_v_blocks,)](
                partial_m,
                partial_l,
                partial_o,
                self.out_oneq,
                q_query,
                self.value_dim,
                num_n_blocks,
                num_v_blocks,
                block_v,
                num_warps=warps,
                num_stages=3,
            )

    def validate_variant(self, block_n: int, block_d: int, block_v: int, block_q: int, warps: int) -> dict:
        self.torch_flat()
        self.triton_grouped(block_n, block_d, block_v, block_q, warps)
        self.triton_onequery_repeated(block_n, block_d, block_v, warps)
        torch.cuda.synchronize()

        reference = self.out_flat.float()
        grouped = self.out_grouped.float()
        oneq = self.out_oneq.float()
        grouped_diff = (reference - grouped).abs()
        oneq_diff = (reference - oneq).abs()
        ref_abs = reference.abs().max().item()
        finite = (
            bool(torch.isfinite(reference).all().item())
            and bool(torch.isfinite(grouped).all().item())
            and bool(torch.isfinite(oneq).all().item())
        )
        return {
            "finite_outputs": finite,
            "max_abs_diff_grouped_vs_flat": float(grouped_diff.max().item()),
            "mean_abs_diff_grouped_vs_flat": float(grouped_diff.mean().item()),
            "relative_max_abs_diff_grouped": float(grouped_diff.max().item() / ref_abs) if ref_abs else 0.0,
            "max_abs_diff_oneq_vs_flat": float(oneq_diff.max().item()),
            "mean_abs_diff_oneq_vs_flat": float(oneq_diff.mean().item()),
            "relative_max_abs_diff_oneq": float(oneq_diff.max().item() / ref_abs) if ref_abs else 0.0,
            "max_ref_abs": float(ref_abs),
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    segment_counts = [int(item) for item in args.segment_counts.split(",") if item.strip()]
    variants = parse_variants(args.variants)
    workload = MultiQuerySparseWorkload(
        segment_counts=segment_counts,
        tokens_per_segment=args.tokens_per_segment,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        query_count=args.query_count,
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
    for block_n, block_d, block_v, block_q, warps in variants:
        common = {
            "block_n": block_n,
            "block_d": block_d,
            "block_v": block_v,
            "block_q": block_q,
            "warps": warps,
            "num_n_blocks": triton.cdiv(workload.selected_tokens, block_n),
            "num_v_blocks": triton.cdiv(workload.value_dim, block_v),
            "num_q_blocks": triton.cdiv(workload.query_count, block_q),
        }
        try:
            grouped_ms = cuda_time(
                lambda bn=block_n, bd=block_d, bv=block_v, bq=block_q, nw=warps: workload.triton_grouped(
                    bn, bd, bv, bq, nw
                ),
                args.warmup,
                args.iters,
            )
            oneq_ms = cuda_time(
                lambda bn=block_n, bd=block_d, bv=block_v, nw=warps: workload.triton_onequery_repeated(
                    bn, bd, bv, nw
                ),
                args.warmup,
                args.iters,
            )
            validation = workload.validate_variant(block_n, block_d, block_v, block_q, warps)
            variant_results.append(
                {
                    **common,
                    "status": "ok",
                    "triton_grouped_ms": grouped_ms,
                    "triton_oneq_repeated_ms": oneq_ms,
                    "triton_grouped_per_query_ms": grouped_ms / workload.query_count,
                    "grouped_vs_flat_speedup": flat_total_ms / grouped_ms,
                    "oneq_repeated_vs_flat_speedup": flat_total_ms / oneq_ms,
                    "grouped_vs_oneq_repeated_speedup": oneq_ms / grouped_ms,
                    "validation": validation,
                }
            )
        except Exception as exc:  # noqa: BLE001 - benchmarking should record failed variants.
            torch.cuda.synchronize()
            variant_results.append(
                {
                    **common,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    valid_results = [item for item in variant_results if item.get("status") == "ok"]
    if not valid_results:
        return {
            "status": "failed_no_valid_variants",
            "order_mode": args.order_mode,
            "variant_results": variant_results,
        }
    best = min(valid_results, key=lambda item: item["triton_grouped_ms"])
    best_oneq = min(valid_results, key=lambda item: item["triton_oneq_repeated_ms"])
    return {
        "status": "ok",
        "order_mode": args.order_mode,
        "segment_counts": segment_counts,
        "segment_count": workload.segment_count,
        "tokens_per_segment": args.tokens_per_segment,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "query_count": args.query_count,
        "total_cache_tokens": workload.total_cache_tokens,
        "selected_tokens": workload.selected_tokens,
        "selection_fraction": workload.selected_tokens / workload.total_cache_tokens,
        "torch_flat_total_ms": flat_total_ms,
        "torch_gather_k_ms": gather_k_ms,
        "torch_gather_v_ms": gather_v_ms,
        "torch_score_softmax_ms": score_softmax_ms,
        "torch_value_ms": value_ms,
        "torch_component_sum_ms": gather_k_ms + gather_v_ms + score_softmax_ms + value_ms,
        "best_triton_grouped_ms": best["triton_grouped_ms"],
        "best_triton_grouped_per_query_ms": best["triton_grouped_per_query_ms"],
        "best_triton_vs_flat_speedup": flat_total_ms / best["triton_grouped_ms"],
        "best_oneq_repeated_ms": best_oneq["triton_oneq_repeated_ms"],
        "best_grouped_vs_best_oneq_repeated_speedup": best_oneq["triton_oneq_repeated_ms"] / best["triton_grouped_ms"],
        "best_variant": {
            "block_n": best["block_n"],
            "block_d": best["block_d"],
            "block_v": best["block_v"],
            "block_q": best["block_q"],
            "warps": best["warps"],
            "num_n_blocks": best["num_n_blocks"],
            "num_v_blocks": best["num_v_blocks"],
            "num_q_blocks": best["num_q_blocks"],
        },
        "best_oneq_variant": {
            "block_n": best_oneq["block_n"],
            "block_d": best_oneq["block_d"],
            "block_v": best_oneq["block_v"],
            "warps": best_oneq["warps"],
            "num_n_blocks": best_oneq["num_n_blocks"],
            "num_v_blocks": best_oneq["num_v_blocks"],
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
                "status",
                "block_n",
                "block_d",
                "block_v",
                "block_q",
                "warps",
                "num_n_blocks",
                "num_v_blocks",
                "num_q_blocks",
                "triton_grouped_ms",
                "triton_oneq_repeated_ms",
                "triton_grouped_per_query_ms",
                "grouped_vs_flat_speedup",
                "oneq_repeated_vs_flat_speedup",
                "grouped_vs_oneq_repeated_speedup",
                "max_abs_diff_grouped_vs_flat",
                "mean_abs_diff_grouped_vs_flat",
                "relative_max_abs_diff_grouped",
                "max_abs_diff_oneq_vs_flat",
                "relative_max_abs_diff_oneq",
                "error_type",
                "error",
            ]
        )
        for item in result.get("variant_results", []):
            validation = item.get("validation", {})
            writer.writerow(
                [
                    result.get("order_mode"),
                    item.get("status"),
                    item.get("block_n"),
                    item.get("block_d"),
                    item.get("block_v"),
                    item.get("block_q"),
                    item.get("warps"),
                    item.get("num_n_blocks"),
                    item.get("num_v_blocks"),
                    item.get("num_q_blocks"),
                    item.get("triton_grouped_ms"),
                    item.get("triton_oneq_repeated_ms"),
                    item.get("triton_grouped_per_query_ms"),
                    item.get("grouped_vs_flat_speedup"),
                    item.get("oneq_repeated_vs_flat_speedup"),
                    item.get("grouped_vs_oneq_repeated_speedup"),
                    validation.get("max_abs_diff_grouped_vs_flat"),
                    validation.get("mean_abs_diff_grouped_vs_flat"),
                    validation.get("relative_max_abs_diff_grouped"),
                    validation.get("max_abs_diff_oneq_vs_flat"),
                    validation.get("relative_max_abs_diff_oneq"),
                    item.get("error_type"),
                    item.get("error"),
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
                "query_count",
                "torch_flat_total_ms",
                "torch_gather_k_ms",
                "torch_gather_v_ms",
                "torch_score_softmax_ms",
                "torch_value_ms",
                "best_triton_grouped_ms",
                "best_triton_grouped_per_query_ms",
                "best_triton_vs_flat_speedup",
                "best_oneq_repeated_ms",
                "best_grouped_vs_best_oneq_repeated_speedup",
                "best_variant",
                "best_oneq_variant",
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
                result.get("query_count"),
                result.get("torch_flat_total_ms"),
                result.get("torch_gather_k_ms"),
                result.get("torch_gather_v_ms"),
                result.get("torch_score_softmax_ms"),
                result.get("torch_value_ms"),
                result.get("best_triton_grouped_ms"),
                result.get("best_triton_grouped_per_query_ms"),
                result.get("best_triton_vs_flat_speedup"),
                result.get("best_oneq_repeated_ms"),
                result.get("best_grouped_vs_best_oneq_repeated_speedup"),
                json.dumps(result.get("best_variant", {}), sort_keys=True),
                json.dumps(result.get("best_oneq_variant", {}), sort_keys=True),
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
