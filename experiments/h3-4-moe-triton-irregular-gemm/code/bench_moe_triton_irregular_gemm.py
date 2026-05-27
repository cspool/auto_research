#!/usr/bin/env python3
"""H3.4 benchmark: no-padding Triton expert matmul for tiny MoE FFN."""

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
    parser.add_argument("--block-ms", default="16,32")
    parser.add_argument("--block-ns", default="32,64")
    parser.add_argument("--block-ks", default="64")
    parser.add_argument("--warps", default="4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2029)
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


class IrregularMoeWorkload:
    def __init__(self, token_counts: list[int], hidden_size: int, ffn_size: int, seed: int):
        torch.manual_seed(seed)
        self.token_counts = token_counts
        self.num_experts = len(token_counts)
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.max_tokens = max(token_counts)
        self.total_tokens = sum(token_counts)
        device = torch.device("cuda:0")

        self.inputs = [
            torch.randn(tokens, hidden_size, device=device, dtype=torch.float16)
            for tokens in token_counts
        ]
        self.w1 = [
            torch.randn(hidden_size, ffn_size, device=device, dtype=torch.float16)
            for _ in token_counts
        ]
        self.w2 = [
            torch.randn(ffn_size, hidden_size, device=device, dtype=torch.float16)
            for _ in token_counts
        ]
        self.hidden_loop = [
            torch.empty(tokens, ffn_size, device=device, dtype=torch.float16)
            for tokens in token_counts
        ]
        self.outputs_loop = [
            torch.empty(tokens, hidden_size, device=device, dtype=torch.float16)
            for tokens in token_counts
        ]

        self.x_concat = torch.cat(self.inputs, dim=0).contiguous()
        self.w1_stack = torch.stack(self.w1).contiguous()
        self.w2_stack = torch.stack(self.w2).contiguous()
        self.hidden_triton = torch.empty(self.total_tokens, ffn_size, device=device, dtype=torch.float16)
        self.y_triton = torch.empty(self.total_tokens, hidden_size, device=device, dtype=torch.float16)

        self.x_pad = torch.zeros(
            self.num_experts, self.max_tokens, hidden_size, device=device, dtype=torch.float16
        )
        for expert_id, tokens in enumerate(token_counts):
            self.x_pad[expert_id, :tokens].copy_(self.inputs[expert_id])
        self.hidden_pad = torch.empty(
            self.num_experts, self.max_tokens, ffn_size, device=device, dtype=torch.float16
        )
        self.y_pad = torch.empty(
            self.num_experts, self.max_tokens, hidden_size, device=device, dtype=torch.float16
        )

        offsets = []
        ends = []
        running = 0
        for tokens in token_counts:
            offsets.append(running)
            running += tokens
            ends.append(running)
        self.expert_offsets = offsets
        self.expert_ends = torch.tensor(ends, device=device, dtype=torch.int32)
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

    def expert_loop(self) -> None:
        for expert_id in range(self.num_experts):
            torch.mm(self.inputs[expert_id], self.w1[expert_id], out=self.hidden_loop[expert_id])
            self.hidden_loop[expert_id].relu_()
            torch.mm(self.hidden_loop[expert_id], self.w2[expert_id], out=self.outputs_loop[expert_id])

    def padded_bmm(self) -> None:
        torch.bmm(self.x_pad, self.w1_stack, out=self.hidden_pad)
        self.hidden_pad.relu_()
        torch.bmm(self.hidden_pad, self.w2_stack, out=self.y_pad)

    def triton_no_pad_ffn(self, block_m: int, block_n: int, block_k: int, warps: int) -> None:
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
            self.y_triton,
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

    def validate(self, block_m: int, block_n: int, block_k: int, warps: int) -> dict:
        self.expert_loop()
        self.triton_no_pad_ffn(block_m, block_n, block_k, warps)
        torch.cuda.synchronize()
        max_abs_diff = 0.0
        max_ref_abs = 0.0
        finite = bool(torch.isfinite(self.y_triton).all().item())
        for expert_id, tokens in enumerate(self.token_counts):
            start = self.expert_offsets[expert_id]
            end = start + tokens
            reference = self.outputs_loop[expert_id].float()
            candidate = self.y_triton[start:end].float()
            diff = (reference - candidate).abs().max().item()
            max_abs_diff = max(max_abs_diff, float(diff))
            max_ref_abs = max(max_ref_abs, float(reference.abs().max().item()))
            finite = finite and bool(torch.isfinite(self.outputs_loop[expert_id]).all().item())
        return {
            "finite_outputs": finite,
            "max_abs_diff_vs_loop": max_abs_diff,
            "max_ref_abs": max_ref_abs,
            "relative_max_abs_diff": max_abs_diff / max_ref_abs if max_ref_abs else 0.0,
        }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable; no GPU result was produced."}

    torch.cuda.set_device(0)
    token_counts = [int(item) for item in args.token_counts.split(",") if item.strip()]
    block_ms = [int(item) for item in args.block_ms.split(",") if item.strip()]
    block_ns = [int(item) for item in args.block_ns.split(",") if item.strip()]
    block_ks = [int(item) for item in args.block_ks.split(",") if item.strip()]
    warps_list = [int(item) for item in args.warps.split(",") if item.strip()]
    workload = IrregularMoeWorkload(token_counts, args.hidden_size, args.ffn_size, args.seed)

    expert_loop_ms = cuda_time(workload.expert_loop, args.warmup, args.iters)
    padded_bmm_ms = cuda_time(workload.padded_bmm, args.warmup, args.iters)
    actual_tokens = sum(token_counts)
    padded_tokens = len(token_counts) * max(token_counts)

    rows = []
    for block_m in block_ms:
        row_tiles = workload.block_map(block_m)[0].numel()
        for block_n in block_ns:
            for block_k in block_ks:
                for warps in warps_list:
                    triton_ms = cuda_time(
                        lambda bm=block_m, bn=block_n, bk=block_k, w=warps: workload.triton_no_pad_ffn(
                            bm, bn, bk, w
                        ),
                        args.warmup,
                        args.iters,
                    )
                    rows.append(
                        {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "warps": warps,
                            "row_tiles": row_tiles,
                            "triton_no_pad_ms": triton_ms,
                            "triton_vs_loop_speedup": expert_loop_ms / triton_ms,
                            "triton_vs_bmm_speedup": padded_bmm_ms / triton_ms,
                        }
                    )

    best_triton = min(rows, key=lambda row: row["triton_no_pad_ms"])
    validation = workload.validate(
        best_triton["block_m"],
        best_triton["block_n"],
        best_triton["block_k"],
        best_triton["warps"],
    )
    return {
        "status": "ok",
        "token_counts": token_counts,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_size,
        "actual_tokens": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_overhead": padded_tokens / actual_tokens,
        "expert_loop_ms": expert_loop_ms,
        "padded_bmm_ms": padded_bmm_ms,
        "padded_bmm_speedup": expert_loop_ms / padded_bmm_ms,
        "variants": rows,
        "best_triton_variant": best_triton,
        "best_triton_beats_loop": best_triton["triton_no_pad_ms"] < expert_loop_ms,
        "best_triton_beats_padded_bmm": best_triton["triton_no_pad_ms"] < padded_bmm_ms,
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
                "block_m",
                "block_n",
                "block_k",
                "warps",
                "row_tiles",
                "triton_no_pad_ms",
                "triton_vs_loop_speedup",
                "triton_vs_bmm_speedup",
            ]
        )
        for row in result.get("variants", []):
            writer.writerow(
                [
                    row["block_m"],
                    row["block_n"],
                    row["block_k"],
                    row["warps"],
                    row["row_tiles"],
                    row["triton_no_pad_ms"],
                    row["triton_vs_loop_speedup"],
                    row["triton_vs_bmm_speedup"],
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
