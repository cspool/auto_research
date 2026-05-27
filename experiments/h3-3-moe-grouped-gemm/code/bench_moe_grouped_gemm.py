#!/usr/bin/env python3
"""H3.3 benchmark: grouped/batched expert GEMM for tiny MoE FFN."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--token-counts", default="64,64,64,64")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2028)
    return parser.parse_args()


def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
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


class GroupedMoeWorkload:
    def __init__(self, token_counts: list[int], hidden_size: int, ffn_size: int, seed: int):
        torch.manual_seed(seed)
        self.token_counts = token_counts
        self.num_experts = len(token_counts)
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.max_tokens = max(token_counts)
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
        self.hidden = [
            torch.empty(tokens, ffn_size, device=device, dtype=torch.float16)
            for tokens in token_counts
        ]
        self.outputs = [
            torch.empty(tokens, hidden_size, device=device, dtype=torch.float16)
            for tokens in token_counts
        ]

        self.x_pad = torch.zeros(
            self.num_experts, self.max_tokens, hidden_size, device=device, dtype=torch.float16
        )
        for expert_id, tokens in enumerate(token_counts):
            self.x_pad[expert_id, :tokens].copy_(self.inputs[expert_id])
        self.w1_stack = torch.stack(self.w1)
        self.w2_stack = torch.stack(self.w2)
        self.hidden_pad = torch.empty(
            self.num_experts, self.max_tokens, ffn_size, device=device, dtype=torch.float16
        )
        self.y_pad = torch.empty(
            self.num_experts, self.max_tokens, hidden_size, device=device, dtype=torch.float16
        )

    def expert_loop(self) -> None:
        for expert_id in range(self.num_experts):
            torch.mm(self.inputs[expert_id], self.w1[expert_id], out=self.hidden[expert_id])
            self.hidden[expert_id].relu_()
            torch.mm(self.hidden[expert_id], self.w2[expert_id], out=self.outputs[expert_id])

    def grouped_bmm(self) -> None:
        torch.bmm(self.x_pad, self.w1_stack, out=self.hidden_pad)
        self.hidden_pad.relu_()
        torch.bmm(self.hidden_pad, self.w2_stack, out=self.y_pad)

    def validate(self) -> dict:
        self.expert_loop()
        self.grouped_bmm()
        torch.cuda.synchronize()
        max_abs_diff = 0.0
        max_ref_abs = 0.0
        finite = True
        for expert_id, tokens in enumerate(self.token_counts):
            reference = self.outputs[expert_id].float()
            candidate = self.y_pad[expert_id, :tokens].float()
            diff = (
                reference
                - candidate
            ).abs().max().item()
            max_abs_diff = max(max_abs_diff, float(diff))
            max_ref_abs = max(max_ref_abs, float(reference.abs().max().item()))
            finite = finite and bool(torch.isfinite(self.outputs[expert_id]).all().item())
            finite = finite and bool(torch.isfinite(self.y_pad[expert_id, :tokens]).all().item())
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
    workload = GroupedMoeWorkload(token_counts, args.hidden_size, args.ffn_size, args.seed)
    expert_loop_ms = cuda_time(workload.expert_loop, args.warmup, args.iters)
    grouped_bmm_ms = cuda_time(workload.grouped_bmm, args.warmup, args.iters)
    actual_tokens = sum(token_counts)
    padded_tokens = len(token_counts) * max(token_counts)
    return {
        "status": "ok",
        "token_counts": token_counts,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_size,
        "max_tokens": max(token_counts),
        "actual_tokens": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_overhead": padded_tokens / actual_tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "expert_loop_ms": expert_loop_ms,
        "grouped_bmm_ms": grouped_bmm_ms,
        "grouped_speedup": expert_loop_ms / grouped_bmm_ms,
        "validation": workload.validate(),
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
                "hidden_size",
                "ffn_size",
                "actual_tokens",
                "padded_tokens",
                "padding_overhead",
                "expert_loop_ms",
                "grouped_bmm_ms",
                "grouped_speedup",
            ]
        )
        writer.writerow(
            [
                ",".join(str(item) for item in result.get("token_counts", [])),
                result.get("hidden_size"),
                result.get("ffn_size"),
                result.get("actual_tokens"),
                result.get("padded_tokens"),
                result.get("padding_overhead"),
                result.get("expert_loop_ms"),
                result.get("grouped_bmm_ms"),
                result.get("grouped_speedup"),
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
