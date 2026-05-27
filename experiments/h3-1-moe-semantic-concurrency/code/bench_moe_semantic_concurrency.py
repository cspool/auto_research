#!/usr/bin/env python3
"""H3.1 benchmark: semantic MoE expert concurrency on one GPU."""

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
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
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


class MoeWorkload:
    def __init__(self, token_counts: list[int], hidden_size: int, ffn_size: int, seed: int):
        torch.manual_seed(seed)
        self.token_counts = token_counts
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_experts = len(token_counts)
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

    def expert(self, expert_id: int) -> None:
        torch.mm(self.inputs[expert_id], self.w1[expert_id], out=self.hidden[expert_id])
        self.hidden[expert_id].relu_()
        torch.mm(self.hidden[expert_id], self.w2[expert_id], out=self.outputs[expert_id])

    def serial(self) -> None:
        for expert_id in range(self.num_experts):
            self.expert(expert_id)

    def concurrent(self, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        for expert_id, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                self.expert(expert_id)
        for stream in streams:
            current.wait_stream(stream)

    def finite_outputs(self) -> bool:
        return all(bool(torch.isfinite(output).all().item()) for output in self.outputs)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "note": "CUDA is unavailable; no GPU result was produced.",
        }

    torch.cuda.set_device(0)
    token_counts = [int(item) for item in args.token_counts.split(",") if item.strip()]
    workload = MoeWorkload(token_counts, args.hidden_size, args.ffn_size, args.seed)
    streams = [torch.cuda.Stream() for _ in token_counts]

    expert_ms = [
        cuda_time(lambda expert_id=expert_id: workload.expert(expert_id), args.warmup, args.iters)
        for expert_id in range(workload.num_experts)
    ]
    serial_ms = cuda_time(workload.serial, args.warmup, args.iters)
    concurrent_ms = cuda_time(lambda: workload.concurrent(streams), args.warmup, args.iters)
    sum_expert_ms = sum(expert_ms)

    expert_rows = []
    for expert_id, latency_ms in enumerate(expert_ms):
        expert_rows.append(
            {
                "expert_id": expert_id,
                "tokens": token_counts[expert_id],
                "isolated_ms": latency_ms,
                "share_of_sum": latency_ms / sum_expert_ms if sum_expert_ms else 0.0,
            }
        )

    return {
        "status": "ok",
        "token_counts": token_counts,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "expert_ms": expert_rows,
        "sum_expert_ms": sum_expert_ms,
        "max_expert_ms": max(expert_ms),
        "serial_ms": serial_ms,
        "concurrent_ms": concurrent_ms,
        "stream_speedup": serial_ms / concurrent_ms,
        "overlap_ratio": 1.0 - concurrent_ms / sum_expert_ms,
        "dominance_ratio": max(expert_ms) / sum_expert_ms if sum_expert_ms else 0.0,
        "finite_outputs": workload.finite_outputs(),
    }


def write_outputs(out_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert_id", "tokens", "isolated_ms", "share_of_sum"])
        for row in result.get("expert_ms", []):
            writer.writerow([row["expert_id"], row["tokens"], row["isolated_ms"], row["share_of_sum"]])


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
