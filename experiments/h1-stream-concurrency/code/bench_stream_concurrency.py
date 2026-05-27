#!/usr/bin/env python3
"""H1 microbenchmark for single-request operator concurrency.

The benchmark is intentionally small and self-contained. It records a skip
artifact when CUDA is unavailable so future runs do not confuse environment
blockers with negative research results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--vector-rows", type=int, default=65536)
    parser.add_argument("--vector-cols", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cpu-dry-run", action="store_true")
    return parser.parse_args()


def environment(device: str) -> dict:
    cuda_available = torch.cuda.is_available()
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = torch.cuda.get_device_capability(0)
    return env


def finite_check(tensors: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


class Workload:
    def __init__(self, device: torch.device, matrix_size: int, vector_rows: int, vector_cols: int):
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.device = device
        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        self.c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        self.x = torch.randn(vector_rows, vector_cols, device=device, dtype=dtype)
        self.bias = torch.randn(vector_rows, vector_cols, device=device, dtype=dtype)
        self.y = torch.empty(vector_rows, vector_cols, device=device, dtype=dtype)
        self.r = torch.empty(vector_rows, device=device, dtype=dtype)

    def compute_gemm(self) -> None:
        torch.matmul(self.a, self.b, out=self.c)

    def elementwise(self) -> None:
        torch.add(self.x, self.bias, out=self.y)

    def reduction(self) -> None:
        torch.sum(self.x, dim=1, out=self.r)

    def serial(self) -> None:
        self.compute_gemm()
        self.elementwise()
        self.reduction()

    def outputs(self) -> list[torch.Tensor]:
        return [self.c, self.y, self.r]


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


def cpu_time(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_cuda(args: argparse.Namespace, out_dir: Path) -> dict:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    work = Workload(device, args.matrix_size, args.vector_rows, args.vector_cols)
    streams = [torch.cuda.Stream() for _ in range(3)]

    def multistream() -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            work.compute_gemm()
        with torch.cuda.stream(streams[1]):
            work.elementwise()
        with torch.cuda.stream(streams[2]):
            work.reduction()
        for stream in streams:
            current.wait_stream(stream)

    isolated = {
        "compute_gemm": cuda_time(work.compute_gemm, args.warmup, args.iters),
        "elementwise": cuda_time(work.elementwise, args.warmup, args.iters),
        "reduction": cuda_time(work.reduction, args.warmup, args.iters),
    }
    serial_ms = cuda_time(work.serial, args.warmup, args.iters)
    multistream_ms = cuda_time(multistream, args.warmup, args.iters)

    graph_ms = math.nan
    try:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            work.serial()

        def replay() -> None:
            graph.replay()

        graph_ms = cuda_time(replay, args.warmup, args.iters)
    except RuntimeError as exc:
        (out_dir / "cuda_graph_error.txt").write_text(str(exc), encoding="utf-8")

    sum_isolated = sum(isolated.values())
    result = {
        "status": "ok",
        "device": "cuda",
        "matrix_size": args.matrix_size,
        "vector_rows": args.vector_rows,
        "vector_cols": args.vector_cols,
        "warmup": args.warmup,
        "iters": args.iters,
        "isolated_ms": isolated,
        "sum_isolated_ms": sum_isolated,
        "serial_ms": serial_ms,
        "multistream_ms": multistream_ms,
        "cuda_graph_serial_ms": graph_ms,
        "overlap_ratio": 1.0 - multistream_ms / sum_isolated,
        "serial_speedup": serial_ms / multistream_ms,
        "graph_speedup": serial_ms / graph_ms if graph_ms and not math.isnan(graph_ms) else math.nan,
        "finite_outputs": finite_check(work.outputs()),
    }
    return result


def run_cpu_dry(args: argparse.Namespace) -> dict:
    rows = min(args.vector_rows, 1024)
    cols = min(args.vector_cols, 64)
    size = min(args.matrix_size, 128)
    device = torch.device("cpu")
    work = Workload(device, size, rows, cols)
    serial_ms = cpu_time(work.serial, warmup=2, iters=3)
    return {
        "status": "cpu_dry_run_only",
        "device": "cpu",
        "note": "CPU run validates code path only; it is not evidence for GPU stream concurrency.",
        "matrix_size": size,
        "vector_rows": rows,
        "vector_cols": cols,
        "serial_ms": serial_ms,
        "finite_outputs": finite_check(work.outputs()),
    }


def write_outputs(out_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f"{key}.{sub_key}", sub_value])
            else:
                writer.writerow([key, value])


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    env = environment(args.device)

    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if wants_cuda and torch.cuda.is_available():
        result = run_cuda(args, out_dir)
    elif args.cpu_dry_run or args.device == "cpu":
        result = run_cpu_dry(args)
    else:
        result = {
            "status": "skipped_no_cuda",
            "device": "none",
            "note": "CUDA is unavailable; no GPU concurrency conclusion was produced.",
        }

    write_outputs(out_dir, env, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

