#!/usr/bin/env python3
"""H1.1 benchmark: does reshaping a dominant GEMM help stream overlap?"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--vector-rows", type=int, default=32768)
    parser.add_argument("--vector-cols", type=int, default=256)
    parser.add_argument("--chunks", default="2,4,8,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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


class Workload:
    def __init__(self, matrix_size: int, vector_rows: int, vector_cols: int):
        device = torch.device("cuda:0")
        dtype = torch.float16
        self.n = matrix_size
        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        self.c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        self.x = torch.randn(vector_rows, vector_cols, device=device, dtype=dtype)
        self.bias = torch.randn(vector_rows, vector_cols, device=device, dtype=dtype)
        self.y = torch.empty(vector_rows, vector_cols, device=device, dtype=dtype)
        self.r = torch.empty(vector_rows, device=device, dtype=dtype)

    def full_gemm(self) -> None:
        torch.matmul(self.a, self.b, out=self.c)

    def elementwise(self) -> None:
        torch.add(self.x, self.bias, out=self.y)

    def reduction(self) -> None:
        torch.sum(self.x, dim=1, out=self.r)

    def full_serial(self) -> None:
        self.full_gemm()
        self.elementwise()
        self.reduction()

    def full_multistream(self, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            self.full_gemm()
        with torch.cuda.stream(streams[1]):
            self.elementwise()
        with torch.cuda.stream(streams[2]):
            self.reduction()
        for stream in streams:
            current.wait_stream(stream)

    def chunk_ranges(self, chunks: int) -> list[tuple[int, int]]:
        rows_per_chunk = math.ceil(self.n / chunks)
        ranges = []
        for start in range(0, self.n, rows_per_chunk):
            end = min(start + rows_per_chunk, self.n)
            ranges.append((start, end))
        return ranges

    def chunked_gemm(self, chunks: int) -> None:
        for start, end in self.chunk_ranges(chunks):
            torch.matmul(self.a[start:end, :], self.b, out=self.c[start:end, :])

    def chunked_serial(self, chunks: int) -> None:
        self.chunked_gemm(chunks)
        self.elementwise()
        self.reduction()

    def chunked_multistream(self, chunks: int, streams: list[torch.cuda.Stream]) -> None:
        current = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(current)
        with torch.cuda.stream(streams[0]):
            self.chunked_gemm(chunks)
        with torch.cuda.stream(streams[1]):
            self.elementwise()
        with torch.cuda.stream(streams[2]):
            self.reduction()
        for stream in streams:
            current.wait_stream(stream)

    def finite_outputs(self) -> bool:
        tensors = [self.c, self.y, self.r]
        return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "note": "CUDA is unavailable; no GPU result was produced.",
        }

    torch.cuda.set_device(0)
    workload = Workload(args.matrix_size, args.vector_rows, args.vector_cols)
    streams = [torch.cuda.Stream() for _ in range(3)]
    chunks = [int(item) for item in args.chunks.split(",") if item.strip()]

    full_gemm_ms = cuda_time(workload.full_gemm, args.warmup, args.iters)
    elementwise_ms = cuda_time(workload.elementwise, args.warmup, args.iters)
    reduction_ms = cuda_time(workload.reduction, args.warmup, args.iters)
    full_serial_ms = cuda_time(workload.full_serial, args.warmup, args.iters)
    full_multistream_ms = cuda_time(lambda: workload.full_multistream(streams), args.warmup, args.iters)
    full_sum_isolated = full_gemm_ms + elementwise_ms + reduction_ms

    chunk_results = []
    for chunk_count in chunks:
        chunked_gemm_ms = cuda_time(lambda c=chunk_count: workload.chunked_gemm(c), args.warmup, args.iters)
        chunked_serial_ms = cuda_time(lambda c=chunk_count: workload.chunked_serial(c), args.warmup, args.iters)
        chunked_multistream_ms = cuda_time(
            lambda c=chunk_count: workload.chunked_multistream(c, streams),
            args.warmup,
            args.iters,
        )
        chunk_sum_isolated = chunked_gemm_ms + elementwise_ms + reduction_ms
        chunk_results.append(
            {
                "chunks": chunk_count,
                "chunked_gemm_ms": chunked_gemm_ms,
                "chunked_serial_ms": chunked_serial_ms,
                "chunked_multistream_ms": chunked_multistream_ms,
                "stream_speedup": chunked_serial_ms / chunked_multistream_ms,
                "overlap_ratio": 1.0 - chunked_multistream_ms / chunk_sum_isolated,
                "chunk_overhead": chunked_gemm_ms / full_gemm_ms,
            }
        )

    best = max(chunk_results, key=lambda item: item["stream_speedup"])
    return {
        "status": "ok",
        "matrix_size": args.matrix_size,
        "vector_rows": args.vector_rows,
        "vector_cols": args.vector_cols,
        "warmup": args.warmup,
        "iters": args.iters,
        "full": {
            "full_gemm_ms": full_gemm_ms,
            "elementwise_ms": elementwise_ms,
            "reduction_ms": reduction_ms,
            "sum_isolated_ms": full_sum_isolated,
            "serial_ms": full_serial_ms,
            "multistream_ms": full_multistream_ms,
            "stream_speedup": full_serial_ms / full_multistream_ms,
            "overlap_ratio": 1.0 - full_multistream_ms / full_sum_isolated,
        },
        "chunks": chunk_results,
        "best_chunk_by_stream_speedup": best,
        "finite_outputs": workload.finite_outputs(),
    }


def write_outputs(out_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "kind",
                "chunks",
                "serial_ms",
                "multistream_ms",
                "gemm_ms",
                "stream_speedup",
                "overlap_ratio",
                "chunk_overhead",
            ]
        )
        if result.get("status") == "ok":
            full = result["full"]
            writer.writerow(
                [
                    "full",
                    1,
                    full["serial_ms"],
                    full["multistream_ms"],
                    full["full_gemm_ms"],
                    full["stream_speedup"],
                    full["overlap_ratio"],
                    1.0,
                ]
            )
            for row in result["chunks"]:
                writer.writerow(
                    [
                        "chunked",
                        row["chunks"],
                        row["chunked_serial_ms"],
                        row["chunked_multistream_ms"],
                        row["chunked_gemm_ms"],
                        row["stream_speedup"],
                        row["overlap_ratio"],
                        row["chunk_overhead"],
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

