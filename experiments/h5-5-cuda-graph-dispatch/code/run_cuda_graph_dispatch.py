#!/usr/bin/env python3
"""H5.5 CUDA Graph replay dispatch substrate experiment."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[3]
H5_1_SCRIPT = ROOT / "experiments" / "h5-1-runtime-selector" / "code" / "run_runtime_selector.py"
DEFAULT_H5_1_RESULT = ROOT / "experiments" / "h5-1-runtime-selector" / "results" / "rtx4090_default" / "result.json"
DEFAULT_OUT_DIR = ROOT / "experiments" / "h5-5-cuda-graph-dispatch" / "results" / "rtx4090_default"
DEFAULT_DATA_DIR = ROOT / "data"


def load_h5_1_module():
    spec = importlib.util.spec_from_file_location("h5_1_runtime_selector", H5_1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load H5.1 script from {H5_1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--h5-1-result", default=str(DEFAULT_H5_1_RESULT))
    parser.add_argument("--policies", default="static_best_average,resource_aware")
    parser.add_argument("--small-elements", type=int, default=8_388_608)
    parser.add_argument("--large-elements", type=int, default=16_777_216)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--memory-elements", type=int, default=67_108_864)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--warmup-queues", type=int, default=12)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=48)
    parser.add_argument("--profile-capture", action="store_true")
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def load_policy_selections(path: Path, policy_names: list[str]) -> dict[str, dict[str, str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["policy"]: row["selections"] for row in result["policy_rows"]}
    return {name: rows[name] for name in policy_names}


def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = torch.cuda.get_device_capability(0)
    return env


def event_time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def wall_time_ms(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def queue_run_plain(workload, tasks, shapes, selections, streams) -> None:
    current = torch.cuda.current_stream()
    bg_stream = streams["background"]
    micro_stream = streams["micro"]
    for task in tasks:
        variant = selections[(task.load_class, task.shape_label)]
        bg_stream.wait_stream(current)
        micro_stream.wait_stream(current)
        if task.load_class != "idle":
            with torch.cuda.stream(bg_stream):
                workload.background(task.load_class)
        with torch.cuda.stream(micro_stream):
            workload.triton_fused(shapes[task.shape_label].n_elements, variant)
        if task.load_class != "idle":
            current.wait_stream(bg_stream)
        current.wait_stream(micro_stream)


class nvtx_range:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.nvtx.range_pop()
        return False


def profiler_start(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStart()


def profiler_stop(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStop()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable"}

    out_dir = Path(args.out_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", str(out_dir / "triton_cache"))

    h5 = load_h5_1_module()
    torch.cuda.set_device(0)
    policy_names = [item.strip() for item in args.policies.split(",") if item.strip()]
    raw_selections = load_policy_selections(Path(args.h5_1_result), policy_names)
    variant_ids = sorted({variant_id for selections in raw_selections.values() for variant_id in selections.values()})
    variants = {}
    for variant_id in variant_ids:
        block = int(variant_id.split("_", 1)[0][1:])
        warps = int(variant_id.split("_", 1)[1][1:])
        variants[variant_id] = h5.Variant(block, warps)

    shapes = {
        "small": h5.TaskShape("small", args.small_elements),
        "large": h5.TaskShape("large", args.large_elements),
    }
    tasks = [
        h5.QueueTask(0, "compute", "small"),
        h5.QueueTask(1, "memory", "large"),
        h5.QueueTask(2, "compute", "large"),
        h5.QueueTask(3, "memory", "small"),
        h5.QueueTask(4, "compute", "small"),
        h5.QueueTask(5, "memory", "large"),
    ]
    workload = h5.Workload(
        max_elements=max(shape.n_elements for shape in shapes.values()),
        matrix_size=args.matrix_size,
        memory_elements=args.memory_elements,
        scale=args.scale,
        seed=args.seed,
    )
    streams = {"background": torch.cuda.Stream(), "micro": torch.cuda.Stream()}
    selections = {
        policy: {tuple(key.split(":", 1)): variants[variant_id] for key, variant_id in rows.items()}
        for policy, rows in raw_selections.items()
    }

    for variant in variants.values():
        for shape in shapes.values():
            workload.triton_fused(shape.n_elements, variant)
    workload.background("compute")
    workload.background("memory")
    torch.cuda.synchronize()

    for _ in range(args.warmup_queues):
        for policy_name in policy_names:
            queue_run_plain(workload, tasks, shapes, selections[policy_name], streams)
    torch.cuda.synchronize()

    graph_rows = []
    measurement_rows = []
    graphs = {}
    for policy_name in policy_names:
        policy_selections = selections[policy_name]
        py_event = []
        py_wall = []
        for _ in range(args.iters):
            py_event.append(event_time_ms(lambda s=policy_selections: queue_run_plain(workload, tasks, shapes, s, streams)))
            py_wall.append(wall_time_ms(lambda s=policy_selections: queue_run_plain(workload, tasks, shapes, s, streams)))
        graph = torch.cuda.CUDAGraph()
        capture_error = ""
        capture_ms = float("nan")
        try:
            torch.cuda.synchronize()
            queue_run_plain(workload, tasks, shapes, policy_selections, streams)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.graph(graph, capture_error_mode="global"):
                queue_run_plain(workload, tasks, shapes, policy_selections, streams)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - start) * 1000.0
            graphs[policy_name] = graph
        except Exception as exc:  # noqa: BLE001 - experiment records backend failure.
            capture_error = repr(exc)

        graph_event = []
        graph_wall = []
        if not capture_error:
            for _ in range(args.iters):
                graph_event.append(event_time_ms(graph.replay))
                graph_wall.append(wall_time_ms(graph.replay))

        py_event_med = median(py_event)
        graph_event_med = median(graph_event)
        speedup_event = py_event_med / graph_event_med if graph_event else float("nan")
        py_wall_med = median(py_wall)
        graph_wall_med = median(graph_wall)
        speedup_wall = py_wall_med / graph_wall_med if graph_wall else float("nan")
        graph_rows.append(
            {
                "policy": policy_name,
                "capture_success": not bool(capture_error),
                "capture_wall_ms": capture_ms,
                "capture_error": capture_error,
                "python_event_median_ms": py_event_med,
                "graph_event_median_ms": graph_event_med,
                "speedup_event": speedup_event,
                "python_wall_median_ms": py_wall_med,
                "graph_wall_median_ms": graph_wall_med,
                "speedup_wall": speedup_wall,
            }
        )
        for idx, value in enumerate(py_event):
            measurement_rows.append({"policy": policy_name, "mode": "python_stream", "repeat": idx, "event_ms": value, "wall_ms": py_wall[idx]})
        for idx, value in enumerate(graph_event):
            measurement_rows.append({"policy": policy_name, "mode": "cuda_graph_replay", "repeat": idx, "event_ms": value, "wall_ms": graph_wall[idx]})

    if args.profile_capture and graphs:
        torch.cuda.synchronize()
        profiler_start(True)
        try:
            with nvtx_range("h5_5_graph_profile_window"):
                for repeat in range(args.profile_repeats):
                    for policy_name in policy_names:
                        graph = graphs.get(policy_name)
                        if graph is None:
                            continue
                        with nvtx_range(f"h5_5_graph_replay_{repeat}_{policy_name}"):
                            graph.replay()
            torch.cuda.synchronize()
        finally:
            profiler_stop(True)

    return {
        "status": "ok",
        "args": vars(args),
        "variant_ids": variant_ids,
        "raw_selections": raw_selections,
        "summary": graph_rows,
        "measurements": measurement_rows,
    }


def write_outputs(out_dir: Path, data_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result.get("status") != "ok":
        return
    summary_fields = [
        "policy",
        "capture_success",
        "capture_wall_ms",
        "capture_error",
        "python_event_median_ms",
        "graph_event_median_ms",
        "speedup_event",
        "python_wall_median_ms",
        "graph_wall_median_ms",
        "speedup_wall",
    ]
    measurement_fields = ["policy", "mode", "repeat", "event_ms", "wall_ms"]
    write_csv(out_dir / "summary.csv", result["summary"], summary_fields)
    write_csv(out_dir / "measurements.csv", result["measurements"], measurement_fields)
    (data_dir / "h5_5_cuda_graph_dispatch_summary.csv").write_text((out_dir / "summary.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (data_dir / "h5_5_cuda_graph_dispatch_measurements.csv").write_text((out_dir / "measurements.csv").read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run(args)
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, Path(args.data_dir), environment(), result)
    print(json.dumps({"status": result.get("status"), "out_dir": str(out_dir), "summary": result.get("summary")}, indent=2))


if __name__ == "__main__":
    main()
