#!/usr/bin/env python3
"""H5.4 warmed persistent queue profiler for H5 runtime limits."""

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
DEFAULT_OUT_DIR = ROOT / "experiments" / "h5-4-runtime-limit-profiling" / "results" / "rtx4090_default"
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
    parser.add_argument("--measurement-repeats", type=int, default=24)
    parser.add_argument("--profile-repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--profile-capture", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def load_policy_selections(path: Path, policy_names: list[str]) -> dict[str, dict[str, str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["policy"]: row["selections"] for row in result["policy_rows"]}
    missing = [name for name in policy_names if name not in rows]
    if missing:
        raise KeyError(f"missing policy rows in {path}: {missing}")
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def profiler_start(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStart()


def profiler_stop(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStop()


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
    queue = [
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
        policy: {
            tuple(key.split(":", 1)): variants[variant_id]
            for key, variant_id in policy_selections.items()
        }
        for policy, policy_selections in raw_selections.items()
    }

    with h5.nvtx_range("h5_4_precompile_variants"):
        for variant in variants.values():
            for shape in shapes.values():
                workload.triton_fused(shape.n_elements, variant)
        workload.background("compute")
        workload.background("memory")
    torch.cuda.synchronize()

    with h5.nvtx_range("h5_4_warmup_queues"):
        for _ in range(args.warmup_queues):
            for policy_name in policy_names:
                workload.queue_run(queue, shapes, policy_name, selections[policy_name], streams)
    torch.cuda.synchronize()

    rows = []
    with h5.nvtx_range("h5_4_measurement_outside_capture"):
        for repeat in range(args.measurement_repeats):
            for policy_name in policy_names:
                event_ms = event_time_ms(
                    lambda name=policy_name: workload.queue_run(
                        queue,
                        shapes,
                        name,
                        selections[name],
                        streams,
                    )
                )
                wall_ms = wall_time_ms(
                    lambda name=policy_name: workload.queue_run(
                        queue,
                        shapes,
                        name,
                        selections[name],
                        streams,
                    )
                )
                rows.append(
                    {
                        "repeat": repeat,
                        "policy": policy_name,
                        "event_ms": event_ms,
                        "wall_ms": wall_ms,
                    }
                )

    torch.cuda.synchronize()
    profiler_start(args.profile_capture)
    try:
        with h5.nvtx_range("h5_4_profile_window"):
            for repeat in range(args.profile_repeats):
                for policy_name in policy_names:
                    with h5.nvtx_range(f"h5_4_profile_repeat_{repeat}_{policy_name}"):
                        workload.queue_run(queue, shapes, policy_name, selections[policy_name], streams)
        torch.cuda.synchronize()
    finally:
        profiler_stop(args.profile_capture)

    summary_rows = []
    baseline_policy = policy_names[0]
    baseline_event = median([row["event_ms"] for row in rows if row["policy"] == baseline_policy])
    for policy_name in policy_names:
        policy_event = [row["event_ms"] for row in rows if row["policy"] == policy_name]
        policy_wall = [row["wall_ms"] for row in rows if row["policy"] == policy_name]
        summary_rows.append(
            {
                "policy": policy_name,
                "event_median_ms": median(policy_event),
                "event_mean_ms": mean(policy_event),
                "event_min_ms": min(policy_event),
                "event_max_ms": max(policy_event),
                "wall_median_ms": median(policy_wall),
                "wall_mean_ms": mean(policy_wall),
                "wall_min_ms": min(policy_wall),
                "wall_max_ms": max(policy_wall),
                "speedup_vs_first_policy_event": baseline_event / median(policy_event),
            }
        )

    return {
        "status": "ok",
        "args": vars(args),
        "policies": policy_names,
        "variant_ids": variant_ids,
        "queue": [task.__dict__ for task in queue],
        "raw_selections": raw_selections,
        "measurements": rows,
        "summary": summary_rows,
    }


def write_outputs(out_dir: Path, data_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result.get("status") != "ok":
        return
    write_csv(out_dir / "measurements.csv", result["measurements"], ["repeat", "policy", "event_ms", "wall_ms"])
    write_csv(
        out_dir / "summary.csv",
        result["summary"],
        [
            "policy",
            "event_median_ms",
            "event_mean_ms",
            "event_min_ms",
            "event_max_ms",
            "wall_median_ms",
            "wall_mean_ms",
            "wall_min_ms",
            "wall_max_ms",
            "speedup_vs_first_policy_event",
        ],
    )
    for name in ["measurements.csv", "summary.csv"]:
        (data_dir / f"h5_4_warmed_queue_{name}").write_text((out_dir / name).read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    result = run(args)
    write_outputs(out_dir, data_dir, environment(), result)
    print(json.dumps({"status": result.get("status"), "out_dir": str(out_dir), "summary": result.get("summary")}, indent=2))


if __name__ == "__main__":
    main()
