#!/usr/bin/env python3
"""H5.1 runtime resource-aware micro-op selector benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "experiments" / "h5-1-runtime-selector" / "results"
DEFAULT_DATA_DIR = ROOT / "data"


@triton.jit
def h5_fused_gelu_residual_kernel(
    x_ptr,
    bias_ptr,
    residual_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    z = x * scale + bias
    z3 = z * z * z
    gelu_arg = 0.7978845608028654 * (z + 0.044715 * z3)
    tanh_arg = 2.0 / (1.0 + tl.exp(-2.0 * gelu_arg)) - 1.0
    gelu = 0.5 * z * (1.0 + tanh_arg)
    tl.store(y_ptr + offsets, gelu + residual, mask=mask)


@dataclass(frozen=True)
class Variant:
    block_size: int
    warps: int

    @property
    def variant_id(self) -> str:
        return f"B{self.block_size}_W{self.warps}"


@dataclass(frozen=True)
class TaskShape:
    label: str
    n_elements: int


@dataclass(frozen=True)
class QueueTask:
    step: int
    load_class: str
    shape_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--block-sizes", default="512,1024,2048,4096,8192")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--small-elements", type=int, default=8_388_608)
    parser.add_argument("--large-elements", type=int, default=16_777_216)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--memory-elements", type=int, default=67_108_864)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iters", type=int, default=48)
    parser.add_argument("--policy-warmup", type=int, default=8)
    parser.add_argument("--policy-iters", type=int, default=40)
    parser.add_argument("--policy-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def split_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


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


def wall_time_cuda(fn: Callable[[], None]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def nvtx_range(name: str):
    class _Range:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.cuda.nvtx.range_pop()
            return False

    return _Range()


class Workload:
    def __init__(
        self,
        max_elements: int,
        matrix_size: int,
        memory_elements: int,
        scale: float,
        seed: int,
    ):
        torch.manual_seed(seed)
        device = torch.device("cuda:0")
        self.max_elements = max_elements
        self.matrix_size = matrix_size
        self.memory_elements = memory_elements
        self.scale = float(scale)

        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        self.c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float16)

        self.mem_a = torch.randn(memory_elements, device=device, dtype=torch.float16)
        self.mem_b = torch.randn(memory_elements, device=device, dtype=torch.float16)
        self.mem_out = torch.empty(memory_elements, device=device, dtype=torch.float16)

        self.x = torch.randn(max_elements, device=device, dtype=torch.float16)
        self.bias = torch.randn(max_elements, device=device, dtype=torch.float16)
        self.residual = torch.randn(max_elements, device=device, dtype=torch.float16)
        self.y = torch.empty(max_elements, device=device, dtype=torch.float16)
        self.y_torch = torch.empty(max_elements, device=device, dtype=torch.float16)

    def background_compute(self) -> None:
        torch.matmul(self.a, self.b, out=self.c)

    def background_memory(self) -> None:
        torch.add(self.mem_a, self.mem_b, out=self.mem_out)

    def background(self, load_class: str) -> None:
        if load_class == "idle":
            return
        if load_class == "compute":
            self.background_compute()
            return
        if load_class == "memory":
            self.background_memory()
            return
        raise ValueError(f"unknown load_class={load_class}")

    def triton_fused(self, n_elements: int, variant: Variant) -> None:
        grid = (triton.cdiv(n_elements, variant.block_size),)
        h5_fused_gelu_residual_kernel[grid](
            self.x,
            self.bias,
            self.residual,
            self.y,
            n_elements,
            self.scale,
            variant.block_size,
            num_warps=variant.warps,
        )

    def torch_fused_reference(self, n_elements: int) -> None:
        self.y_torch[:n_elements] = (
            F.gelu(self.x[:n_elements] * self.scale + self.bias[:n_elements], approximate="tanh")
            + self.residual[:n_elements]
        )

    def concurrent_step(
        self,
        load_class: str,
        n_elements: int,
        variant: Variant,
        streams: dict[str, torch.cuda.Stream],
    ) -> None:
        current = torch.cuda.current_stream()
        bg_stream = streams["background"]
        micro_stream = streams["micro"]
        bg_stream.wait_stream(current)
        micro_stream.wait_stream(current)

        if load_class != "idle":
            with torch.cuda.stream(bg_stream), nvtx_range(f"h5_bg_{load_class}"):
                self.background(load_class)
        with torch.cuda.stream(micro_stream), nvtx_range(f"h5_micro_{variant.variant_id}"):
            self.triton_fused(n_elements, variant)

        if load_class != "idle":
            current.wait_stream(bg_stream)
        current.wait_stream(micro_stream)

    def queue_run(
        self,
        tasks: list[QueueTask],
        shapes: dict[str, TaskShape],
        policy_name: str,
        selections: dict[tuple[str, str], Variant],
        streams: dict[str, torch.cuda.Stream],
    ) -> None:
        with nvtx_range(f"h5_policy_{policy_name}"):
            for task in tasks:
                key = (task.load_class, task.shape_label)
                variant = selections[key]
                with nvtx_range(f"h5_step_{task.step}_{task.load_class}_{task.shape_label}_{variant.variant_id}"):
                    self.concurrent_step(
                        task.load_class,
                        shapes[task.shape_label].n_elements,
                        variant,
                        streams,
                    )

    def validate(self, shape: TaskShape, variant: Variant) -> dict:
        self.triton_fused(shape.n_elements, variant)
        self.torch_fused_reference(shape.n_elements)
        torch.cuda.synchronize()
        max_abs_diff = (self.y[: shape.n_elements].float() - self.y_torch[: shape.n_elements].float()).abs().max()
        finite = bool(torch.isfinite(self.y[: shape.n_elements]).all().item())
        return {
            "shape_label": shape.label,
            "variant_id": variant.variant_id,
            "finite_outputs": finite,
            "max_abs_diff_vs_torch": float(max_abs_diff.item()),
        }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def build_policies(rows: list[dict], variants: list[Variant], shapes: dict[str, TaskShape]) -> dict[str, dict]:
    by_variant = {variant.variant_id: variant for variant in variants}
    variant_ids = [variant.variant_id for variant in variants]
    load_classes = sorted({row["load_class"] for row in rows})
    shape_labels = sorted(shapes)

    isolated_mean = {
        variant_id: mean(
            [row["isolated_ms"] for row in rows if row["variant_id"] == variant_id and row["load_class"] == "idle"]
        )
        for variant_id in variant_ids
    }
    average_mean = {
        variant_id: mean([row["step_ms"] for row in rows if row["variant_id"] == variant_id])
        for variant_id in variant_ids
    }
    load_mean = {
        (load_class, variant_id): mean(
            [row["step_ms"] for row in rows if row["load_class"] == load_class and row["variant_id"] == variant_id]
        )
        for load_class in load_classes
        for variant_id in variant_ids
    }
    context_mean = {
        (load_class, shape_label, variant_id): mean(
            [
                row["step_ms"]
                for row in rows
                if row["load_class"] == load_class
                and row["shape_label"] == shape_label
                and row["variant_id"] == variant_id
            ]
        )
        for load_class in load_classes
        for shape_label in shape_labels
        for variant_id in variant_ids
    }

    static_isolated = min(variant_ids, key=lambda vid: isolated_mean[vid])
    static_average = min(variant_ids, key=lambda vid: average_mean[vid])

    policies = {
        "static_best_isolated": {
            (load_class, shape_label): by_variant[static_isolated]
            for load_class in load_classes
            for shape_label in shape_labels
        },
        "static_best_average": {
            (load_class, shape_label): by_variant[static_average]
            for load_class in load_classes
            for shape_label in shape_labels
        },
        "load_aware": {
            (load_class, shape_label): by_variant[
                min(variant_ids, key=lambda vid, load=load_class: load_mean[(load, vid)])
            ]
            for load_class in load_classes
            for shape_label in shape_labels
        },
        "resource_aware": {
            (load_class, shape_label): by_variant[
                min(variant_ids, key=lambda vid, load=load_class, shape=shape_label: context_mean[(load, shape, vid)])
            ]
            for load_class in load_classes
            for shape_label in shape_labels
        },
        "oracle_context": {
            (load_class, shape_label): by_variant[
                min(variant_ids, key=lambda vid, load=load_class, shape=shape_label: context_mean[(load, shape, vid)])
            ]
            for load_class in load_classes
            for shape_label in shape_labels
        },
    }
    return policies


def serialize_selections(selections: dict[tuple[str, str], Variant]) -> dict[str, str]:
    return {f"{load}:{shape}": variant.variant_id for (load, shape), variant in sorted(selections.items())}


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda", "note": "CUDA is unavailable."}

    torch.cuda.set_device(0)
    variants = [Variant(block_size, warps) for block_size in split_ints(args.block_sizes) for warps in split_ints(args.warps)]
    shapes = {
        "small": TaskShape("small", args.small_elements),
        "large": TaskShape("large", args.large_elements),
    }
    load_classes = ["idle", "compute", "memory"]
    queue = [
        QueueTask(0, "compute", "small"),
        QueueTask(1, "memory", "large"),
        QueueTask(2, "compute", "large"),
        QueueTask(3, "memory", "small"),
        QueueTask(4, "compute", "small"),
        QueueTask(5, "memory", "large"),
    ]

    workload = Workload(
        max_elements=max(shape.n_elements for shape in shapes.values()),
        matrix_size=args.matrix_size,
        memory_elements=args.memory_elements,
        scale=args.scale,
        seed=args.seed,
    )
    streams = {
        "background": torch.cuda.Stream(),
        "micro": torch.cuda.Stream(),
    }

    first_call_rows = []
    for variant in variants:
        wall_ms = wall_time_cuda(lambda v=variant: workload.triton_fused(shapes["small"].n_elements, v))
        first_call_rows.append(
            {
                "variant_id": variant.variant_id,
                "block_size": variant.block_size,
                "warps": variant.warps,
                "first_invocation_wall_ms": wall_ms,
            }
        )

    background_rows = []
    for load_class in load_classes:
        if load_class == "idle":
            latency = 0.0
        else:
            latency = cuda_time(lambda lc=load_class: workload.background(lc), args.warmup, args.iters)
        background_rows.append({"load_class": load_class, "background_only_ms": latency})

    calibration_rows = []
    for shape in shapes.values():
        for variant in variants:
            isolated_ms = cuda_time(
                lambda s=shape, v=variant: workload.triton_fused(s.n_elements, v),
                args.warmup,
                args.iters,
            )
            for load_class in load_classes:
                step_ms = cuda_time(
                    lambda lc=load_class, s=shape, v=variant: workload.concurrent_step(
                        lc,
                        s.n_elements,
                        v,
                        streams,
                    ),
                    args.warmup,
                    args.iters,
                )
                bg_ms = next(row["background_only_ms"] for row in background_rows if row["load_class"] == load_class)
                overlap_ratio = 0.0 if load_class == "idle" else 1.0 - step_ms / (bg_ms + isolated_ms)
                calibration_rows.append(
                    {
                        "load_class": load_class,
                        "shape_label": shape.label,
                        "n_elements": shape.n_elements,
                        "variant_id": variant.variant_id,
                        "block_size": variant.block_size,
                        "warps": variant.warps,
                        "isolated_ms": isolated_ms,
                        "background_only_ms": bg_ms,
                        "step_ms": step_ms,
                        "overlap_ratio": overlap_ratio,
                    }
                )

    policies = build_policies(calibration_rows, variants, shapes)
    policy_measurements = []
    policy_names = list(policies)
    rng = random.Random(args.seed)
    for repeat in range(args.policy_repeats):
        ordered = policy_names[:]
        rng.shuffle(ordered)
        for policy_name in ordered:
            selections = policies[policy_name]
            queue_ms = cuda_time(
                lambda name=policy_name, table=selections: workload.queue_run(queue, shapes, name, table, streams),
                args.policy_warmup,
                args.policy_iters,
            )
            policy_measurements.append(
                {
                    "repeat": repeat,
                    "policy": policy_name,
                    "queue_ms": queue_ms,
                }
            )

    policy_rows = []
    for policy_name, selections in policies.items():
        measurements = [row["queue_ms"] for row in policy_measurements if row["policy"] == policy_name]
        policy_rows.append(
            {
                "policy": policy_name,
                "queue_ms": median(measurements),
                "queue_mean_ms": mean(measurements),
                "queue_min_ms": min(measurements),
                "queue_max_ms": max(measurements),
                "selections": serialize_selections(selections),
            }
        )

    baseline_ms = next(row["queue_ms"] for row in policy_rows if row["policy"] == "static_best_isolated")
    oracle_ms = next(row["queue_ms"] for row in policy_rows if row["policy"] == "oracle_context")
    for row in policy_rows:
        row["speedup_vs_static_isolated"] = baseline_ms / row["queue_ms"]
        row["regret_vs_oracle_pct"] = (row["queue_ms"] / oracle_ms - 1.0) * 100.0

    best_policy = min(policy_rows, key=lambda row: row["queue_ms"])
    validation_variant_id = min(
        {row["variant_id"] for row in calibration_rows},
        key=lambda vid: mean(
            [
                row["step_ms"]
                for row in calibration_rows
                if row["variant_id"] == vid and row["load_class"] == "idle"
            ]
        ),
    )
    validation_variant = next(variant for variant in variants if variant.variant_id == validation_variant_id)
    validations = [workload.validate(shape, validation_variant) for shape in shapes.values()]

    return {
        "status": "ok",
        "args": vars(args),
        "queue": [task.__dict__ for task in queue],
        "shapes": {label: shape.__dict__ for label, shape in shapes.items()},
        "variants": [variant.__dict__ | {"variant_id": variant.variant_id} for variant in variants],
        "first_call_rows": first_call_rows,
        "background_rows": background_rows,
        "calibration_rows": calibration_rows,
        "policy_measurements": policy_measurements,
        "policy_rows": policy_rows,
        "best_policy": best_policy,
        "compile_cache_proxy": {
            "first_invocation_min_ms": min(row["first_invocation_wall_ms"] for row in first_call_rows),
            "first_invocation_median_ms": median([row["first_invocation_wall_ms"] for row in first_call_rows]),
            "first_invocation_max_ms": max(row["first_invocation_wall_ms"] for row in first_call_rows),
            "first_invocation_total_ms": sum(row["first_invocation_wall_ms"] for row in first_call_rows),
        },
        "validation": validations,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_outputs(out_dir: Path, data_dir: Path, env: dict, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result.get("status") != "ok":
        return

    write_csv(
        out_dir / "first_call.csv",
        result["first_call_rows"],
        ["variant_id", "block_size", "warps", "first_invocation_wall_ms"],
    )
    write_csv(
        out_dir / "background.csv",
        result["background_rows"],
        ["load_class", "background_only_ms"],
    )
    write_csv(
        out_dir / "calibration.csv",
        result["calibration_rows"],
        [
            "load_class",
            "shape_label",
            "n_elements",
            "variant_id",
            "block_size",
            "warps",
            "isolated_ms",
            "background_only_ms",
            "step_ms",
            "overlap_ratio",
        ],
    )
    policy_rows = [
        {
            **row,
            "selections": json.dumps(row["selections"], sort_keys=True),
        }
        for row in result["policy_rows"]
    ]
    write_csv(
        out_dir / "policy_results.csv",
        policy_rows,
        [
            "policy",
            "queue_ms",
            "queue_mean_ms",
            "queue_min_ms",
            "queue_max_ms",
            "speedup_vs_static_isolated",
            "regret_vs_oracle_pct",
            "selections",
        ],
    )
    write_csv(
        out_dir / "policy_measurements.csv",
        result["policy_measurements"],
        ["repeat", "policy", "queue_ms"],
    )
    write_csv(
        out_dir / "summary.csv",
        [
            {
                "best_policy": result["best_policy"]["policy"],
                "best_queue_ms": result["best_policy"]["queue_ms"],
                "resource_aware_queue_ms": next(
                    row["queue_ms"] for row in result["policy_rows"] if row["policy"] == "resource_aware"
                ),
                "static_best_isolated_queue_ms": next(
                    row["queue_ms"] for row in result["policy_rows"] if row["policy"] == "static_best_isolated"
                ),
                "static_best_average_queue_ms": next(
                    row["queue_ms"] for row in result["policy_rows"] if row["policy"] == "static_best_average"
                ),
                "load_aware_queue_ms": next(
                    row["queue_ms"] for row in result["policy_rows"] if row["policy"] == "load_aware"
                ),
                "oracle_queue_ms": next(
                    row["queue_ms"] for row in result["policy_rows"] if row["policy"] == "oracle_context"
                ),
                **result["compile_cache_proxy"],
            }
        ],
        [
            "best_policy",
            "best_queue_ms",
            "resource_aware_queue_ms",
            "static_best_isolated_queue_ms",
            "static_best_average_queue_ms",
            "load_aware_queue_ms",
            "oracle_queue_ms",
            "first_invocation_min_ms",
            "first_invocation_median_ms",
            "first_invocation_max_ms",
            "first_invocation_total_ms",
        ],
    )

    for source, target_name in [
        (out_dir / "first_call.csv", "h5_1_runtime_selector_first_call.csv"),
        (out_dir / "calibration.csv", "h5_1_runtime_selector_calibration.csv"),
        (out_dir / "policy_measurements.csv", "h5_1_runtime_selector_policy_measurements.csv"),
        (out_dir / "policy_results.csv", "h5_1_runtime_selector_policy_results.csv"),
        (out_dir / "summary.csv", "h5_1_runtime_selector_summary.csv"),
    ]:
        (data_dir / target_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    args = parse_args()
    env = environment()
    result = run(args)
    write_outputs(Path(args.out_dir), Path(args.data_dir), env, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
