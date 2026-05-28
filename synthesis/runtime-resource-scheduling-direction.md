# Direction Update: Runtime Resource-Aware Micro-Operator Scheduling

Date: 2026-05-27
Tooling update: local GPU profiling/developer packages were installed after
this direction update. Nsight Systems is now on `PATH`; Nsight Compute and CUDA
Toolkit 12.8 developer binaries are available under `/usr/local/cuda-12.8/bin`;
TensorRT `trtexec` is on `PATH`. The default sandbox still does not expose GPU
device nodes, so GPU/profiler runs should use the same escalated GPU execution
path or a GPU-visible shell.

## Motivation

The project should not be framed as "find one more faster kernel." The single
kernel acceleration space is already relatively constrained: once movement,
softmax, value aggregation, or MoE expert tiles are fused, additional kernel
speedups are incremental and workload-specific.

The more important research direction is runtime orchestration:

> Given a single request on one accelerator, dynamically compile or select
> multiple operator/micro-operator variants and schedule them according to
> currently available hardware resources, while managing task queues, migration
> boundaries, and backend resource limits.

This reframes H2-H4 as evidence for the substrate needed by such a runtime:
multi-version kernels, compiler metadata, semantic micro-operator units,
resource feasibility filters, and measured full-fragment latency.

## Revised Research Claim

Useful single-request inference concurrency comes from a runtime system with
four cooperating layers:

1. **Dynamic micro-op generation**: compile or retrieve variants with different
   tile shapes, warps, shared-memory footprints, value-blocking, query-blocking,
   and launch shapes.
2. **Resource-state observation**: estimate available SM, memory bandwidth,
   shared memory, launch-queue pressure, and occupancy from profiler counters or
   lightweight runtime proxies.
3. **Task scheduling and migration**: split model work into migration-safe
   micro-tasks, enqueue them with dependencies, and move remaining tasks to
   variants/streams/partitions that fit current resources.
4. **Backend resource management**: expose hardware/runtime controls such as
   CUDA streams, CUDA Graphs, MPS/SM partitioning, persistent work queues, NPU
   pipeline ports, or accelerator-specific command queues.

## How Existing Experiments Support This

- H1/H1.1 show that ordinary streams and framework-level chunking do not provide
  enough control over idle resources.
- H2/H2.1 show that the best kernel variant depends on co-scheduling context,
  not just isolated latency.
- H2.2 shows that dynamic/runtime selection needs compiler metadata, even when
  hardware counters are unavailable.
- H3 shows that model semantics must be lowered into no-padding grouped
  micro-tasks, and that movement micro-ops are part of the schedule.
- H4 shows that sparse attention needs resource feasibility filtering and
  shape/order-aware variant selection. H4.6-H4.9 are already a small runtime
  selector prototype.

## New Hypothesis H5

Runtime resource-aware task management can outperform static best-variant
selection for single-request inference fragments when the accelerator has
changing residual resources.

Prediction:

- a scheduler that chooses micro-op variants based on current co-running load
  will beat a static best-isolated or static best-average selector;
- tile-level task boundaries will provide a practical migration granularity;
- dynamic compilation/caching overhead will be acceptable only when variants are
  reused or compiled ahead of the critical path;
- software runtime limits, such as launch overhead and weak resource visibility,
  will explain why naive operator concurrency underperforms.

## Proposed Experiment Sequence

### H5.1 Resource-Shaped Runtime Selector

Build a small runtime that schedules one primary operator plus one or two
micro-op chains. Use existing H2/H4 variants as tasks. Compare:

- static best-isolated variant;
- static best-average variant;
- oracle best-concurrent variant;
- runtime selector keyed by current co-running load class.

Primary metric: end-to-end single-request latency. Secondary metrics: overlap
ratio, variant regret, launch count, and invalid/resource-rejected variants.

Updated profiling parameters:

- wrap representative runs with `nsys profile --trace=cuda,nvtx,osrt`;
- mark scheduler decisions and variant IDs with NVTX ranges if practical;
- record launch count, launch gaps, stream timeline overlap, and CPU submission
  overhead alongside latency.

### H5.2 Dynamic Compilation and Cache Overhead

Measure Triton compile latency, cache lookup latency, and warm-run latency for
the same variant table. Distinguish:

- cold compile on critical path;
- ahead-of-time warmup;
- lazy compile with fallback variant;
- cache-hit steady state.

This tells whether dynamic compilation is a realistic runtime mechanism or only
an offline autotuning mechanism.

Updated dependency parameters:

- explicitly capture the Triton cache directory and compile wall time;
- keep `/usr/local/cuda-12.8/bin` available for later cubin/SASS inspection;
- report cold compile, cache-hit launch, and steady-state launch separately.

### H5.3 Task Migration Boundary

Implement migration at micro-task granularity rather than in-kernel preemption.
For MoE or sparse attention, split remaining work into tile tasks and allow the
runtime to rebind later tiles to a different variant or stream after observing
resource pressure.

This models what a software runtime can plausibly do on current GPUs. True
in-kernel task migration remains a hardware/runtime co-design topic.

### H5.4 Software/Hardware Runtime Limit Analysis

Nsight Systems/Compute is now available. Measure:

- launch gaps and CPU submission overhead;
- achieved occupancy and active warps;
- SM busy, Tensor Core, LDST, SFU utilization;
- HBM read/write traffic;
- shared-memory pressure and occupancy limits;
- stream overlap versus hardware execution overlap.

This directly addresses which runtime and hardware limits block performance.

Initial command shapes:

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true \
  -o experiments/h5-1-runtime-selector/results/nsys_<case> \
  python3 experiments/h5-1-runtime-selector/code/run_runtime_selector.py

/usr/local/cuda-12.8/bin/ncu --set speed-of-light --target-processes all \
  --force-overwrite --export experiments/h5-4-runtime-limits/results/ncu_<case> \
  python3 experiments/h5-4-runtime-limits/code/profile_runtime_limits.py
```

Use Nsight Compute only on representative kernels/cases first; full sweeps would
distort runtimes and produce too much profiler output.

## Role of H4.10

H4.10 value-dimension held-outs are still useful, but they should be positioned
as a selector stress test, not the main research direction. If value dimension
breaks the selector, it adds `value_dim`, `num_v_blocks`, and value pressure to
the runtime feature table. If it does not, the H4 selector is mature enough to
serve as an input to H5's runtime scheduler.

## Expected Contribution Shape

The paper/story should target this form:

> Single-request modern-model inference is limited less by individual kernel
> optimization than by missing runtime support for resource-aware micro-task
> orchestration. A practical system needs dynamic variant generation, runtime
> resource-state observation, dependency-aware micro-task queues, and migration
> boundaries that align with compiler-visible model semantics.
