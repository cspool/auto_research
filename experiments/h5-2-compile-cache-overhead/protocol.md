# H5.2 Protocol - Dynamic Compilation and Cache Overhead

Date: 2026-05-28

## Hypothesis

Dynamic compilation and cache state are first-class runtime constraints for
resource-aware micro-operator scheduling. If variant generation happens on the
critical path, compile/cache overhead will dominate the steady-state latency of
fine-grained single-request micro-tasks.

## Prediction

For H5.1 Triton fused GELU/residual variants:

- cold-cache first launch will be orders of magnitude slower than steady-state
  launch latency;
- cache-hit first launch will be much cheaper than cold compile but still much
  slower than steady-state event timing because it includes module loading and
  runtime setup;
- after warmup, steady-state launch+kernel latency will return to sub-ms scale;
- therefore H5 runtimes should precompile, warm caches ahead of the request, or
  use a fallback variant while asynchronous compilation happens off the critical
  path.

## Method

A parent script launches a worker Python process for each variant and cache
state. The worker imports Torch/Triton, allocates tensors, then measures:

1. `first_launch_wall_ms`: synchronized wall time for the first kernel launch in
   that worker process;
2. `steady_event_ms`: CUDA event timing after warmup for repeated launches;
3. Triton cache file counts and bytes after the run.

For each variant, the parent runs:

- `cold`: empty per-variant `TRITON_CACHE_DIR`;
- `cache_hit`: same cache directory after the cold run;
- optional repeated cache-hit workers to expose module-load/process variance.

## Metrics

- cold first launch wall time;
- cache-hit first launch wall time;
- steady-state CUDA event latency;
- cold/cache-hit overhead ratios relative to steady state;
- cache artifact count and bytes.

## Sanity Checks

- CUDA must be available.
- Outputs must be finite.
- The same kernel and shapes are used across cold/cache-hit states.
