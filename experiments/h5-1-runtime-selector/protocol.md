# H5.1 Protocol — Resource-Shaped Runtime Selector

Date: 2026-05-28

## Hypothesis

Runtime resource-aware task management can beat static kernel-variant selection
for a single-request inference fragment when the co-running accelerator load
changes across micro-tasks.

## Prediction

A scheduler that selects a Triton micro-operator variant from the current
resource/load state will reduce end-to-end queue latency versus:

- one global best-isolated variant;
- one global best-average variant;
- a load-aware selector that ignores task shape.

The oracle selector that knows the measured best variant for each
`(load_class, task_shape)` pair is the upper bound. If resource-aware selection
does not beat the static baselines, then the current micro-op/variant set is too
weak or the runtime's observable state is not predictive enough.

## Workload

The synthetic fragment models a dependency-aware single-request queue:

1. each micro-task is a fused model-shaped epilogue
   `Y = GELU(X * scale + Bias) + Residual`;
2. each task runs while the rest of the accelerator is in one of three observed
   states: `idle`, `compute` GEMM pressure, or `memory` vector-add pressure;
3. tasks execute sequentially through the queue, while each task may overlap
   with its current background load on another CUDA stream.

The queue deliberately changes resource state between tasks. This keeps H5.1
focused on runtime selection instead of another single-kernel speedup.

## Variants

The fused Triton micro-op is generated with multiple `BLOCK_SIZE` and
`num_warps` variants. The benchmark records first invocation wall time as a
dynamic compile/cache proxy, then measures steady-state isolated and concurrent
latency.

## Selectors

- `static_best_isolated`: one variant with lowest mean isolated micro-op
  latency.
- `static_best_average`: one variant with lowest mean latency across all
  calibration states.
- `load_aware`: one variant per current load class, averaged across task
  shapes.
- `resource_aware`: one variant per `(load_class, task_shape)`.
- `oracle_context`: measured best variant per queue step; this should match
  the best possible table selector for the calibrated state space.

## Metrics

Primary metric:

- end-to-end queue latency in milliseconds.

Secondary metrics:

- policy speedup versus `static_best_isolated`;
- policy regret versus `oracle_context`;
- variant choices per queue step;
- first invocation wall time for compile/cache overhead;
- background-only latency and overlap diagnostics.

## Sanity Checks

- CUDA must be available.
- Outputs must remain finite.
- A Triton fused output is compared against PyTorch GELU for max absolute error.
- Timings are recorded after warmup; first invocation timing is reported
  separately and not mixed with steady-state latency.

## Notes

Nsight Systems was expected from the Docker handoff, but the current shell does
not expose `docker` or `nsys`. The benchmark still emits NVTX ranges so the same
script can be profiled later in the GPU Docker environment.
