# H5.4 Analysis - Warmed Persistent Queue Runtime Limits

Date: 2026-05-28

## Question

After cold compile and fresh-process cache lookup are removed from the critical
path, what limits the H5.1 fine-grained runtime queue: policy selection, CPU/CUDA
submission overhead, stream dependency management, or GPU kernel time?

## Setup

H5.4 reuses H5.1's six-step queue and policy selections, but runs in one warmed
persistent process. Before the measured/profiler windows it allocates tensors,
precompiles every selected Triton variant, runs background compute and memory
operators, and warms both policy queues.

The final Nsight Systems capture uses `cudaProfilerStart/Stop`, so the profile
contains only warmed queue submission plus one final synchronization. CUDA event
and wall timing are measured outside the profiler capture to avoid per-repeat
synchronization pollution.

Captured artifacts:

```text
experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.nsys-rep
experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.sqlite
experiments/h5-4-runtime-limit-profiling/results/rtx4090_default/nsys_summary.json
```

## Warmed Policy Timing

| Policy | Event median ms | Event min ms | Event max ms | Wall median ms | Relative speed |
|---|---:|---:|---:|---:|---:|
| static_best_average | 2.2662 | 2.2310 | 2.3839 | 2.2664 | 1.0000x |
| resource_aware | 2.2338 | 2.1596 | 2.3578 | 2.2475 | 1.0145x |

Resource-aware selection is slightly faster in this warmed run, by about
1.45%. That is still smaller than the min-to-max timing
spread within each policy: 6.85% for `static_best_average` and
9.17% for `resource_aware`. The selector signal remains real
but small at this granularity.

## Nsight Systems Summary

The clean queue-only capture ran 48 repeats of each policy, 96 queue submissions
total. Each queue has six steps and launches one background kernel plus one
Triton fused micro-op per step, for 1152 GPU kernels in the capture.

| Component | Count | Total ms | Mean |
|---|---:|---:|---:|
| CUDA runtime API calls | 14978 | 96.13 | - |
| Non-sync CUDA API time | - | 24.77 | - |
| Final `cudaDeviceSynchronize` wait | 1 | 71.36 | - |
| CUDA runtime+driver launch wrappers | 576+576 | 8.56 | 14.87 us per launch pair |
| PyTorch stream/event management calls | - | 16.21 | - |
| GPU kernels | 1152 | 241.89 | 209.97 us per kernel |
| GPU kernel span | - | 206.92 | overlap proxy 1.17x |
| CPU enqueue NVTX window | 1 | 136.16 | 1.418 ms per queue |

Kernel breakdown:

| Kernel | Count | Total ms | Mean us |
|---|---:|---:|---:|
| memory background elementwise | 288 | 122.59 | 425.65 |
| fused Triton micro-op | 576 | 67.20 | 116.66 |
| compute background GEMM | 288 | 52.10 | 180.91 |

The CPU enqueue NVTX window is 136.16 ms, while GPU kernels span
206.92 ms. That means the Python/PyTorch stream queue can enqueue
work faster than the GPU drains it in this warmed setting, but the enqueue path
is still large: about 1.418 ms per six-step queue before
final synchronization. The stream/event management calls alone account for about
16.21 ms in the capture, and launch wrappers add another
8.56 ms.

## Interpretation

H5.4 refines H5.1 and H5.2 rather than overturning them. Once cold compile and
fresh-process cache lookup are removed, the queue is no longer dominated by
Triton dynamic loading. However, a Python/PyTorch stream-level scheduler still
pays substantial host-side overhead to express fine-grained dependencies:
CUDA events, stream waits, stream-capture checks, and launch wrappers.

The remaining policy difference is about 1.45% in favor of resource-aware
selection in this run, but intra-policy spread is several times larger. A better
runtime-selector story therefore needs a cheaper dispatch substrate before more
complex policy rules or migration decisions can be trusted.

## Decision

H5.4 supports moving away from a Python/CUDA-stream queue for fine-grained
micro-ops. The next experiment should test whether CUDA Graph replay or another
persistent/batched dispatch path reduces queue overhead enough for the
resource-aware selector to matter. H5.3 task migration should use larger tiles or
that cheaper dispatch substrate; otherwise migration overhead will likely hide
any resource-placement benefit.
