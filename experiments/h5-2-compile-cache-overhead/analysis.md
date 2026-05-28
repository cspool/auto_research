# H5.2 Analysis - Dynamic Compilation and Cache Overhead

Date: 2026-05-28

## Question

How large are Triton cold-compile, cache-hit first-launch, and steady-state
launch costs for the same fused micro-operator variants used in H5.1?

## Setup

H5.2 reuses the fused GELU/residual Triton micro-op from H5.1 with five
representative variants:

```text
B512_W4, B1024_W8, B2048_W8, B4096_W4, B8192_W8
```

For each variant, a parent process launches workers under a per-variant
`TRITON_CACHE_DIR`:

- `cold`: empty cache directory;
- `cache_hit`: same directory after the cold run, repeated twice;
- steady state: CUDA event timing after warmup inside each worker.

The worker imports Torch/Triton before timing the first launch, allocates tensors,
then times a synchronized first kernel launch. Process wall time is also recorded
for the full worker invocation.

## Results

Aggregate over five variants:

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| cold first launch wall ms | 1319.75 | 1391.29 | 1414.86 |
| cache-hit first launch wall ms | 608.68 | 661.03 | 740.17 |
| steady event ms | 0.1424 | 0.1439 | 0.1564 |
| cold overhead ratio vs steady | 9015x | 9259x | 9835x |
| cache-hit overhead ratio vs steady | 4231x | 4637x | 5117x |
| cold process wall ms | 3856.38 | 4126.22 | 4359.65 |
| cache-hit process wall ms | 3128.74 | 3383.79 | 3719.06 |

Per-variant summary:

| Variant | Cold first ms | Cache-hit first ms | Steady ms | Cold ratio | Hit ratio |
|---|---:|---:|---:|---:|---:|
| B1024_W8 | 1375.76 | 728.53 | 0.1424 | 9662x | 5117x |
| B2048_W8 | 1319.75 | 661.03 | 0.1425 | 9259x | 4637x |
| B4096_W4 | 1414.86 | 608.68 | 0.1439 | 9835x | 4231x |
| B512_W4 | 1409.72 | 740.17 | 0.1564 | 9015x | 4733x |
| B8192_W8 | 1391.29 | 649.45 | 0.1531 | 9090x | 4243x |

All runs produced finite outputs. Each variant generated 10 cache files; cache
bytes ranged from about 115 KB to 228 KB.

## Interpretation

H5.2 strongly supports the overhead diagnosis from H5.1. Dynamic compilation on
the critical path is impossible for this micro-task scale: the cold first launch
is roughly 1.3-1.4 seconds, while steady-state launch+kernel latency is about
0.14-0.16 milliseconds.

Cache hits help, but not enough if the first cache-hit launch happens inside the
request. A cache-hit worker still spends about 0.61-0.74 seconds on its first
synchronized launch. The full worker process wall time remains 3.1-3.7 seconds,
which includes Python/Torch/Triton process startup and module setup.

This means an H5 runtime needs at least one of the following:

- ahead-of-time compilation and cache warming before the request;
- a persistent runtime process that keeps modules loaded;
- a fallback variant while new variants compile asynchronously;
- coarser migration-safe tasks so each scheduling decision amortizes dispatch;
- CUDA Graph or persistent-kernel/batched dispatch to reduce launch overhead.

## What This Rules Out

- Per-request cold compilation for fine-grained micro-op variants.
- Treating Triton cache hits as free in a fresh worker/process.
- Measuring scheduler policies without separating process startup, module load,
  first launch, and steady-state launch costs.

## Next

H5.4 should profile a warmed persistent process and queue-only execution with
Nsight Systems. The immediate target is to remove compile and process startup
from the critical path, then quantify remaining launch gaps and stream overlap.
After that, H5.3 can test migration boundaries using larger task tiles or a
persistent/batched dispatch substrate.
