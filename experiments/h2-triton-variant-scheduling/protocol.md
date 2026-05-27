# H2 Protocol: Triton Variant Scheduling

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H2: Micro-operator tiling plus multi-version kernels improves scheduling.

For this first H2 test, the narrower prediction is:

> The micro-kernel variant with the best standalone latency is not necessarily
> the best variant for concurrent execution with a dominant GEMM. Runtime
> scheduling should consider concurrency behavior, not just isolated kernel time.

## Workload

The workload has two independent operators:

1. a GEMM (`A @ B`) representing a compute-heavy model operator;
2. a Triton vector add (`Y = X + Bias`) representing a memory-bound micro-op.

The vector add is generated with multiple Triton variants:

- `BLOCK_SIZE`: 128, 256, 512, 1024, 2048, 4096, 8192;
- `num_warps`: 4, 8.

For each variant, measure:

- standalone add latency;
- serial latency: GEMM then add;
- concurrent latency: GEMM and add launched on separate CUDA streams;
- overlap ratio: `1 - concurrent / (gemm_alone + add_alone)`;
- stream speedup: `serial / concurrent`.

## Metrics

| Metric | Meaning |
|---|---|
| `add_ms` | Isolated Triton add latency. |
| `serial_ms` | GEMM then add latency. |
| `concurrent_ms` | Separate-stream latency. |
| `stream_speedup` | `serial_ms / concurrent_ms`. |
| `overlap_ratio` | Fraction of isolated work hidden by concurrency. |
| `best_isolated_variant` | Variant with minimum add latency. |
| `best_concurrent_variant` | Variant with minimum concurrent latency. |

## Prediction

If H2 is supported:

- variants will differ measurably in concurrent latency and overlap ratio;
- the best concurrent variant may differ from the fastest isolated variant;
- a scheduler that selects by isolated latency alone may be suboptimal.

If H2 is not supported in this setup:

- all variants behave similarly under concurrency, or
- the fastest isolated variant is also the best concurrent variant.

## Sanity Checks

- CUDA events time all GPU work.
- Warm up each variant to trigger Triton compilation before measurement.
- Exclude default sandbox runs because `/dev/nvidia*` is not visible there.
- Check output is finite.

