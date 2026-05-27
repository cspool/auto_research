# H2.1 Protocol: Triton Fused Micro-Kernel Variants

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H2: Micro-operator tiling plus multi-version kernels improves scheduling.

This follow-up narrows the prediction:

> For a model-shaped fused micro-kernel, the fastest standalone Triton variant
> will not necessarily be the best variant when co-scheduled with a dominant
> GEMM. Fusion reduces launch and global-memory traffic, but concurrency still
> needs variant-level scheduling metadata.

## Workload

The workload has two independent operators:

1. a GEMM (`A @ B`) representing a compute-heavy model operator;
2. a fused Triton micro-kernel representing residual/MLP epilogue work:
   `Y = GELU(X * scale + Bias) + Residual`.

The fused micro-kernel is model-shaped because it combines elementwise scale,
bias, activation, and residual addition into one launch. This approximates
common transformer and diffusion epilogue fragments without requiring a full
model graph.

The Triton micro-kernel is generated with multiple variants:

- `BLOCK_SIZE`: 128, 256, 512, 1024, 2048, 4096, 8192;
- `num_warps`: 4, 8.

For each variant, measure:

- standalone fused Triton latency;
- eager PyTorch unfused chain latency;
- serial latency: GEMM then fused Triton kernel;
- concurrent latency: GEMM and fused Triton kernel launched on separate CUDA
  streams;
- overlap ratio: `1 - concurrent / (gemm_alone + fused_alone)`;
- stream speedup: `serial / concurrent`;
- nominal fused-kernel bandwidth from global bytes touched.

## Metrics

| Metric | Meaning |
|---|---|
| `fused_ms` | Isolated Triton fused micro-kernel latency. |
| `torch_unfused_ms` | Isolated eager PyTorch chain latency. |
| `serial_ms` | GEMM then fused Triton latency. |
| `concurrent_ms` | Separate-stream latency. |
| `stream_speedup` | `serial_ms / concurrent_ms`. |
| `overlap_ratio` | Fraction of isolated work hidden by concurrency. |
| `nominal_fused_gbps` | Nominal global-memory bytes per fused latency. |
| `best_isolated_variant` | Variant with minimum fused latency. |
| `best_concurrent_variant` | Variant with minimum concurrent latency. |

## Prediction

If H2 generalizes beyond vector add:

- Triton fusion will beat the eager PyTorch chain for the micro-kernel itself;
- variants will differ measurably in concurrent latency and overlap ratio;
- the best concurrent variant may differ from the fastest isolated variant;
- the scheduler objective will matter: best isolated latency, best concurrent
  latency, and best stream speedup may select different variants.

If H2 does not generalize in this setup:

- all variants behave similarly under concurrency, or
- the fastest isolated fused variant is also the best concurrent and best
  stream-speedup variant.

## Sanity Checks

- CUDA events time all GPU work.
- Warm up each variant to trigger Triton compilation before measurement.
- Compare the Triton fused output with PyTorch `GELU(..., approximate="tanh")`.
- Check outputs are finite.
- Run GPU measurements outside the default sandbox because `/dev/nvidia*` is
  not visible there.
