# H2.1 Analysis: Triton Fused Micro-Kernel Variants

Date: 2026-05-26

## Status

H2 is strengthened, with an important caveat about metrics.

The experiment replaced the simple H2 vector add with a model-shaped fused
micro-kernel:

`Y = GELU(X * scale + Bias) + Residual`.

This is closer to transformer/diffusion epilogue work than a pure memory copy
or add. The result preserves the main H2 pattern: variant choice depends on the
scheduling objective.

## Core Result

Triton fusion was clearly faster than eager PyTorch for the micro-kernel itself,
and the best standalone fused variant did not reliably identify the best
co-scheduled behavior.

| Run | PyTorch unfused ms | Best fused ms | Best isolated variant | Best concurrent variant | Best concurrent ms | Best stream-speedup variant |
|---|---:|---:|---|---|---:|---|
| balanced 2048 | 0.2721 | 0.1422 | block=2048, warps=8 | block=2048, warps=4 | 0.2271 | block=4096, warps=4 |
| memory-heavy 1024 | 1.4640 | 0.5614 | block=256, warps=8 | block=512, warps=4 | 0.5726 | block=2048, warps=4 |

For the balanced shape, the best concurrent variant reduced latency from
`0.2766 ms` for the fastest isolated variant under concurrency to `0.2271 ms`.
That is a meaningful scheduling-context effect, not just noise.

For the memory-heavy shape, the best isolated and best concurrent variants were
effectively tied in absolute concurrent latency (`0.5726208 ms` vs
`0.5726037 ms`). The stronger signal there is metric separation: the best
stream-speedup variant reported `1.1427x` speedup but had worse absolute
concurrent latency (`0.5920 ms`) than the best concurrent variant.

## Interpretation

The fused micro-kernel confirms that operator fusion and concurrent scheduling
are complementary, not substitutes. Fusion removes framework-level launch and
global-memory overhead: the best Triton fused latency was about `1.91x` faster
than PyTorch unfused on the balanced shape and about `2.61x` faster on the
memory-heavy shape. But once fused kernels become scheduling units, their
variant resource profile still matters.

The most useful refinement is about objective functions:

- `best isolated latency` is good for single-op optimization;
- `best concurrent latency` is the direct scheduling target for latency;
- `stream_speedup` can be inflated by a slow serial baseline and should be
  treated as a diagnostic, not the primary scheduler objective.

This aligns with Infera/MetaAttention/PAT-style thinking: a compiler/runtime
should keep multi-version kernel metadata and select variants according to the
current co-location context.

## Sanity Checks

- GPU: RTX 4090 visible only outside the default sandbox.
- PyTorch: 2.11.0+cu128; Triton: 3.6.0.
- Output finite checks passed.
- Max absolute difference versus PyTorch tanh-GELU was `0.0078125`, acceptable
  for fp16 epilogue timing.
- Runs were sequential to avoid multi-process GPU interference.

## Limitations

- The fused kernel is still a synthetic epilogue, not a full attention, MoE, or
  DiT block.
- Hardware counters were not collected; derived bandwidth is only nominal.
- The memory-heavy best-isolated/best-concurrent difference is numerically tiny.
- CUDA event timing is enough for latency trends but not for explaining SM/SFU
  occupancy, DRAM traffic, or scheduler stalls.

## Next Step

H2.2 should add profiler-facing resource evidence. Two practical paths:

1. collect Nsight Compute counters for the balanced variants
   `(2048,8)`, `(2048,4)`, and `(4096,4)`;
2. move to a model-semantic workload, such as a tiny MoE expert epilogue or a
   DiT patch/MLP fragment, and preserve the same variant-vs-scheduler metrics.
