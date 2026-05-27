# H2 Analysis: Triton Variant Scheduling

Date: 2026-05-26

## Status

Preliminary support for H2.

The experiment tested a controlled Triton memory micro-kernel (`Y = X + Bias`)
with multiple `BLOCK_SIZE` and `num_warps` variants. Each variant was measured
in isolation and when running concurrently with an independent GEMM on separate
CUDA streams.

## Core Result

The fastest isolated variant was not the best concurrent variant in both tested
shapes. This supports the H2 claim that runtime scheduling needs multi-version
kernel metadata and should not select variants solely by standalone latency.

## Results

| Run | Best isolated variant | Best isolated add ms | Best concurrent variant | Best concurrent ms | Best concurrent overlap |
|---|---|---:|---|---:|---:|
| balanced 2048 | block=4096, warps=4 | 0.1019 | block=2048, warps=8 | 0.1864 | 0.1652 |
| memory-heavy 1024 | block=256, warps=8 | 0.4227 | block=128, warps=4 | 0.4349 | 0.0197 |

For the balanced shape, selecting the best concurrent variant instead of the
fastest isolated variant reduced concurrent latency from `0.1949 ms` to
`0.1864 ms` and increased overlap ratio from `0.0917` to `0.1652`.

For the memory-heavy shape, the difference is smaller but still present: the
best isolated variant had almost no stream speedup (`0.9995x`), while another
variant gave the best concurrent latency and a separate variant gave the best
stream speedup (`1.0990x`).

## Interpretation

This is the first experiment where the positive result matches the compiler
literature pattern from Infera/MetaAttention/PAT: micro-kernels should exist in
multiple variants, and runtime selection should use scheduling context.

The result also refines H1.1. PyTorch-level row chunking was too blunt because it
changed GEMM shape and added launches without exposing clean resource metadata.
Triton variants are still simple, but they expose a controllable knob
(`BLOCK_SIZE`, `num_warps`) and show that isolated kernel speed is not a complete
selection criterion.

## Limitations

- The Triton micro-kernel is a simple vector add, not yet attention/MoE/DiT.
- We measured latency and overlap, not Nsight hardware counters.
- Variant behavior may include Triton/codegen and CUDA scheduling effects that
  need profiler confirmation.
- Best-stream-speedup and best-concurrent-latency can differ; the scheduling
  objective must be explicit.

## Next Step

H2.1 should move from vector add to a model-shaped fused micro-kernel:

1. add + activation + scale, or
2. dequantize + add, or
3. small expert FFN tile.

The next measurement should include Nsight Compute or at least derived bandwidth
to connect variant choice to resource pressure.

