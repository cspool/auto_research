# H3.2 Analysis: Triton MoE Expert Epilogue Micro-Operators

Date: 2026-05-26

## Status

Strong support for the refined H3.

H3.1 showed that MoE expert semantics alone do not help if each expert remains a
PyTorch-level stream launch. H3.2 lowers the expert epilogue to a Triton
micro-operator and compares per-expert launches against a grouped single kernel.
The grouped kernel wins decisively.

## Results

| Run | Token counts | PyTorch loop ms | Best grouped ms | Grouped vs PyTorch | Grouped vs Triton serial | Grouped vs Triton concurrent |
|---|---|---:|---:|---:|---:|---:|
| balanced 4x64 | 64,64,64,64 | 0.2906 | 0.0172 | 16.88x | 7.90x | 17.72x |
| skewed | 160,64,24,8 | 0.2560 | 0.0135 | 18.98x | 7.40x | 16.37x |
| tiny 4x16 | 16,16,16,16 | 0.2480 | 0.0175 | 14.15x | 7.48x | 16.95x |

Per-expert Triton stream concurrency remained slower than per-expert Triton
serial execution. The best concurrent implementation was still about 16x to 18x
slower than the grouped single-kernel implementation.

## Interpretation

This is the clearest positive result so far for model-semantic decomposition:
MoE experts are useful units, but the runtime should not schedule them as
independent framework-level kernels. The useful implementation unit is a grouped
compiler-visible micro-operator that removes per-expert launch overhead and lets
the compiler generate one contiguous tile schedule.

This refines the project story:

- H3.1 negative: expert branches on streams are too coarse;
- H3.2 positive: grouped expert epilogue micro-operators are effective;
- H2/H2.2 explain how to make this a runtime system: generate variants, keep
  metadata, and select by absolute grouped/concurrent latency.

The result also shows a boundary for "operator concurrency": sometimes the best
way to exploit semantic parallelism is not simultaneous kernels, but collapsing
many semantic fragments into one compiler-visible grouped kernel.

## Variant Selection

Best grouped variant was not always the best per-expert concurrent variant:

| Run | Best grouped | Best concurrent | Different? |
|---|---|---|---|
| balanced 4x64 | block=256, warps=8 | block=128, warps=4 | yes |
| skewed | block=2048, warps=8 | block=2048, warps=8 | no |
| tiny 4x16 | block=2048, warps=8 | block=4096, warps=8 | yes |

This preserves the H2 lesson: the scheduling objective must be explicit. For
H3.2, the main objective is grouped latency, not stream speedup.

## Sanity Checks

- GPU runs were performed outside the default sandbox.
- Output finite checks passed.
- Max absolute difference versus PyTorch tanh-GELU was `0.00390625`, acceptable
  for fp16 timing.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- H3.2 isolates expert epilogue/routing-adjacent work; it does not implement
  grouped expert GEMM.
- Routing/top-k and scatter/gather are still excluded.
- The grouped kernel uses contiguous expert segments, so it does not measure
  irregular gather/scatter overhead.
- Hardware counters are still unavailable.

## Next Step

H3.3 should move one level deeper into actual MoE compute:

1. compare PyTorch expert GEMM loop with a grouped GEMM or tiled Triton expert
   matmul microbenchmark;
2. add routing/scatter overhead as a separate micro-op;
3. test whether grouped compute plus grouped epilogue beats the H3.1 PyTorch
   expert-loop baseline end to end.
