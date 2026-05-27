# H3.1 Analysis: Tiny MoE Semantic Concurrency

Date: 2026-05-26

## Status

Negative for naive PyTorch-level expert stream concurrency; useful refinement
for H3.

MoE experts are good semantic decomposition units, but this experiment shows
that simply launching each expert on its own CUDA stream is not enough to turn
that semantic parallelism into single-request latency improvement.

## Results

| Run | Token counts | Sum expert ms | Serial ms | Concurrent ms | Stream speedup | Overlap |
|---|---|---:|---:|---:|---:|---:|
| balanced 4x64 | 64,64,64,64 | 0.1749 | 0.2052 | 0.3110 | 0.6596x | -0.7784 |
| skewed | 160,64,24,8 | 0.2008 | 0.2192 | 0.3226 | 0.6795x | -0.6065 |
| tiny 4x16 | 16,16,16,16 | 0.1755 | 0.2065 | 0.3028 | 0.6818x | -0.7255 |

All three distributions were slower with one stream per expert.

## Interpretation

This result weakens the naive form of H3:

> Model-semantic decomposition alone does not guarantee concurrency.

The refined version is stronger and more actionable:

> Model semantics identify the right units, but those units must be lowered into
> compiler-visible micro-operators or spatially controlled kernels before a
> runtime can exploit them.

The isolated expert latencies are small and clustered around a few dozen
microseconds. That suggests a floor from small GEMM launch/library behavior. In
that regime, four PyTorch streams add scheduling overhead and contention rather
than hiding work. The skewed distribution also behaves as expected: the largest
expert dominates more of the work, but even balanced experts do not overlap
profitably at this abstraction level.

## Connection to Earlier Experiments

H3.1 is consistent with H1 and H1.1:

- H1: streams alone do not create useful overlap;
- H1.1: framework-level decomposition is too blunt;
- H3.1: even model-semantic branches need lower-level resource control.

It also motivates the H2 path: MoE should be tested next with fused/routed
micro-kernel variants and compiler metadata, not just expert-level PyTorch
streams.

## Limitations

- Routing/top-k and scatter/gather costs were excluded to isolate expert compute.
- ReLU FFN was used instead of GELU to avoid allocation noise.
- No grouped GEMM or Triton expert kernel was tested yet.
- The experiment is a latency microbenchmark, not a full MoE layer.

## Next Step

H3.2 should implement a compiler-visible MoE fragment:

1. group expert tokens into small tiles;
2. use Triton for the expert epilogue or a small grouped-matmul-like tile;
3. retain variant metadata as in H2.2;
4. compare against this H3.1 PyTorch expert-stream baseline.
