# Outer Loop 1: From Stream Overlap to Compiler/Runtime Variant Scheduling

Date: 2026-05-26

## Scope

This synthesis covers the first five experiments:

1. H1 stream concurrency;
2. H1.1 PyTorch row-chunk resource shaping;
3. H2 Triton vector-add variant scheduling;
4. H2.1 Triton fused GELU/residual micro-kernel;
5. H2.2 Triton compiler metadata extraction.

## Main Claim

Single-request, single-accelerator operator concurrency is not primarily a
stream API problem. It is a compiler/runtime co-scheduling problem.

The useful abstraction is:

1. decompose model work into schedulable micro-operators;
2. generate multiple resource-shaped kernel variants;
3. retain compiler metadata for each variant;
4. choose variants by absolute concurrent latency under the current co-location
   context.

## Evidence

H1 showed that naive CUDA stream concurrency is weak. Across three RTX 4090
shapes, multi-stream speedup ranged from 0.997x to 1.043x and overlap stayed
near zero.

H1.1 showed that coarse framework-level shaping is not enough. Splitting GEMM
rows in PyTorch sometimes improved relative stream speedup, but it changed GEMM
library behavior, added launches, and did not reliably improve absolute
latency.

H2 showed the first positive mechanism. For a simple Triton vector add, the
fastest isolated variant was not the best concurrent variant. This means
runtime selection needs scheduling-context information.

H2.1 generalized that mechanism to a model-shaped fused epilogue:
`GELU(X * scale + Bias) + Residual`. Triton fusion improved micro-op latency by
1.91x to 2.61x over eager PyTorch, but variant choice still depended on the
co-location objective.

H2.2 showed that a compiler framework can expose the static side of the needed
metadata. Triton cache artifacts recovered tile size, warps, thread count,
shared memory, PTX register declarations, and code size for every measured
variant.

## Refined Taxonomy

The original taxonomy has four axes: granularity, mechanism, hardware
architecture, and model archetype. The experiments refine the granularity and
mechanism axes:

- **Stream-level overlap** is too weak as the main lever.
- **Framework operator chunking** is useful for exploration but too blunt for
  reliable conclusions.
- **Compiler-generated micro-kernel variants** are the most promising current
  unit.
- **Variant metadata plus empirical co-scheduling measurements** is the runtime
  interface.

## Metric Rule

The primary scheduler objective should be absolute concurrent latency.

Secondary diagnostics:

- isolated latency: useful for single-kernel optimization;
- overlap ratio: useful for reasoning about hidden work;
- stream speedup: useful but dangerous because it can be inflated by a slow
  serial baseline.

## Next Direction

Move from synthetic micro-kernels to model-semantic units.

Recommended next hypothesis:

**H3.1: Single-request MoE exposes concurrency only when expert work is
represented as compiler-visible micro-operator events, not as generic PyTorch
expert loops.**

Minimal experiment:

1. build a tiny single-GPU MoE inference fragment with 2-4 experts;
2. compare sequential expert execution, naive multi-stream expert execution,
   and Triton epilogue/routing micro-kernel variants;
3. measure absolute latency, overlap, variant choice, and compiler metadata;
4. keep batch size/request count fixed at one to preserve the project scope.

This is the right pivot because MoE is explicitly in the user's target model
set and gives semantic decomposition units that synthetic epilogues do not.
