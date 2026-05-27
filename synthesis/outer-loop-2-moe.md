# Outer Loop 2: MoE Branch Synthesis

Date: 2026-05-26

## Scope

This synthesis covers the MoE sequence:

- H3.1: PyTorch expert streams;
- H3.2: grouped Triton expert epilogue;
- H3.3: padded grouped expert GEMM;
- H3.4: no-padding Triton expert matmul;
- H3.5: routed no-padding MoE fragment with scatter/gather.

## Main Claim

For single-request MoE inference on one accelerator, useful concurrency does not
come from launching each expert on a separate stream. It comes from preserving
expert semantics long enough to form compact compiler-visible work units, then
scheduling those units as no-padding tiles with explicit routing movement.

The implementation unit should be:

> routed semantic segment -> row-tile map -> grouped no-padding kernel variant
> -> measured scatter/compute/gather full-fragment latency.

## Evidence Ladder

| Step | Result | Decision |
|---|---|---|
| H3.1 expert streams | Best speedup only 0.6818x | Streams are the wrong abstraction. |
| H3.2 grouped epilogue | 14.15x-18.98x vs PyTorch loop | Grouped compiler-visible micro-ops are strong. |
| H3.3 padded grouped GEMM | 1.08x-1.39x vs expert loop | Grouped compute helps but padding hurts skew. |
| H3.4 no-padding matmul | 1.28x-1.48x vs loop, 1.12x-1.31x vs padded bmm | Irregular valid-tile scheduling is the right compute unit. |
| H3.5 routed fragment | 1.39x-1.64x vs PyTorch routed loop | Scatter/gather is meaningful but does not erase the gain. |

## Architecture View

The MoE branch points to three hardware-facing constraints:

1. **Launch overhead and framework boundaries dominate small expert fragments.**
   Per-expert streams add overhead faster than they expose useful spatial
   sharing.
2. **Padding waste is a first-order scheduler problem under skew.** Padded bmm
   hides routing irregularity from the compiler and makes the hardware execute
   invalid token rows.
3. **Routing movement is a real micro-op.** On the tested RTX 4090 setup,
   scatter/gather costs about `0.037-0.042 ms`, or `21%-23%` of the full routed
   Triton fragment.

## Compiler/Runtime Pattern

A practical compiler framework for this class should expose:

- expert-token counts and offsets after routing;
- a row-tile map over valid expert segments;
- multiple grouped kernel variants with tile/warp/register metadata;
- explicit scatter/gather or fused movement-compute kernels;
- full-fragment latency as the scheduler objective.

This generalizes the earlier H2 result. H2 showed that fastest isolated variant
is not necessarily the best co-scheduled variant. H3 adds that the variants
should be generated from model-semantic sparse segments, not only from generic
operator graphs.

## Decision

H3 is strongly supported for the MoE branch in refined form:

> model semantics are useful when they are lowered into compiler-visible,
> no-padding grouped tiles and measured as full routed fragments.

The next high-value branch is H4: irregular Video/VLM token selection and KV
retrieval. It should reuse the same protocol shape:

1. define one-request sparse semantic segments;
2. compare framework loop/stream baselines against grouped no-padding kernels;
3. measure movement micro-ops separately and as part of the full fragment;
4. report absolute full-fragment latency as the primary metric.

If continuing inside MoE first, the best H3.6 experiment is a fused routing
variant: combine scatter with the first expert matmul tile, then compare against
the H3.5 three-launch scatter/compute/gather path.
