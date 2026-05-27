# Outer Loop 3: Sparse Video/VLM KV Retrieval Synthesis

Date: 2026-05-26

## Scope

This synthesis covers the H4 sequence:

- H4.1: sparse Video/VLM KV retrieval score proxy;
- H4.2: locality/order and Triton tile variant sweep;
- H4.3: sparse score-softmax-value fragment.

## Main Claim

For single-request Video/VLM sparse retrieval, semantic selection must not be
executed as per-frame or per-cluster framework work. The useful unit is a
compiler-visible sparse attention fragment that fuses selected-token movement,
score computation, softmax/reduction, and value aggregation.

The implementation unit should be:

> selected token set -> order/locality metadata -> fused sparse attention kernel
> variants -> full-fragment latency selection.

## Evidence Ladder

| Step | Result | Decision |
|---|---|---|
| H4.1 segment loop | Fused Triton was 52x-64x faster than per-segment PyTorch loops | Semantic video segments are metadata, not launch units. |
| H4.1 flat baseline | Fused gather-score was 1.15x-1.35x faster than flat PyTorch gather+GEMM | Selected-K materialization is avoidable overhead. |
| H4.2 order sweep | Fused gather-score stayed 1.37x-1.42x faster across random/sorted/shuffled/clustered orders | Order matters, but only together with kernel variant metadata. |
| H4.3 sparse SSV | Fused score-softmax-value was 5.78x-6.79x faster than flat PyTorch eager | The benefit grows when movement, reduction, and value aggregation are fused. |

## Architecture View

H4 points to three architecture/compiler constraints:

1. **Branch granularity is too fine for framework execution.** A 32-segment
   retrieval loop creates launch pressure and loses by more than 50x.
2. **Movement and reduction must be fused.** Score-only fusion helps modestly,
   but score-softmax-value fusion is much stronger because it removes selected
   K/V materialization and keeps reduction local.
3. **Locality is contextual.** Sorting selected tokens drastically changes
   neighbor-span statistics, but latency does not follow a simple monotonic
   rule. The runtime needs empirical variant tables keyed by order statistics,
   tile shape, warps, and full-fragment latency.

This explains why V-Rex and Focus-style systems are relevant: they turn dynamic
visual-token selection and KV retrieval into local pipeline or memory-interface
work. The GPU proxy shows that some of that idea can be captured by fused Triton
kernels, but the next boundary is larger selected-token sets that require
multi-block online softmax.

## Compiler/Runtime Contract

A practical runtime for sparse Video/VLM retrieval should expose:

- selected-token count and selected-token index tensor;
- segment/frame boundaries as metadata, not as launch boundaries;
- order/locality statistics such as neighbor span and monotonicity;
- K/V cache layout and value dimension;
- multi-version fused sparse attention kernels;
- validation and empirical latency records for each variant/order class.

The primary metric remains full-fragment latency. Component timings and order
statistics are diagnostics.

## Decision

H4 is strongly supported for the sparse attention branch in refined form:

> irregular Video/VLM retrieval benefits from compiler-visible movement-compute
> fusion; locality/order metadata should guide variant selection, but the
> decisive unit is the fused score-softmax-value fragment.

The next step should **deepen H4**, not broaden yet. H4.3's limitation is clear:
the fused kernel is single-tile and requires `selected_tokens <= BLOCK_N`.
H4.4 should implement a multi-block online-softmax sparse attention kernel for
larger selected-token counts, then compare:

1. flat PyTorch materialization;
2. current single-tile Triton when valid;
3. multi-block Triton online softmax;
4. order-aware variant selection.

If H4.4 succeeds, the project will have a stronger GPU-side compiler argument.
If it fails, the hardware-side KVPU/KVMU or concentration-unit argument becomes
more compelling.
