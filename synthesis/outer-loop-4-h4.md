# Outer Loop 4: H4 Sparse Attention Synthesis

Date: 2026-05-26

## Scope

This synthesis reviews H4.1 through H4.5:

- H4.1: sparse Video/VLM KV retrieval score.
- H4.2: selected-token order and locality sweep.
- H4.3: single-tile score-softmax-value sparse attention.
- H4.4: multi-block online-softmax sparse attention for 1024 selected tokens.
- H4.5: multi-query sparse attention with shared selected K/V.

## Main Claim

For single-request Video/VLM-style sparse retrieval, the useful execution unit
is not a segment loop, not a selected-token gather, and not even a good
single-query sparse attention kernel. The useful unit is a compiler-visible
grouped sparse attention fragment:

1. selected-token index set and order statistics;
2. K/V layout and segment boundaries;
3. score, online softmax, and value aggregation;
4. query/head grouping over shared selected K/V;
5. static resource feasibility and multi-version tile variants;
6. empirical full-fragment latency as the scheduling objective.

This shifts H4 from a broad "irregular video/VLM workloads may need hardware
help" statement to a more precise compiler/runtime contract.

## Evidence Ladder

| Step | Question | Result | What It Adds |
|---|---|---:|---|
| H4.1 | Can per-segment retrieval loops work? | Per-segment PyTorch was 52x-64x slower than fused Triton. | Segment/framework loops are the wrong unit. |
| H4.1 | Is flat gather+GEMM enough? | Fused Triton beat flat PyTorch by 1.15x-1.35x. | Selected-K materialization still matters. |
| H4.2 | Does order/locality determine the kernel? | Fused Triton beat flat PyTorch by 1.37x-1.42x, but best tile changed by order. | Order statistics need variant metadata. |
| H4.3 | Does fusion still help after softmax/value? | Single-tile Triton beat flat PyTorch by 5.78x-6.79x. | Full sparse-attention fusion is much stronger than score-only fusion. |
| H4.4 | Does it scale beyond one tile? | Two-stage online-softmax Triton beat flat PyTorch by 2.99x-4.76x. | The path survives 1024 selected tokens. |
| H4.5 | Does it scale across queries? | Grouped-query Triton beat flat PyTorch by 2.86x-3.64x and repeated one-query kernels by 3.41x-3.66x. | Query/head grouping must be in the lowering unit. |

## Pattern

The H4 branch now mirrors the MoE branch from H3. Semantic structure is useful
only when it becomes a compiler/runtime scheduling unit below the framework
loop:

- MoE: route segments -> valid row tiles -> no-padding grouped matmul ->
  explicit movement micro-ops.
- Sparse Video/VLM: selected-token segments -> order-aware sparse attention
  tiles -> online-softmax reduction -> grouped query/head execution.

Both branches point away from naive stream concurrency and toward semantic
micro-operator lowering with runtime variant selection.

## Hardware/Compiler Interpretation

The current GPU evidence is strong enough to support a compiler/runtime path:
Triton-style custom kernels can recover large wins if they see the sparse
attention fragment as one unit. However, H4.5 also strengthens the hardware-side
question rather than dismissing it. The full sparse fragment is shaped by:

- irregular selected-token memory movement;
- repeated K/V reads across value blocks and query groups;
- online-softmax partial state;
- shared-memory feasibility limits;
- launch count and partial-buffer traffic.

Those are exactly the places where accelerator or NPU designs might add
memory-interface gather units, KV-side concentration, or online-reduction
support. The research story should therefore present GPU compiler/runtime as the
first viable implementation path and hardware support as the next co-design
question, not as a competing claim.

## Direction Decision

Deepen H4 one more step before broadening.

The next experiment should be H4.6: build a compiler/runtime variant table for
the H4.5 sparse attention kernels. It should join:

- measured latency from `data/h4_5_kv_multi_query_variants.csv`;
- static feasibility outcomes such as shared-memory OOR;
- tile parameters (`BLOCK_N`, `BLOCK_D`, `BLOCK_V`, `BLOCK_Q`, warps);
- selected-token order statistics;
- query count and value-block count.

The goal is not just another speedup number. It is to demonstrate how a compiler
framework would choose or reject sparse attention micro-kernels without
exhaustive online benchmarking.

## Candidate H4.6 Protocol

Hypothesis: a small static-plus-measured runtime table can filter infeasible
grouped sparse attention variants and choose a near-best valid kernel across
selected-token orders.

Prediction: using static feasibility plus simple shape/order features should
select a valid variant within `10%` of the measured best in at least three of
four H4.5 order modes, while always rejecting the shared-memory-infeasible
`N256 D64 V64 Q4 W4` variant.

## Updated Research Story

The project is converging on a vertical method:

1. identify semantic concurrency in the model workload;
2. avoid framework-level streams/loops as the execution substrate;
3. lower semantics into grouped micro-operator tiles;
4. measure full-fragment latency, not only isolated kernel latency;
5. feed static compiler metadata and empirical measurements into a runtime
   variant selector;
6. use hardware counters or accelerator primitives only after the compiler path
   exposes a persistent movement/reduction bottleneck.

