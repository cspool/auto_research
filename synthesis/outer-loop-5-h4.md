# Outer Loop 5: H4 Runtime Selector Synthesis

Date: 2026-05-27

## Scope

This synthesis reviews H4.1 through H4.9:

- H4.1-H4.3 established sparse Video/VLM retrieval as a fused sparse-attention
  fragment rather than a framework segment loop.
- H4.4-H4.5 scaled that fragment to multi-block online softmax and grouped
  query execution.
- H4.6-H4.9 tested whether a compiler/runtime variant table can reject invalid
  kernels and choose near-best valid kernels across order and held-out shape
  axes.

## Main Claim

The H4 branch now supports a specific compiler/runtime method:

> Lower irregular Video/VLM selected-token retrieval into a grouped sparse
> attention fragment, then choose among resource-diverse Triton variants using
> static feasibility filtering plus empirical tables keyed by order statistics
> and shape pressure.

This is stronger than a kernel benchmark claim. It is a vertical method:

1. expose selected-token metadata instead of launching per segment;
2. fuse selected K/V movement, score, online softmax, and value reduction;
3. group queries that share selected K/V;
4. generate tile/warp variants;
5. filter shared-memory-infeasible variants before launch;
6. select by measured full-fragment latency, not isolated kernel latency.

## Evidence Ladder

| Step | Question | Result | Meaning |
|---|---|---:|---|
| H4.1 | Are segment loops viable? | Per-segment PyTorch was 52x-64x slower than fused Triton. | Segment/frame semantics should be metadata, not launch units. |
| H4.2 | Does order/locality matter? | Best tile changed with order, while fused speedup stayed 1.37x-1.42x. | Order stats need to feed variant selection. |
| H4.3 | Does full sparse attention fusion help? | Single-tile score-softmax-value beat flat PyTorch by 5.78x-6.79x. | Movement, softmax, and value aggregation should be one compiler-visible unit. |
| H4.4 | Does the path scale beyond one tile? | Multi-block online softmax beat flat PyTorch by 2.99x-4.76x. | Partial-state reduction overhead is manageable. |
| H4.5 | Does query grouping matter? | Grouped Q=4 beat flat PyTorch by 2.86x-3.64x and repeated one-query kernels by 3.41x-3.66x. | Query/head grouping belongs in the lowering unit. |
| H4.6 | Can a runtime table choose variants? | Static filter rejected all OOR rows; order-aware selector hit 4/4 near-best. | The compiler/runtime contract is implementable. |
| H4.7 | Does it generalize to held-out Q? | H4.6 selector hit 7/8; Q=8 clustered exposed query pressure. | Query count is a required selector feature. |
| H4.8 | Can query features repair the miss? | Query-aware selector hit 8/8 exactly on H4.7 cases. | `query_count`, `num_q_blocks`, and query pressure are useful table keys. |
| H4.9 | Does it generalize to held-out selected-token counts? | H4.8 selector hit 8/8 within 15%, mean regret 4.66%, max 9.92%. | Selected-token count refines exact tuning but is not yet a near-best failure mode. |

## Pattern

Two independent patterns now line up:

- **MoE branch**: semantic routing only helped after lowering into compact
  no-padding expert tiles plus explicit movement micro-ops.
- **Sparse Video/VLM branch**: selected-token sparsity only helped after
  lowering into grouped sparse-attention tiles plus runtime variant selection.

The common principle is that single-request concurrency should be expressed as
compiler-visible micro-operator fragments, not as framework-level stream
parallelism.

## Selector Lessons

The current H4 selector stack has four layers:

1. static shared-memory feasibility;
2. selected-token order class/statistics;
3. query-count and query-block pressure;
4. measured full-fragment latency by variant.

H4.9 shows selected-token count is useful but not mandatory for the current
near-best threshold. The remaining untested axis is value dimension. That axis
is more structurally important than selected-token count because it changes:

- number of value blocks;
- shared-memory pressure through `BLOCK_V`;
- partial value-buffer traffic;
- the choice between `BLOCK_V=64` and `BLOCK_V=128` variants;
- the hardware-side interpretation of online reduction and K/V movement.

## Hardware/Compiler Interpretation

The GPU compiler/runtime path is now credible enough to be the primary
implementation path. But the same results also identify the accelerator
co-design surface:

- selected-token movement wants gather/concentration support close to memory;
- online softmax wants partial max/denominator/value state support;
- grouped query reuse wants K/V movement amortized across queries;
- static feasibility failures expose fixed shared-memory ceilings;
- value-blocking may expose whether the bottleneck is compute, shared memory, or
  global partial-buffer traffic.

Because Nsight counters are unavailable in this environment, the next best
compiler-facing evidence is a value-dimension hold-out plus static metadata and
latency tables. Hardware-counter work should be added later when the profiling
environment is available.

## Direction Decision

Deepen H4 one more step before broadening to NPU/accelerator primitives.

The next experiment should be H4.10: value-dimension held-out shapes. It should
reuse the H4.5-H4.9 grouped sparse-attention benchmark and selector evaluation,
but vary value dimension:

- `V=64` and `V=256`;
- `selected_tokens=1024`;
- `query_count=4`;
- four selected-token orders;
- same six candidate variants, including `BLOCK_V=64` and `BLOCK_V=128`.

The selector under test should be the H4.8/H4.9 order/query selector. A
diagnostic value-pressure oracle should add `value_dim`, `num_v_blocks`, and
`value_dim / BLOCK_V`. If the base selector misses, that reveals the next
required table feature. If it still stays near-best, the H4 selector story is
strong enough to move to a learned/table selector over all H4 data or to a
hardware/NPU mapping section.

## Candidate H4.10 Protocol

Hypothesis: value dimension is the next likely failure axis for grouped sparse
attention runtime selection because it changes value-block count, partial-buffer
traffic, and shared-memory pressure.

Prediction:

1. static feasibility filtering will reject all observed shared-memory OOR
   variants;
2. the H4.8/H4.9 selector will choose valid variants, but may miss the 15%
   near-best threshold on `V=256` if value-block pressure changes the best
   `BLOCK_V`/warp choice;
3. a value-pressure oracle should reduce regret if such misses occur.

## Updated Research Story

The project's current contribution is becoming:

> For single-request modern-model inference, useful concurrency comes from
> semantic micro-operator lowering plus hardware-aware runtime variant
> selection. GPU compiler frameworks can realize much of this today with
> Triton-style grouped kernels and metadata tables; accelerator/NPU support is
> most justified where selected-token movement and online reduction remain
> persistent bottlenecks after that lowering.
