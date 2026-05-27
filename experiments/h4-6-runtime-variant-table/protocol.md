# H4.6 Protocol: Runtime Variant Table for Grouped Sparse Attention

Date: 2026-05-27

## Hypothesis

H4.6 tests the compiler/runtime layer implied by H4.5. A small variant table
that combines static feasibility, tile metadata, selected-token order
statistics, and measured latency can reject infeasible grouped sparse-attention
kernels and select a near-best valid variant without benchmarking every variant
online for every request.

## Confirmatory Prediction

Using the H4.5 grouped sparse-attention measurements:

1. A static resource proxy will reject every observed shared-memory-infeasible
   variant.
2. A static-safe empirical selector will choose a valid variant within `10%` of
   measured best latency in at least three of four selected-token order modes.
3. Adding order statistics to the selector will improve over a global mean
   latency selector, especially for the sorted/order-sensitive case.

## Inputs

- `data/h4_5_kv_multi_query_summary.csv`
- `data/h4_5_kv_multi_query_variants.csv`

The input contains four selected-token order modes, six candidate variants per
mode, measured grouped-query latency for valid variants, and recorded launch
failures for shared-memory-infeasible variants.

## Static Feasibility Proxy

The prototype resource proxy is:

```text
shared_proxy = BLOCK_N * BLOCK_Q * (BLOCK_D + BLOCK_V)
```

The hardware limit is parsed from observed Triton `OutOfResources` errors when
available. For the H4.5 RTX 4090 runs, the reported limit is `101376`. Variants
with `shared_proxy > limit` are rejected before latency selection.

This is intentionally a simple compiler-table proxy, not a full Triton shared
memory estimator. The goal is to test whether static metadata is useful enough
to prevent known-bad launches.

## Selectors

1. **global_mean_latency**:
   - filter static-infeasible variants;
   - compute mean measured latency per variant across all H4.5 orders;
   - choose the lowest mean latency variant for every order.

2. **order_aware_rule_table**:
   - filter static-infeasible variants;
   - classify request order by `order_span_mean`, `order_span_p95`, and
     `monotonic_fraction`;
   - use a small table of order classes to choose variants:
     - globally shuffled: high span, non-monotonic -> `N128 D64 V64 Q4 W8`;
     - random segment: moderate span, non-monotonic -> `N64 D64 V64 Q4 W4`;
     - clustered segment: monotonic, low p95 -> `N128 D64 V128 Q4 W4`;
     - globally sorted: monotonic, larger p95 -> `N128 D128 V64 Q4 W4`.

The rule table is intentionally transparent. It demonstrates what a compiler
framework could encode after profiling a small workload family.

## Metrics

- rejected observed OOR variants / total observed OOR variants;
- selected invalid variants;
- per-order selected latency versus measured best latency;
- regret percentage: `selected_latency / best_latency - 1`;
- number of order modes within `10%` of measured best.

## Decision Rule

- **Supported** if static filtering rejects all observed OOR variants and at
  least one static-safe selector chooses within `10%` of best in at least three
  of four order modes.
- **Strongly supported** if the order-aware selector improves the global selector
  and reaches within `10%` in all four modes.
- **Not supported** if filtering misses observed OOR variants or selectors choose
  invalid/poor variants in most modes.

