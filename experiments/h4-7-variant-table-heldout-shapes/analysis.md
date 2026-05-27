# H4.7 Analysis: Held-Out Query Counts for the Runtime Variant Table

Date: 2026-05-27

## Status

Supported, with a clear next refinement.

H4.7 tests the H4.6 runtime variant table on held-out query counts `Q=2` and
`Q=8`, while keeping the H4.5 selected-token count, dimensions, order modes, and
variant sweep. Static filtering continues to reject every observed
shared-memory-infeasible variant. The H4.6 order-aware selector remains within
`15%` of measured best in `7/8` held-out cases and improves mean regret versus
the global selector.

## Results

### Selector Summary

| Selector | Near-best cases | Mean regret | Max regret | Fallbacks |
|---|---:|---:|---:|---:|
| global_mean_latency_h4_6 | 7/8 | 9.92% | 27.44% | 0 |
| order_aware_rule_table_h4_6 | 7/8 | 7.79% | 18.72% | 0 |

The confirmatory prediction passes: the order-aware selector reaches the `15%`
threshold in `7/8` held-out cases and has lower mean regret than the global
selector.

### Static Feasibility

The static proxy rejected all `8/8` observed OOR rows. The same
`N256 D64 V64 Q4 W4` variant failed for every held-out query-count/order pair,
again with `133120` required shared-memory bytes against a `101376` limit.

### Held-Out Decisions

| Q | Order | Order-aware selected | Best measured | Regret |
|---:|---|---|---|---:|
| 2 | clustered segment | N128 D64 V128 Q4 W4 | N128 D128 V64 Q4 W4 | 4.55% |
| 2 | random segment | N64 D64 V64 Q4 W4 | N128 D64 V128 Q4 W4 | 12.87% |
| 2 | random shuffled | N128 D64 V64 Q4 W8 | N128 D64 V128 Q4 W4 | 1.51% |
| 2 | random sorted | N128 D128 V64 Q4 W4 | N128 D64 V128 Q4 W4 | 4.77% |
| 8 | clustered segment | N128 D64 V128 Q4 W4 | N128 D64 V64 Q4 W8 | 18.72% |
| 8 | random segment | N64 D64 V64 Q4 W4 | N128 D64 V64 Q4 W4 | 11.18% |
| 8 | random shuffled | N128 D64 V64 Q4 W8 | N128 D64 V128 Q4 W4 | 2.10% |
| 8 | random sorted | N128 D128 V64 Q4 W4 | N128 D64 V64 Q4 W8 | 6.57% |

## Interpretation

H4.7 strengthens H4.6 because the table is no longer evaluated only on held-in
H4.5 order modes. The static resource feature generalizes cleanly across
`Q=2` and `Q=8`: the same high-`BLOCK_N` grouped tile is rejected in every
case. The order-aware selector also remains useful, improving mean regret over
the global selector.

The single miss is informative. For `Q=8 clustered_segment`, the best measured
variant is `N128 D64 V64 Q4 W8`, while the H4.6 order-aware rule chooses
`N128 D64 V128 Q4 W4`. This indicates that query-count pressure changes the
best value-blocking/warps choice even when order statistics are identical. The
selector needs query-count and number-of-query-blocks features, not only order
features.

This is a good systems result: the H4.6 table was not merely memorizing one
shape, but it is also not complete. It exposes the next feature that a compiler
runtime table should include.

## Sanity Checks

- GPU sweeps ran sequentially outside the default sandbox because CUDA devices
  are not visible inside it.
- All eight held-out sweeps completed and recorded one OOR variant each.
- Evaluation reused the H4.6 selector rules without retuning.
- Selector evaluation script passed Python bytecode compilation.

## Limitations

- Held-out shapes only changed `query_count`; selected-token count, key/value
  dimensions, and variant set stayed fixed.
- The order-aware selector is still a transparent hand-built rule table.
- No hardware counters are available to explain why `Q=8 clustered_segment`
  prefers `W8`.
- The benchmark still uses synthetic selected-token orders rather than real
  Video/VLM traces.

## Next Step

H4.8 should add a query-count-aware selector:

- include `query_count`, `num_q_blocks`, and `query_count / BLOCK_Q`;
- keep the same static feasibility proxy;
- evaluate on held-out selected-token counts or value dimensions;
- test whether the new table fixes the `Q=8 clustered_segment` miss without
  hurting the other seven cases.

