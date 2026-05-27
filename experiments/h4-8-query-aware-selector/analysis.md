# H4.8 Analysis: Query-Count-Aware Runtime Selector

Date: 2026-05-27

## Status

Supported as a targeted retrospective selector test.

H4.8 adds query-count features to the H4.6/H4.7 runtime variant table. The new
rule table fixes the `Q=8 clustered_segment` miss from H4.7 and selects the
measured-best valid variant in all eight held-out query-count cases. The result
is intentionally not a broad generalization claim; it shows that query-count
pressure is the missing selector dimension exposed by H4.7.

## Results

| Selector | Cases | Near-best cases | Mean regret | Max regret | Q=8 clustered fixed |
|---|---:|---:|---:|---:|---|
| H4.6 order-aware | 8 | 7 | 7.79% | 18.72% | no |
| H4.8 query-aware | 8 | 8 | 0.00% | 0.00% | yes |

The H4.8 rule table includes `query_count`, `num_q_blocks`, and
`query_pressure = query_count / BLOCK_Q`. It keeps the static feasibility filter
from H4.6.

## Decisions

| Q | Order | H4.8 selected | Measured best | Regret |
|---:|---|---|---|---:|
| 2 | clustered segment | N128 D128 V64 Q4 W4 | N128 D128 V64 Q4 W4 | 0.00% |
| 2 | random segment | N128 D64 V128 Q4 W4 | N128 D64 V128 Q4 W4 | 0.00% |
| 2 | random shuffled | N128 D64 V128 Q4 W4 | N128 D64 V128 Q4 W4 | 0.00% |
| 2 | random sorted | N128 D64 V128 Q4 W4 | N128 D64 V128 Q4 W4 | 0.00% |
| 8 | clustered segment | N128 D64 V64 Q4 W8 | N128 D64 V64 Q4 W8 | 0.00% |
| 8 | random segment | N128 D64 V64 Q4 W4 | N128 D64 V64 Q4 W4 | 0.00% |
| 8 | random shuffled | N128 D64 V128 Q4 W4 | N128 D64 V128 Q4 W4 | 0.00% |
| 8 | random sorted | N128 D64 V64 Q4 W8 | N128 D64 V64 Q4 W8 | 0.00% |

## Interpretation

H4.8 clarifies the mechanism behind H4.7's miss. Selected-token order statistics
are not enough once the query dimension changes. For `Q=8`, two query blocks are
launched per value block, so the best variant shifts toward different
value-blocking and warp choices. That is why `Q=8 clustered_segment` prefers
`N128 D64 V64 Q4 W8` instead of the H4.6 clustered choice
`N128 D64 V128 Q4 W4`.

The compiler/runtime contract is now more concrete:

1. reject static resource-infeasible variants;
2. classify selected-token order;
3. include query-shape pressure (`query_count`, `num_q_blocks`, `query_count /
   BLOCK_Q`);
4. choose from measured full-fragment variant tables.

This is exactly the kind of vertical method the project is trying to extract:
semantic sparse retrieval becomes a multi-version micro-operator family, and a
runtime table chooses an implementation using both static compiler features and
request-shape metadata.

## Sanity Checks

- H4.8 uses the already validated H4.7 GPU measurements; no new GPU execution was
  required.
- The selector script passed Python bytecode compilation.
- No fallback was used in any case.
- The selected variants are all static-feasible and measured-valid.

## Limitations

- The rule table is retrospective and hand-built from the H4.7 measurements.
- It only covers `Q=2` and `Q=8` with fixed `selected_tokens=1024`,
  `key_dim=128`, `value_dim=128`, and the same six candidate variants.
- It fixes held-out query-count cases, but it has not been tested on held-out
  selected-token counts or value dimensions.

## Next Step

H4.9 should test true shape generalization beyond query count. Recommended next
axis: selected-token count.

- Run `selected_tokens=512` and `selected_tokens=2048` at `Q=4`.
- Keep the same order modes and candidate variants where feasible.
- Evaluate whether static feasibility plus order/query features still choose
  near-best valid variants.
- If the table misses, add `num_n_blocks` and selected-token/blocking pressure
  as the next selector features.

