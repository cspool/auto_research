# H4.8 Protocol: Query-Count-Aware Runtime Selector

Date: 2026-05-27

## Hypothesis

H4.8 tests the specific failure mode discovered in H4.7. The H4.6 selector used
selected-token order statistics but did not explicitly model query-count
pressure. Adding query-count features should fix the `Q=8 clustered_segment`
miss and reduce held-out regret without breaking the other H4.7 cases.

## Confirmatory Prediction

On the H4.7 held-out query-count data:

1. A query-count-aware selector will remain static-feasible for every selected
   case.
2. It will choose within `15%` of measured best in all `8/8` held-out cases.
3. It will reduce mean regret versus the H4.6 order-aware selector.
4. It will fix the `Q=8 clustered_segment` miss by selecting
   `N128 D64 V64 Q4 W8`.

## Inputs

- `data/h4_7_heldout_variant_rows.csv`
- `data/h4_7_heldout_selector_decisions.csv`

These inputs contain the measured H4.7 `Q=2` and `Q=8` grouped sparse-attention
variant latencies and the H4.6 selector decisions.

## Selector Features

The H4.8 selector uses:

- selected-token order statistics:
  - `order_span_mean`
  - `order_span_p95`
  - `monotonic_fraction`
- query-shape features:
  - `query_count`
  - `BLOCK_Q`
  - `num_q_blocks = ceil(query_count / BLOCK_Q)`
  - `query_pressure = query_count / BLOCK_Q`
- static feasibility:
  - `shared_proxy = BLOCK_N * BLOCK_Q * (BLOCK_D + BLOCK_V)`
  - reject variants above the known H4.5/H4.7 shared-memory limit.

## Rule Table

The rule table is intentionally transparent:

- For `query_count <= 2`:
  - clustered monotonic low-p95 order -> `N128 D128 V64 Q4 W4`
  - otherwise -> `N128 D64 V128 Q4 W4`
- For `query_count >= 8`:
  - clustered monotonic low-p95 order -> `N128 D64 V64 Q4 W8`
  - globally sorted monotonic order -> `N128 D64 V64 Q4 W8`
  - high-span shuffled order -> `N128 D64 V128 Q4 W4`
  - random segment order -> `N128 D64 V64 Q4 W4`
- For the original `query_count = 4` case:
  - reuse the H4.6 order-aware rule table.

This is a retrospective rule table built to test whether query-shape features
are the missing dimension, not a claim of learned generalization.

## Metrics

- selected invalid variants;
- regret percentage versus measured best;
- cases within `15%` of measured best;
- mean and max regret;
- whether `Q=8 clustered_segment` is fixed.

## Decision Rule

- **Supported** if H4.8 selects valid variants within `15%` in all eight H4.7
  held-out cases and reduces mean regret versus H4.6 order-aware selection.
- **Partially supported** if it fixes the Q=8 clustered miss but introduces new
  misses elsewhere.
- **Not supported** if query-aware rules do not improve mean regret or choose
  invalid variants.

