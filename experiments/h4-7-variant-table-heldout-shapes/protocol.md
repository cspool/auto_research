# H4.7 Protocol: Held-Out Query Counts for the Runtime Variant Table

Date: 2026-05-27

## Hypothesis

H4.7 tests whether the H4.6 runtime variant-table features generalize beyond the
held-in H4.5 shape. If the table is capturing useful compiler/runtime structure,
then static feasibility plus order-aware variant selection should remain
competitive when `query_count` changes from the H4.5 training value `Q=4` to
held-out values `Q=2` and `Q=8`.

## Confirmatory Prediction

Using the H4.6 selectors without retuning:

1. static feasibility filtering will reject all observed shared-memory OOR
   variants for held-out shapes;
2. the order-aware selector will choose a valid variant within `15%` of measured
   best latency in at least six of eight held-out cases;
3. the order-aware selector will match or improve the global mean latency
   selector on mean regret.

The threshold is relaxed from H4.6's `10%` because H4.7 changes shape, not only
selected-token order.

## Workload

- One request.
- Shared selected-token indices across query vectors.
- `32` segments, `32` selected tokens per segment.
- `1024` selected tokens total.
- `tokens_per_segment=256`.
- `key_dim=128`, `value_dim=128`.
- Held-out query counts: `Q=2`, `Q=8`.
- Order modes:
  - `random_segment`
  - `random_sorted`
  - `random_shuffled`
  - `clustered_segment`

## Variant Sweep

Use the same grouped sparse-attention variants as H4.5:

- `128,64,64,4,4`
- `128,64,128,4,4`
- `256,64,64,4,4`
- `128,128,64,4,4`
- `128,64,64,4,8`
- `64,64,64,4,4`

The selector is therefore tested out-of-shape, not out-of-variant.

## Selectors Under Test

- **global_mean_latency**: H4.6 global table choice, `N64 D64 V64 Q4 W4`.
- **order_aware_rule_table**: H4.6 order-statistics rule table.

Both selectors use the same H4.6 static feasibility proxy:

```text
shared_proxy = BLOCK_N * BLOCK_Q * (BLOCK_D + BLOCK_V)
```

## Metrics

- static rejected OOR rows / observed OOR rows;
- selected invalid variants;
- selected latency versus measured best latency for each held-out case;
- regret percentage;
- number of held-out cases within `15%` of measured best.

## Decision Rule

- **Supported** if static filtering rejects all observed OOR rows and the
  order-aware selector is within `15%` in at least six of eight cases.
- **Partially supported** if the selector remains valid but misses the near-best
  threshold in several cases.
- **Not supported** if the selector frequently chooses invalid variants or loses
  badly to the global selector.

