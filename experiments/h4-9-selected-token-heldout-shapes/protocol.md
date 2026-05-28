# H4.9 Protocol: Held-Out Selected-Token Counts for Runtime Variant Selection

Date: 2026-05-27

## Hypothesis

H4.9 tests whether the H4.8 runtime variant table generalizes beyond query-count
changes to selected-token-count changes. If the table is still missing a shape
axis, changing selected-token count should expose it through different
`BLOCK_N`, value-blocking, or warp preferences.

## Confirmatory Prediction

Using the H4.8 query-count-aware selector without retuning:

1. static feasibility filtering will reject all observed shared-memory OOR
   variants for held-out selected-token counts;
2. the selector will choose a valid variant within `15%` of measured best in at
   least six of eight held-out cases;
3. if the selector misses, the miss will correlate with selected-token blocking
   pressure such as `num_n_blocks = ceil(selected_tokens / BLOCK_N)`.

## Workload

- One request.
- Shared selected-token indices across query vectors.
- `32` segments.
- Held-out selected-token counts:
  - `512` total: `16` selected tokens per segment.
  - `2048` total: `64` selected tokens per segment.
- `tokens_per_segment=256`.
- `query_count=4`.
- `key_dim=128`, `value_dim=128`.
- Order modes:
  - `random_segment`
  - `random_sorted`
  - `random_shuffled`
  - `clustered_segment`

## Variant Sweep

Use the same grouped sparse-attention variants as H4.5-H4.8:

- `128,64,64,4,4`
- `128,64,128,4,4`
- `256,64,64,4,4`
- `128,128,64,4,4`
- `128,64,64,4,8`
- `64,64,64,4,4`

## Selector Under Test

Use the H4.8 selector:

- static shared-memory feasibility proxy;
- selected-token order statistics;
- query-count features (`query_count`, `num_q_blocks`, `query_count / BLOCK_Q`).

Because `Q=4`, the selector reduces to the H4.6 order-aware table for this
experiment. That is intentional: H4.9 asks whether selected-token count creates
a new missing feature.

## Metrics

- static rejected OOR rows / observed OOR rows;
- selected invalid variants;
- selected latency versus measured best latency;
- regret percentage;
- number of cases within `15%` of measured best;
- miss pattern versus `selected_tokens` and `num_n_blocks`.

## Decision Rule

- **Supported** if static filtering rejects all OOR rows and the selector is
  within `15%` in at least six of eight held-out cases.
- **Partially supported** if it remains valid but misses several cases in a
  coherent selected-token-count pattern.
- **Not supported** if it frequently selects invalid or far-from-best variants.

