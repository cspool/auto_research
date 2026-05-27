# H4.2 Protocol: Locality-Aware Sparse KV Retrieval Ordering

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H4.1 showed that a fused Triton gather-score kernel beats flat PyTorch
gather+GEMM, but it used only random selected indices in segment order. H4.2
tests whether the runtime/compiler should treat selected-token ordering as a
first-class scheduling decision:

> Locality-aware ordering of selected KV tokens should reduce gather-score
> latency for movement-heavy sparse retrieval. Triton tile shape should also
> interact with the selected-token count and ordering, so a multi-version table
> should outperform a single fixed tile.

## Workload

Same single-request retrieval-score proxy as H4.1:

`S[selected_tokens, Qn] = K[selected_indices, D] @ Q[D, Qn]`

Default shape:

- 32 semantic segments;
- `tokens_per_segment=256`;
- selected tokens: 32 per segment, 1024 total;
- `D=256`;
- `Qn=8`;
- fp16 inputs and outputs.

## Index Orders

All modes keep request count fixed at one:

1. **random_segment**: random selected tokens within each segment, concatenated
   by semantic segment.
2. **random_sorted**: same random selected set, globally sorted by cache index.
3. **random_shuffled**: same random selected set, globally shuffled, breaking
   segment locality.
4. **clustered_segment**: contiguous selected spans within each segment,
   concatenated by segment.

These represent a spectrum from locality-poor to locality-friendly retrieval.

## Compared Implementations

1. **Flat PyTorch gather+GEMM**:
   - `index_select(K, selected_indices)` into a materialized buffer;
   - `torch.mm(gathered_K, Q)`.
2. **Triton fused gather-score variants**:
   - one kernel loads K rows through selected indices and multiplies by `Q`;
   - no materialized selected-K buffer;
   - tile sweep over `(BLOCK_M, BLOCK_D, BLOCK_Q, num_warps)`.

## Tile Variants

Default sweep:

- `(8, 64, 8, 4)`;
- `(16, 64, 8, 4)`;
- `(32, 64, 8, 4)`;
- `(16, 128, 8, 4)`;
- `(32, 128, 8, 4)`;
- `(16, 64, 8, 8)`.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_flat_total_ms` | PyTorch materialized gather plus GEMM latency. |
| `torch_gather_ms` | PyTorch selected-K materialization latency. |
| `torch_score_ms` | GEMM latency after materialization. |
| `best_triton_fused_ms` | Best Triton fused gather-score variant latency. |
| `best_triton_vs_flat_speedup` | `torch_flat_total_ms / best_triton_fused_ms`. |
| `best_variant` | Tile and warp configuration selected by latency. |
| `order_span_mean` | Mean absolute neighbor index distance in selected order. |

## Prediction

If locality/order metadata matters:

- random_shuffled should be slower than random_segment and random_sorted;
- clustered_segment should be fastest or near-fastest;
- best Triton tile may differ across orders.

If ordering barely matters:

- launch and arithmetic dominate this proxy;
- H4 should move next to softmax/value mixing and larger memory hierarchy
  effects before making architecture claims.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Validate every Triton variant against flat PyTorch for the same selected order.
- Keep request count fixed at one.
