# H4.4 Protocol: Multi-Block Online-Softmax Sparse Attention

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H4.3 showed a strong win for a single-tile sparse score-softmax-value kernel,
but it required `selected_tokens <= BLOCK_N`. H4.4 tests the next boundary:

> A two-stage multi-block online-softmax Triton kernel should extend fused
> sparse attention to larger selected-token counts while retaining a speedup
> over flat PyTorch materialization. The speedup will likely be lower than H4.3
> because partial reductions require an additional kernel and partial buffers.

## Workload

Single request, single query, one accelerator:

`scores = K_selected @ q`, `p = softmax(scores)`, `out = p @ V_selected`.

Default full shape:

- 32 semantic segments;
- 32 selected tokens per segment, 1024 selected tokens total;
- `tokens_per_segment=256`;
- `D=128`;
- `Vd=128`;
- fp16 K/V/q inputs, fp32 softmax/value accumulation.

This full shape is intentionally larger than H4.3's single-tile setting.

## Index Orders

Reuse H4.2/H4.3 modes:

1. `random_segment`;
2. `random_sorted`;
3. `random_shuffled`;
4. `clustered_segment`.

## Compared Implementations

1. **Flat PyTorch eager**:
   - materialize selected K and V with `index_select`;
   - score, softmax, and value aggregation in eager PyTorch.
2. **Triton multi-block online softmax**:
   - partial kernel over selected-token blocks:
     local max, local denominator, and local weighted value partials;
   - reduce kernel combines partials with online-softmax rescaling.

The H4.3 single-tile kernel is not valid for the full 1024-token shape unless
`BLOCK_N >= 1024`, which is intentionally excluded to keep the per-program tile
reasonable.

## Tile Variants

Default sweep:

- `(64, 64, 64, 4)`;
- `(128, 64, 64, 4)`;
- `(256, 64, 64, 4)`;
- `(128, 128, 64, 4)`;
- `(256, 128, 64, 4)`;
- `(128, 64, 128, 4)`;
- `(128, 64, 64, 8)`.

Tuple format is `(BLOCK_N, BLOCK_D, BLOCK_V, num_warps)`.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_flat_total_ms` | Full PyTorch gather + score + softmax + value latency. |
| `best_triton_online_ms` | Best two-kernel online-softmax Triton latency. |
| `best_triton_vs_flat_speedup` | `torch_flat_total_ms / best_triton_online_ms`. |
| `best_variant` | Tile and warp configuration selected by latency. |
| `num_n_blocks` | Number of selected-token blocks. |
| `order_span_mean` | Mean absolute neighbor distance in selected-token order. |

## Prediction

If H4.4 succeeds:

- multi-block Triton should beat flat PyTorch on the 1024-token shape;
- best tile should vary with order and selected-token locality;
- H4 has a credible GPU compiler/runtime path beyond single-tile toy cases.

If H4.4 fails:

- partial buffers, extra launches, or score recomputation dominate;
- the H4 claim shifts toward hardware-side retrieval/concentration or a more
  production FlashAttention-like persistent kernel.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Validate every Triton variant against PyTorch eager output.
- Keep request count fixed at one.
