# H4.3 Protocol: Sparse KV Score-Softmax-Value Fragment

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H4.1 and H4.2 used score-only retrieval: `K_selected @ Q`. H4.3 adds the next
attention stages:

`scores = K_selected @ q`, `p = softmax(scores)`, `out = p @ V_selected`.

Prediction:

> Adding value gathering and softmax/reduction should make materialization and
> movement costs more visible. A fused Triton score-softmax-value kernel should
> beat flat PyTorch eager execution if selected-token count fits in one
> reduction tile, but the win may depend on value block size because naive
> fused variants can recompute scores per value block.

## Workload

Single request, single query, one accelerator:

1. key cache `K[segments * tokens_per_segment, D]`;
2. value cache `V[segments * tokens_per_segment, Vd]`;
3. selected token indices from video/VLM retrieval;
4. query vector `q[D]`;
5. output vector `out[Vd]`.

Default full shape:

- 16 semantic segments;
- 16 selected tokens per segment, 256 selected tokens total;
- `tokens_per_segment=256`;
- `D=128`;
- `Vd=128`;
- fp16 K/V/q inputs, fp32 softmax/value accumulation.

## Index Orders

Reuse the H4.2 modes:

1. `random_segment`;
2. `random_sorted`;
3. `random_shuffled`;
4. `clustered_segment`.

## Compared Implementations

1. **Flat PyTorch eager**:
   - materialize `K_selected` and `V_selected` with `index_select`;
   - compute scores and softmax;
   - compute value aggregation.
2. **Triton fused SSV**:
   - one fused score-softmax-value kernel;
   - loads K and V through selected indices;
   - avoids materialized selected-K and selected-V buffers;
   - sweeps `(BLOCK_N, BLOCK_D, BLOCK_V, num_warps)`.

## Tile Variants

Default sweep:

- `(256, 32, 32, 4)`;
- `(256, 64, 32, 4)`;
- `(256, 64, 64, 4)`;
- `(256, 64, 128, 4)`;
- `(256, 128, 64, 4)`;
- `(256, 64, 64, 8)`.

`BLOCK_N` must be at least the selected-token count. This first version is a
single-tile reduction, not a production multi-block attention kernel.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_flat_total_ms` | Full PyTorch gather + score + softmax + value latency. |
| `torch_gather_k_ms` | Key gather latency. |
| `torch_gather_v_ms` | Value gather latency. |
| `torch_score_softmax_ms` | Score and softmax latency after K gather. |
| `torch_value_ms` | Value aggregation latency after V gather and softmax. |
| `best_triton_ssv_ms` | Best fused Triton score-softmax-value latency. |
| `best_triton_vs_flat_speedup` | `torch_flat_total_ms / best_triton_ssv_ms`. |
| `best_variant` | Tile and warp configuration selected by latency. |

## Prediction

If fused SSV wins:

- H4 evidence strengthens: sparse video/VLM retrieval benefits from movement
  and compute fusion below framework operators.
- Next step should implement a multi-block online-softmax variant for larger
  selected-token counts.

If fused SSV loses:

- the single-tile fused kernel is too naive or recomputes scores too often;
- next step should separate score/softmax and value aggregation or use a
  FlashAttention-like online algorithm.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Validate every Triton variant against PyTorch eager output.
- Keep request count fixed at one.
