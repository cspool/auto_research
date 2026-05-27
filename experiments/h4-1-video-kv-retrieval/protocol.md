# H4.1 Protocol: Sparse Video/VLM KV Retrieval Score

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H4: Irregular video/VLM workloads need hardware-aware or hardware-assisted
scheduling.

The MoE branch showed that semantic sparse segments become useful only after
they are lowered into compact compiler-visible work units. H4.1 tests the same
idea on a video/VLM-style KV retrieval fragment:

> A per-frame/per-cluster PyTorch loop will be launch-bound and slow. Flattening
> selected KV tokens into one PyTorch gather+GEMM will help, but it still
> materializes selected K rows. A fused Triton gather-score kernel should reduce
> launch and materialization cost for single-request sparse retrieval.

## Workload

Single request, one accelerator:

1. video/VLM key cache `K[segments * tokens_per_segment, D]`;
2. selected token indices grouped by semantic segment/frame;
3. query matrix `Q[D, Qn]` for a small number of decoder/query vectors;
4. output score matrix `S[selected_tokens, Qn] = K_selected @ Q`.

This approximates the front half of sparse video KV retrieval: dynamic token
selection plus light attention scoring. It deliberately omits softmax and value
mixing so the first H4 experiment isolates irregular gather + score.

## Compared Implementations

1. **PyTorch segment loop**:
   - for each segment, `index_select(K, indices_e)`;
   - `torch.mm(K_e, Q)`;
   - copy result into the contiguous score buffer.
2. **PyTorch flat gather+GEMM**:
   - concatenate all selected indices;
   - `index_select(K, selected_indices)` into a materialized buffer;
   - one `torch.mm(gathered_K, Q)`.
3. **Triton fused gather-score**:
   - one kernel loads K rows through selected indices and multiplies by `Q`;
   - no materialized gathered-K buffer.

## Token Distributions

- balanced random: 32 segments, 32 selected tokens each;
- skewed random: 32 segments with a few hot segments and many small segments;
- tiny random: 32 segments, 4 selected tokens each.

Default cache shape:

- `tokens_per_segment=256`;
- `D=256`;
- `Qn=8`;
- fp16 inputs and outputs.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_segment_loop_ms` | Framework per-segment loop latency. |
| `torch_flat_total_ms` | Flat PyTorch gather + GEMM latency. |
| `torch_gather_ms` | Materialized selected-K gather latency. |
| `torch_score_ms` | GEMM over already materialized selected-K. |
| `triton_fused_ms` | Single fused gather-score kernel latency. |
| `triton_vs_segment_speedup` | `torch_segment_loop_ms / triton_fused_ms`. |
| `triton_vs_flat_speedup` | `torch_flat_total_ms / triton_fused_ms`. |
| `gather_fraction_of_flat` | `torch_gather_ms / torch_flat_total_ms`. |

## Prediction

If H4.1 matches the H3 pattern:

- segment loop will be much slower than flat/fused paths;
- flat PyTorch will improve launch pressure but still pay materialization;
- Triton fused gather-score should beat flat PyTorch when query count is small
  and retrieval is movement-heavy.

If Triton fused loses:

- cuBLAS GEMM over materialized selected K is strong enough to offset the gather
  materialization cost;
- the next H4 step should keep flat gather+GEMM as baseline and explore
  locality-aware index ordering, persistent kernels, or hardware-assisted gather.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Validate segment loop, flat PyTorch, and Triton fused outputs.
- Keep request count fixed at one.
