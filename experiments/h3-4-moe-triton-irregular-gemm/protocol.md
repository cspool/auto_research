# H3.4 Protocol: Irregular No-Padding Triton Expert Matmul

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H3: Model-semantic decomposition exposes stronger concurrency.

H3.3 showed that padded grouped GEMM is positive but skew-sensitive. H3.4 tests
the next implementation step:

> A no-padding expert-tiled Triton matmul should recover skewed-routing waste by
> scheduling only valid expert token tiles. However, a naive Triton matmul may
> still lose to cuBLAS-backed PyTorch/bmm unless the custom kernel is
> sufficiently optimized.

## Workload

Four-expert MoE FFN fragment:

`Y_e = ReLU(X_e @ W1_e) @ W2_e`

Compare three implementations:

1. **Expert loop**: one `torch.mm` pair per expert.
2. **Padded grouped bmm**: stack experts to `[experts, max_tokens, hidden]` and
   use two `torch.bmm` calls.
3. **No-padding Triton tiled matmul**: concatenate expert tokens, build a row
   tile map for valid expert segments only, and run two custom Triton grouped
   matmul kernels plus one ReLU.

## Token Distributions

- balanced: `64,64,64,64`;
- skewed: `160,64,24,8`;
- tiny: `16,16,16,16`.

## Triton Variants

Default sweep:

- `BLOCK_M`: 16, 32;
- `BLOCK_N`: 32, 64;
- `BLOCK_K`: 64;
- `num_warps`: 4.

The goal is not to beat cuBLAS with a production GEMM, but to test whether the
no-padding scheduling unit is promising and how much optimization gap remains.

## Metrics

| Metric | Meaning |
|---|---|
| `expert_loop_ms` | Serial per-expert PyTorch FFN latency. |
| `padded_bmm_ms` | Padded strided-batched FFN latency. |
| `triton_no_pad_ms` | Two no-padding Triton matmuls plus ReLU. |
| `triton_vs_loop_speedup` | `expert_loop_ms / triton_no_pad_ms`. |
| `triton_vs_bmm_speedup` | `padded_bmm_ms / triton_no_pad_ms`. |
| `row_tiles` | Number of valid expert row tiles scheduled by Triton. |
| `padding_overhead` | Padded bmm work expansion. |

## Prediction

If no-padding custom grouped matmul is already useful:

- Triton no-padding should beat padded bmm on the skewed distribution;
- balanced/tiny may be closer because padding overhead is absent there.

If it is not yet useful:

- Triton no-padding may lose to cuBLAS-backed baselines, showing that the next
  step is kernel optimization rather than merely changing the schedule.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Check Triton output against the expert loop on valid token ranges.
- Keep request count fixed at one.
