# H3.3 Protocol: MoE Grouped Expert GEMM

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H3: Model-semantic decomposition exposes stronger concurrency.

H3.2 showed that grouped expert epilogues are much faster than per-expert
launches. H3.3 tests whether the same grouping idea helps actual expert compute:

> For single-request MoE expert FFN compute, grouped/batched GEMM should beat a
> Python/PyTorch expert loop when the token distribution is balanced. Skewed
> routing may reduce or reverse the benefit if padding wastes too much work.

## Workload

Four-expert MoE FFN fragment:

`Y_e = ReLU(X_e @ W1_e) @ W2_e`

Compare two implementations:

1. **Expert loop**: one `torch.mm` pair per expert, as in H3.1.
2. **Grouped padded GEMM**: pad expert token tensors to `max_tokens`, stack into
   shape `[experts, max_tokens, hidden]`, and execute two `torch.bmm` calls.

The grouped path represents a practical framework-level grouped GEMM baseline.
It is not a custom Triton grouped matmul, but it tests whether combining expert
compute into a single batched GEMM unit is promising before writing lower-level
kernels.

## Token Distributions

- balanced: `64,64,64,64`;
- skewed: `160,64,24,8`;
- tiny: `16,16,16,16`.

## Metrics

| Metric | Meaning |
|---|---|
| `expert_loop_ms` | Serial per-expert PyTorch FFN latency. |
| `grouped_bmm_ms` | Padded strided-batched FFN latency. |
| `grouped_speedup` | `expert_loop_ms / grouped_bmm_ms`. |
| `padding_overhead` | `experts * max_tokens / actual_tokens`. |

## Prediction

If grouped expert compute is the right direction:

- balanced and tiny distributions should show grouped speedup;
- skewed may weaken because padding overhead is large;
- this will justify H3.4: custom grouped/tiled Triton expert matmul without
  padding waste.

If grouped bmm does not help:

- the next path should focus on routing/epilogue grouping only, or on custom
  spatial partitioning rather than standard batched GEMM.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Check grouped output against the expert loop on valid token ranges.
- Keep request count fixed at one.
