# H3.1 Protocol: Tiny MoE Semantic Concurrency

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H3: Model-semantic decomposition exposes stronger concurrency.

This first H3 test narrows the prediction:

> MoE expert branches are semantic concurrency units, but naive framework-level
> multi-stream expert execution will only help when individual expert GEMMs are
> small enough not to saturate the GPU and the routing distribution is balanced.

## Workload

Simulate a single-request MoE FFN fragment with four experts. Routing is
pre-materialized into per-expert token tensors so that this experiment isolates
expert compute concurrency rather than top-k/routing overhead.

Each expert computes:

`Y_e = ReLU(X_e @ W1_e) @ W2_e`

The ReLU is in-place to avoid allocator noise. All tensors are fp16 on one RTX
4090. This is intentionally a PyTorch-level expert-loop baseline before moving
to Triton or fused MoE epilogues.

## Variants

Run three token distributions:

- balanced: `64,64,64,64`;
- skewed: `160,64,24,8`;
- tiny: `16,16,16,16`.

For each distribution, measure:

- isolated latency of each expert;
- serial loop over experts;
- concurrent expert execution with one CUDA stream per expert;
- stream speedup: `serial_ms / concurrent_ms`;
- overlap ratio: `1 - concurrent_ms / sum(expert_isolated_ms)`.

## Prediction

If model-semantic decomposition is useful at this level:

- balanced/tiny experts should show better overlap than previous generic H1
  operator streams;
- skewed routing should reduce overlap because the largest expert dominates;
- large experts may still serialize or contend because each GEMM can already
  fill the GPU.

If naive expert streams are insufficient:

- all distributions will be neutral or slower, reinforcing the need for
  compiler-level expert micro-kernels or spatial partitioning.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Check every expert output is finite.
- Keep request count fixed at one; token counts represent tokens inside one
  request/sequence, not batched requests.
