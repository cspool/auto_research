# H3.2 Protocol: Triton MoE Expert Epilogue Micro-Operators

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H3: Model-semantic decomposition exposes stronger concurrency.

H3.1 showed that PyTorch one-stream-per-expert execution is too coarse. H3.2
tests the next lower level:

> Once MoE expert work is represented as compiler-visible epilogue
> micro-operators, a grouped single Triton kernel should outperform multiple
> per-expert launches and naive per-expert stream concurrency for small
> single-request expert fragments.

## Workload

Simulate the epilogue after expert FFN compute for a single request. For each
expert segment:

`Y_e = GELU(X_e * scale + Bias_e) + Residual_e`

This isolates expert epilogue/routing-adjacent work, not expert GEMM. It is a
deliberate bridge from H2.1's fused epilogue to H3.1's MoE semantic units.

## Compared Implementations

For each token distribution and each Triton variant:

1. **PyTorch expert loop**: eager PyTorch per expert.
2. **Triton expert serial**: launch one Triton kernel per expert sequentially.
3. **Triton expert concurrent**: launch one Triton kernel per expert on separate
   CUDA streams.
4. **Triton grouped**: concatenate all expert segments and launch one Triton
   kernel over the grouped buffer.

Triton variants:

- `BLOCK_SIZE`: 128, 256, 512, 1024, 2048, 4096;
- `num_warps`: 4, 8.

Token distributions:

- balanced: `64,64,64,64`;
- skewed: `160,64,24,8`;
- tiny: `16,16,16,16`.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_loop_ms` | Eager PyTorch expert loop latency. |
| `serial_ms` | Four per-expert Triton launches in sequence. |
| `concurrent_ms` | Four per-expert Triton launches on separate streams. |
| `grouped_ms` | One grouped Triton launch over all expert segments. |
| `grouped_vs_serial_speedup` | `serial_ms / grouped_ms`. |
| `grouped_vs_concurrent_speedup` | `concurrent_ms / grouped_ms`. |
| `concurrent_vs_serial_speedup` | `serial_ms / concurrent_ms`. |

## Prediction

If H3.2 supports the refined H3:

- grouped single-kernel execution will beat per-expert serial and concurrent
  launches, especially for tiny/balanced epilogues;
- per-expert concurrent streams may remain slower because launch and scheduling
  overhead dominate;
- best `grouped_ms` may choose a different Triton variant than best per-expert
  serial/concurrent latency.

If H3.2 does not support the refined H3:

- grouped execution does not beat the per-expert baselines, suggesting the
  useful unit is not epilogue fusion but expert GEMM/grouped GEMM itself.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Check grouped Triton output against PyTorch tanh-GELU output.
- Keep request count fixed at one; token counts are per-expert tokens inside one
  request/sequence.
