# H3.5 Protocol: Routed No-Padding MoE Fragment

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H3: Model-semantic decomposition exposes stronger concurrency.

H3.4 showed that no-padding expert tiles beat padded bmm when expert-token
segments are already materialized. H3.5 adds the surrounding routing movement:

> No-padding grouped expert tiles should remain beneficial after adding
> scatter/gather around the expert buffer, but routing movement will consume a
> meaningful fraction of the gain. Scatter/gather should be measured as
> micro-operators, not hidden inside the compute kernel.

## Workload

Single-request, top-1 routed MoE FFN fragment:

1. original token buffer `X[T, H]`;
2. token-to-expert assignment with a fixed expert-load distribution;
3. scatter tokens into expert-contiguous buffer;
4. run no-padding grouped FFN:
   `Y_e = ReLU(X_e @ W1_e) @ W2_e`;
5. gather expert outputs back to original token order.

## Compared Implementations

1. **PyTorch routed loop**:
   - `index_select` tokens per expert;
   - two `torch.mm` calls per expert;
   - `index_copy_` output back to original order.
2. **Triton routed fragment**:
   - Triton scatter micro-op;
   - H3.4 no-padding grouped matmul FFN;
   - Triton gather micro-op.

H3.4's best tile shape is used by default:

- `BLOCK_M=32`, `BLOCK_N=32`, `BLOCK_K=64`, `num_warps=4`.

## Token Distributions

- balanced: `64,64,64,64`;
- skewed: `160,64,24,8`;
- tiny: `16,16,16,16`.

## Metrics

| Metric | Meaning |
|---|---|
| `torch_routed_loop_ms` | Full PyTorch routed expert loop latency. |
| `triton_compute_ms` | No-padding grouped FFN latency with prepacked expert buffer. |
| `triton_scatter_ms` | Scatter original tokens into expert-contiguous buffer. |
| `triton_gather_ms` | Gather expert output back to original token order. |
| `triton_routed_ms` | Scatter + grouped FFN + gather latency. |
| `triton_routed_speedup` | `torch_routed_loop_ms / triton_routed_ms`. |
| `movement_fraction` | `(scatter + gather) / routed_total`. |

## Prediction

If the H3.4 result survives routing:

- Triton routed fragment should beat PyTorch routed loop for balanced and
  skewed cases;
- routing movement will be most visible in tiny cases where compute is small;
- skewed routing should still benefit because no-padding compute avoids padded
  work.

If routing erases the gain:

- Triton compute remains fast, but scatter/gather dominates total latency;
- the next design should fuse routing movement with compute tiles or use
  hardware-assisted gather/scatter.

## Sanity Checks

- Run outside the default sandbox because CUDA is not visible there.
- Use CUDA events and sequential benchmark processes.
- Check Triton routed output against PyTorch routed loop in original token
  order.
- Keep request count fixed at one.
