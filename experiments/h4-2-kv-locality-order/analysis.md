# H4.2 Analysis: Locality-Aware Sparse KV Retrieval Ordering

Date: 2026-05-26

## Status

H4.2 refines the H4 story: locality/order metadata matters, but not in a simple
"more sorted is always faster" way for this small score-only proxy.

The strongest signal is the interaction between selected-token order and Triton
tile variant. The best fused kernel remained faster than flat PyTorch
gather+GEMM in every order, but the best tile changed across orders.

## Results

| Order | Mean neighbor span | Monotonic | Flat PyTorch | Best Triton | Speedup | Best variant |
|---|---:|---:|---:|---:|---:|---|
| random segment | 90.6 | 52.8% | 0.0299 ms | 0.0211 ms | 1.42x | M32 D128 W4 |
| random sorted | 8.0 | 100.0% | 0.0294 ms | 0.0213 ms | 1.38x | M16 D64 W8 |
| random shuffled | 2673.7 | 48.1% | 0.0299 ms | 0.0214 ms | 1.40x | M32 D64 W4 |
| clustered segment | 7.7 | 100.0% | 0.0295 ms | 0.0216 ms | 1.37x | M16 D128 W4 |

All runs used one request, `32` segments, `32` selected tokens per segment,
`tokens_per_segment=256`, `D=256`, `Qn=8`, and fp16 inputs/outputs.

## Interpretation

There are three takeaways.

First, fused movement-compute remains useful. Across all order modes, the best
Triton fused gather-score kernel beats flat PyTorch gather+GEMM by `1.37x` to
`1.42x`. This supports the H4.1 conclusion that materializing selected K rows is
avoidable overhead.

Second, ordering alone is not a silver bullet at this scale. Global sorting
reduced mean neighbor span from `90.6` to `8.0`, and global shuffling increased
it to `2673.7`, but the best fused Triton latency stayed within about `2.1%`
across all orders. The score-only proxy is likely too small and too cacheable
for locality to dominate absolute latency.

Third, tile/order interaction is real. The best tile was:

- `M32 D128 W4` for random segment order;
- `M16 D64 W8` for globally sorted order;
- `M32 D64 W4` for globally shuffled order;
- `M16 D128 W4` for clustered segment order.

This is the compiler/runtime contract emerging from H4: selected-token order,
locality statistics, and kernel resource metadata should be considered jointly.
A runtime that applies only an index-sort transform without retuning the fused
kernel may not improve latency.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke test and all full runs passed validation for every Triton variant.
- Triton outputs matched flat PyTorch exactly for all measured variants.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- The fragment still covers only score computation, not softmax or value
  aggregation.
- The selected-token set is synthetic and deterministic from the seed.
- Hardware counters are unavailable, so cache/locality effects are inferred from
  latency and index-order statistics.
- The sweep is small: six tile variants and one problem size.

## Next Step

H4.3 should add value aggregation or a score-softmax-value chain:

1. compute `K_selected @ Q`;
2. reduce/softmax over selected tokens;
3. gather selected V rows and accumulate `softmax(S) @ V_selected`.

That fragment will increase memory traffic and inter-operator dependencies,
making it a better test for whether GPU compiler/runtime ordering is sufficient
or whether the V-Rex/Focus hardware-side retrieval argument becomes stronger.
