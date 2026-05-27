# H4.4 Analysis: Multi-Block Online-Softmax Sparse Attention

Date: 2026-05-26

## Status

Support for the H4 GPU compiler/runtime path beyond the H4.3 single-tile
setting.

H4.4 extends the sparse score-softmax-value fragment to `1024` selected tokens,
which is intentionally larger than the previous single-tile setting. A two-stage
Triton implementation computes local softmax/value partials per selected-token
block, then combines them with online-softmax rescaling. It beats flat PyTorch
materialization in every tested order.

## Results

| Order | Flat PyTorch | Best Triton online | Speedup | Best variant | Mean span |
|---|---:|---:|---:|---|---:|
| random segment | 0.1330 ms | 0.0286 ms | 4.65x | N128 D64 V128 W4 | 88.0 |
| random sorted | 0.1319 ms | 0.0436 ms | 3.02x | N128 D64 V64 W8 | 8.0 |
| random shuffled | 0.1364 ms | 0.0287 ms | 4.76x | N128 D64 V64 W4 | 2675.7 |
| clustered segment | 0.1234 ms | 0.0413 ms | 2.99x | N128 D64 V64 W8 | 7.9 |

All runs used one request, one query, `32` segments, `32` selected tokens per
segment, `1024` selected tokens total, `tokens_per_segment=256`, `D=128`,
`Vd=128`, and fp16 K/V/q inputs with fp32 softmax/value accumulation.

## Interpretation

H4.4 answers the main question left by Outer Loop 3: the fused sparse attention
story survives beyond a single-tile toy case. Multi-block online-softmax Triton
beats flat PyTorch by `2.99x-4.76x` on a 1024-token sparse retrieval fragment.

The speedup is lower than H4.3's `5.78x-6.79x`, which is expected. H4.4 uses two
kernel launches, stores partial max/denominator/value buffers, and then reduces
partials. That extra machinery is the cost of scaling beyond
`selected_tokens <= BLOCK_N`.

The order result is deliberately not simple. PyTorch eager benefits from the
more locality-friendly clustered order (`0.1234 ms`, the fastest PyTorch case).
Triton online does not monotonically improve with locality: random segment and
random shuffled are the fastest Triton cases, while sorted and clustered are
slower. Without hardware counters this should not be overinterpreted, but it
does reinforce the H4.2/H4.3 contract: order statistics alone do not select the
best implementation. The scheduler needs order statistics, tile shape, number of
partial blocks, value-blocking, warps, and measured full-fragment latency.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke test and all full runs passed validation for every Triton variant.
- Relative max difference versus PyTorch eager was below `4e-7`.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- The implementation is a simple two-kernel prototype, not a production
  FlashAttention-like persistent kernel.
- It still handles one query only. Multi-query/multi-head shapes may change the
  best tiling and the amortization of partial reductions.
- The partial kernel recomputes scores per value block when `num_v_blocks > 1`.
  A more mature implementation should separate score metadata from value-block
  accumulation or use shared/persistent state.
- Hardware counters are unavailable, so memory locality claims remain based on
  latency and index-order diagnostics.

## Next Step

H4.5 should test multi-query or multi-head sparse attention:

1. reuse selected indices across multiple query vectors;
2. compare per-query online kernels versus a grouped query kernel;
3. measure whether score computation and K/V movement amortize across queries.

This will decide whether the H4 compiler/runtime path can scale from a
single-query retrieval proxy toward a realistic Video/VLM decoder attention
fragment.
