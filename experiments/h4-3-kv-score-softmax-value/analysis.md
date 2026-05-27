# H4.3 Analysis: Sparse KV Score-Softmax-Value Fragment

Date: 2026-05-26

## Status

Strong support for the H4 fused movement-compute direction on the tested
single-tile sparse attention fragment.

H4.3 extends H4.1/H4.2 from score-only retrieval to a single-query sparse
attention chain: gather selected K/V rows, compute scores, softmax, and
aggregate selected V rows. The fused Triton score-softmax-value kernel beats
flat PyTorch eager in every tested selected-token order.

## Results

| Order | Flat PyTorch | Best Triton SSV | Speedup | Best variant | Mean span |
|---|---:|---:|---:|---|---:|
| random segment | 0.1402 ms | 0.0207 ms | 6.79x | N256 D64 V64 W8 | 95.1 |
| random sorted | 0.1250 ms | 0.0216 ms | 5.78x | N256 D64 V32 W4 | 15.9 |
| random shuffled | 0.1285 ms | 0.0210 ms | 6.12x | N256 D64 V128 W4 | 1312.0 |
| clustered segment | 0.1286 ms | 0.0214 ms | 6.00x | N256 D64 V128 W4 | 15.1 |

All runs used one request, one query, `16` segments, `16` selected tokens per
segment, `256` selected tokens total, `tokens_per_segment=256`, `D=128`,
`Vd=128`, and fp16 K/V/q inputs with fp32 softmax/value accumulation.

## Interpretation

H4.3 is the strongest positive H4 result so far. In H4.1/H4.2, fused
gather-score beat flat PyTorch by roughly `1.15x-1.42x`. After adding softmax
and value aggregation, fused Triton beats flat PyTorch by `5.78x-6.79x`.

The reason is visible in the PyTorch component timings. For the full fragment,
each K and V gather costs about `0.013 ms`, but score+softmax costs about
`0.060-0.064 ms` and value aggregation costs about `0.030-0.033 ms`. The fused
kernel removes K/V materialization and keeps the score, softmax, and value
reduction in one compiler-visible unit. That is exactly the intra-kernel
movement-compute fusion pattern suggested by V-Rex/Focus-style hardware
co-design, implemented here as a GPU compiler proxy.

Ordering remains nontrivial. Sorted and clustered orders improve PyTorch eager
latency relative to random segment order, but they do not monotonically improve
fused Triton. The best Triton latency stays in a narrow band
(`0.0207-0.0216 ms`) while the best tile changes by order. This repeats the H4.2
lesson: order/locality metadata matters, but it must be joined with kernel
variant metadata and measured full-fragment latency.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke test and all full runs passed validation for every Triton variant.
- Relative max difference versus PyTorch eager was below `8.3e-4`.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- The fused Triton implementation is a single-tile reduction and requires
  `selected_tokens <= BLOCK_N`; it is not yet a production FlashAttention-style
  online softmax over many blocks.
- The workload uses one query. Multi-query or multi-head retrieval may change
  the best tiling strategy.
- PyTorch eager uses materialized selected K/V and fp32 score/value operations;
  a library sparse attention baseline could be stronger.
- Hardware counters are unavailable, so architecture claims remain based on
  latency decomposition.

## Next Step

H4 now has enough inner-loop evidence for a third outer-loop synthesis covering
H4.1-H4.3. The synthesis should decide whether to:

1. deepen H4 with a multi-block online-softmax sparse attention kernel; or
2. broaden to DiT/diffusion or multimodal module pipelines using the same
   full-fragment fusion protocol.
