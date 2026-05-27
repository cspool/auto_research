# H4.5 Analysis: Multi-Query Sparse KV Online Softmax

Date: 2026-05-26

## Status

Supported.

H4.5 extends H4.4 from one query to `Q=4` query vectors sharing the same
selected K/V set. The grouped-query two-stage Triton implementation beats flat
PyTorch eager in all four selected-token orders and also beats a measured
per-query repeated online-softmax baseline by about `3.4x-3.7x`.

## Results

| Order | Flat PyTorch | Best grouped Triton | Speedup vs flat | Best per-query repeated | Grouped vs repeated | Best grouped variant | Mean span |
|---|---:|---:|---:|---:|---:|---|---:|
| random segment | 0.1231 ms | 0.0338 ms | 3.64x | 0.1238 ms | 3.66x | N64 D64 V64 Q4 W4 | 92.7 |
| random sorted | 0.1228 ms | 0.0429 ms | 2.86x | 0.1470 ms | 3.42x | N128 D128 V64 Q4 W4 | 8.0 |
| random shuffled | 0.1306 ms | 0.0433 ms | 3.02x | 0.1475 ms | 3.41x | N128 D64 V64 Q4 W8 | 2767.7 |
| clustered segment | 0.1280 ms | 0.0425 ms | 3.01x | 0.1473 ms | 3.46x | N128 D64 V128 Q4 W4 | 7.8 |

All full runs used one request, `32` segments, `32` selected tokens per segment,
`1024` selected tokens total, `tokens_per_segment=256`, `D=128`, `Vd=128`,
`Q=4`, and fp16 K/V/Q inputs with fp32 online-softmax/value accumulation.

## Interpretation

The confirmatory prediction passes: grouped-query Triton beats flat PyTorch by
more than `2.0x` in all four order modes. This makes the H4 sparse-attention
branch stronger than the H4.4 single-query result because it shows that the
compiler-visible sparse fragment scales in the query/head dimension rather than
requiring independent sparse-attention launches for each query.

The most important comparison is not just Triton versus PyTorch. Repeating the
single-query online-softmax path four times costs `0.1238-0.1475 ms`, which is
roughly flat-PyTorch latency and often slower than flat PyTorch. Grouping the
four queries into one partial kernel plus one reduce kernel lowers latency to
`0.0338-0.0433 ms`. That is the concrete scheduling lesson: query/head reuse
must be represented in the compiler/runtime unit, not left as a framework loop
over otherwise good single-query kernels.

The best tile still depends on order. Random segment prefers `N64 D64 V64 Q4
W4`, sorted prefers `N128 D128 V64 Q4 W4`, shuffled prefers `N128 D64 V64 Q4
W8`, and clustered prefers `N128 D64 V128 Q4 W4`. Order statistics alone do not
select the best implementation. They need to be joined with query grouping,
value blocking, warps, and resource limits.

One hardware-facing result is especially useful: the `N256 D64 V64 Q4 W4`
variant failed in all full runs with shared-memory overuse (`133120` bytes
required versus `101376` hardware limit). H4.5 therefore adds a concrete
compiler/runtime constraint to the H4 story: grouped-query sparse attention
should not only rank measured variants; it must also filter variants by static
resource feasibility before launch.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke and full runs completed sequentially to avoid cross-process GPU
  interference.
- The grouped kernel uses explicit fp32/IEEE dot paths after an exploratory
  tensor-core version showed larger but still small fp16-style differences.
- Full-run relative max difference versus PyTorch eager was below `7e-7` for
  every best grouped variant.
- Failed variants are preserved in `data/h4_5_kv_multi_query_variants.csv`
  rather than silently dropped.

## Protocol Deviations

The protocol described the per-query comparison as a simple estimate. The final
implementation records a stronger measured baseline: it invokes the one-query
H4.4-style partial and reduce kernels once per query and times the full repeated
sequence.

## Limitations

- The grouped kernel is still a two-kernel prototype. A production
  FlashAttention-like implementation would likely reduce partial-buffer traffic
  and avoid recomputing scores per value block.
- The run tests `Q=4`; broader query counts may shift the best `BLOCK_Q` and
  shared-memory boundary.
- The benchmark still uses synthetic K/V/Q and synthetic selected-token orders,
  not a real video-token selector or VLM decoder trace.
- Hardware counters are still unavailable, so resource interpretation comes
  from latency, Triton launch failures, and index-order diagnostics.

## Next Step

Run an outer-loop synthesis over H4.1-H4.5. The emerging H4 contribution is now
clear enough to compare against a stronger production-style baseline or to
broaden toward accelerator/NPU primitives:

1. GPU compiler path: sparse segment metadata, order statistics, query grouping,
   variant feasibility, and empirical full-fragment latency.
2. Hardware path: selected-token movement and online-softmax reduction are
   concrete candidates for KV-side gather/reduction units or memory-interface
   scheduling support.

