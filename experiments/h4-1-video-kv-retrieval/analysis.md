# H4.1 Analysis: Sparse Video/VLM KV Retrieval Score

Date: 2026-05-26

## Status

Preliminary support for H4, with a refinement.

The first H4 experiment tests a sparse video/VLM retrieval fragment:
`selected K rows @ Q`, where selected rows are grouped by frame/segment. The
result follows the pattern learned in the MoE branch: semantic segments are
useful metadata, but executing them as many framework operations is disastrous.
Flattening and fusing are the useful GPU path.

## Results

| Run | Selected tokens | Segment loop | Flat gather+GEMM | Fused Triton | Fused vs loop | Fused vs flat | Gather fraction of flat |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced random | 1024 | 1.4908 ms | 0.0289 ms | 0.0232 ms | 64.37x | 1.25x | 43.6% |
| skewed random | 1088 | 1.3370 ms | 0.0293 ms | 0.0217 ms | 61.60x | 1.35x | 43.3% |
| tiny random | 128 | 1.1967 ms | 0.0264 ms | 0.0229 ms | 52.25x | 1.15x | 46.2% |

All runs used one request, `32` semantic segments, `tokens_per_segment=256`,
`D=256`, `Qn=8`, random selected indices, and fp16 inputs/outputs.

## Interpretation

H4.1 gives the strongest launch-pressure result so far. The per-segment PyTorch
loop has about `96` launches per request in this proxy and is `45x-52x` slower
than even the flat PyTorch path. This is the same lesson as H3.1, but amplified:
semantic sparse branches cannot be directly mapped to framework loops or
streams.

The flat PyTorch baseline is already strong because it collapses retrieval into
two operations: one `index_select` and one small GEMM. Still, the selected-K
materialization is not free. Gather alone accounts for `43%-46%` of flat
gather+GEMM latency. The fused Triton gather-score kernel removes that
materialized intermediate and wins by `1.15x-1.35x`.

This refines H4. The current evidence does not yet prove that custom hardware is
necessary for this fragment. It does show that irregular video/VLM retrieval
needs a compiler/runtime representation below framework operators: selected
indices, segment boundaries, locality/order metadata, and fused movement-compute
kernels. Hardware units like V-Rex's KVPU/KVMU or Focus's memory-interface
concentration unit are plausible next-stage designs once the fragment expands
to dynamic prediction, value mixing, and larger memory hierarchies.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke test and all full runs passed validation.
- Segment loop, flat PyTorch, and Triton fused outputs matched exactly for the
  tested fp16 shapes.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- H4.1 only computes retrieval scores, not softmax or value aggregation.
- Selection indices are synthetic, not produced by a real video-token predictor.
- The benchmark does not yet compare random versus locality-aware ordering.
- Hardware counters are unavailable, so memory-system claims are based on
  latency decomposition rather than direct HBM/LDST utilization.
- The Triton kernel uses one static tile shape; no multi-version scheduler has
  been added yet.

## Next Step

H4.2 should test locality and ordering:

1. random selected indices versus clustered indices;
2. original semantic segment order versus globally sorted selected indices;
3. multiple Triton tile variants for `D`, `Qn`, and selected-token count.

The goal is to decide whether a GPU compiler/runtime can recover enough memory
locality through ordering and fusion, or whether the V-Rex/Focus style
hardware-side retrieval/concentration argument becomes stronger.
