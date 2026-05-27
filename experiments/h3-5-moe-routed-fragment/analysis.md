# H3.5 Analysis: Routed No-Padding MoE Fragment

Date: 2026-05-26

## Status

Support for the refined MoE branch of H3.

H3.4 showed that a no-padding Triton row-tile map beats padded grouped bmm when
expert-token segments are already materialized. H3.5 adds the surrounding
routing movement: scatter from original token order into expert-contiguous
layout, run the no-padding grouped FFN, then gather back to original token
order.

The no-padding grouped fragment still beats the PyTorch routed expert loop in
all tested distributions. Routing movement is meaningful, but it does not erase
the gain.

## Results

| Run | PyTorch routed loop | Triton scatter | Triton compute | Triton gather | Triton routed | Speedup | Movement fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced 4x64 | 0.2718 ms | 0.0183 ms | 0.1579 ms | 0.0185 ms | 0.1753 ms | 1.55x | 21.0% |
| skewed 160/64/24/8 | 0.2728 ms | 0.0223 ms | 0.2024 ms | 0.0198 ms | 0.1959 ms | 1.39x | 21.5% |
| tiny 4x16 | 0.2744 ms | 0.0183 ms | 0.1561 ms | 0.0196 ms | 0.1671 ms | 1.64x | 22.6% |

The full routed latency is the primary metric. The scatter, compute, and gather
numbers are separately measured diagnostics; their sum can exceed the full
routed latency because the composed path benefits from immediate producer to
consumer locality and has different launch/timing boundaries.

## Interpretation

H3.5 closes a key gap in the MoE branch. The earlier H3.4 result could have
been an artifact of assuming an already packed expert buffer. It was not:
explicit scatter/gather adds roughly `0.037-0.042 ms`, or about one fifth of the
full Triton routed fragment, but the end-to-end fragment still beats the
framework-level routed loop by `1.39x-1.64x`.

The skewed case is the most important boundary condition. It remains faster
than the PyTorch routed loop, but its speedup is the lowest because it schedules
more row tiles and has slightly higher movement cost. That supports a practical
compiler/runtime rule: routing skew should be represented in the tile map and
variant selector, not hidden behind padded dense batches or independent expert
streams.

The MoE branch now has a coherent implementation ladder:

- naive expert streams are slower than serial framework execution;
- grouped epilogue micro-operators produce very large wins;
- padded grouped GEMM is useful but skew-sensitive;
- no-padding expert tiles recover skewed routing waste;
- adding scatter/gather shows the full routed fragment still wins.

## Sanity Checks

- GPU runs were performed outside the default sandbox because CUDA devices are
  not visible inside it.
- Smoke test and all full runs passed validation against the PyTorch routed loop
  in original token order.
- Relative max difference versus PyTorch was below `9e-4`, acceptable for fp16
  GEMM ordering differences.
- Runs were sequential to avoid cross-process GPU interference.
- The PyTorch baseline does not zero the output buffer before `index_copy_`
  because top-1 routing covers every token exactly once.

## Limitations

- The fragment uses fixed top-1 assignments, not a measured gating softmax or
  top-k combiner weights.
- Scatter/gather are separate flat copy kernels. A production design should fuse
  routing movement with the first or second expert matmul where possible.
- Only one RTX 4090 class GPU backend was measured; no NPU/Ascend/ROCm counters
  are available yet.
- The Triton grouped matmul is a simple baseline, not a tuned production grouped
  GEMM.
- Hardware counters were unavailable, so movement and compute attribution is
  latency-based.

## Next Step

The MoE branch is mature enough for a second outer-loop synthesis. The next
research branch should either:

1. extend MoE with top-k routing/combiner weights and fused scatter-first-matmul
   kernels; or
2. move to H4 with video/VLM token selection and KV retrieval, using the same
   decomposition: semantic sparse segments, no-padding tiles, explicit movement
   micro-ops, and full-fragment latency as the primary metric.
