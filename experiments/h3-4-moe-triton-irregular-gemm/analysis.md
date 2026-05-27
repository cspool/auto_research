# H3.4 Analysis: Irregular No-Padding Triton Expert Matmul

Date: 2026-05-26

## Status

Strong support for the no-padding grouped/tiled expert matmul direction.

H3.3 showed that padded grouped bmm is positive but loses much of its benefit
under skewed routing. H3.4 replaces padding with a Triton row-tile map that
schedules only valid expert token tiles. Even this simple Triton kernel beats
both the expert loop and padded bmm across all tested token distributions.

## Results

| Run | Padding overhead | Expert loop ms | Padded bmm ms | Best Triton no-pad ms | vs loop | vs padded bmm |
|---|---:|---:|---:|---:|---:|---:|
| balanced 4x64 | 1.00x | 0.2189 | 0.1914 | 0.1709 | 1.28x | 1.12x |
| skewed | 2.50x | 0.2421 | 0.2135 | 0.1634 | 1.48x | 1.31x |
| tiny 4x16 | 1.00x | 0.2154 | 0.1735 | 0.1550 | 1.39x | 1.12x |

The best variant in all three runs was `BLOCK_M=32`, `BLOCK_N=32`,
`BLOCK_K=64`, `num_warps=4`.

## Interpretation

This result closes the loop opened by H3.1-H3.3:

- H3.1: expert streams are the wrong abstraction;
- H3.2: grouped epilogue micro-operators are very effective;
- H3.3: padded grouped GEMM is promising but skew-sensitive;
- H3.4: irregular no-padding tiled matmul recovers the skewed case and beats
  padded bmm.

The strongest evidence is the skewed distribution. H3.3's padded bmm did 2.5x
token work and only achieved 1.08x speedup over the expert loop. H3.4 scheduled
9 valid row tiles and improved to 1.48x over the expert loop and 1.31x over
padded bmm.

This supports the refined MoE story: semantic decomposition becomes useful when
the compiler/runtime keeps the semantic segments but schedules them as compact
valid tiles, not as independent framework kernels and not as padded dense
batches.

## Sanity Checks

- GPU runs were performed outside the default sandbox.
- Output finite checks passed.
- Relative max difference versus the PyTorch expert loop was below `9e-4`,
  acceptable for different fp16 GEMM accumulation paths.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- The Triton kernel is still a simple baseline, not a production grouped GEMM.
- No routing/top-k/scatter/gather/combiner weights are included.
- Only four experts and three token distributions are tested.
- Hardware counters are unavailable, so the result is latency-based.
- The implementation uses a separate ReLU launch between the two matmuls; a
  fused FFN tile could improve this further.

## Next Step

H3.5 should add routing/scatter micro-ops around this no-padding grouped matmul:

1. generate token-to-expert assignments for one request;
2. measure scatter into concatenated expert-token buffer;
3. run H3.4 no-padding grouped FFN;
4. gather/combine back to original token order;
5. compare full routed MoE fragment with the H3.1 expert-loop baseline.
