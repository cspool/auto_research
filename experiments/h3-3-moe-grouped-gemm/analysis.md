# H3.3 Analysis: MoE Grouped Expert GEMM

Date: 2026-05-26

## Status

Moderate support for grouped expert compute, with an important boundary.

H3.2 showed a very large win for grouped expert epilogues. H3.3 tests actual
expert FFN compute using PyTorch strided batched GEMM (`torch.bmm`) as a
framework-level grouped GEMM baseline.

## Results

| Run | Token counts | Padding overhead | Expert loop ms | Grouped bmm ms | Speedup |
|---|---|---:|---:|---:|---:|
| balanced 4x64 | 64,64,64,64 | 1.00x | 0.2265 | 0.1842 | 1.23x |
| skewed | 160,64,24,8 | 2.50x | 0.2409 | 0.2232 | 1.08x |
| tiny 4x16 | 16,16,16,16 | 1.00x | 0.2073 | 0.1496 | 1.39x |

Grouped GEMM is positive across all three distributions, but the win is much
smaller than the H3.2 epilogue win.

## Interpretation

The result supports the refined H3 direction but also clarifies where the hard
part is:

- Epilogue/routing-adjacent work benefits massively from being grouped into one
  compiler-visible micro-operator.
- Expert GEMM compute benefits from grouping, but standard padded batched GEMM
  only gives modest speedups.
- Skewed routing erodes the benefit because padded grouped GEMM performs extra
  work. In the skewed run, padding overhead was `2.5x`, and speedup fell to
  `1.08x`.

This suggests that a strong MoE implementation needs a custom grouped or tiled
expert matmul that avoids padding waste while still reducing launch overhead and
preserving compiler-visible scheduling metadata.

## Connection to H3.1/H3.2

H3.1 negative result:

- PyTorch expert streams are the wrong implementation unit.

H3.2 positive result:

- grouped expert epilogue micro-operators are very effective.

H3.3 moderate result:

- grouped expert compute is promising, but standard padded batched GEMM is only
  a baseline. The next step should be irregular grouped/tiled matmul.

## Sanity Checks

- GPU runs were performed outside the default sandbox.
- Outputs were finite.
- Grouped output matched expert-loop output within fp16 GEMM tolerance.
  Max relative difference was below `9e-4`.
- Runs were sequential to avoid cross-process GPU interference.

## Limitations

- Uses padded strided batched GEMM, not a custom no-padding grouped GEMM.
- Does not include routing, top-k, scatter/gather, or combine weights.
- Only four experts are tested.
- PyTorch/cuBLAS kernel choices can vary by shape; this is a baseline, not a
  final implementation.

## Next Step

The next useful experiment is an irregular grouped expert matmul benchmark:

1. avoid padding by scheduling expert tiles over a concatenated token buffer;
2. keep per-expert weight pointers;
3. compare against H3.3 padded grouped bmm and H3.1 expert loop;
4. extract Triton metadata for tile size, warps, registers, and code size.
