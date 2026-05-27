# H2.2 Analysis: Triton Compiler Metadata for Variant Scheduling

Date: 2026-05-26

## Status

Supported as a compiler-interface result, with limits.

Nsight Compute/System was not available in the current environment, so H2.2 used
Triton cache artifacts instead. This is weaker than hardware counters, but it is
directly relevant to the compiler/runtime boundary: the cache exposes metadata a
runtime scheduler could attach to each generated variant.

## Core Result

The extractor recovered all H2.1 fused micro-kernel cache entries:

- 29 unique compiled variants total;
- 1 smoke-test variant;
- 14 balanced-shape variants;
- 14 memory-heavy variants.

For every measured H2.1 variant, the cache artifacts recover:

- `block_size` from TTGIR tensor shape;
- `num_warps`, `num_ctas`, `num_stages`, `shared` from Triton JSON metadata;
- `.reqntid` and `.reg` declarations from PTX;
- PTX/cubin file sizes as codegen footprint proxies.

The joined scheduler table is saved at
`experiments/h2-2-triton-compiler-metadata/results/joined_latency_metadata.csv`.

## Metadata Pattern

For this fused epilogue kernel:

- `num_warps=4` maps to `.reqntid 128`;
- `num_warps=8` maps to `.reqntid 256`;
- static shared memory is `0` for all variants;
- `num_ctas=1` and `num_stages=3` for all variants;
- PTX register declarations and code size grow strongly with tile size.

Representative balanced-shape rows:

| Variant | Concurrent ms | reqntid | reg_b32_decl | PTX bytes | Role |
|---|---:|---:|---:|---:|---|
| block=2048, warps=8 | 0.2766 | 256 | 161 | 16609 | best isolated |
| block=2048, warps=4 | 0.2271 | 128 | 313 | 28252 | best concurrent |
| block=4096, warps=4 | 0.2374 | 128 | 617 | 51535 | best stream speedup |

The important observation is not that static metadata directly predicts latency.
It does not. The useful point is that a compiler/runtime can build a variant
table with both static metadata and empirical scheduling measurements. This is
exactly the missing interface implied by H2 and H2.1.

## Interpretation

H2.2 connects the latency experiments to an implementation path:

1. generate multiple Triton variants;
2. retain compiler metadata for each variant;
3. benchmark or profile representative co-location contexts;
4. select variants by absolute concurrent latency, using overlap/speedup as
   diagnostics.

This mirrors the literature pattern from Infera-style micro-operators and
MetaAttention/PAT-style compiler metadata. The scheduler needs a feature table,
not just a single fused kernel.

## Limitations

- PTX `.reg` declarations are static declarations, not achieved occupancy.
- No SM/SFU/DRAM counters were collected because Nsight tools were unavailable.
- Cache artifacts depend on Triton version and backend codegen.
- The metadata extractor currently targets one kernel name and should be
  generalized before becoming a reusable benchmark tool.

## Next Step

The next inner-loop step should move from synthetic epilogue fragments to a
model-semantic unit:

- H3.1: tiny single-request MoE with two or four experts, measuring expert
  epilogue and router/top-k micro-op co-scheduling; or
- H3.2: DiT/Video-style patch MLP fragment, measuring per-patch micro-op
  concurrency and compiler variant metadata.
