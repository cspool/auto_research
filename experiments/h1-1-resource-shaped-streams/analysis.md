# H1.1 Analysis: Resource-Shaped Stream Concurrency

Date: 2026-05-26

## Status

H1.1 is mixed-to-negative for naive PyTorch-level GEMM chunking.

The experiment tested whether splitting a dominant GEMM into row chunks makes
independent elementwise/reduction operators overlap better in separate CUDA
streams. It ran on an NVIDIA GeForce RTX 4090 with PyTorch 2.11.0+cu128.

## Summary

Chunking can create more apparent overlap, but PyTorch-level row chunking does
not reliably improve end-to-end latency. The overhead of extra GEMM launches and
less favorable GEMM shapes usually consumes the overlap benefit.

## Results

| Shape | Full multi ms | Full stream speedup | Best chunk | Chunk multi ms | Chunk stream speedup | Chunk overlap | Chunk overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| compute-heavy 4096 | 0.9213 | 1.0512 | 4 | 0.9599 | 1.0242 | -0.0300 | 0.9654 |
| balanced 2048 | 0.2430 | 1.0207 | 2 | 0.2515 | 1.0589 | 0.0128 | 1.0733 |
| memory-heavy 1024 | 0.5972 | 1.0177 | 4 | 0.6075 | 1.1147 | 0.1876 | 4.8901 |

Notes:

- "Best chunk" above is chosen by chunked stream speedup relative to its own
  chunked serial baseline.
- Balanced shape had an absolute fastest chunked variant at 4 chunks
  (`0.2273 ms`), but its stream speedup was only `1.0057x`; the improvement
  appears to come from the chunked GEMM shape itself, not stream overlap.
- Memory-heavy shape showed high overlap ratio for chunks, but GEMM chunk
  overhead was very large (`4.89x` at the best stream-speedup point), so absolute
  latency did not beat full multi-stream.

## Interpretation

This result refines H1 rather than overturning it. "Resource shaping" is likely
real, but naive framework-level chunking is too blunt:

1. Smaller GEMMs can expose scheduling gaps or change library kernel choices.
2. Extra launches and worse GEMM aspect ratios can erase the benefit.
3. Apparent overlap ratio can be misleading if the isolated chunked GEMM baseline
   is much slower than full GEMM.
4. Useful resource shaping probably needs compiler/kernel-level control:
   persistent kernels, fused kernels, explicit SM masks, launch-parameter
   shaping, or generated micro-kernels with known resource profiles.

The strongest practical lesson is that micro-operator granularity cannot be
chosen only by graph partitioning. It must be co-designed with kernel generation
and runtime scheduling metadata.

## What This Rules Out

For this benchmark, simply splitting GEMM into PyTorch row chunks is not a
sufficient method for robust single-request operator concurrency.

## Next Step

Move from "framework-level chunking" to one of two more controlled paths:

1. H2 direction: write Triton micro-kernels with explicit tile shapes and compare
   resource-diverse variants.
2. H1.2 direction: use a true spatial-control mechanism, such as stream SM masks
   or launch/block shaping, if a controllable userspace API is available.

