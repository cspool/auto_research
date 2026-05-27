# H1 Analysis: Stream Concurrency Baseline

Date: 2026-05-26

## Status

Preliminary support for H1.

The experiment ran on an NVIDIA GeForce RTX 4090 using PyTorch 2.11.0+cu128 and
CUDA 12.8. Default sandbox execution could not see `/dev/nvidia*`; GPU runs were
executed outside the sandbox after verifying GPU visibility.

## Valid Runs

| Run | Shape | Serial ms | Multi-stream ms | CUDA Graph serial ms | Sum isolated ms | Overlap ratio | Serial speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | mat=2048, vec=65536x256 | 0.2308 | 0.2212 | 0.2351 | 0.2353 | 0.0599 | 1.0433 |
| memory-heavy | mat=1024, vec=262144x256 | 0.5930 | 0.5883 | 0.5880 | 0.5876 | -0.0011 | 1.0080 |
| compute-heavy | mat=4096, vec=32768x256 | 0.9558 | 0.9591 | 0.9387 | 0.9649 | 0.0060 | 0.9966 |

## Interpretation

Naive multi-stream execution did not create strong operator concurrency. Across
three shapes, the best speedup was only 1.043x and the best overlap ratio was
about 6.0%. Two variants were essentially neutral or slightly worse.

The results match the local-notes expectation from uShare/Bullet/Infera:
independent streams alone do not force useful spatial sharing when one dominant
kernel occupies the GPU. They mostly expose launch-tail overlap or small runtime
effects.

CUDA Graph replay did not create overlap either. It sometimes reduced overhead
slightly, but as expected it preserves the serial execution dependency pattern.

## Excluded Runs

Two exploratory variants were accidentally launched concurrently:

- `results/gpu_rtx4090_memory_heavy`
- `results/gpu_rtx4090_compute_heavy`

Those are excluded from the conclusion because two benchmark processes can
interfere with each other on the same GPU. The valid versions are the `_seq`
runs.

## What This Rules Out

For this operator mix and RTX 4090 environment, simply placing independent
operators on separate CUDA streams is not enough to obtain meaningful
single-request parallelism.

This does not rule out:

- resource-shaped co-location such as uShare-style block-size shaping;
- explicit SM partitioning such as Bullet-style stream masks;
- intra-kernel fusion such as FlashFuser or JanusQuant;
- semantic micro-operator scheduling such as Infera-style tiling.

## Next Step

H1.1 should test whether resource shaping changes the result. The next
microbenchmark should compare:

1. naive multi-stream,
2. intentionally smaller GEMM tiles or chunked GEMM,
3. stream priority,
4. optional SM partitioning if a controllable API is available,
5. fused/micro-tiled implementation for one chain.

