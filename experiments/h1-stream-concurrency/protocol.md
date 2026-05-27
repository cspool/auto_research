# H1 Protocol: Stream Concurrency Baseline

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H1: For single-request inference on one GPU, naive multi-stream execution gives
limited overlap unless the kernels are resource-complementary or spatially
constrained.

## Motivation

The local-notes survey found that multiple systems treat plain CUDA stream
concurrency as insufficient:

- μShare argues that same-kernel blocks tend to stack on SMs, leaving only weak
  tail overlap unless launch parameters shape spatial placement.
- Bullet uses explicit SM partitioning rather than relying on normal stream
  scheduling.
- Infera and FlashFuser expose smaller or fused units to avoid whole-kernel
  monopolization.

This experiment tests the lowest baseline first: independent GPU operations in
separate streams.

## Workload

The microbenchmark uses independent model-shaped GPU operators:

1. compute-bound matrix multiply;
2. memory/elementwise operation;
3. reduction operation;
4. optional lightweight elementwise operation.

The operators are independent so multi-stream execution is legally allowed. The
workload does not claim to be a full model; it is a controlled proxy for the
operator mix found in attention, normalization, dequantization, and FFN chains.

## Conditions

| Condition | Description |
|---|---|
| isolated | Time each operator alone. |
| serial | Run all operators sequentially on the default stream. |
| multistream | Launch independent operators on separate CUDA streams. |
| cuda_graph_serial | Capture and replay the serial sequence to reduce launch overhead. |

## Metrics

- `latency_ms`: end-to-end time for each condition.
- `sum_isolated_ms`: sum of isolated operator latencies.
- `overlap_ratio`: `1 - multistream_ms / sum_isolated_ms`.
- `serial_speedup`: `serial_ms / multistream_ms`.
- `graph_speedup`: `serial_ms / cuda_graph_serial_ms`.
- environment: GPU name, CUDA availability, PyTorch version.

## Prediction

If H1 is correct:

- `multistream_ms` will be close to `serial_ms` for shapes where a dominant
  kernel occupies most SM resources.
- `overlap_ratio` will be modest even though operations are independent.
- CUDA Graph may reduce launch overhead but will not create true operator
  overlap.

If H1 is wrong:

- `multistream_ms` will approach the maximum isolated operator latency rather
  than the sum or serial latency.
- `overlap_ratio` will be high across the default shapes.

## Sanity Checks

- If CUDA is unavailable, do not report a GPU conclusion. Record the environment
  as `skipped_no_cuda`.
- For CUDA runs, preallocate tensors and use CUDA events for timing.
- Run warmup iterations before measurement.
- Check numerical outputs are finite after each condition.

