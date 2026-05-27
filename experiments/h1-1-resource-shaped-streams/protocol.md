# H1.1 Protocol: Resource-Shaped Stream Concurrency

Date: 2026-05-26

Status: protocol locked in file. Git pre-registration is unavailable because
`/data3/auto_research` is not a git repository.

## Hypothesis

H1.1: If naive CUDA streams fail because a dominant kernel monopolizes GPU
resources, then reshaping the dominant operator into smaller micro-operators
should improve overlap with independent memory/reduction operators.

## Relation to H1

H1 showed that full-kernel multi-stream execution on RTX 4090 achieved only
0.997x to 1.043x speedup across three shapes. H1.1 tests whether this is a
problem of stream semantics in general or a problem of resource shape and kernel
granularity.

## Method

Compute the same operator mix as H1:

1. matrix multiply;
2. elementwise add;
3. row reduction.

The GEMM is evaluated in two forms:

- full GEMM: one `A @ B` kernel;
- chunked GEMM: split `A` by row chunks and run multiple smaller GEMMs that
  write non-overlapping rows of `C`.

For each chunk count, compare:

- `chunked_serial`: chunked GEMM, elementwise, reduction on one stream;
- `chunked_multistream`: chunked GEMM sequence on one stream, elementwise and
  reduction on separate streams;
- `chunked_gemm_only`: the chunked GEMM alone.

Full-kernel serial and full-kernel multi-stream are measured as anchors.

## Metrics

- `latency_ms`
- `stream_speedup = chunked_serial_ms / chunked_multistream_ms`
- `full_stream_speedup = full_serial_ms / full_multistream_ms`
- `overlap_ratio = 1 - chunked_multistream_ms / (chunked_gemm_only_ms + elementwise_ms + reduction_ms)`
- `chunk_overhead = chunked_gemm_only_ms / full_gemm_ms`

## Prediction

If resource shaping helps:

- at some moderate chunk count, `stream_speedup` should exceed H1's full-kernel
  stream speedup;
- `overlap_ratio` should increase versus the full-kernel baseline;
- too many chunks should eventually lose to launch overhead.

If resource shaping does not help:

- chunked multi-stream speedup will remain near 1.0 or become slower;
- chunk overhead will dominate any overlap benefit.

## Sanity Checks

- Use CUDA events and synchronize between trials.
- Use row chunks so chunked GEMM computes exactly the same output tensor layout.
- Check finite outputs after the run.
- Exclude results if benchmark processes run concurrently on the same GPU.

