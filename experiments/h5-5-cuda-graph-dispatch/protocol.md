# H5.5 Protocol - CUDA Graph Replay as a Cheaper Dispatch Substrate

Date: 2026-05-28

## Hypothesis

The H5.4 warmed queue still pays substantial Python/PyTorch stream-event and
kernel-launch overhead. Capturing the fixed six-step dependency queue as a CUDA
Graph and replaying it should reduce host dispatch overhead enough that runtime
variant selection has a cleaner substrate.

## Prediction

Compared with the warmed Python/PyTorch stream queue:

- CUDA Graph replay will lower wall/event queue latency for the same fixed work;
- graph capture will be a one-time cost and must be outside the request path;
- if graph capture fails or gives little benefit, H5 should move toward larger
  task tiles or lower-level persistent/batched dispatch rather than Python-side
  fine-grained migration.

## Method

Reuse the H5.1 queue, shapes, and policy selections from H5.1 result JSON.
For each policy:

1. allocate persistent tensors/streams;
2. precompile all selected Triton variants and warm background work;
3. measure the plain Python/PyTorch stream queue without NVTX inside the queue;
4. capture the same queue into a `torch.cuda.CUDAGraph`;
5. measure `graph.replay()` latency with CUDA events and wall timing;
6. record graph capture success/failure and capture wall time.

The first H5.5 run is a functional latency experiment. If graph replay succeeds
and is promising, the next run should add an Nsight Systems graph replay profile.

## Metrics

- Python stream queue event/wall median per policy;
- graph capture wall time per policy;
- graph replay event/wall median per policy;
- replay speedup over plain Python stream queue;
- graph capture errors, if any.

## Decision Rule

If graph replay gives a meaningful speedup without changing GPU work, continue
H5 on graph/persistent dispatch and then revisit H5.3 migration. If it fails,
H5.3 should use larger task tiles or a lower-level persistent-kernel/batched
runtime instead of PyTorch/CUDA graph capture.
