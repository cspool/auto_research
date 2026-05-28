# H5.5 Analysis - CUDA Graph Replay Dispatch Substrate

Date: 2026-05-28

## Question

Can CUDA Graph replay reduce the warmed Python/PyTorch stream queue overhead
observed in H5.4 enough to become a better dispatch substrate for H5 runtime
resource-aware scheduling?

## Setup

H5.5 reuses H5.1's six-step queue, shapes, and policy selections. It first runs
a plain Python/PyTorch stream queue without internal NVTX ranges, then captures
the same fixed queue into one `torch.cuda.CUDAGraph` per policy and measures
`graph.replay()`.

All Triton variants and background kernels are warmed before measurement. The
one-time graph capture cost is recorded separately and is not treated as request
latency.

## Latency Results

| Policy | Capture success | Capture wall ms | Python event ms | Graph event ms | Event speedup | Python wall ms | Graph wall ms | Wall speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static_best_average | True | 11.02 | 2.2421 | 2.1368 | 1.0493x | 2.2620 | 2.1416 | 1.0562x |
| resource_aware | True | 1.55 | 2.2110 | 2.1418 | 1.0323x | 2.2125 | 2.5077 | 0.8823x |

Both policy graphs captured successfully. CUDA event timing shows graph replay is
faster than the plain Python stream queue: 1.05x
for `static_best_average` and 1.03x for
`resource_aware`. The `resource_aware` wall timing is noisier than event timing,
so event timing is the primary comparison.

The graph replay medians for both policies converge near 2.14 ms. That suggests
most remaining latency is GPU work and dependency structure, not Python launch
submission. Graph replay mainly removes host dispatch overhead.

## Nsight Systems Graph Replay Profile

The profile window replays 48 repeats of two policy graphs, so it submits 96
complete six-step queues.

| Metric | H5.4 warmed stream queue | H5.5 graph replay |
|---|---:|---:|
| Queues in profile | 96 | 96 |
| CUDA runtime API calls | 14978 | 194 |
| CUDA kernel launch API calls | 576 runtime + 576 driver | 0 |
| CUDA graph launches | 0 | 96 |
| Graph launch API total | - | 3.12 ms |
| CPU enqueue/NVTX window | 136.16 ms | 4.44 ms |
| CPU enqueue per queue | 1.418 ms | 0.046 ms |
| Enqueue improvement | 1.00x | 30.6x |
| API-call reduction | 1.00x | 77.2x |

Nsight Systems confirms the mechanism. CUDA Graph replay removes the 12 per-step
kernel launch submissions and most stream/event plumbing from the host-visible
queue path. The capture contains 96 `cudaGraphLaunch` calls, no
`cudaLaunchKernel` calls, and only 194 CUDA runtime API calls total. The final
`cudaDeviceSynchronize` still waits for GPU drain, which is expected because the
profile submits graphs asynchronously and synchronizes once at the end.

## Interpretation

H5.5 is the first clearly positive dispatch-substrate result in H5. It does not
make the GPU work disappear, so end-to-end event speedup is modest at 3%-5%.
But it sharply reduces host submission overhead: about 30x lower CPU enqueue
window per queue than the H5.4 Python/PyTorch stream queue, and about 77x fewer
CUDA runtime API calls in the comparable 96-queue profile.

This changes the next H5 direction. Runtime resource-aware scheduling should not
be built as a Python loop over individual kernels. A practical design should use
a bank of pre-warmed graph variants, persistent/batched dispatch, or larger
migration-safe tasks. Dynamic compilation still belongs outside the request path
as H5.2 showed.

## Decision

H5.5 supports CUDA Graph replay as a viable cheaper dispatch substrate for fixed
micro-op queues. The next migration experiment should either:

- build a small graph bank for alternative queue/variant choices and measure
  switching overhead; or
- test H5.3 task migration on larger tiles using graph replay where the migrated
  units are graph-level or batched units rather than individual Python-launched
  kernels.
