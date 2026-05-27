# Experiment Framework

## Goal

Create a repeatable framework to evaluate multi-operator and micro-operator
concurrency for a single inference request on one accelerator backend.

## Workloads

| Workload | Proxy model shape | Concurrency source |
|---|---|---|
| W1 attention chain | QKV, attention, norm, dequant, projection | memory-bound and compute-bound micro-op overlap |
| W2 MoE FFN | top-k small expert GEMMs plus gate and combine | expert-level parallelism and prefetch/compute overlap |
| W3 DiT/diffusion block | attention, MLP, timestep/modulation, elementwise ops | denoising-step and block-internal fusion |
| W4 multimodal pipeline | visual encoder, projector, text/LLM decode | module-level concurrency |
| W5 video KV retrieval | dynamic token/cluster selection plus light attention | irregular selection and memory movement overlap |

## Baselines

1. Serial framework execution: PyTorch eager or exported graph.
2. Compiler baseline: `torch.compile`/Inductor when available.
3. Multi-stream baseline: independent CUDA streams for legal subgraphs.
4. CUDA Graph baseline: capture/replay to reduce launch overhead.
5. Fused kernel baseline: Triton or custom CUDA for selected chains.
6. Micro-operator scheduler: tile/subgraph variants with runtime selection.

## Metrics

| Metric | Definition |
|---|---|
| Latency | End-to-end single-request wall time. |
| Speedup | `T_serial / T_method`. |
| Overlap ratio | `1 - T_method / sum(T_isolated_micro_ops)`. |
| Launch pressure | Number of GPU launches per request or per block. |
| Global memory traffic | HBM bytes read/write from profiler. |
| Unit utilization | SM, Tensor Core, LDST, SFU, memory-pipeline utilization. |
| Scheduler overhead | CPU/runtime overhead for choosing and launching work. |
| Numerical drift | Max/mean error versus serial baseline. |

## Tools

- Nsight Systems: timeline, launch gaps, stream overlap.
- Nsight Compute: HBM traffic, occupancy, Tensor Core/LDST utilization.
- PyTorch profiler: framework-level operator timeline.
- Triton profiler or generated IR inspection for custom kernels.
- Local notes for NPU/custom-accelerator comparisons when hardware is absent.

## First Inner-Loop Candidate

H1 experiment: construct a small set of resource-complementary kernels:

- compute-bound GEMM/attention tile,
- memory-bound layernorm or dequantization,
- elementwise activation,
- small reduction.

Run serial, multi-stream, CUDA Graph, and block/resource-shaped schedules. The
prediction is that multi-stream alone gives weak overlap, while resource-shaped
or fused/micro-tiled schedules improve overlap ratio and reduce latency.

