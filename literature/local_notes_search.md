# Local Notes Search Log

Date: 2026-05-25

Scope: local-notes only. The Obsidian MCP search API was not available in this
session, so the bootstrap used filesystem search over:

- `/data3/paper_analysis/paper_secs`
- `/data3/paper_analysis/knowledge_notes`
- `/data3/paper_analysis/experiment_notes`
- `/data3/paper_analysis/idea_notes`

## Query Segments

| Segment | Keywords | Role |
|---|---|---|
| S1 | CUDA stream, CUDA Graph, concurrent kernel, MPS, co-location, SM mask | GPU runtime concurrency mechanisms |
| S2 | micro operator, tile, multi-version kernel, operator fusion, kernel fusion | compiler and kernel generation methods |
| S3 | MoE, expert, expert offloading, grouped GEMM | sparse expert model bottlenecks |
| S4 | Diffusion, DiT, denoising, patch, wavelet | image/video generation model bottlenecks |
| S5 | multimodal, VLM, video LLM, visual token, KV retrieval | multimodal and video inference |
| S6 | NPU, accelerator, Tensor Core, SME, PIM, systolic array, chiplet | hardware architecture and simulator methods |

## Selected Local Evidence

| Topic | Local note path | Why it matters |
|---|---|---|
| Micro-operator compiler/scheduler | `/data3/paper_analysis/experiment_notes/编译实验笔记/Automated End-to-End Model Serving with Cooperative Compilation and Scheduling.md` | Most direct note for tile-based micro operators, multi-version micro-kernels, and runtime kernel scheduling/fusion. |
| Inter-SM kernel fusion | `/data3/paper_analysis/experiment_notes/编译实验笔记/FlashFuser_ Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Conn.md` | Shows how Hopper DSM and TMA expose inter-core communication for compute-intensive operator chains. |
| Non-intrusive kernel co-location | `/data3/paper_analysis/idea_notes/μShare_ Non-Intrusive Kernel Co-Locating on NVIDIA GPUs.md` | Directly studies concurrent kernels on NVIDIA GPUs through launch-parameter shaping. |
| Dynamic SM partitioning | `/data3/paper_analysis/idea_notes/Bullet- Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration.md` | Provides intra-GPU prefill/decode spatial-temporal orchestration using stream SM masks. |
| Attention-specific compiler | `/data3/paper_analysis/idea_notes/MetaAttention_ A Unified and Performant Attention Framework Across Hardware Backends.md` | Shows a semantics-aware compiler/runtime for attention variants and hardware backends. |
| Prefix-aware multi-tile attention | `/data3/paper_analysis/idea_notes/PAT_ Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel.md` | Provides a pack-forward-merge attention kernel with multi-tile runtime selection and multi-stream execution. |
| Sparse Transformer fusion | `/data3/paper_analysis/experiment_notes/编译实验笔记/Accelerating Sparse Transformer Inference on GPU (STOF).md` | Demonstrates Triton/TileLang template fusion for sparse Transformer downstream operators. |
| Diffusion compiler | `/data3/paper_analysis/experiment_notes/编译实验笔记/Difflow_ A Data-Characteristic-Aware Serving System for Diffusion Models.md` | Diffusion-specific dGraph/dEngine compiler with symbolic data-property propagation. |
| Mixed-resolution diffusion | `/data3/paper_analysis/idea_notes/MixFusion_ A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models.md` | Patch-level decomposition and fused patch stitching for diffusion serving. |
| Mixture-of-Diffusion serving | `/data3/paper_analysis/idea_notes/MoDM_ Efficient Serving for Image Generation via Mixture-of-Diffusion Models.md` | Useful for system-level routing and GPU resource allocation, less central for single-request micro-operator concurrency. |
| Wavelet Diffusion | `/data3/paper_analysis/idea_notes/Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis.md` | Model-side signal analysis; no inference speedup, but useful for wavelet/spatial saliency decomposition. |
| Multimodal module multiplexing | `/data3/paper_analysis/idea_notes/Efficient Multimodal Serving via Module Multiplexing.md` | Module-level GPU concurrency via MPS and per-module SM allocation. |
| VLM accelerator | `/data3/paper_analysis/idea_notes/Focus_ A Streaming Concentration Architecture for Efficient Vision-Language Models.md` | Hardware/software co-design for streaming visual-token compression on systolic-array accelerators. |
| Video LLM accelerator | `/data3/paper_analysis/experiment_notes/kernel实验笔记/V-Rex_ Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval.md` | Runtime pipeline and custom hardware blocks for irregular KV retrieval in video LLMs. |
| MoE offloading | `/data3/paper_analysis/idea_notes/Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading.md` | Fine-grained expert prefetch/cache scheduling; mostly multi-GPU/offload but relevant to MoE control/data overlap. |
| Fused quantized attention | `/data3/paper_analysis/idea_notes/JanusQuant_ Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference.md` | Strong example of fusing micro-ops to remove launch and memory round trips in single-GPU inference. |
| Custom AI accelerator compiler | `/data3/paper_analysis/experiment_notes/编译实验笔记/RPU - A Reasoning Processing Unit.md` | Shows static micro-kernel lowering to memory/compute/network instruction streams. |
| ARM SME microkernel | `/data3/paper_analysis/idea_notes/ASM-SpMM_ Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration.md` | Non-GPU backend example: SME outer-product microkernel, multi-tile concurrency, explicit prefetch. |

## Search Outcome

The local notes contain enough material for a first bootstrap. The strongest
coverage is on NVIDIA GPU systems and compiler/runtime methods. NPU/Ascend/CANN
coverage is currently weak in the selected evidence and remains an explicit
follow-up search target inside local-notes.

