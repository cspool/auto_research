# FlashFuser

Local note: `/data3/paper_analysis/experiment_notes/编译实验笔记/FlashFuser_ Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Conn.md`

Relevance: strong evidence for intra-kernel concurrency and inter-SM dataflow.

Key method: FlashFuser uses Hopper Distributed Shared Memory, TMA, and mbarrier
to fuse compute-intensive operator chains whose intermediates do not fit inside
one SM. A search engine enumerates loop schedule, tiling, and resource mapping,
then uses a dataflow analyzer and cost model to choose candidate kernels.

Environment: H100 SXM, CUDA 12.4, PyTorch 2.6, TVM 0.9, Triton 3.2.

Why it matters: it shows that single-request concurrency can be created inside a
fused kernel by using inter-core communication rather than relying on independent
kernel overlap.

