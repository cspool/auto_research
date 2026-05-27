# STOF

Local note: `/data3/paper_analysis/experiment_notes/编译实验笔记/Accelerating Sparse Transformer Inference on GPU (STOF).md`

Relevance: adaptive operator fusion for sparse Transformer inference.

Key method: encode fusion schemes as binary numerical expressions, map them to
Triton/TileLang templates, and search fusion boundaries plus kernel parameters
with performance caching.

Environment: RTX 4090, A100, CUDA 12.6, PyTorch 2.7.0.

Why it matters: STOF is useful for the compiler side of the project: template
fusion can be made searchable and graph-aware without hand-writing every fused
operator chain from scratch.

