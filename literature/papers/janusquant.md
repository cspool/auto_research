# JanusQuant

Local note: `/data3/paper_analysis/idea_notes/JanusQuant_ Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference.md`

Relevance: fused micro-ops for long-context inference.

Key method: runtime smoothing, fast absmax positioning, ring-buffer token cache,
and mixed-precision attention. Dequantization is fused into attention and cache
management avoids repeated tensor concatenation.

Environment: A100-PCIE-40GB, PyTorch 2.4.0, CUDA 12.6.

Why it matters: it turns multiple small overhead-heavy kernels into fused
attention-side work, eliminating launch overhead and global memory round trips.

