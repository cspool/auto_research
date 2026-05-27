# MetaAttention

Local note: `/data3/paper_analysis/idea_notes/MetaAttention_ A Unified and Performant Attention Framework Across Hardware Backends.md`

Relevance: attention-specific compiler/runtime for modern model variants.

Key method: represent attention as relevance scoring plus aggregation, with
customizable functions and intermediate-tensor scheduling. It supports parallel
and recurrent patterns, online normalization, chunk parallelism, and backend
device configs.

Environment: H100 SXM5 with CUDA 12.4; also notes AMD MI250 backend coverage.

Why it matters: generic operator fusion misses attention semantics. This is a
clear example where domain-specific IR exposes concurrency and fusion legality
that general compilers do not infer.

