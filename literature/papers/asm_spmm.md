# ASM-SpMM

Local note: `/data3/paper_analysis/idea_notes/ASM-SpMM_ Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration.md`

Relevance: non-GPU backend microkernel concurrency.

Key method: OP-MCF sparse format, ARM SME outer-product microkernel, multi-tile
concurrency, explicit prefetch, SVE/Neon fallback for low-density blocks, and
heterogeneous-core work stealing.

Environment: Apple M4 and other ARM SME platforms in local note.

Why it matters: it broadens the project beyond CUDA: backend-specific
instructions and register files can create useful micro-op concurrency when the
compiler/runtime exposes the right format and mapping.

