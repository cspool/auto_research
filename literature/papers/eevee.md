# Efficient Multimodal Serving via Module Multiplexing

Local note: `/data3/paper_analysis/idea_notes/Efficient Multimodal Serving via Module Multiplexing.md`

Relevance: module-level concurrency for multimodal inference.

Key method: split visual encoder, text encoder, and decoder into independent
processes with different batch sizes and SM allocations, using NVIDIA MPS and a
modal cache.

Environment: RTX 3090/A100, NVIDIA MPS.

Why it matters: multimodal models naturally expose module-level heterogeneity.
For single requests, the transferable idea is resource partitioning across
independent or pipeline-overlappable modules.

