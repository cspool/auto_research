# Bullet

Local note: `/data3/paper_analysis/idea_notes/Bullet- Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration.md`

Relevance: useful runtime pattern for spatial-temporal orchestration on a single
GPU.

Key method: split prefill and decode into separate engines, use dynamic SM
partitioning with stream masks, estimate interference through an SM-scaling
roofline model, and schedule prefill layers/decode steps under SLO constraints.

Environment: A100/H100/H20, SGLang v0.4.6, PyTorch 2.6.0, CUDA Graph for decode.

Why it matters: even though it targets serving, the mechanism is relevant to a
single backend: it treats subgraphs as concurrent engines and changes resource
partitioning at microsecond scale.

