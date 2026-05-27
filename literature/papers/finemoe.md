# FineMoE

Local note: `/data3/paper_analysis/idea_notes/Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading.md`

Relevance: MoE-specific overlap and cache scheduling.

Key method: iteration-level expert probability maps, semantic/trajectory search,
similarity-aware expert selection, async prefetch, and probability-aware
prefetch/eviction priority.

Environment: Mixtral-8x7B, RTX 3090/A100 setups in local note.

Why it matters: it is not a pure single-GPU method, but it identifies the right
MoE granularity: iteration, layer, expert probability, prefetch urgency, and
cache residency.

