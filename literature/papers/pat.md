# PAT

Local note: `/data3/paper_analysis/idea_notes/PAT_ Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel.md`

Relevance: resource-efficient attention kernel with runtime tile selection.

Key method: pack-forward-merge execution. A memory-centric scheduler builds a
prefix tree, packs queries sharing KV blocks, selects multi-tile kernel variants
at runtime, uses multi-stream forward execution, and merges partial attention
with online softmax.

Environment: A100-80GB, vLLM v0.9.0, FlashAttention/FlashInfer baselines.

Why it matters: it is a concrete template for single-request or small-batch
attention micro-operator design: expose reuse, select tile shapes dynamically,
and keep memory traffic as the primary objective.

