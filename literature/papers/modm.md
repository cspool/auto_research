# MoDM

Local note: `/data3/paper_analysis/idea_notes/MoDM_ Efficient Serving for Image Generation via Mixture-of-Diffusion Models.md`

Relevance: system-level diffusion routing and resource allocation.

Key method: final-image cache, CLIP text-image retrieval, small-model refinement,
large-model fallback, and PID-driven GPU worker allocation.

Environment: A40 and MI210 GPUs.

Why it matters: less central for single-request micro-op concurrency, but useful
for understanding diffusion serving control planes and model-mixture scheduling.

