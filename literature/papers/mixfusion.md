# MixFusion

Local note: `/data3/paper_analysis/idea_notes/MixFusion_ A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models.md`

Relevance: patch-level decomposition for diffusion workloads.

Key method: convert mixed-resolution images into uniform patches, use CSP patch
management, fuse patch-edge stitching into GroupNorm, apply patch-level caching,
and schedule under SLO constraints.

Environment: local note reports H100-80GB.

Why it matters: patching is a model-semantic decomposition unit. For single
requests, the interesting follow-up is whether one high-resolution request can
be split into patches to expose intra-request parallelism without excessive
boundary overhead.

