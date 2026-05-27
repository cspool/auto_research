# Difflow

Local note: `/data3/paper_analysis/experiment_notes/编译实验笔记/Difflow_ A Data-Characteristic-Aware Serving System for Diffusion Models.md`

Relevance: diffusion-specific compiler decomposition.

Key method: symbolic data-property propagation, dGraph identification, selective
dEngine generation, redundancy elimination, ragged operation regularization, and
invariant tensor elimination.

Environment: A100 40GB PCIe, H100 80GB PCIe, CUDA 12.1/12.8, PyTorch 2.9 release.

Why it matters: diffusion pipelines have loop and data-property structure that a
generic operator graph hides. Difflow shows how semantic properties can decide
which subgraphs to specialize and where to remove redundant computation.

