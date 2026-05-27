# Latent Wavelet Diffusion

Local note: `/data3/paper_analysis/idea_notes/Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis.md`

Relevance: model-side wavelet saliency for diffusion.

Key method: scale-consistent VAE fine-tuning plus wavelet-masked flow matching
where high-frequency spatial regions receive more supervision.

Environment: A100 GPUs; no inference-time modification.

Why it matters: it does not solve inference concurrency directly. Its value is
as a possible signal for spatially selective micro-operator scheduling in
wavelet/diffusion models.

