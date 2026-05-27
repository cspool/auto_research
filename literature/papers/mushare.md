# uShare / microShare

Local note: `/data3/paper_analysis/idea_notes/μShare_ Non-Intrusive Kernel Co-Locating on NVIDIA GPUs.md`

Relevance: direct evidence for non-intrusive concurrent kernels on NVIDIA GPUs.

Key method: intercept kernel launches and reshape block sizes so the closed GPU
scheduler scatters blocks across SMs instead of stacking same-kernel blocks. It
also delays launches to co-locate kernels with complementary hardware resource
profiles.

Environment: NVIDIA A40/A800-class GPUs, PyTorch 2.2.0, CUDA 11.8.

Why it matters: it explains why nominal CUDA concurrency often fails and gives a
practical mechanism for manipulating spatial sharing without modifying kernels
or GPU hardware.

