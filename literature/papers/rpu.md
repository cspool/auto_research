# RPU

Local note: `/data3/paper_analysis/experiment_notes/编译实验笔记/RPU - A Reasoning Processing Unit.md`

Relevance: custom accelerator compiler and static micro-kernel scheduling.

Key method: trace PyTorch graphs into a CISC-style RPU ISA. Linear layers are
lowered into Loading, Looping, and Launching micro-kernels. The compiler statics
orders DMA and compute instructions and embeds pipeline-arbiter flags.

Environment: custom RPU hardware model; RTL verification projected from TSMC N16
to N2 in local note.

Why it matters: it provides an accelerator-side contrast to GPU kernel launches:
single-request execution can become autonomous instruction-stream execution with
memory/compute/network pipelines synchronized by the compiler.

