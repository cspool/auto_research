# Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

Local note: `/data3/paper_analysis/experiment_notes/编译实验笔记/Automated End-to-End Model Serving with Cooperative Compilation and Scheduling.md`

Relevance: central evidence for compiler/runtime co-design with micro operators.

Key method: Infera uses TVM 0.16.0 to partition operators into tiles/micro
operators, merge small operators into shepherd operators, generate multi-version
micro-kernels, and schedule/fuse kernels at runtime. It also applies warp
specialization and cut-and-patch SASS instruction scheduling.

Environment: A100-PCIE-40GB, CUDA 12.0, Linux 6.1.0.

Why it matters: it directly supports the project's core thesis that concurrency
must be exposed below framework operator granularity and paired with
scheduler-friendly kernel variants.

