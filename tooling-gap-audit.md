# Tooling Status Audit

Date: 2026-05-27
Updated: 2026-05-27 after local GPU tooling installation

Scope: audited the current AutoResearch workspace: 19 experiment analyses, 23
experiment scripts, synthesis notes, literature setup, and the current host
tool/Python environment.

## Current Environment Snapshot

- Host GPU is available outside the default sandbox: two NVIDIA GeForce RTX
  4090 GPUs, driver 570.211.01.
- Default sandbox cannot access `/dev/nvidia*`; inside it, `nvidia-smi` fails
  and `torch.cuda.is_available()` is false.
- Python stack is usable for current Triton experiments:
  - Python 3.13.12
  - PyTorch 2.11.0+cu128
  - Triton 3.6.0
  - pandas, PyYAML, matplotlib, numpy, scipy, scikit-learn installed
- `git` is installed and `/data3/auto_research` is now a git repository.
- NVIDIA MPS commands are installed:
  - `nvidia-cuda-mps-control`
  - `nvidia-cuda-mps-server`
- Updated GPU tooling:
  - `nsys` is on `PATH`: NVIDIA Nsight Systems 2024.6.2.
  - `ncu` is installed but not on the current `PATH`:
    `/usr/local/cuda-12.8/bin/ncu`, NVIDIA Nsight Compute 2026.2.0.
  - CUDA Toolkit 12.8 developer binaries are installed under
    `/usr/local/cuda-12.8/bin`: `nvcc`, `cuobjdump`, `nvdisasm`, and
    `compute-sanitizer`.
  - `trtexec` is on `PATH`: TensorRT v11.0.0 CLI.
  - Plot/report packages are installed: plotly 6.7.0, seaborn 0.13.2,
    WeasyPrint 68.1, Playwright 1.60.0.
  - Literature API packages are installed: semanticscholar 0.12.0, arxiv 4.0.0.

Recommended shell update for interactive runs:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
```

For Codex-run experiments, prefer explicit tool paths such as
`/usr/local/cuda-12.8/bin/ncu` so results do not depend on shell startup files.

## High-Priority Tooling Status

### 1. NVIDIA Nsight Systems and Nsight Compute

Status:

- `nsys`: installed and on `PATH`.
- `ncu`: installed at `/usr/local/cuda-12.8/bin/ncu`, but not on current `PATH`.
- `nvprof`: still missing and not needed for new experiments; it is deprecated
  for modern CUDA profiling.

Compromise in experiments:

- H2.2 used Triton cache artifacts instead of Nsight Compute/System.
- H2/H3/H4 conclusions are latency-based, with no SM/SFU/DRAM/HBM traffic,
  achieved occupancy, Tensor Core, LDST, or launch-gap counters.
- H4 memory-locality and online-reduction explanations are currently inferred
  from latency and tile metadata.

Updated experiment recommendation:

- Add Nsight Systems timeline capture to H5.1/H5.2 to measure launch gaps,
  CPU submission overhead, stream overlap, and CUDA Graph behavior.
- Add Nsight Compute selected-kernel captures to H5.4 and follow-up H4 selector
  stress tests to measure occupancy, memory throughput, shared-memory limits,
  and achieved instruction-pipeline utilization.
- Use narrow metric sets first to keep profiling overhead manageable.

Verification:

```bash
nsys --version
/usr/local/cuda-12.8/bin/ncu --version
nvidia-smi
```

### 2. CUDA Toolkit Developer Binaries

Status:

- Installed under `/usr/local/cuda-12.8/bin`:
  - `nvcc` 12.8.93
  - `cuobjdump` 12.8.90
  - `nvdisasm`
  - `compute-sanitizer` 2025.1.0
- Not all of these are on the current `PATH`.

Compromise in experiments:

- No custom CUDA C++ kernels were built; experiments stayed in PyTorch/Triton.
- SASS/cubin inspection was not available; H2.2 used PTX/Triton JSON metadata.
- No compute-sanitizer checks for custom kernels.

Updated experiment recommendation:

- New custom CUDA baselines are now possible when Triton cannot express a
  runtime scheduling primitive cleanly.
- Use `cuobjdump`/`nvdisasm` to strengthen compiler-metadata analysis beyond PTX
  declarations.
- Use `compute-sanitizer` on new custom kernels and on any Triton-generated
  interop experiments that touch manual pointer arithmetic.

Verification:

```bash
/usr/local/cuda-12.8/bin/nvcc --version
/usr/local/cuda-12.8/bin/cuobjdump --version
/usr/local/cuda-12.8/bin/nvdisasm --version
/usr/local/cuda-12.8/bin/compute-sanitizer --version
```

### 3. Default Sandbox GPU Visibility

This is mostly a configuration issue, not a missing Python package.

Observed:

- Outside sandbox: `nvidia-smi` sees two RTX 4090 GPUs.
- Inside sandbox: `/dev/nvidia*` is absent, `nvidia-smi` cannot communicate with
  the driver, and PyTorch CUDA reports zero devices.

Compromise in experiments:

- GPU sweeps had to run via escalated execution.
- CPU/sandbox probe outputs were kept only as code-path checks, not evidence.

Fix recommendation:

- If running in a container, install/configure NVIDIA Container Toolkit and
  launch with GPU device access, e.g. `--gpus all`.
- If this is the Codex sandbox policy, keep using approved escalated commands
  for GPU experiments.

Verification:

```bash
ls -l /dev/nvidia*
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Medium-Priority Tooling Status

### 4. TensorRT CLI

Status:

- `trtexec` is installed at `/usr/bin/trtexec`.

Compromise:

- No TensorRT production-style baseline for fused attention/LLM inference.
- Current baselines are PyTorch eager, PyTorch flat materialization, and Triton.

Updated experiment recommendation:

- Add TensorRT only where an ONNX/exportable baseline is meaningful. It is less
  direct for the custom irregular sparse-attention fragments, but useful for
  dense operator chains and production-style comparison.

Verification:

```bash
trtexec --version
```

### 5. Report Rendering and Interactive Plotting

Status:

- Installed: `plotly`, `seaborn`, `weasyprint`, `playwright`.
- Available browser: `google-chrome`.
- `wkhtmltopdf` remains absent, but WeasyPrint and Playwright are enough.

Updated experiment/report recommendation:

- Future progress reports can include richer interactive/static plots.
- Use Playwright screenshot/PDF checks for important HTML reports.

```bash
python3 -c "import plotly, seaborn, playwright, weasyprint; print('ok')"
```

### 6. Literature API Packages

Status:

- Installed: `semanticscholar`, `arxiv`.

Updated recommendation:

- The initial literature policy remains local-notes-only because that was the
  user request. If the project later broadens, Semantic Scholar and arXiv API
  passes are now available.

```bash
python3 -c "import semanticscholar, arxiv; print('ok')"
```

## Hardware-Specific Optional Toolchains

### 7. Ascend/NPU Toolchain

Missing commands:

- `atc`
- `msprof`
- `acltracert`
- `ascend-smi`
- `omc`

Compromise:

- NPU/Ascend/CANN evidence remains literature/local-note based.
- No Ascend runtime, compiler, profiler, or device-counter experiments were run.

Install only if Ascend hardware is actually available:

- Ascend driver/firmware
- CANN Toolkit
- CANN NNRT/runtime
- profiling tools that provide `msprof` or the current replacement profiler
- `atc` compiler

Verification:

```bash
ascend-smi
atc --version
msprof --version
acltracert --help
```

### 8. ROCm/AMD Toolchain

Missing commands:

- `rocminfo`
- `rocprof`

Compromise:

- No AMD GPU comparison was run.

Install only if AMD GPU hardware is available:

- ROCm runtime/toolkit
- rocprofiler tools

Verification:

```bash
rocminfo
rocprof --version
```

## Not Missing, But Needs Process Cleanup

### Git Pre-Registration

Earlier protocols say `/data3/auto_research` was not a git repository. Current
check shows it is now a git repo and `git` is installed.

No install needed. For future experiments, commit protocol files before GPU
runs if pre-registration history matters.

### PyTorch Profiler and Inductor

Available through the current PyTorch install:

- `torch.profiler`
- TorchDynamo/Inductor

No install needed. These remain useful for framework-level operator timelines
and `torch.compile` baselines, but Nsight should now be preferred for
SM/HBM-level runtime-limit claims.

## Remaining Actions

1. Add `/usr/local/cuda-12.8/bin` to the persistent shell `PATH`, or keep using
   explicit paths in experiment scripts.
2. Configure default execution environment for GPU visibility, or keep using
   approved escalated GPU commands. The sandbox still cannot see `/dev/nvidia*`.
3. For H5, add profiler parameters:
   - `nsys profile --trace=cuda,nvtx,osrt`
   - `/usr/local/cuda-12.8/bin/ncu --set speed-of-light`
   - targeted Nsight Compute sections for memory, occupancy, scheduler, and
     launch statistics.
4. Ascend CANN or ROCm remain optional and hardware-dependent.
