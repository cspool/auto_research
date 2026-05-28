# H5.4 Protocol - Warmed Persistent Queue Runtime Limits

Date: 2026-05-28

## Hypothesis

After H5.2 removes cold compile and fresh-process cache lookup from the
critical path, the remaining runtime limit for fine-grained single-request
micro-op scheduling is CPU/CUDA submission overhead and launch-gap structure,
not the fused Triton kernel time itself.

## Prediction

In a warmed persistent process:

- H5.1 queue latency will remain in the millisecond range even though each fused
  micro-op kernel is only sub-millisecond;
- Nsight Systems will show many small CUDA API launches and stream dependency
  gaps inside the queue window;
- resource-aware and static-average policy windows will remain close unless the
  dispatch substrate is changed or task granularity increases.

## Method

Reuse the H5.1 fused GELU/residual Triton micro-op, task queue, and policy
selections. Do not recalibrate variants inside H5.4. Load policy selections from
`experiments/h5-1-runtime-selector/results/rtx4090_default/result.json`.

Before the profiled window:

1. allocate persistent tensors and streams;
2. launch every variant required by the selected policies on both small and
   large task shapes;
3. run background compute/memory operators;
4. run several queue warmups for each policy;
5. synchronize.

Inside the profiled window:

1. call `cudaProfilerStart()`;
2. emit NVTX ranges for the H5.4 profile window, each policy, and each queue
   repeat;
3. run warmed queues without per-repeat event timing or host synchronization;
4. synchronize once at the end and call `cudaProfilerStop()`.

CUDA event and wall timing are recorded outside the profiler capture window so
that measurement synchronizations do not dominate the Nsight trace.

Use Nsight Systems with CUDA profiler API capture range:

```bash
nsys profile --trace=cuda,nvtx,osrt   --capture-range=cudaProfilerApi --capture-range-end=stop   --export=sqlite --force-overwrite=true   --output experiments/h5-4-runtime-limit-profiling/results/h5_4_warmed_queue_profile   python3 experiments/h5-4-runtime-limit-profiling/code/run_warmed_queue_profile.py --profile-capture
```

## Metrics

- warmed queue event latency per policy;
- warmed queue wall latency per policy;
- number and total duration of CUDA API launch calls inside the capture;
- number and total duration of GPU kernels inside the capture;
- GPU fused micro-op time versus CPU/API submission time;
- qualitative NVTX timeline structure for policy and step windows.

## Decision Rule

If CPU/API launch time and gaps remain large relative to GPU kernel time, H5.3
migration should not be implemented as another Python/CUDA-stream queue. It
should use larger task tiles, CUDA Graph replay, persistent kernel/batched
dispatch, or a lower-level runtime substrate.
