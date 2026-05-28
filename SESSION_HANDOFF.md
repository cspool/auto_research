# AutoResearch Session Handoff

Date: 2026-05-28

## Resume Order

1. Read `research-state.yaml`.
2. Read `findings.md`.
3. Read `synthesis/runtime-resource-scheduling-direction.md`.
4. Read `docker/README.md` for the GPU container environment.
5. Continue next H5 dispatch-substrate experiment: CUDA Graph replay or persistent/batched queue dispatch.

## Current Research Direction

The project has pivoted from "make one kernel faster" to runtime
resource-aware scheduling for single-request inference on one accelerator.

H5.1, H5.2, H5.4, and H5.5 are now complete. The next phase is graph-bank switching or graph-level migration:

- keep H5.2 compile/cache overhead as the explanation for H5.1 queue noise;
- use H5.5 CUDA Graph evidence as the cheaper dispatch substrate;
- keep resource-aware variant selection, but treat H5.1 as evidence that
  dispatch overhead can hide selector wins;
- evaluate task migration with graph-level/batched units or first measure graph-bank
  switching overhead.

H4.10 value-dimension held-outs remain useful as a selector stress test, but
they are no longer the main story.

## H5.1 Result

H5.1 created `experiments/h5-1-runtime-selector/` and ran successfully on the
current GPU-visible container environment. Key result:

```text
static_best_average queue median: 2.1977 ms
resource_aware queue median:     2.2004 ms
static_best_isolated median:     2.2101 ms
policy repeat spread:            8.3% to 10.6%
first invocation median:         105.69 ms
first invocation max:            824.63 ms
```

Interpretation: calibrated best variants differ by resource state and task shape,
but the queue-level policy differences are smaller than timing jitter. Nsight
Systems smoke profiling showed CUDA launch/loading API time dominates the tiny
fused-kernel GPU work. H5.4 has now confirmed the remaining warmed-queue dispatch overhead, so the next step is graph/persistent dispatch rather than another selector rule.

Artifacts:

```text
experiments/h5-1-runtime-selector/protocol.md
experiments/h5-1-runtime-selector/analysis.md
experiments/h5-1-runtime-selector/code/run_runtime_selector.py
experiments/h5-1-runtime-selector/results/rtx4090_default/
experiments/h5-1-runtime-selector/results/nsys_h5_1_tooling_smoke.nsys-rep
experiments/h5-1-runtime-selector/results/nsys_h5_1_tooling_smoke.sqlite
data/h5_1_runtime_selector_*.csv
to_human/2026-05-28-h5-1-runtime-selector.html
```


## H5.2 Result

H5.2 created `experiments/h5-2-compile-cache-overhead/` and measured Triton
cold compile/cache-hit/steady-state overhead for five representative H5.1
variants. Key result:

```text
cold first launch median:       1391.29 ms
cache-hit first launch median:   661.03 ms
steady-state event median:         0.1439 ms
cold overhead ratio median:      9258.75x
cache-hit overhead ratio median: 4637.46x
cold process wall median:       4126.22 ms
cache-hit process wall median:  3383.79 ms
```

Interpretation: dynamic compilation is impossible on the critical path for this
micro-task scale, and a cache hit in a fresh process is still far from free. The
runtime needs ahead-of-time cache warming, a persistent process with modules
loaded, async compilation behind a fallback variant, larger migration-safe tiles,
or CUDA Graph/persistent-kernel/batched dispatch.

Artifacts:

```text
experiments/h5-2-compile-cache-overhead/protocol.md
experiments/h5-2-compile-cache-overhead/analysis.md
experiments/h5-2-compile-cache-overhead/code/run_compile_cache_overhead.py
experiments/h5-2-compile-cache-overhead/results/rtx4090_default/
data/h5_2_compile_cache_*.csv
to_human/2026-05-28-h5-2-compile-cache-overhead.html
```


## H5.4 Result

H5.4 created `experiments/h5-4-runtime-limit-profiling/` and profiled a warmed
persistent process. Cold compile and fresh-process cache lookup were kept out of
the measured/profiler windows. Key result:

```text
static_best_average event median: 2.2662 ms
resource_aware event median:     2.2338 ms
resource_aware speedup:          1.0145x
static policy spread:            6.85%
resource policy spread:          9.17%
CPU enqueue window:              136.16 ms for 96 queues
CPU enqueue per queue:           1.418 ms
non-sync CUDA API time:          24.77 ms
stream/event management:         16.21 ms
launch wrapper time:              8.56 ms across 576 launches
GPU kernel span:                 206.92 ms
GPU kernel total:                241.89 ms
```

Interpretation: warming fixes the extreme H5.2 cold/cache overhead, but the
Python/PyTorch stream queue still spends enough time on event/wait/launch
plumbing that sub-2% selector wins are fragile. The next step should test CUDA
Graph replay or persistent/batched dispatch before H5.3 task migration.

Artifacts:

```text
experiments/h5-4-runtime-limit-profiling/protocol.md
experiments/h5-4-runtime-limit-profiling/analysis.md
experiments/h5-4-runtime-limit-profiling/code/run_warmed_queue_profile.py
experiments/h5-4-runtime-limit-profiling/results/rtx4090_default/
experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.nsys-rep
experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.sqlite
data/h5_4_warmed_queue_*.csv
to_human/2026-05-28-h5-4-warmed-queue-runtime-limits.html
```


## H5.5 Result

H5.5 created `experiments/h5-5-cuda-graph-dispatch/` and tested CUDA Graph replay
for the fixed H5 queue. Key result:

```text
static python event median:      2.2421 ms
static graph event median:       2.1368 ms
static graph speedup:            1.0493x
resource python event median:    2.2110 ms
resource graph event median:     2.1418 ms
resource graph speedup:          1.0323x
CUDA runtime API calls:          194 for 96 graph queues
cudaGraphLaunch calls:           96
cudaLaunchKernel calls:          0
CPU enqueue window:              4.44 ms for 96 graph queues
CPU enqueue per graph queue:     0.046 ms
H5.4 stream enqueue per queue:   1.418 ms
```

Interpretation: CUDA Graph replay does not remove GPU work, so end-to-end event
speedups are modest. But it strongly fixes host dispatch: about 30.6x lower CPU
enqueue window per queue and 77x fewer CUDA runtime API calls than H5.4's warmed
Python/PyTorch stream queue profile. Next step should be graph-bank switching or
H5.3 migration with graph-level/batched units.

Artifacts:

```text
experiments/h5-5-cuda-graph-dispatch/protocol.md
experiments/h5-5-cuda-graph-dispatch/analysis.md
experiments/h5-5-cuda-graph-dispatch/code/run_cuda_graph_dispatch.py
experiments/h5-5-cuda-graph-dispatch/results/rtx4090_default/
experiments/h5-5-cuda-graph-dispatch/results/nsys_h5_5_graph_replay.nsys-rep
experiments/h5-5-cuda-graph-dispatch/results/nsys_h5_5_graph_replay.sqlite
data/h5_5_cuda_graph_dispatch_*.csv
to_human/2026-05-28-h5-5-cuda-graph-dispatch.html
```

## Local Tooling Activation

The current container user cannot run root `apt`, and `.bashrc` is read-only, so
missing tools were installed persistently under `/home/descfly/.local`: Python
3.10 dev headers for Triton helper compilation, `column`, and wrappers for
`nsys` and `ncu`. Activate them in every new shell/command before Triton or
profiler work:

```bash
source /home/descfly/.local/devtools/activate-auto-research.sh
```

This is no longer using `/tmp` for headers.

## Docker GPU Environment

The GPU Docker image has been built and smoke-tested.

```text
image: auto-research-gpu:cu128
image id: 292c3482f072
size: 17.6GB
docker data-root: /data3/docker
base image: nvidia/cuda:12.8.0-devel-ubuntu22.04
```

Important package versions:

```text
node v20.20.2
codex-cli 0.134.0
torch 2.11.0+cu128
triton 3.6.0
```

The container currently running for development:

```text
name: auto-research-gpu-dev
network: host
ipc: host
project bind mount: /data3/auto_research -> /workspace/auto_research
cache bind mount: /data3/auto_research_docker_cache
codex home bind mount: /data3/auto_research_codex_home
```

Verified smoke test:

```text
torch 2.11.0+cu128
cuda_available True
device_count 2
device0 NVIDIA GeForce RTX 4090
matmul_sum 1073741824.0
```

## How To Continue Experiments

Recommended short-term path: keep the existing outer VS Code/Codex session and
run all experiments through the container with `docker exec`.

Examples:

```bash
docker exec auto-research-gpu-dev python3 experiments/some-experiment/code/run.py
docker exec auto-research-gpu-dev nsys profile --trace=cuda,nvtx,osrt --stats=true ...
docker exec auto-research-gpu-dev /usr/local/cuda-12.8/bin/ncu --set speed-of-light ...
```

For an interactive shell:

```bash
docker exec -it auto-research-gpu-dev bash
cd /workspace/auto_research
```

If the container is not running, start it with:

```bash
bash docker/run_auto_research_gpu.sh
```

That helper is interactive and uses:

```text
--gpus all
--network host
--ipc host
--ulimit memlock=-1
--ulimit stack=67108864
```

For a detached long-lived container, use the explicit command recorded in the
chat history or recreate the existing pattern with `sleep infinity`.

## Agent Placement

There are two valid continuation modes:

1. Outer agent, container experiments:
   - current VS Code Codex session remains outside the container;
   - every GPU/profiler experiment runs inside the container via `docker exec`;
   - this preserves the current conversation context.

2. Agent inside container:
   - attach VS Code to `auto-research-gpu-dev`, open `/workspace/auto_research`;
   - run `codex -C /workspace/auto_research` inside the container;
   - the agent and experiments both run inside Docker;
   - first run may require Codex login/config in `/data3/auto_research_codex_home`.

Claude CLI is not installed in the image yet. Codex CLI is installed.

## Files Changed For Containerization

- `docker/Dockerfile.gpu`
- `docker/run_auto_research_gpu.sh`
- `docker/README.md`
- `research-state.yaml`
- `research-log.md`
- `SESSION_HANDOFF.md`

## Next Suggested Experiment

Create the next H5 graph-bank/migration experiment:

- reuse H5.1 Triton variants;
- reuse H5.5's graph replay code and H5.1 policy selections;
- build a small graph bank for static/resource-aware/alternate variants and
  measure graph switching overhead, or implement H5.3 migration at graph-level
  task boundaries;
- keep dynamic compilation and fresh-process cache lookup outside the measured
  request path;
- compare against H5.5 graph replay and H5.4 Python/PyTorch stream queue.

