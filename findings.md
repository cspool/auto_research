# Findings — Multi-Operator Concurrency for Single-Accelerator Inference

## Current Understanding
Local-notes bootstrap is complete. The strongest pattern is vertical co-design:
single-request operator concurrency is rarely obtained by simply launching
framework operators in multiple streams. Useful concurrency appears when the
system exposes smaller scheduling units and then controls where they execute:
tiles, micro-operators, fused operator chains, attention CTAs, diffusion patches,
multimodal modules, MoE expert events, or accelerator pipeline stages.

The core mechanism stack is:
1. identify a semantic decomposition unit;
2. generate resource-diverse kernels or hardware micro-ops;
3. schedule/fuse using hardware metadata;
4. verify that launch overhead, global-memory traffic, and resource contention
   do not erase the concurrency benefit.

The research emphasis is now sharper: single-kernel acceleration is not the
main remaining opportunity. The more important target is runtime support for
resource-aware task orchestration: dynamically selecting or compiling
operator/micro-operator variants, observing available accelerator resources,
managing dependency-aware task queues, and allowing remaining work to migrate at
tile or micro-op boundaries. The existing H2-H4 results should be treated as
building blocks for this runtime rather than as isolated kernel wins.

H5.1 tested that runtime direction directly. It produced a useful mixed result:
a context-aware table did recover different best Triton fused micro-op variants
for different load and task-shape contexts, but the Python/CUDA-stream queue did
not show a robust end-to-end win. Static-best-average, resource-aware,
load-aware, and static-isolated policies all landed within about 0.6% median
queue latency, while repeated queue timings varied by 8.3%-10.6%. A lightweight
Nsight Systems profile confirmed the mechanism: launch and dynamic loading API
time dominates the tiny fused-kernel GPU work. This moves H5 from "choose the
right table entry" toward "make dispatch, cache state, and resource observation
cheap enough for the table to matter."

H1 now has preliminary experimental support on an RTX 4090. A microbenchmark
with independent compute, elementwise/memory, and reduction operators showed
that naive multi-stream execution produced only -0.3% to +4.3% serial speedup
across three shapes. The best overlap ratio was about 6.0%, and two shapes were
essentially neutral. This supports the view that stream concurrency alone is not
the right abstraction for meaningful single-request operator concurrency.

H1.1 tested a simple resource-shaping idea: split the dominant GEMM into PyTorch
row chunks and run that chunk sequence beside independent elementwise/reduction
streams. This did not robustly improve absolute latency. The best relative
chunked stream speedup was 1.1147x on the memory-heavy shape, but GEMM chunk
overhead was 4.89x and absolute latency remained slower than full multi-stream.
The balanced shape had a faster 4-chunk absolute result, but its stream speedup
was only 1.0057x, suggesting the win came from GEMM shape/library selection, not
true cross-operator overlap.

H2 has preliminary support from a Triton variant scheduling experiment. A simple
memory-bound vector-add micro-kernel was generated with multiple `BLOCK_SIZE`
and `num_warps` variants and run beside a GEMM. In both balanced and
memory-heavy shapes, the fastest isolated variant was not the lowest-latency
concurrent variant. For the balanced shape, the fastest isolated variant
(`block=4096, warps=4`) had concurrent latency 0.1949 ms, while the best
concurrent variant (`block=2048, warps=8`) reached 0.1864 ms and increased
overlap ratio from 0.0917 to 0.1652.

H2.1 strengthens this result with a model-shaped fused micro-kernel:
`Y = GELU(X * scale + Bias) + Residual`. Triton fusion beat eager PyTorch
unfused execution by about 1.91x on the balanced shape and 2.61x on the
memory-heavy shape. The balanced shape again separated fastest isolated variant
from best concurrent variant: isolated best was `block=2048, warps=8`, but
concurrent best was `block=2048, warps=4` (0.2271 ms concurrent latency). The
memory-heavy shape added a metric warning: the best stream-speedup variant
reported 1.1427x but had worse absolute concurrent latency than the best
concurrent-latency variant.

H2.2 links the latency result to a compiler/runtime implementation path. Triton
cache artifacts for the fused micro-kernel exposed `block_size`, `num_warps`,
PTX `.reqntid`, static shared memory, PTX register declarations, and PTX/cubin
size for all 29 compiled variants. This does not replace Nsight counters, but it
shows how a compiler can emit the static half of a scheduler variant table that
is then joined with empirical concurrent-latency measurements.

Outer-loop synthesis after five experiments reframes the project direction:
the next step should test model-semantic decomposition, not more synthetic
stream overlap. H3.1 should use a tiny single-request MoE fragment because MoE
experts are explicit semantic units and can reveal whether the current
micro-kernel/metadata story survives a newer-model workload.

H3.1 tested that pivot and produced a useful negative result. A tiny
single-request MoE FFN fragment with four experts was run with pre-materialized
per-expert token tensors. Naive PyTorch one-stream-per-expert execution was
slower than serial expert execution for balanced (`0.6596x`), skewed (`0.6795x`),
and tiny (`0.6818x`) token distributions. This refines H3: model semantics
identify useful decomposition units, but those units must be lowered into
compiler-visible micro-operators, grouped kernels, or spatially controlled
execution units before they become latency wins.

H3.2 supports this refined H3. A Triton MoE expert epilogue benchmark compared
PyTorch expert loops, per-expert Triton serial launches, per-expert Triton
streams, and one grouped Triton kernel over all expert segments. Grouped Triton
won decisively across balanced, skewed, and tiny token distributions: 14.15x to
18.98x faster than PyTorch expert epilogue loops and 16.37x to 17.72x faster
than per-expert Triton stream launches. This shows that some model-semantic
parallelism is best exploited by collapsing semantic fragments into one
compiler-visible grouped kernel rather than by launching many concurrent
kernels.

H3.3 moved from epilogue work to actual expert FFN compute using padded
strided-batched GEMM (`torch.bmm`) as a grouped compute baseline. Grouped GEMM
was positive but modest: 1.23x speedup for balanced `64,64,64,64`, 1.08x for
skewed `160,64,24,8`, and 1.39x for tiny `16,16,16,16`. The skewed case had
2.5x padding overhead, explaining why standard padded grouped GEMM is only a
baseline. The next MoE step needs irregular grouped/tiled expert matmul that
avoids padding waste.

H3.4 implemented that no-padding direction with a simple Triton row-tile map
over valid expert token segments. It beat both expert loops and padded bmm in
all three distributions. The best no-padding Triton variant reached 1.28x over
expert loop and 1.12x over padded bmm for balanced, 1.48x and 1.31x for skewed,
and 1.39x and 1.12x for tiny. This is the strongest MoE compute result so far:
semantic segments should be kept, but execution should schedule compact valid
tiles instead of independent expert kernels or padded dense batches.

## Taxonomy Framework
The analysis is organized along four axes:

### Axis 1: Concurrency Granularity
1. **Kernel-level** — Individual CUDA/HIP kernel launch overlap
2. **Operator-level** — Framework operator (conv, matmul, attention) concurrency
3. **Subgraph-level** — Fused operator groups executed concurrently
4. **Micro-operator level** — Fine-grained fused primitives (element-wise, reduce)

### Axis 2: Mechanism
1. **Stream concurrency** — CUDA streams, HIP streams, SYCL queues
2. **Graph capture & replay** — CUDA Graphs, XPU Graphs
3. **Compiler fusion** — TVM, XLA, MLIR, Triton automatic fusion
4. **Hardware MPS** — NVIDIA Multi-Process Service, AMD CU masking
5. **Async execution** — Async memory copies, compute/prefetch overlap

### Axis 3: Hardware Architecture
1. **NVIDIA GPU** (CUDA, TensorRT, Triton)
2. **Ascend NPU** (CANN, TBE, Ascend Graph)
3. **AMD GPU** (ROCm, MIGraphX)
4. **TPU/Google** (XLA, SPMD)
5. **Generic/Cross-platform** (MLIR, IREE, OpenCL/SYCL)

### Axis 4: Model Archetype
1. **MoE** — Expert dispatch concurrency, all-to-all overlap
2. **Diffusion/DiT** — Denoising step fusion, attention+MLP overlap
3. **Multimodal** — Multi-encoder concurrent execution
4. **Video** — Spatial-temporal operator fusion

## Open Questions
- What is the theoretical maximum concurrency achievable under single-request constraints?
- How do NPU architectures differ from GPU in operator concurrency support?
- Can compiler-based fusion fully replace hand-written fused kernels for newer model types?
- What profiling tools exist for measuring operator-level concurrency efficiency?

## Patterns and Insights

- Infera is the most direct anchor for this project: split operators into
  micro-operators, generate multi-version kernels, and schedule/fuse at runtime.
- FlashFuser and JanusQuant show that many wins are intra-kernel: fuse away
  global memory round trips and separate launch boundaries.
- μShare and Bullet show that GPU spatial sharing needs explicit control:
  block-size shaping, launch timing, MPS, or stream SM masks.
- MetaAttention suggests that semantics-aware compiler IR is essential for
  newer attention variants because generic graph fusion does not understand
  online softmax, recurrent state, MLA shape asymmetry, or chunk parallelism.
- Difflow and MixFusion show that diffusion/DiT workloads should be decomposed
  by data property, dGraph, patch, or denoising-step structure rather than only
  by framework operator.
- Focus, V-Rex, RPU, and ASM-SpMM show how custom accelerators/NPU-like
  backends express micro-op concurrency through memory-interface units,
  pipeline instruction streams, and backend-specific intrinsics.
- The first GPU microbenchmark matches the literature pattern: independent
  streams do not automatically create strong spatial sharing. Better next steps
  are resource-shaped co-location, explicit SM partitioning, or intra-kernel
  fusion.
- H1.1 shows that resource shaping has to be lower-level than naive PyTorch graph
  chunking. Chunking can expose overlap but also changes GEMM kernel choice and
  adds launches, so the net effect is unstable. The next serious path should be
  compiler/kernel-controlled micro-kernels or explicit spatial partitioning.
- H2's first positive signal is about selection criteria: fastest standalone
  micro-kernel is not necessarily the best concurrent micro-kernel. Runtime
  schedulers need per-variant metadata for the actual objective: isolated
  latency, concurrent latency, overlap, or speedup.
- H2.1 shows that fusion and co-scheduling are complementary. Fusion shrinks
  the micro-op itself, but a fused kernel still needs resource-aware variant
  selection when it is co-located with a dominant operator.
- H2.2 shows a concrete compiler-framework implementation route: Triton JSON,
  TTGIR, and PTX artifacts can be parsed into variant metadata. Static metadata
  alone cannot predict performance, but it can seed a runtime table for
  empirical or profiler-guided scheduling.
- H3.1 shows that semantic branches are not enough by themselves. MoE experts
  should be treated as a source of compiler-visible micro-op events or grouped
  kernel tiles, not merely as a Python/framework loop to put on streams.
- H3.2 gives the positive counterpart: grouped expert epilogue micro-operators
  can turn MoE semantic structure into latency wins. This shifts the target from
  "operator concurrency" toward "semantic grouping plus compiler-visible
  scheduling units."
- H3.3 shows that grouped expert compute is useful but harder than grouped
  epilogues. Standard padded batched GEMM gives only modest speedups and is
  vulnerable to routing skew, so the interesting implementation target is
  irregular no-padding grouped/tiled matmul.
- H3.4 confirms the no-padding target. A simple row-tile map already beats
  padded bmm, especially under skew, so the next MoE system should preserve
  expert semantic boundaries while scheduling only valid tiles.
- H3.5 confirms that the no-padding MoE target survives routing movement. A
  routed Triton fragment with explicit scatter, grouped no-padding FFN, and
  gather beat the PyTorch routed expert loop by 1.39x to 1.64x. Scatter/gather
  cost about 0.037-0.042 ms, or 21%-23% of the full routed Triton latency, so
  movement must be a first-class micro-op rather than an implementation detail.
- The MoE branch now has a stable vertical implementation pattern: semantic
  routing segments, row-tile maps over valid token ranges, grouped no-padding
  kernels, explicit movement micro-ops, and full-fragment latency as the
  scheduler objective.
- H4.1 transfers the MoE lesson to Video/VLM KV retrieval. Per-frame/per-segment
  PyTorch retrieval loops are untenable for single-request sparse selection:
  a 32-segment loop was 52x to 64x slower than a fused Triton gather-score
  kernel. Flat PyTorch gather+GEMM collapses launch pressure, but selected-K
  materialization still consumes 43%-46% of flat latency; fused Triton removes
  that intermediate and beats flat PyTorch by 1.15x to 1.35x.
- H4 is refined rather than fully proven. H4.1 does not show that custom
  hardware is strictly necessary for sparse retrieval scores; it shows that the
  runtime/compiler must expose selected indices, segment boundaries, locality
  order, and movement-compute fusion. The hardware co-design claim should be
  retested after adding value aggregation, larger memory hierarchy effects, and
  locality-sensitive index ordering.
- H4.2 shows that locality/order metadata must be paired with kernel-variant
  metadata. Sorting selected KV indices reduced mean neighbor span from 90.6 to
  8.0 and shuffling increased it to 2673.7, but best fused Triton latency stayed
  within about 2.1% across orders. The best tile changed by order, so the runtime
  contract should join selected-token order statistics with Triton tile/warp
  variants and empirical full-fragment latency.
- H4.3 strengthens the sparse attention branch. Extending KV retrieval from
  score-only to score-softmax-value changed the fused-vs-flat win from roughly
  1.x to 5.78x-6.79x. PyTorch eager spends about 0.060-0.064 ms in
  score+softmax and 0.030-0.033 ms in value aggregation after K/V gather; a
  single-tile Triton fused kernel keeps these stages inside one compiler-visible
  unit and avoids selected-K/V materialization.
- Outer Loop 3 consolidates H4: irregular Video/VLM retrieval should be lowered
  to fused sparse attention fragments, not executed as per-frame/per-cluster
  framework work. The compiler/runtime contract is selected-token set,
  order/locality statistics, K/V layout, multi-version fused kernels, and
  empirical full-fragment latency. The immediate boundary is no longer whether
  fusion helps, but whether the single-tile result extends to multi-block
  online softmax for larger selected-token counts.
- H4.4 answers the Outer Loop 3 boundary positively. A two-stage multi-block
  online-softmax Triton prototype scales sparse score-softmax-value to 1024
  selected tokens and still beats flat PyTorch eager by 2.99x to 4.76x. The
  speedup is lower than H4.3's 5.78x-6.79x because the implementation pays for
  an extra reduce launch and partial buffers, but the GPU compiler/runtime path
  remains viable beyond the single-tile toy case.
- H4.5 extends the sparse attention branch from one query to four query vectors
  sharing the same selected K/V set. Grouped-query Triton beats flat PyTorch by
  2.86x to 3.64x and beats measured per-query repeated online-softmax kernels by
  3.41x to 3.66x. This shows the compiler-visible unit should include
  query/head grouping, not just a good single-query sparse kernel. It also adds
  a hard resource constraint: `N256 D64 V64 Q4 W4` failed on RTX 4090 because
  the fp32 grouped kernel required 133120 bytes of shared memory against a
  101376-byte hardware limit.
- Outer Loop 4 consolidates H4 into a compiler/runtime contract: sparse
  Video/VLM retrieval should be lowered as a grouped sparse attention fragment
  with selected-token metadata, order statistics, online-softmax/value
  reduction, query/head grouping, static resource feasibility, multi-version
  tile variants, and empirical full-fragment latency. The next step should
  demonstrate variant-table selection rather than only another hand-picked
  kernel speedup.
- H4.6 demonstrates that runtime-table layer on H4.5 data. A simple static
  proxy `BLOCK_N * BLOCK_Q * (BLOCK_D + BLOCK_V)` rejected all observed
  shared-memory OOR rows. A global mean latency table selected within 10% of
  measured best in 3/4 order modes, while an order-aware rule table selected the
  measured best in all four modes. The caveat is important: the rule table is
  retrospective and needs held-out shapes before becoming a stronger compiler
  selection claim.
- H4.7 tests that caveat with held-out query counts `Q=2` and `Q=8`. Static
  filtering still rejects all observed OOR rows, and the H4.6 order-aware table
  stays within 15% of measured best in 7/8 cases with lower mean regret than the
  global selector. The miss, `Q=8 clustered_segment`, shows that query-count
  pressure changes the best value-blocking/warps choice; a compiler/runtime
  table needs query-count and query-block features in addition to order stats.
- H4.8 adds those query-count features retrospectively. With `query_count`,
  `num_q_blocks`, and `query_count / BLOCK_Q`, the selector fixes the
  `Q=8 clustered_segment` miss and selects measured-best valid variants in all
  8/8 H4.7 held-out query-count cases. This supports the feature diagnosis, but
  not broad generalization yet: the next test must hold out selected-token count
  or value dimension.
- H4.9 provides that selected-token-count hold-out. At `512` and `2048`
  selected tokens with `Q=4`, static feasibility rejected all 8/8 observed OOR
  rows and the H4.8 order/query selector stayed within 15% of measured best in
  8/8 cases. Mean regret was 4.66% and max regret was 9.92%. A diagnostic
  selected-token pressure oracle selected exact best in 8/8, so selected-token
  count is useful for exact tuning but not yet a near-best failure mode.
- Outer Loop 5 consolidates H4 into a runtime-selector method. The current
  compiler/runtime contract has static shared-memory feasibility, selected-token
  order statistics, query-pressure features, shape metadata, and measured
  full-fragment latency. The next best stress test is value dimension because it
  changes `num_v_blocks`, partial-buffer traffic, and `BLOCK_V` choice more
  directly than selected-token count did.
- A user-guided direction update reframes the next phase as H5: runtime
  resource-aware task management. The key system question is how to schedule
  different operators or micro-operators according to idle resources when those
  tasks may be dynamically compiled, queued, and migrated at runtime. This
  shifts the central contribution from single-kernel speedups toward software
  and hardware runtime support: resource visibility, task queues, dependency
  tracking, variant caches, task migration boundaries, and backend controls such
  as streams, graphs, MPS/SM partitioning, persistent queues, or NPU pipeline
  ports.

- H5.1 shows the first runtime-selector boundary. Calibrated context matters:
  best variants changed across compute, memory, idle, small, and large tasks,
  with 11.8%-74.4% worst/best spread in the calibration table. But end-to-end
  Python-level queue policies differed by less than the 8.3%-10.6% repeated
  timing spread, and Nsight Systems showed launch/loading API time dominating
  the tiny GPU kernels. The next H5 experiments must measure and reduce runtime
  overhead, not just improve selector rules.

- H5.2 strongly confirms the overhead diagnosis. Across five representative
  H5.1 Triton variants, cold first launch took 1.32-1.41 s, cache-hit first
  launch in a fresh worker still took 0.61-0.74 s, and warmed steady-state
  event latency was only 0.142-0.156 ms. That puts cold and cache-hit first
  launches around 9,000x and 4,600x above the warmed micro-op path, so per-
  request compilation and fresh-process cache hits are not viable scheduling
  mechanisms for fine-grained single-request micro-operators.

- H5.4 removes cold-start noise and exposes the next boundary. In a warmed
  persistent process, `resource_aware` was 1.0145x faster than
  `static_best_average` by median event timing, but min-to-max spread was still
  6.85%-9.17%. Nsight Systems showed the Python/PyTorch stream queue enqueued
  each six-step queue in about 1.418 ms, with 16.21 ms of stream/event
  management and 8.56 ms of launch-wrapper overhead across 576 captured
  launches. The selector can help, but the dispatch substrate is now the main
  limiter.

- H5.5 is the first clearly positive dispatch-substrate result. CUDA Graph replay
  captured both fixed policy queues successfully and improved median event
  latency by 1.0493x for `static_best_average` and 1.0323x for
  `resource_aware`. Nsight Systems showed the comparable 96-queue host path
  dropping from 14978 CUDA runtime API calls in H5.4 to 194, with zero
  `cudaLaunchKernel` calls, 96 `cudaGraphLaunch` calls, and CPU enqueue per
  queue falling from 1.418 ms to about 0.046 ms.

## Lessons and Constraints

- H5.1 rules out a naive Python-level offline variant-table scheduler for
  fine-grained micro-tasks. The selector signal exists in calibration, but
  dispatch jitter, launch overhead, and dynamic loading hide sub-1% policy
  differences at queue level. H5.2 confirms that compile/cache/first-launch
  overhead dominates by orders of magnitude; H5.4 should now measure warmed
  persistent queue execution before another scheduler policy is trusted.
- H5.2 rules out per-request cold compilation and fresh-process cache-hit lookup
  for micro-op variant selection. A practical runtime needs ahead-of-time cache
  warming, a persistent process with loaded modules, asynchronous compilation
  behind a fallback variant, coarser migration-safe task tiles, or CUDA
  Graph/persistent-kernel/batched dispatch.
- H5.4 rules out treating a Python/PyTorch stream queue as the final runtime
  substrate for fine-grained migration. Even warmed execution spends enough on
  event/wait/launch plumbing that sub-2% policy differences are fragile.
- H5.5 supports CUDA Graph replay as a viable fixed-queue dispatch substrate.
  The next challenge is not raw launch overhead but flexibility: graph-bank
  switching, graph-level task migration, or larger batched units that preserve
  resource-aware choices without returning to per-kernel Python dispatch.
- Container tooling is now locally available without `/tmp` header workarounds:
  source `/home/descfly/.local/devtools/activate-auto-research.sh` to expose
  Python 3.10 development headers for Triton, `column`, `nsys`, and `ncu`.
  `.bashrc` is read-only in this container, so activation must be explicit per
  shell or command.

- Multi-request serving papers are useful but must be filtered: the project
  targets single request and single accelerator, so batching/goodput claims are
  only secondary evidence.
- NPU/Ascend/CANN local-note evidence remains thin and needs another targeted
  local-notes pass.
- A future experiment should report overlap ratio, launch count, global memory
  traffic, and hardware unit utilization, not just latency.
- Protocol commits are not available yet because `/data3/auto_research` is not a
  git repository.
- The default Codex sandbox does not expose `/dev/nvidia*`; GPU experiments must
  be run outside the sandbox or in an environment with visible device nodes.
- Two parallel exploratory H1 runs were excluded because benchmark processes can
  interfere on the same GPU. Use `_seq` result directories for conclusions.
- Overlap ratio must be interpreted with the isolated baseline: chunked isolated
  work can become much slower, making overlap ratio look better while absolute
  latency gets worse.
- Best-stream-speedup and best-concurrent-latency can differ. Future experiment
  protocols should name the scheduling objective before comparing variants.
- Stream speedup can be misleading when the serial baseline differs across
  variants. The scheduler's primary metric for single-request latency should be
  absolute concurrent latency, with speedup and overlap as diagnostics.
- Nsight Systems is now available on `PATH`, and Nsight Compute is available at
  `/usr/local/cuda-12.8/bin/ncu`. Older H2-H4 conclusions remain latency-based,
  but new H5/H4 follow-ups should collect launch-gap, occupancy, SM/SFU/DRAM,
  LDST/Tensor Core, and HBM traffic counters where profiling overhead is
  acceptable.
- The current MoE experiment excludes routing/top-k and scatter/gather costs.
  Future MoE experiments should add those back after the expert compute unit is
  represented below the PyTorch loop level.
- H3.2 still isolates epilogue/routing-adjacent work, not expert GEMM. The next
  MoE step must include grouped GEMM or Triton tiled expert matmul before making
  full-layer claims.
- H3.3 includes expert GEMM but still excludes routing/top-k/scatter/gather and
  uses padding. Full-layer claims require adding routing costs and avoiding
  padded work.
- H3.4 includes no-padding expert matmul but still excludes routing, top-k,
  scatter/gather, and combine weights. The next full-fragment test should add
  these surrounding micro-ops and measure whether routing overhead erases the
  no-padding matmul gain.
- H3.5 adds scatter/gather for fixed top-1 routing, but still excludes measured
  gating, top-k combine weights, and fused movement-compute kernels. The next
  MoE-specific step is scatter-first-matmul fusion; the next broader step is
  applying the same sparse-segment protocol to Video/VLM token selection and KV
  retrieval.
- H4.1 only covers score computation `K_selected @ Q`, not softmax/value mixing
  or real video-token predictors. The next H4 step should compare random,
  clustered, and globally sorted retrieval orders and add tile-variant selection
  before making stronger hardware-architecture claims.
- H4.2 still uses a score-only proxy. The next experiment should add softmax and
  value aggregation so locality affects both K and V access and reduction shape.
  That will be a better test of whether GPU compiler/runtime ordering is enough
  or whether hardware-side retrieval/concentration becomes necessary.
- H4.3 is still a single-query, single-tile sparse attention fragment with
  `selected_tokens <= BLOCK_N`. It is strong evidence for movement-compute
  fusion, but production claims require a multi-block online-softmax kernel,
  multi-query/multi-head shapes, and stronger sparse-attention baselines.
- After Outer Loop 3, the next experiment is H4.4: multi-block online-softmax
  sparse attention. If it succeeds, the GPU compiler/runtime story strengthens;
  if it fails, the hardware-side KVPU/KVMU or concentration-unit argument
  becomes more compelling.
- H4.4 still handles one query. The next H4 test should add multi-query or
  multi-head sparse attention to see whether K/V movement and score computation
  can be amortized across queries, or whether partial-buffer/reduction overhead
  grows too quickly.
- H4.5 covers `Q=4` but not larger query/head counts. The next synthesis should
  decide whether to deepen with a production-style sparse attention baseline and
  hardware counters, or broaden toward accelerator/NPU primitives for selected
  K/V movement and online-softmax reduction.
- H4.6 covers variant selection on held-in H4.5 order modes only. The next H4
  step should add held-out shape axes such as `Q=2/8`, selected tokens
  `512/2048`, or `V=64/256` and test whether the same table features still
  choose near-best valid variants.
- H4.7 adds held-out query counts but keeps selected-token count, key/value
  dimensions, and candidate variants fixed. The next shape-generalization step
  should add selected-token or value-dimension variation and include
  query-count-aware features in the selector.
- H4.9 makes Outer Loop 5 the right next move. The H4 compiler/runtime story now
  has evidence for sparse retrieval fusion, online-softmax scaling,
  query-grouping, static resource filtering, query-aware selection, and
  selected-token-count hold-out generalization. The next decision is whether to
  deepen with value-dimension held-outs and a learned/table selector, or broaden
  toward accelerator/NPU selected-token movement and online-reduction
  primitives.
- Outer Loop 5 chooses to deepen once more with H4.10 value-dimension held-outs
  before broadening. If `V=64/256` still stays near-best, the next step can move
  to a learned/table selector over all H4 data or to hardware/NPU mapping. If it
  misses, value pressure becomes the next required selector feature.
- H4.10 should now be treated as a selector stress test, not the main research
  story. The next main branch is H5: build experiments around runtime resource
  observation, dynamic compilation/cache overhead, micro-task queueing, and
  migration boundaries. Performance limitations should be attributed to
  software/hardware runtime support whenever possible, especially launch
  overhead, weak resource visibility, shared-memory/occupancy ceilings, and
  limited control over SM or accelerator pipeline allocation.
