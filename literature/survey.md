# Survey: Multi-Operator Concurrency for Single-Accelerator Inference

## Research Question

For a single request running on one GPU, NPU, or custom accelerator backend, how
can modern models expose and exploit multi-operator or micro-operator
concurrency? The emphasis is on hardware architecture and compiler/runtime
implementation rather than multi-request batching or multi-device distribution.

## Core Finding

Local notes suggest that "just launch kernels in multiple streams" is too weak
for single-request inference. The stronger pattern is vertical co-design:

1. Decompose the model graph into scheduler-visible units: tiles, micro
   operators, modules, patches, attention CTAs, or hardware pipeline stages.
2. Generate multiple kernel versions with different resource profiles.
3. Schedule or fuse those units using hardware-aware metadata: SM occupancy,
   Tensor Core pressure, shared memory, registers, HBM traffic, DMA lanes, or
   NPU pipeline ports.
4. Preserve semantic structure when generic graph fusion loses important
   optimization opportunities.

## Taxonomy

### 1. Runtime Co-Location and Spatial Partitioning

Representative notes: μShare, Bullet, EEVEE.

Methods:

- CUDA MPS or stream-level spatial sharing.
- SM mask control for prefill/decode or module partitioning.
- Kernel launch interception and block-size shaping to influence the closed GPU
  scheduler.
- Time-shifted launches to pair resource-complementary kernels.

Main lesson: hardware dispatch is not a neutral executor. If one kernel fills
all SMs, nominal stream concurrency degenerates into tiny tail overlap. Effective
co-location requires manipulating spatial occupancy or creating resource
complementarity.

### 2. Compiler-Generated Micro Operators

Representative notes: Infera, STOF, MetaAttention.

Methods:

- Split large operators into micro-operators/tiles.
- Merge very small operators into shepherd operators to avoid scheduling
  overhead.
- Generate multi-version kernels with different ILP/TLP/intensity trade-offs.
- Carry compile-time metadata to runtime scheduling.
- Use semantic templates for attention instead of generic matmul/softmax/matmul
  graph fusion.

Main lesson: micro-operator concurrency is useful only when the scheduler has a
large but controlled action space. Over-fragmentation creates launch and
scheduling overhead; under-fragmentation recreates kernel monopolization.

### 3. Intra-Kernel Fusion and Inter-Core Communication

Representative notes: FlashFuser, PAT, JanusQuant.

Methods:

- Fuse producer/consumer operators so intermediates stay in register, shared
  memory, DSM, or local pipeline buffers.
- Use Hopper DSM/TMA/mbarrier for inter-SM communication in fused GEMM chains.
- Use pack-forward-merge attention kernels to reuse KV blocks across queries.
- Fuse dequantization/smoothing/cache management into attention or quantization
  kernels.

Main lesson: the best concurrency often appears inside one larger kernel rather
than across separately launched kernels. The key is to remove global-memory
round trips and CPU launch boundaries while still preserving enough parallelism.

### 4. Model-Specific Semantic Decomposition

Representative notes: Difflow, MixFusion, EEVEE, FineMoE, LWD.

Methods:

- Diffusion: dGraph/dEngine decomposition, symbolic property propagation,
  invariant tensor hoisting, patch-level batching.
- DiT: step-level and patch/token-level scheduling ideas; current local evidence
  is stronger for multi-GPU scheduling than single-GPU micro-operator execution.
- MoE: iteration-level expert prediction, async prefetch, cache priority,
  grouped/sparse expert execution.
- Multimodal: split visual/text/projector/decoder modules and allocate resources
  per module.
- Wavelet diffusion: training-time spatial saliency, potentially useful as a
  signal for spatially selective inference even though LWD itself has no
  inference-time change.

Main lesson: newer models expose concurrency through semantics: experts,
patches, denoising steps, modality modules, visual-token redundancy, and KV
retrieval. Generic compilers see operators; model-specific systems see why
operators can be reordered, fused, cached, or partially skipped.

### 5. Hardware/Compiler Co-Design

Representative notes: Focus, V-Rex, RPU, ASM-SpMM.

Methods:

- Put concentration/compression units near systolic-array memory interfaces.
- Move irregular KV retrieval into dedicated hardware blocks.
- Compile model graphs into static DMA/compute/network instruction streams.
- Use backend-specific intrinsics such as ARM SME outer-product instructions.

Main lesson: when irregular control, sparse access, or token-level branching is
too costly on a GPU, hardware co-design turns irregularity into structured local
work. For NPU/accelerator research, the compiler IR must expose pipelines,
memory instructions, and micro-op dependencies, not just tensor shapes.

## Gaps From Local Notes

1. Single-request focus is under-covered. Many notes optimize multi-request
   batching, serving goodput, or multi-GPU scheduling. We need to isolate what
   still works when batch size is one.
2. NPU/Ascend/CANN evidence is thin. Local notes currently emphasize NVIDIA GPU,
   custom accelerators, PIM, and ARM SME.
3. There is no unified metric for micro-operator concurrency. Papers report
   speedup, goodput, utilization, or energy, but not a common overlap efficiency.
4. Cross-model comparison is missing. MoE, diffusion/DiT, multimodal, and video
   systems use different decomposition units, making a vertical framework useful.
5. Compiler-runtime contracts are underspecified. The strongest systems require
   kernel resource metadata, stream/SM placement, fusion legality, and semantic
   dependency metadata.

## Initial Hypotheses

H1: For single-request inference, naive multi-stream execution improves little
unless kernels are resource-complementary or spatially constrained.

H2: Micro-operator tiling plus multi-version kernels can outperform generic
operator fusion when runtime chooses variants using resource pressure and
dependency metadata.

H3: Model-semantic decomposition exposes concurrency that generic compiler
passes miss: MoE experts, diffusion patches/dGraphs, multimodal modules, and
video KV retrieval.

H4: Hardware co-design becomes necessary when the concurrency source is
irregular and branch-heavy, such as video KV retrieval or fine-grained visual
token concentration.

## Proposed Experimental Direction

Start with a single-GPU benchmark suite because it is the most locally
actionable. Build synthetic but model-shaped kernels for:

- attention + normalization + dequantization chains,
- MoE top-k expert FFN with small experts,
- DiT/diffusion block with attention, MLP, and elementwise stages,
- multimodal visual/text/module pipeline,
- video KV retrieval proxy with irregular selection.

Compare:

- serial PyTorch/Triton baseline,
- CUDA streams only,
- CUDA Graph replay,
- resource-shaped co-location,
- fused/micro-tiled kernels,
- semantic decomposition schedule.

Primary metrics:

- single-request latency,
- kernel launch count,
- overlap ratio,
- achieved HBM bandwidth and FLOPS,
- SM/Tensor Core/LDST utilization,
- global memory traffic,
- scheduler overhead,
- accuracy or numerical equivalence where applicable.

