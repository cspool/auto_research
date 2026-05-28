# Research Log — Multi-Operator Concurrency for Single-Accelerator Inference

## 2026-05-25 — Bootstrap Initiated

**Decision**: Begin systematic literature survey and taxonomy construction for
multi-operator/micro-operator concurrency in single-request, single-accelerator
inference scenarios.

**Scope defined**:
- **Hardware focus**: GPU (NVIDIA CUDA), NPU (Ascend, TPU), generic accelerators
- **Model types**: MoE, wavelet-Diffusion, DiT, multimodal, Video models
- **Concurrency level**: Operator-level and micro-operator-level (fused kernels, 
  kernel launch overlap, graph-level concurrency)
- **Framework focus**: TVM, Triton, XLA, MLIR, CUDA Graphs, TensorRT, OpenVINO, 
  Ascend CANN, IREE
- **Excluded**: Multi-request batching, multi-device distribution (single accelerator only)

**Initial search directions**:
1. CUDA Graph / kernel launch overlap for reducing launch overhead
2. Operator fusion (horizontal/vertical) in TVM, XLA, MLIR
3. Micro-kernel concurrency via CUDA streams and MPS
4. Compiler-driven concurrency (Triton, Apache TVM, MLIR-based)
5. Hardware-specific: NVIDIA MPS, Ascend TBE, TPU XLA
6. Model-specific optimizations (MoE expert parallelism on single GPU, DiT attention+temporal fusion)

## 2026-05-25 — Local Notes Literature Setup

**Decision**: Perform literature setup using local-notes only, per user request.
No web search was used.

**Continuity note**: The autoresearch skill requests `/loop` or `cron.add`
first. This session exposes neither tool, so continuity could not be installed.
Fallback was recorded in `research-state.yaml`.

**Artifacts created**:
- `literature/local_notes_search.md`
- `literature/survey.md`
- `literature/experiment_framework.md`
- per-paper summaries under `literature/papers/`
- `to_human/2026-05-25-local-notes-bootstrap.html`

**Synthesis**:
- The most relevant core systems are Infera, FlashFuser, μShare, Bullet,
  MetaAttention, PAT, JanusQuant, Difflow, MixFusion, EEVEE, Focus, V-Rex,
  FineMoE, RPU, ASM-SpMM, and STOF.
- The working taxonomy is: runtime co-location, compiler-generated micro
  operators, intra-kernel fusion, model-semantic decomposition, and
  hardware/compiler co-design.
- The next research action should be an H1 microbenchmark testing whether
  multi-stream execution alone is insufficient for single-request inference.

## 2026-05-26 — H1 Stream Concurrency Microbenchmark

**Protocol**: Locked in `experiments/h1-stream-concurrency/protocol.md`.
Git pre-registration is unavailable because the project directory is not a git
repository.

**Environment discovery**:
- Default sandbox cannot see `/dev/nvidia*`, so `nvidia-smi` and PyTorch CUDA
  fail there.
- Outside the sandbox, the host exposes two NVIDIA GeForce RTX 4090 GPUs.
- PyTorch 2.11.0+cu128 reports CUDA 12.8 and `cuda_available=True`.

**Valid results**:
- `default`: serial 0.2308 ms, multi-stream 0.2212 ms, speedup 1.0433x,
  overlap ratio 0.0599.
- `memory-heavy`: serial 0.5930 ms, multi-stream 0.5883 ms, speedup 1.0080x,
  overlap ratio -0.0011.
- `compute-heavy`: serial 0.9558 ms, multi-stream 0.9591 ms, speedup 0.9966x,
  overlap ratio 0.0060.

**Decision**: Mark H1 as preliminarily supported. Naive multi-stream execution
does not produce meaningful single-request operator concurrency in this setup.

**Next**: Test H1.1 with resource-shaped co-location or explicit partitioning.

## 2026-05-26 — H1.1 Resource-Shaped Stream Benchmark

**Protocol**: Locked in
`experiments/h1-1-resource-shaped-streams/protocol.md`.

**Question**: If full-kernel streams do not overlap well, does splitting the
dominant GEMM into row chunks create useful stream overlap?

**Valid results**:
- compute-heavy 4096: full multi-stream 0.9213 ms, best chunked multi-stream
  0.9599 ms, best chunk stream speedup 1.0242x.
- balanced 2048: full multi-stream 0.2430 ms, best relative chunk stream
  speedup 1.0589x at 2 chunks, but absolute fastest chunked result was 0.2273 ms
  at 4 chunks with only 1.0057x stream speedup.
- memory-heavy 1024: full multi-stream 0.5972 ms, best chunk stream speedup
  1.1147x at 4 chunks, but chunk GEMM overhead was 4.89x and absolute latency
  did not beat full multi-stream.

**Decision**: Mark H1.1 as mixed-to-negative for naive PyTorch-level chunking.
Resource shaping likely matters, but it needs compiler/kernel-level control or
explicit spatial partitioning. Framework-level row chunking is too blunt.

## 2026-05-26 — H2 Triton Variant Scheduling

**Protocol**: Locked in
`experiments/h2-triton-variant-scheduling/protocol.md`.

**Question**: Does the best isolated micro-kernel variant differ from the best
variant under concurrent execution with a GEMM?

**Method**: Implemented a Triton vector-add micro-kernel with `BLOCK_SIZE` in
{128, 256, 512, 1024, 2048, 4096, 8192} and `num_warps` in {4, 8}. Measured
isolated add, GEMM+add serial, and GEMM/add concurrent streams.

**Valid results**:
- Balanced 2048: best isolated variant was `(block=4096, warps=4)` with
  `add_ms=0.1019`; best concurrent variant was `(block=2048, warps=8)` with
  `concurrent_ms=0.1864` and overlap `0.1652`.
- Memory-heavy 1024: best isolated variant was `(block=256, warps=8)` with
  `add_ms=0.4227`; best concurrent variant was `(block=128, warps=4)` with
  `concurrent_ms=0.4349`.

**Decision**: Mark H2 as preliminarily supported. Kernel variant selection must
consider scheduling context; isolated latency alone is not a sufficient runtime
selection metric.

## 2026-05-26 — H2.1 Triton Fused Micro-Kernel

**Protocol**: Locked in
`experiments/h2-1-triton-fused-microkernel/protocol.md`.

**Question**: Does the H2 variant-scheduling result hold when the micro-kernel
is a model-shaped fused epilogue instead of a simple vector add?

**Method**: Implemented a Triton fused micro-kernel
`Y = GELU(X * scale + Bias) + Residual` with `BLOCK_SIZE` in
{128, 256, 512, 1024, 2048, 4096, 8192} and `num_warps` in {4, 8}. Measured
isolated fused latency, PyTorch eager unfused latency, GEMM+fused serial, and
GEMM/fused concurrent streams.

**Valid results**:
- Balanced 2048: PyTorch unfused latency was `0.2721 ms`; best Triton fused
  latency was `0.1422 ms`. Best isolated variant was `(block=2048, warps=8)`,
  while best concurrent variant was `(block=2048, warps=4)` with
  `concurrent_ms=0.2271`.
- Memory-heavy 1024: PyTorch unfused latency was `1.4640 ms`; best Triton fused
  latency was `0.5614 ms`. Best isolated and best concurrent variants were
  effectively tied in concurrent latency, but the best stream-speedup variant
  had worse absolute concurrent latency.

**Decision**: H2 is strengthened. Fusion improves the micro-op, but runtime
scheduling still needs variant metadata and a clear objective. Absolute
concurrent latency should be the primary scheduler metric; stream speedup is a
diagnostic because it can be inflated by a slow serial baseline.

## 2026-05-26 — H2.2 Triton Compiler Metadata Extraction

**Protocol**: Locked in
`experiments/h2-2-triton-compiler-metadata/protocol.md`.

**Question**: If Nsight counters are unavailable, can Triton compiler artifacts
still expose useful variant metadata for a runtime scheduler?

**Method**: Parsed H2.1 Triton cache artifacts for
`fused_gelu_residual_kernel`: JSON metadata, TTGIR, PTX, and cubin/PTX file
sizes. Joined the recovered metadata with H2.1 latency and overlap results.

**Valid results**:
- Recovered 29 unique compiled variants: one smoke-test variant, 14 balanced
  variants, and 14 memory-heavy variants.
- Recovered `block_size`, `num_warps`, `.reqntid`, `shared`, PTX register
  declarations, and code-size proxies for every measured variant.
- For balanced 2048, the best isolated variant `(block=2048, warps=8)` had
  `.reqntid=256` and `reg_b32_decl=161`, while the best concurrent variant
  `(block=2048, warps=4)` had `.reqntid=128` and `reg_b32_decl=313`.

**Decision**: H2 now has an implementation-level path: generate multi-version
Triton kernels, retain compiler metadata, and join it with empirical
co-scheduling measurements. Static compiler metadata is not enough by itself,
but it is a usable scheduler-table substrate.

## 2026-05-26 — Outer Loop 1 Synthesis

**Scope**: Synthesized the first five experiments: H1, H1.1, H2, H2.1, and
H2.2.

**Main claim**: Single-request, single-accelerator operator concurrency is not
primarily a stream API problem. It is a compiler/runtime co-scheduling problem:
decompose model work into schedulable micro-operators, generate multiple
resource-shaped kernel variants, retain compiler metadata, and select by
absolute concurrent latency under the current co-location context.

**Metric decision**: Absolute concurrent latency is the primary scheduler
objective. Isolated latency, overlap ratio, and stream speedup are diagnostics.

**Next direction**: Move to H3.1: a tiny single-request MoE fragment with
semantic expert units, comparing sequential expert loops, naive multi-stream
expert execution, and Triton routing/epilogue micro-kernel variants.

**Artifacts**:
- `synthesis/outer-loop-1.md`
- `to_human/2026-05-26-outer-loop-1.html`

## 2026-05-26 — H3.1 Tiny MoE Semantic Concurrency

**Protocol**: Locked in
`experiments/h3-1-moe-semantic-concurrency/protocol.md`.

**Question**: Do MoE expert branches, as model-semantic units, produce useful
single-request concurrency when each expert is launched on a separate CUDA
stream?

**Method**: Built a four-expert single-request MoE FFN fragment with
pre-materialized routing. Each expert computes
`Y_e = ReLU(X_e @ W1_e) @ W2_e` in fp16 on one RTX 4090. Compared serial expert
loop versus one CUDA stream per expert for balanced, skewed, and tiny token
distributions.

**Valid results**:
- Balanced `64,64,64,64`: serial `0.2052 ms`, concurrent `0.3110 ms`,
  speedup `0.6596x`.
- Skewed `160,64,24,8`: serial `0.2192 ms`, concurrent `0.3226 ms`, speedup
  `0.6795x`.
- Tiny `16,16,16,16`: serial `0.2065 ms`, concurrent `0.3028 ms`, speedup
  `0.6818x`.

**Decision**: Refine H3. MoE experts are useful semantic decomposition units,
but PyTorch expert streams are not a sufficient implementation. The next MoE
step should lower expert tiles/epilogues into Triton or grouped kernels and
reuse the H2 variant metadata method.

## 2026-05-26 — H3.2 Triton MoE Expert Epilogue

**Protocol**: Locked in
`experiments/h3-2-moe-triton-epilogue/protocol.md`.

**Question**: If MoE expert epilogues are lowered into Triton micro-operators,
is it better to launch per-expert kernels or group all expert segments into one
compiler-visible kernel?

**Method**: Simulated the expert epilogue
`Y_e = GELU(X_e * scale + Bias_e) + Residual_e` for one request. Compared
PyTorch expert loop, per-expert Triton serial launches, per-expert Triton
streams, and a grouped single Triton kernel. Swept `BLOCK_SIZE` in
{128, 256, 512, 1024, 2048, 4096} and `num_warps` in {4, 8}.

**Valid results**:
- Balanced `64,64,64,64`: PyTorch loop `0.2906 ms`, best grouped `0.0172 ms`,
  grouped-vs-PyTorch `16.88x`.
- Skewed `160,64,24,8`: PyTorch loop `0.2560 ms`, best grouped `0.0135 ms`,
  grouped-vs-PyTorch `18.98x`.
- Tiny `16,16,16,16`: PyTorch loop `0.2480 ms`, best grouped `0.0175 ms`,
  grouped-vs-PyTorch `14.15x`.
- Per-expert Triton streams remained slower than per-expert Triton serial.

**Decision**: H3 is partially supported in its refined form. MoE semantics help
when lowered to grouped compiler-visible micro-operators. The next step should
extend this from epilogue work to expert compute via grouped GEMM or Triton
tiled expert matmul plus routing/scatter micro-ops.

## 2026-05-26 — H3.3 MoE Grouped Expert GEMM

**Protocol**: Locked in
`experiments/h3-3-moe-grouped-gemm/protocol.md`.

**Question**: Does grouping help actual MoE expert FFN compute, beyond the
epilogue-only win in H3.2?

**Method**: Compared a serial PyTorch expert loop with a padded strided-batched
GEMM implementation using two `torch.bmm` calls over stacked experts. Tested
balanced, skewed, and tiny token distributions.

**Valid results**:
- Balanced `64,64,64,64`: expert loop `0.2265 ms`, grouped bmm `0.1842 ms`,
  speedup `1.23x`.
- Skewed `160,64,24,8`: expert loop `0.2409 ms`, grouped bmm `0.2232 ms`,
  speedup `1.08x`, padding overhead `2.5x`.
- Tiny `16,16,16,16`: expert loop `0.2073 ms`, grouped bmm `0.1496 ms`,
  speedup `1.39x`.

**Decision**: Grouped expert compute is promising but standard padded grouped
GEMM is only a baseline. The next implementation target should be irregular
grouped/tiled expert matmul that avoids padding waste and exposes compiler
metadata.

## 2026-05-26 — H3.4 Irregular No-Padding Triton Expert Matmul

**Protocol**: Locked in
`experiments/h3-4-moe-triton-irregular-gemm/protocol.md`.

**Question**: Can a simple no-padding Triton expert matmul recover the skewed
routing waste observed in H3.3 padded grouped bmm?

**Method**: Implemented a Triton row-tile map over valid expert token segments.
The FFN runs two custom grouped matmul kernels plus one ReLU, scheduling only
actual expert token rows. Compared against the expert loop and padded bmm.

**Valid results**:
- Balanced `64,64,64,64`: expert loop `0.2189 ms`, padded bmm `0.1914 ms`,
  no-padding Triton `0.1709 ms`, speedup `1.28x` over loop and `1.12x` over
  padded bmm.
- Skewed `160,64,24,8`: expert loop `0.2421 ms`, padded bmm `0.2135 ms`,
  no-padding Triton `0.1634 ms`, speedup `1.48x` over loop and `1.31x` over
  padded bmm.
- Tiny `16,16,16,16`: expert loop `0.2154 ms`, padded bmm `0.1735 ms`,
  no-padding Triton `0.1550 ms`, speedup `1.39x` over loop and `1.12x` over
  padded bmm.

**Decision**: H3 is now strongly supported for the MoE branch in refined form.
The best implementation unit is neither expert streams nor padded dense batches,
but no-padding compiler-visible expert tiles. Next: add routing/scatter/gather
micro-ops around H3.4 to test a fuller single-request MoE fragment.

## 2026-05-26 — H3.5 Routed No-Padding MoE Fragment

**Protocol**: Locked in
`experiments/h3-5-moe-routed-fragment/protocol.md`.

**Question**: Does the H3.4 no-padding grouped expert matmul remain beneficial
after adding routing movement around the expert buffer?

**Method**: Built a fixed top-1 routed single-request MoE fragment. The Triton
path scatters original tokens into expert-contiguous layout, runs the H3.4
no-padding grouped FFN, then gathers outputs back to original token order. The
baseline is a PyTorch routed expert loop using `index_select`, two `torch.mm`
calls per expert, and `index_copy_`.

**Valid results**:
- Balanced `64,64,64,64`: PyTorch routed loop `0.2718 ms`, Triton routed
  `0.1753 ms`, speedup `1.55x`, movement fraction `21.0%`.
- Skewed `160,64,24,8`: PyTorch routed loop `0.2728 ms`, Triton routed
  `0.1959 ms`, speedup `1.39x`, movement fraction `21.5%`.
- Tiny `16,16,16,16`: PyTorch routed loop `0.2744 ms`, Triton routed
  `0.1671 ms`, speedup `1.64x`, movement fraction `22.6%`.

**Decision**: H3.5 supports the full-fragment MoE claim. Routing movement is a
real micro-op, costing about one fifth of routed Triton latency, but it does not
erase the no-padding grouped-tile gain. The MoE branch is ready for a second
outer-loop synthesis.

## 2026-05-26 — Outer Loop 2 MoE Synthesis

**Scope**: Synthesized H3.1 through H3.5.

**Main claim**: For single-request MoE inference on one accelerator, useful
concurrency does not come from per-expert streams. It comes from preserving
expert semantics long enough to create no-padding compiler-visible tile maps,
then scheduling those tiles with explicit routing movement and full-fragment
latency as the objective.

**Decision**: H3 is strongly supported for the MoE branch in refined form. The
next broad branch should be H4: Video/VLM token selection and KV retrieval using
the same sparse-segment and movement-micro-op protocol. If continuing inside
MoE first, run H3.6 by fusing scatter with the first expert matmul tile.

**Artifacts**:
- `experiments/h3-5-moe-routed-fragment/analysis.md`
- `data/h3_5_moe_routed_fragment_summary.csv`
- `synthesis/outer-loop-2-moe.md`
- `to_human/2026-05-26-h3-5-moe-routed-fragment.html`
- `to_human/2026-05-26-outer-loop-2-moe.html`

## 2026-05-26 — H4.1 Sparse Video/VLM KV Retrieval Score

**Protocol**: Locked in
`experiments/h4-1-video-kv-retrieval/protocol.md`.

**Question**: Does the sparse-segment lesson from MoE transfer to Video/VLM KV
retrieval, where dynamic selected tokens must be gathered and scored against a
small set of query vectors?

**Method**: Built a single-request retrieval-score fragment:
`S = K_selected @ Q`, with selected K rows grouped into 32 semantic
frame/segment groups. Compared three implementations: per-segment PyTorch
`index_select + mm + copy`, flat PyTorch `index_select + mm`, and a fused Triton
gather-score kernel that avoids materializing `K_selected`.

**Valid results**:
- Balanced random: 1024 selected tokens, segment loop `1.4908 ms`, flat PyTorch
  `0.0289 ms`, fused Triton `0.0232 ms`; fused speedup `64.37x` over segment
  loop and `1.25x` over flat.
- Skewed random: 1088 selected tokens, segment loop `1.3370 ms`, flat PyTorch
  `0.0293 ms`, fused Triton `0.0217 ms`; fused speedup `61.60x` over segment
  loop and `1.35x` over flat.
- Tiny random: 128 selected tokens, segment loop `1.1967 ms`, flat PyTorch
  `0.0264 ms`, fused Triton `0.0229 ms`; fused speedup `52.25x` over segment
  loop and `1.15x` over flat.

**Decision**: H4 is preliminarily supported but refined. The evidence strongly
rules out framework-level segment loops for sparse Video/VLM retrieval and
supports fused movement-compute kernels. It does not yet prove that custom
hardware is necessary; the next step should test locality-aware index ordering,
value mixing, and multi-version Triton tiles.

**Artifacts**:
- `experiments/h4-1-video-kv-retrieval/analysis.md`
- `data/h4_1_video_kv_retrieval_summary.csv`
- `to_human/2026-05-26-h4-1-video-kv-retrieval.html`

## 2026-05-26 — H4.2 Locality-Aware Sparse KV Retrieval Ordering

**Protocol**: Locked in
`experiments/h4-2-kv-locality-order/protocol.md`.

**Question**: Does selected-token locality/order materially change sparse
Video/VLM KV retrieval latency, and does the best Triton tile variant depend on
that order?

**Method**: Reused the H4.1 score proxy `K_selected @ Q` with 32 segments and
1024 selected tokens. Compared flat PyTorch gather+GEMM with six fused Triton
tile variants under four order modes: random segment order, globally sorted
random selections, globally shuffled random selections, and clustered segment
spans.

**Valid results**:
- Random segment: flat `0.0299 ms`, best Triton `0.0211 ms`, speedup `1.42x`,
  best variant `M32 D128 W4`, mean neighbor span `90.6`.
- Random sorted: flat `0.0294 ms`, best Triton `0.0213 ms`, speedup `1.38x`,
  best variant `M16 D64 W8`, mean neighbor span `8.0`.
- Random shuffled: flat `0.0299 ms`, best Triton `0.0214 ms`, speedup `1.40x`,
  best variant `M32 D64 W4`, mean neighbor span `2673.7`.
- Clustered segment: flat `0.0295 ms`, best Triton `0.0216 ms`, speedup
  `1.37x`, best variant `M16 D128 W4`, mean neighbor span `7.7`.

**Decision**: H4 is strengthened but refined. Locality transforms alone do not
monotonically predict latency in this score-only proxy, but tile/order
interaction is real. The compiler/runtime contract should join selected-token
order statistics with kernel-variant metadata and empirical full-fragment
latency. Next: add softmax and value aggregation.

**Artifacts**:
- `experiments/h4-2-kv-locality-order/analysis.md`
- `data/h4_2_kv_locality_order_summary.csv`
- `to_human/2026-05-26-h4-2-kv-locality-order.html`

## 2026-05-26 — H4.3 Sparse KV Score-Softmax-Value Fragment

**Protocol**: Locked in
`experiments/h4-3-kv-score-softmax-value/protocol.md`.

**Question**: Does fused movement-compute remain beneficial after extending the
Video/VLM sparse KV retrieval proxy from score-only to score, softmax, and value
aggregation?

**Method**: Built a single-request, single-query sparse attention fragment:
`scores = K_selected @ q`, `p = softmax(scores)`, `out = p @ V_selected`.
Compared flat PyTorch eager, which materializes selected K/V, against a
single-tile Triton fused score-softmax-value kernel. Swept six
`BLOCK_N/BLOCK_D/BLOCK_V/warps` variants across four selected-token orders.

**Valid results**:
- Random segment: flat PyTorch `0.1402 ms`, best Triton `0.0207 ms`, speedup
  `6.79x`, best variant `N256 D64 V64 W8`.
- Random sorted: flat PyTorch `0.1250 ms`, best Triton `0.0216 ms`, speedup
  `5.78x`, best variant `N256 D64 V32 W4`.
- Random shuffled: flat PyTorch `0.1285 ms`, best Triton `0.0210 ms`, speedup
  `6.12x`, best variant `N256 D64 V128 W4`.
- Clustered segment: flat PyTorch `0.1286 ms`, best Triton `0.0214 ms`,
  speedup `6.00x`, best variant `N256 D64 V128 W4`.

**Decision**: H4 is strongly supported for the sparse attention branch in its
refined form. Score-softmax-value fusion makes the movement-compute benefit much
larger than score-only retrieval. The limitation is that the current Triton
kernel is single-tile and single-query; the next decision should come from an
outer-loop synthesis over H4.1-H4.3.

**Artifacts**:
- `experiments/h4-3-kv-score-softmax-value/analysis.md`
- `data/h4_3_kv_score_softmax_value_summary.csv`
- `to_human/2026-05-26-h4-3-kv-score-softmax-value.html`

## 2026-05-26 — Outer Loop 3 H4 Synthesis

**Scope**: Synthesized H4.1 through H4.3.

**Main claim**: For single-request Video/VLM sparse retrieval, semantic
selection must not be executed as per-frame or per-cluster framework work. The
useful unit is a compiler-visible sparse attention fragment that fuses
selected-token movement, score computation, softmax/reduction, and value
aggregation.

**Evidence ladder**:
- H4.1: fused Triton gather-score beat per-segment PyTorch loops by `52x-64x`.
- H4.1: fused Triton gather-score beat flat PyTorch gather+GEMM by `1.15x-1.35x`.
- H4.2: fused Triton gather-score beat flat PyTorch by `1.37x-1.42x` across
  random, sorted, shuffled, and clustered orders; best tile changed by order.
- H4.3: fused Triton score-softmax-value beat flat PyTorch eager by
  `5.78x-6.79x`.

**Decision**: H4 is strongly supported for the sparse attention branch in
refined form. Deepen H4 next with H4.4: a multi-block online-softmax sparse
attention kernel for larger selected-token counts. This is the key boundary
between a GPU compiler/runtime story and a stronger hardware-side retrieval
argument.

**Artifacts**:
- `synthesis/outer-loop-3-h4.md`
- `to_human/2026-05-26-outer-loop-3-h4.html`

## 2026-05-26 — H4.4 Multi-Block Online-Softmax Sparse Attention

**Protocol**: Locked in
`experiments/h4-4-kv-online-softmax/protocol.md`.

**Question**: Does the H4.3 fused sparse attention result extend beyond the
single-tile limit to larger selected-token counts?

**Method**: Built a two-stage Triton online-softmax sparse attention prototype
for one request and one query. The partial kernel computes local max,
denominator, and weighted value partials per selected-token block; the reduce
kernel combines partials with online-softmax rescaling. Tested 1024 selected
tokens, four selected-token orders, and seven tile variants.

**Valid results**:
- Random segment: flat PyTorch `0.1330 ms`, best Triton `0.0286 ms`, speedup
  `4.65x`, best variant `N128 D64 V128 W4`.
- Random sorted: flat PyTorch `0.1319 ms`, best Triton `0.0436 ms`, speedup
  `3.02x`, best variant `N128 D64 V64 W8`.
- Random shuffled: flat PyTorch `0.1364 ms`, best Triton `0.0287 ms`, speedup
  `4.76x`, best variant `N128 D64 V64 W4`.
- Clustered segment: flat PyTorch `0.1234 ms`, best Triton `0.0413 ms`,
  speedup `2.99x`, best variant `N128 D64 V64 W8`.

**Decision**: H4.4 supports the GPU compiler/runtime path beyond a single-tile
toy case. The cost of multi-block partials is visible, but the prototype still
beats flat PyTorch by `2.99x-4.76x`. Next: test multi-query/multi-head sparse
attention to see whether K/V movement and score computation amortize.

**Artifacts**:
- `experiments/h4-4-kv-online-softmax/analysis.md`
- `data/h4_4_kv_online_softmax_summary.csv`
- `to_human/2026-05-26-h4-4-kv-online-softmax.html`

## 2026-05-26 — H4.5 Multi-Query Sparse KV Online Softmax

**Protocol**: Locked in
`experiments/h4-5-kv-multi-query/protocol.md`.

**Question**: Does the H4 sparse attention compiler/runtime path scale from one
query to a small multi-query/multi-head fragment when all queries reuse the same
selected K/V set?

**Method**: Built a `Q=4` sparse attention benchmark with shared selected
indices, `1024` selected tokens, fp16 K/V/Q inputs, and fp32 online-softmax/value
accumulation. Compared flat PyTorch eager against a grouped-query two-stage
Triton implementation and a measured per-query repeated H4.4-style baseline.

**Valid results**:
- Random segment: flat PyTorch `0.1231 ms`, best grouped Triton `0.0338 ms`,
  speedup `3.64x`, grouped versus repeated one-query `3.66x`.
- Random sorted: flat PyTorch `0.1228 ms`, best grouped Triton `0.0429 ms`,
  speedup `2.86x`, grouped versus repeated one-query `3.42x`.
- Random shuffled: flat PyTorch `0.1306 ms`, best grouped Triton `0.0433 ms`,
  speedup `3.02x`, grouped versus repeated one-query `3.41x`.
- Clustered segment: flat PyTorch `0.1280 ms`, best grouped Triton `0.0425 ms`,
  speedup `3.01x`, grouped versus repeated one-query `3.46x`.

**Decision**: H4.5 supports the sparse-attention compiler/runtime path. The
query/head dimension must be grouped into the sparse attention lowering unit;
otherwise repeating good single-query kernels pays repeated launch, score, K/V
movement, and partial-reduction costs. One large grouped tile
(`N256 D64 V64 Q4 W4`) failed with shared-memory overuse, adding static resource
feasibility to the runtime-table contract.

**Artifacts**:
- `experiments/h4-5-kv-multi-query/analysis.md`
- `data/h4_5_kv_multi_query_summary.csv`
- `data/h4_5_kv_multi_query_variants.csv`
- `to_human/2026-05-26-h4-5-kv-multi-query.html`

## 2026-05-26 — Outer Loop 4 H4 Synthesis

**Scope**: Synthesized H4.1 through H4.5.

**Main claim**: For single-request Video/VLM sparse retrieval, the useful unit
is a compiler-visible grouped sparse attention fragment: selected-token indices
and order statistics, K/V layout, score, online softmax, value aggregation,
query/head grouping, static resource feasibility, multi-version tile variants,
and empirical full-fragment latency.

**Evidence ladder**:
- H4.1: per-segment PyTorch retrieval was `52x-64x` slower than fused Triton.
- H4.1: fused gather-score beat flat gather+GEMM by `1.15x-1.35x`.
- H4.2: best tile changed with selected-token order, so locality alone is not
  enough.
- H4.3: fused score-softmax-value beat flat PyTorch by `5.78x-6.79x`.
- H4.4: two-stage online softmax scaled to `1024` selected tokens and beat flat
  PyTorch by `2.99x-4.76x`.
- H4.5: grouped `Q=4` sparse attention beat flat PyTorch by `2.86x-3.64x` and
  repeated one-query kernels by `3.41x-3.66x`.

**Decision**: Deepen H4 once more with H4.6, but shift the objective from raw
kernel speed to compiler/runtime selection. Build a variant table that joins
measured latency, static resource feasibility, tile metadata, order statistics,
and query/value-block shape. The next test should ask whether this table can
reject infeasible kernels and select near-best valid variants without exhaustive
online benchmarking.

**Artifacts**:
- `synthesis/outer-loop-4-h4.md`
- `to_human/2026-05-26-outer-loop-4-h4.html`

## 2026-05-27 — H4.6 Runtime Variant Table

**Protocol**: Locked in
`experiments/h4-6-runtime-variant-table/protocol.md`.

**Question**: Can the H4 sparse-attention path move from hand-picked kernel
speedups to a compiler/runtime table that rejects infeasible variants and
selects near-best valid variants?

**Method**: Built a table-selection experiment over the H4.5 CSV artifacts. The
selector joins tile metadata, a static shared-memory proxy, selected-token order
statistics, and measured grouped-query latency. Compared a static-safe global
mean latency selector against an order-aware rule table.

**Valid results**:
- Static proxy rejected `4/4` observed shared-memory OOR rows. The failing
  variant was `N256 D64 V64 Q4 W4`, with proxy `131072` over hardware limit
  `101376`.
- Global mean latency selector chose within `10%` of measured best in `3/4`
  order modes; max regret was `12.06%` on random sorted.
- Order-aware rule table chose within `10%` in `4/4` order modes and matched
  measured best for all four H4.5 orders.

**Decision**: H4.6 supports the compiler/runtime selection layer. Static
resource filtering plus measured full-fragment latency is already useful, and
order statistics fix the case that a global latency table misses. The result is
small and retrospective, so the next step should test held-out shapes rather
than only held-in order modes.

**Artifacts**:
- `experiments/h4-6-runtime-variant-table/analysis.md`
- `data/h4_6_runtime_variant_table.csv`
- `data/h4_6_runtime_variant_decisions.csv`
- `data/h4_6_runtime_variant_summary.csv`
- `to_human/2026-05-27-h4-6-runtime-variant-table.html`

## 2026-05-27 — H4.7 Held-Out Query Counts

**Protocol**: Locked in
`experiments/h4-7-variant-table-heldout-shapes/protocol.md`.

**Question**: Does the H4.6 runtime variant table generalize beyond the held-in
`Q=4` shape to held-out query counts `Q=2` and `Q=8`?

**Method**: Reused the H4.5 grouped sparse-attention benchmark and variant set.
Ran `Q=2` and `Q=8` across four selected-token orders. Evaluated the H4.6 global
mean latency selector and H4.6 order-aware rule table without retuning.

**Valid results**:
- Static feasibility rejected `8/8` observed OOR rows.
- Global mean latency selector was within `15%` of measured best in `7/8`
  held-out cases, with mean regret `9.92%` and max regret `27.44%`.
- Order-aware rule table was within `15%` in `7/8` cases, with mean regret
  `7.79%` and max regret `18.72%`.
- The main miss was `Q=8 clustered_segment`: the rule table selected
  `N128 D64 V128 Q4 W4`, but measured best was `N128 D64 V64 Q4 W8`.

**Decision**: H4.7 supports the runtime-table path under held-out query counts,
but shows the table needs query-count features. The next selector should include
`query_count`, `num_q_blocks`, and `query_count / BLOCK_Q`, then test held-out
selected-token counts or value dimensions.

**Artifacts**:
- `experiments/h4-7-variant-table-heldout-shapes/analysis.md`
- `data/h4_7_heldout_variant_rows.csv`
- `data/h4_7_heldout_selector_decisions.csv`
- `data/h4_7_heldout_selector_summary.csv`
- `to_human/2026-05-27-h4-7-heldout-query-counts.html`

## 2026-05-27 — H4.8 Query-Count-Aware Runtime Selector

**Protocol**: Locked in
`experiments/h4-8-query-aware-selector/protocol.md`.

**Question**: Can query-count features fix the H4.7 selector miss without
introducing new misses?

**Method**: Built a retrospective query-aware rule table over the H4.7 held-out
query-count measurements. Added `query_count`, `num_q_blocks`, and
`query_count / BLOCK_Q` to the H4.6 order-statistics selector while preserving
static shared-memory feasibility filtering.

**Valid results**:
- H4.6 order-aware selector: `7/8` cases within `15%`, mean regret `7.79%`,
  max regret `18.72%`.
- H4.8 query-aware selector: `8/8` cases within `15%`, mean regret `0.00%`,
  max regret `0.00%`.
- The `Q=8 clustered_segment` miss was fixed: H4.8 selected
  `N128 D64 V64 Q4 W8`, matching the measured best.

**Decision**: H4.8 supports the query-shape feature hypothesis. The selector
result is retrospective, so the next step should move to a new held-out axis:
selected-token count or value dimension. If that misses, add `num_n_blocks` or
value-block pressure to the runtime table.

**Artifacts**:
- `experiments/h4-8-query-aware-selector/analysis.md`
- `data/h4_8_query_aware_decisions.csv`
- `data/h4_8_query_aware_comparison.csv`
- `data/h4_8_query_aware_summary.csv`
- `to_human/2026-05-27-h4-8-query-aware-selector.html`

## 2026-05-27 — H4.9 Held-Out Selected-Token Counts

**Protocol**: Locked in
`experiments/h4-9-selected-token-heldout-shapes/protocol.md`.

**Question**: Does the H4.8 order/query runtime selector generalize to
held-out selected-token counts `512` and `2048` at `Q=4`?

**Method**: Reused the H4.5 grouped sparse-attention benchmark and six-variant
candidate set. Ran eight GPU sweeps across two selected-token counts and four
selected-token orders, then evaluated the H4.8 order/query selector against a
diagnostic selected-token pressure oracle.

**Valid results**:
- Static feasibility rejected `8/8` observed shared-memory OOR rows.
- The H4.8 order/query selector was within `15%` of measured best in `8/8`
  held-out cases, with mean regret `4.66%` and max regret `9.92%`.
- The selected-token pressure oracle selected exact measured best in `8/8`
  cases, showing selected-token count can refine exact tuning but is not an
  immediate near-best failure mode.

**Decision**: H4.9 supports selected-token-count shape generalization for the
current selector. Outer Loop 5 is warranted to synthesize H4.1-H4.9 and decide
whether the next step should deepen with value-dimension held-outs, add a
learned/table selector over all H4 data, or broaden toward accelerator/NPU
selected-token movement primitives.

**Artifacts**:
- `experiments/h4-9-selected-token-heldout-shapes/analysis.md`
- `data/h4_9_selected_token_variant_rows.csv`
- `data/h4_9_selected_token_selector_decisions.csv`
- `data/h4_9_selected_token_selector_summary.csv`
- `to_human/2026-05-27-h4-9-selected-token-heldouts.html`

## 2026-05-27 — Outer Loop 5 H4 Runtime Selector Synthesis

**Scope**: Synthesized H4.1-H4.9, with emphasis on the H4.6-H4.9
compiler/runtime selector path.

**Pattern**: H4 now supports a vertical method rather than only a faster kernel:
lower selected-token retrieval into a grouped sparse-attention fragment, filter
shared-memory-infeasible variants statically, and select tile/warp variants from
order/query/shape metadata plus measured full-fragment latency.

**Decision**: Deepen H4 one more step before broadening to NPU/accelerator
primitives. The next experiment should be H4.10 value-dimension held-outs
(`V=64/256`, `selected_tokens=1024`, `Q=4`, four selected-token orders). Value
dimension is the most informative remaining axis because it changes
`num_v_blocks`, partial value-buffer traffic, and the `BLOCK_V`/warp choice.

**Artifacts**:
- `synthesis/outer-loop-5-h4.md`
- `to_human/2026-05-27-outer-loop-5-h4.html`

## 2026-05-27 — Direction Update: Runtime Resource-Aware Scheduling

**User steering**: Refocus the research away from only pursuing single-kernel
speedups. The more important direction is runtime scheduling of different
operators or micro-operators according to idle resources. These tasks may
involve dynamic compilation, runtime task management, backend resource
management, and task migration. The project should also analyze which software
and hardware runtime limits constrain performance.

**Interpretation**: This direction is consistent with the existing evidence.
H1/H1.1 show that ordinary streams and framework chunking provide too little
control. H2/H2.1 show that the best variant depends on co-scheduling context.
H2.2 shows compiler metadata can seed a variant table. H3/H4 show that model
semantics can be lowered into micro-task boundaries suitable for runtime
queueing and migration.

**Decision**: Add H5: runtime resource-aware task management. H4.10
value-dimension held-outs remain useful as a selector stress test, but the main
next phase should measure:
- resource-aware runtime variant scheduling;
- dynamic compilation and variant-cache overhead;
- tile/micro-op task migration boundaries;
- software/hardware runtime limits such as launch overhead, resource
  visibility, shared-memory/occupancy ceilings, and SM or pipeline allocation
  control.

**Artifact**:
- `synthesis/runtime-resource-scheduling-direction.md`

## 2026-05-27 — GPU Tooling Update

**User update**: Local GPU-related packages were updated/installed to support
the next experiments.

**Detected tooling**:
- `nsys` is available on `PATH`: Nsight Systems 2024.6.2.
- Nsight Compute is installed at `/usr/local/cuda-12.8/bin/ncu`: version
  2026.2.0.
- CUDA Toolkit 12.8 developer binaries are installed under
  `/usr/local/cuda-12.8/bin`: `nvcc`, `cuobjdump`, `nvdisasm`, and
  `compute-sanitizer`.
- `trtexec` is available on `PATH`: TensorRT v11.0.0 CLI.
- Plot/report and literature API packages are now importable: `plotly`,
  `seaborn`, `weasyprint`, `playwright`, `semanticscholar`, and `arxiv`.

**Parameter update**: Future H5 protocols should include profiler paths and
keep profiling narrow:
- use `nsys profile --trace=cuda,nvtx,osrt --stats=true` for launch gaps,
  CPU submission overhead, stream overlap, and CUDA timeline evidence;
- use `/usr/local/cuda-12.8/bin/ncu --set speed-of-light` first for
  representative kernels, then add targeted memory/occupancy sections if needed;
- use explicit `/usr/local/cuda-12.8/bin/*` paths until the persistent shell
  `PATH` includes CUDA Toolkit binaries.

**Remaining caveat**: The default sandbox still does not expose `/dev/nvidia*`.
GPU/profiler runs should continue through the GPU-visible escalated path unless
the sandbox configuration changes.

**Artifact**:
- `tooling-gap-audit.md`

## 2026-05-28 — Docker GPU Container Handoff

**Purpose**: Preserve the current auto-research state so a new session can
continue H5 experiments without rediscovering the GPU/container setup.

**Containerization result**:
- Built `auto-research-gpu:cu128` from `docker/Dockerfile.gpu`.
- Docker daemon stores image layers under `/data3/docker`.
- Final image ID is `292c3482f072`; image size is `17.6GB`.
- Base image is `nvidia/cuda:12.8.0-devel-ubuntu22.04`.
- The Dockerfile was corrected so PyTorch installs from the CUDA 12.8 wheel
  channel instead of accidentally pulling CUDA 13 wheels from default PyPI.

**Verified versions**:
- `node v20.20.2`
- `codex-cli 0.134.0`
- `torch 2.11.0+cu128`
- `triton 3.6.0`

**Running container**:
- Name: `auto-research-gpu-dev`
- Network: `host`
- IPC: `host`
- Project bind mount:
  `/data3/auto_research -> /workspace/auto_research`
- Cache mount: `/data3/auto_research_docker_cache`
- Codex home mount: `/data3/auto_research_codex_home`

**GPU smoke test**:
- Container PyTorch reported `cuda_available True`.
- It saw `2` GPUs.
- Device 0 was `NVIDIA GeForce RTX 4090`.
- A CUDA matmul smoke test completed successfully.

**Continuation modes**:
- Method 2: keep the outer VS Code/Codex session and run experiments with
  `docker exec auto-research-gpu-dev ...`. This preserves conversation context
  while making GPU/profiler runs occur inside the container.
- Method 3: attach VS Code to `auto-research-gpu-dev`, open
  `/workspace/auto_research`, and start `codex -C /workspace/auto_research`
  inside the container. This makes the agent process and experiments both run
  inside Docker. Codex CLI is installed; Claude CLI is not installed yet.

**Decision**: New sessions should first read `SESSION_HANDOFF.md`,
`research-state.yaml`, `findings.md`, and
`synthesis/runtime-resource-scheduling-direction.md`. The next research step
remains H5.1: a narrow runtime resource-aware micro-task scheduling experiment
with `nsys` timeline evidence and steady-state versus dynamic compilation/cache
overhead separated.

**Artifacts**:
- `SESSION_HANDOFF.md`
- `docker/Dockerfile.gpu`
- `docker/run_auto_research_gpu.sh`
- `docker/README.md`
## 2026-05-28 — H5.1 Runtime Resource-Aware Selector

**Protocol**: Created `experiments/h5-1-runtime-selector/protocol.md`.

**Tooling fix**: The active container could see CUDA/Triton but lacked Python
3.10 development headers, `column`, and PATH entries for Nsight tools. Because
root `apt` was unavailable and `.bashrc` is read-only, persistent local tools
were installed under `/home/descfly/.local/devtools` and activated with:

```bash
source /home/descfly/.local/devtools/activate-auto-research.sh
```

This exposes local Python headers for Triton helper compilation plus `column`,
`nsys`, and `ncu` wrappers.

**Method**: Built a six-step dependency-aware queue over a Triton fused
GELU/residual micro-op. Calibrated 10 variants across idle, compute-load, and
memory-load states for small and large task shapes. Compared static isolated,
static average, load-aware, resource-aware, and oracle/context table policies.

**Valid results**:
- Static-best-average median queue latency: `2.1977 ms`.
- Resource-aware median queue latency: `2.2004 ms` (`1.0044x` vs static isolated).
- Static-best-isolated median queue latency: `2.2101 ms`.
- Policy repeats had `8.3%` to `10.6%` min-to-max spread, much larger than the
  sub-1% median policy differences.
- Calibration still showed real context dependence: best variants changed across
  contexts, with `11.8%` to `74.4%` worst/best step-latency spread.
- First invocation compile/cache proxy was large: median `105.69 ms`, max
  `824.63 ms`, total `1938.47 ms` across 10 variants.

**Nsight Systems smoke**: Captured
`experiments/h5-1-runtime-selector/results/nsys_h5_1_tooling_smoke.nsys-rep`
and `.sqlite`. In the profiled reduced run, CUDA API time was dominated by
`cudaLaunchKernel` and `cuLibraryLoadData`, while total GPU time for
`h5_fused_gelu_residual_kernel` was only hundreds of microseconds.

**Decision**: Mark H5.1 as mixed. Runtime context affects best variant choice,
but Python/CUDA-stream queue dispatch and dynamic loading currently dominate the
end-to-end objective. Move next to H5.2 compile/cache/launch overhead and H5.4
Nsight runtime-limit analysis before trusting another scheduler policy.

**Artifacts**:
- `experiments/h5-1-runtime-selector/code/run_runtime_selector.py`
- `experiments/h5-1-runtime-selector/analysis.md`
- `data/h5_1_runtime_selector_summary.csv`
- `data/h5_1_runtime_selector_policy_results.csv`
- `to_human/2026-05-28-h5-1-runtime-selector.html`


## 2026-05-28 — H5.2 Compile/Cache Overhead

**Protocol**: Created `experiments/h5-2-compile-cache-overhead/protocol.md`.

**Method**: Reused five representative H5.1 fused GELU/residual Triton variants
and launched worker subprocesses with per-variant `TRITON_CACHE_DIR` directories.
For each variant, measured an empty-cache cold first launch, two cache-hit first
launches in a fresh worker, full worker process wall time, and warmed steady-
state CUDA event latency after warmup.

**Valid results**:
- Cold first launch wall time: min `1319.75 ms`, median `1391.29 ms`, max
  `1414.86 ms`.
- Cache-hit first launch wall time: min `608.68 ms`, median `661.03 ms`, max
  `740.17 ms`.
- Warmed steady-state event latency: min `0.1424 ms`, median `0.1439 ms`, max
  `0.1564 ms`.
- Median cold overhead versus steady state: `9258.75x`.
- Median cache-hit first-launch overhead versus steady state: `4637.46x`.
- Cold process wall median: `4126.22 ms`; cache-hit process wall median:
  `3383.79 ms`.

**Interpretation**: H5.2 strongly supports the H5.1 overhead diagnosis. Dynamic
compilation cannot sit on the critical path for fine-grained micro-op variants,
and even a cache hit in a fresh process is not cheap enough for per-request
selection. Runtime selector work now needs a warmed persistent process,
ahead-of-time cache warming, or a coarser/persistent dispatch substrate.

**Decision**: Mark H5.2 complete and move to H5.4: Nsight Systems profiling of a
warmed queue-only run to separate CPU submission/API time, launch gaps, stream
overlap, and GPU kernel time. Delay H5.3 task migration until dispatch overhead
is either reduced or amortized with larger task tiles.

**Artifacts**:
- `experiments/h5-2-compile-cache-overhead/code/run_compile_cache_overhead.py`
- `experiments/h5-2-compile-cache-overhead/analysis.md`
- `experiments/h5-2-compile-cache-overhead/results/rtx4090_default/result.json`
- `data/h5_2_compile_cache_summary.csv`
- `data/h5_2_compile_cache_measurements.csv`
- `to_human/2026-05-28-h5-2-compile-cache-overhead.html`


## 2026-05-28 — H5.4 Warmed Queue Runtime Limits

**Protocol**: Created `experiments/h5-4-runtime-limit-profiling/protocol.md`.

**Method**: Reused H5.1 static-best-average and resource-aware policy selections
inside one persistent process. Precompiled all selected Triton variants, warmed
background compute/memory work, and warmed both policy queues before measuring.
The Nsight Systems run used `cudaProfilerStart/Stop` so the clean capture window
contained warmed queue submission plus one final synchronization; CUDA event and
wall timings were recorded outside the capture window.

**Valid results**:
- Static-best-average event median: `2.2662 ms`.
- Resource-aware event median: `2.2338 ms` (`1.0145x` vs static-best-average).
- Static-best-average min-to-max event spread: `6.85%`; resource-aware spread:
  `9.17%`.
- Clean Nsight capture: `96` queue submissions, `1152` kernels, `136.16 ms` CPU
  enqueue NVTX window, and `206.92 ms` GPU kernel span.
- Non-sync CUDA API time was `24.77 ms`; final synchronization wait was
  `71.36 ms`.
- PyTorch stream/event management accounted for about `16.21 ms`; CUDA
  runtime+driver launch wrappers accounted for `8.56 ms` across `576` launches.

**Interpretation**: H5.4 confirms that warming removes the extreme H5.2
compile/cache costs, but the Python/PyTorch stream-level queue is still too
expensive for fine-grained runtime-resource decisions. The resource-aware policy
can win slightly, but the effect is smaller than repeated timing spread.

**Decision**: Move next to a cheaper dispatch substrate: CUDA Graph replay over
the fixed dependency queue, persistent-kernel/batched dispatch, or larger task
tiles. Delay H5.3 task migration until dispatch overhead is reduced or amortized.

**Artifacts**:
- `experiments/h5-4-runtime-limit-profiling/code/run_warmed_queue_profile.py`
- `experiments/h5-4-runtime-limit-profiling/analysis.md`
- `experiments/h5-4-runtime-limit-profiling/results/rtx4090_default/nsys_summary.json`
- `experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.nsys-rep`
- `experiments/h5-4-runtime-limit-profiling/results/nsys_h5_4_warmed_queue.sqlite`
- `data/h5_4_warmed_queue_summary.csv`
- `data/h5_4_warmed_queue_measurements.csv`
- `to_human/2026-05-28-h5-4-warmed-queue-runtime-limits.html`


## 2026-05-28 — H5.5 CUDA Graph Dispatch

**Protocol**: Created `experiments/h5-5-cuda-graph-dispatch/protocol.md`.

**Method**: Reused H5.1 queue shapes and policy selections. Measured a plain
Python/PyTorch stream queue without internal NVTX ranges, captured one CUDA Graph
per policy, measured `graph.replay()` latency, and profiled graph replay with
Nsight Systems.

**Valid results**:
- Both `static_best_average` and `resource_aware` queues captured successfully.
- Static-best-average graph replay event median: `2.1368 ms` versus Python
  stream `2.2421 ms`, for `1.0493x` speedup.
- Resource-aware graph replay event median: `2.1418 ms` versus Python stream
  `2.2110 ms`, for `1.0323x` speedup.
- One-time capture wall time was `11.02 ms` for the first/static graph and
  `1.55 ms` for the second/resource graph.
- Graph replay profile for 96 queues had `194` CUDA runtime API calls total,
  `96` `cudaGraphLaunch` calls, and zero `cudaLaunchKernel` calls.
- CPU enqueue/NVTX window fell from H5.4's `136.16 ms` for 96 stream queues to
  `4.44 ms` for 96 graph replays, or from `1.418 ms` to `0.046 ms` per queue.

**Interpretation**: Graph replay gives only modest end-to-end event speedup
because GPU work still dominates, but it strongly fixes the host dispatch path.
H5 should use graph/persistent/batched dispatch as the substrate for future
resource-aware scheduling instead of a Python per-kernel queue.

**Decision**: Next H5 step should test graph-bank switching or task migration at
larger graph-level/batched units. Dynamic compilation remains out of the request
path, and individual-kernel migration through Python streams is no longer the
right baseline.

**Artifacts**:
- `experiments/h5-5-cuda-graph-dispatch/code/run_cuda_graph_dispatch.py`
- `experiments/h5-5-cuda-graph-dispatch/analysis.md`
- `experiments/h5-5-cuda-graph-dispatch/results/rtx4090_default/nsys_summary.json`
- `experiments/h5-5-cuda-graph-dispatch/results/nsys_h5_5_graph_replay.nsys-rep`
- `experiments/h5-5-cuda-graph-dispatch/results/nsys_h5_5_graph_replay.sqlite`
- `data/h5_5_cuda_graph_dispatch_summary.csv`
- `data/h5_5_cuda_graph_dispatch_measurements.csv`
- `to_human/2026-05-28-h5-5-cuda-graph-dispatch.html`
