# H5.1 Analysis - Resource-Shaped Runtime Selector

Date: 2026-05-28

## Question

Can a runtime choose Triton micro-operator variants from the current resource
state and beat static variant selection for a dependency-aware single-request
queue?

## Setup

The benchmark uses a fused Triton micro-op:

```text
Y = GELU(X * scale + Bias) + Residual
```

It calibrates 10 variants, `BLOCK_SIZE in {512,1024,2048,4096,8192}` and
`num_warps in {4,8}`, over two task shapes and three co-running resource states:
`idle`, `compute`, and `memory`. The queue has six dependent steps with changing
resource state:

```text
compute/small -> memory/large -> compute/large -> memory/small ->
compute/small -> memory/large
```

Policies compared:

- `static_best_isolated`: one variant chosen by isolated micro-op latency.
- `static_best_average`: one variant chosen by mean calibrated step latency.
- `load_aware`: one variant per current load class.
- `resource_aware`: one variant per `(load_class, task_shape)`.
- `oracle_context`: the calibrated context table upper-bound label.

The container was missing Python development headers and command wrappers. The
persistent local tool environment is now installed under
`/home/descfly/.local/devtools`; activate it with:

```bash
source /home/descfly/.local/devtools/activate-auto-research.sh
```

## Results

Primary queue results use the median of five randomly interleaved repeated
measurements:

| Policy | Queue ms | Speedup vs isolated static | Selection summary |
|---|---:|---:|---|
| static_best_average | 2.1977 | 1.0056x | one global `B8192_W8` |
| resource_aware | 2.2004 | 1.0044x | per load+shape variants |
| load_aware | 2.2058 | 1.0019x | per load variants |
| static_best_isolated | 2.2101 | 1.0000x | one global `B2048_W8` |
| oracle_context | 2.2561 | 0.9796x | same table as resource-aware |

The policy ordering should not be overinterpreted. The repeated queue
measurements had 8.3%-10.6% min-to-max spread per policy, much larger than the
sub-1% median differences among the practical policies.

The calibration table did show real context dependence. Best variants differed
by resource state and task shape:

| Context | Best variant | Best step ms | Worst/Best spread |
|---|---|---:|---:|
| compute, large | B512_W4 | 0.2456 | 18.7% |
| compute, small | B8192_W8 | 0.1718 | 74.4% |
| idle, large | B4096_W4 | 0.1425 | 17.1% |
| idle, small | B2048_W8 | 0.0995 | 11.8% |
| memory, large | B8192_W8 | 0.5654 | 15.2% |
| memory, small | B1024_W4 | 0.4900 | 15.1% |

Dynamic compile/cache proxy was large relative to steady-state execution:

- first invocation min: 70.97 ms;
- first invocation median: 105.69 ms;
- first invocation max: 824.63 ms;
- total first invocation over 10 variants: 1938.47 ms.

Sanity checks passed: CUDA was available, outputs were finite, and max absolute
error versus PyTorch GELU was `0.00390625` for the small shape and `0.0078125`
for the large shape.

## Nsight Systems Smoke Profile

A lightweight Nsight Systems run was captured after installing `nsys` locally:

- `experiments/h5-1-runtime-selector/results/nsys_h5_1_tooling_smoke.nsys-rep`
- `experiments/h5-1-runtime-selector/results/nsys_h5_1_tooling_smoke.sqlite`

The profile confirms that CPU launch/runtime overhead is a serious part of this
experiment. In the small profiled run, CUDA API time was dominated by
`cudaLaunchKernel` (95 calls, 110.4 ms total API time) and `cuLibraryLoadData`
(14 calls, 41.7 ms). GPU kernel execution itself was tiny by comparison:
`h5_fused_gelu_residual_kernel` had 142 instances with 272 us total GPU time.
This supports the interpretation that Python/CUDA submission and dynamic loading
can dominate fine-grained queue scheduling.

## Interpretation

H5.1 is a mixed result.

The positive signal is that resource context changes the best micro-op variant.
A static isolated selector chose `B2048_W8`, a static average selector chose
`B8192_W8`, and the resource-aware table chose different variants for compute,
memory, small, and large contexts.

The limiting signal is that a tiny Python/CUDA-stream queue does not turn those
calibrated differences into a robust end-to-end win. Queue-level jitter and
launch/submission overhead dominate the small policy differences. The fact that
the calibrated `oracle_context` table can measure slower than the identical
resource-aware table on repeated queue timing is not a scheduler insight; it is
a measurement/runtime-overhead warning.

This sharpens H5: runtime selection is not enough by itself. A useful runtime
must also manage launch overhead, submission jitter, warmup/cache state, and
resource observation. H5.2 and H5.4 should therefore be treated as first-class
experiments rather than diagnostics.

## What This Rules Out

- A simple offline variant table plus Python-level per-task dispatch is not a
  convincing runtime system for fine-grained single-request scheduling.
- Calibration best variants cannot be trusted as an end-to-end improvement
  unless queue execution overhead and measurement variance are controlled.
- Dynamic compilation on the critical path is untenable for this microbenchmark
  scale: first invocation costs tens to hundreds of milliseconds while queue
  execution is about 2.2 ms.

## Next

H5.2 should isolate cold compile, cache-hit lookup, and steady-state launch
costs for the same Triton variants. H5.4 should deepen the Nsight Systems
evidence for launch gaps, CPU submission overhead, stream overlap, and queue
jitter. A later H5.3 migration experiment should use larger task tiles or
persistent/batched submission so the scheduler has enough work per decision to
overcome dispatch noise.
