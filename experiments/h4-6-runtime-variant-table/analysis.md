# H4.6 Analysis: Runtime Variant Table for Grouped Sparse Attention

Date: 2026-05-27

## Status

Supported, with a small-data caveat.

H4.6 converts the H4.5 grouped sparse-attention measurements into a prototype
compiler/runtime variant table. The table uses tile metadata, a simple
shared-memory feasibility proxy, selected-token order statistics, and measured
latency. It successfully rejects every observed shared-memory-infeasible variant
and selects near-best valid variants.

## Results

### Static Feasibility

The static proxy
`shared_proxy = BLOCK_N * BLOCK_Q * (BLOCK_D + BLOCK_V)` rejected the only
observed failing variant family:

| Variant | Proxy | Limit | Observed status |
|---|---:|---:|---|
| N256 D64 V64 Q4 W4 | 131072 | 101376 | 4/4 OutOfResources |

Observed Triton error text reported `133120` bytes required versus a `101376`
byte hardware limit. The proxy is not an exact byte model, but it cleanly
separates the failing tile from all valid H4.5 tiles.

### Selector Quality

| Selector | Near-best orders | Mean regret | Max regret | Invalid selections |
|---|---:|---:|---:|---:|
| global_mean_latency | 3/4 | 3.47% | 12.06% | 0 |
| order_aware_rule_table | 4/4 | 0.00% | 0.00% | 0 |

The global selector chooses `N64 D64 V64 Q4 W4`, the lowest mean-latency valid
variant across H4.5. It is within `10%` of the best measured variant for
clustered, random segment, and random shuffled orders, but misses random sorted
by `12.06%`.

The order-aware table uses `order_span_mean`, `order_span_p95`, and
`monotonic_fraction`. It selects the measured-best valid variant for all four
H4.5 order modes:

| Order | Selected variant | Best variant | Regret |
|---|---|---|---:|
| clustered segment | N128 D64 V128 Q4 W4 | N128 D64 V128 Q4 W4 | 0.00% |
| random segment | N64 D64 V64 Q4 W4 | N64 D64 V64 Q4 W4 | 0.00% |
| random shuffled | N128 D64 V64 Q4 W8 | N128 D64 V64 Q4 W8 | 0.00% |
| random sorted | N128 D128 V64 Q4 W4 | N128 D128 V64 Q4 W4 | 0.00% |

## Interpretation

H4.6 supports the Outer Loop 4 claim that H4 is now a compiler/runtime problem,
not only a kernel-writing problem. H4.5 showed that grouped-query sparse
attention is faster than flat PyTorch and per-query repeated kernels. H4.6 shows
how a runtime could choose among grouped kernels:

1. filter resource-infeasible variants before launch;
2. use measured full-fragment latency rather than isolated-kernel intuition;
3. use selected-token order statistics to avoid the global selector's sorted
   order miss.

The sorted case is the useful discriminator. A global mean latency table picks
the small `N64 D64 V64 Q4 W4` tile because it is excellent for random segment
and acceptable elsewhere. But random sorted prefers `N128 D128 V64 Q4 W4`.
This is precisely the kind of shape-dependent decision that a compiler/runtime
table should make.

## Sanity Checks

- H4.6 is a table-selection experiment; no GPU execution was required.
- Inputs were the checked H4.5 CSV artifacts.
- The selector recorded zero invalid selections after static filtering.
- Static filtering rejected all four observed OOR rows.
- `evaluate_variant_table.py` passes Python bytecode compilation.

## Limitations

- The order-aware rule table is retrospective and hand-built from the H4.5
  order families. It demonstrates the mechanism, not generalization to unseen
  workloads.
- Only four order modes and one shape (`1024` selected tokens, `Q=4`,
  `D=128`, `V=128`) are covered.
- The shared-memory proxy is deliberately simple. A production compiler path
  should ingest actual Triton/LLVM metadata where available.
- The current table does not include hardware counters, occupancy, register
  pressure, or partial-buffer traffic.

## Next Step

Deepen H4.6 into H4.7 by adding one new shape axis and testing whether the table
continues to work:

- query counts such as `Q=2` and `Q=8`;
- selected-token counts such as `512` and `2048`;
- value dimensions such as `V=64` and `V=256`.

The next protocol should use held-out shapes, not only held-in order modes. That
would turn H4.6 from a retrospective runtime table into a stronger compiler
selection experiment.

