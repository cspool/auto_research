# H4.9 Analysis: Held-Out Selected-Token Counts

Date: 2026-05-27

## Status

Supported.

H4.9 tests the H4.8 runtime selector on held-out selected-token counts: `512`
and `2048` selected tokens at `Q=4`. Static filtering again rejects every
observed shared-memory OOR variant. The H4.8 order/query selector remains
within `15%` of measured best in all `8/8` held-out cases, with mean regret
`4.66%` and max regret `9.92%`.

## Results

| Selector | Cases | Near-best cases | Mean regret | Max regret |
|---|---:|---:|---:|---:|
| H4.8 order/query selector | 8 | 8 | 4.66% | 9.92% |
| selected-token pressure oracle | 8 | 8 | 0.00% | 0.00% |

The oracle is retrospective and diagnostic. It shows that selected-token count
can improve exact variant choice, but H4.8 already satisfies the near-best
criterion without selected-token-count features.

## Held-Out Decisions

| Selected tokens | Order | H4.8 selected | Best measured | Regret |
|---:|---|---|---|---:|
| 512 | random segment | N64 D64 V64 Q4 W4 | N128 D64 V64 Q4 W8 | 4.02% |
| 512 | random sorted | N128 D128 V64 Q4 W4 | N64 D64 V64 Q4 W4 | 3.92% |
| 512 | random shuffled | N128 D64 V64 Q4 W8 | N128 D64 V64 Q4 W4 | 9.92% |
| 512 | clustered segment | N128 D128 V64 Q4 W4 | N128 D64 V64 Q4 W8 | 8.46% |
| 2048 | random segment | N64 D64 V64 Q4 W4 | N128 D64 V64 Q4 W8 | 1.64% |
| 2048 | random sorted | N128 D128 V64 Q4 W4 | N64 D64 V64 Q4 W4 | 3.58% |
| 2048 | random shuffled | N128 D64 V64 Q4 W8 | N64 D64 V64 Q4 W4 | 3.64% |
| 2048 | clustered segment | N128 D64 V128 Q4 W4 | N128 D64 V64 Q4 W8 | 2.08% |

## Interpretation

H4.9 is a stronger generalization result than H4.8. H4.8 was a retrospective
repair on held-out query counts. H4.9 changes a new shape axis and finds that
the existing order/query selector still chooses near-best valid variants in all
cases.

The exact best variant still shifts with selected-token count. The diagnostic
oracle reveals two pressure patterns:

- `512` selected tokens often favors `N128 D64 V64 Q4 W8` or a smaller `N64`
  tile, because fewer selected-token blocks make launch/reduction overhead more
  visible.
- `2048` selected tokens often favors `N64` for sorted/shuffled cases or `W8`
  for random/clustered cases, suggesting selected-token blocking and warp choice
  matter for exact tuning.

But these shifts are within the `15%` threshold for the H4.8 selector. That
means selected-token count is a refinement feature, not an immediate failure
mode for near-best runtime selection.

## Sanity Checks

- GPU sweeps ran sequentially outside the default sandbox because CUDA devices
  are not visible inside it.
- All eight held-out selected-token sweeps completed.
- Static feasibility rejected `8/8` observed OOR rows.
- The selected variants were all static-feasible and measured-valid.
- H4.9 scripts passed Python bytecode compilation.

## Limitations

- Only two selected-token counts were tested: `512` and `2048`.
- `Q=4`, `D=128`, `V=128`, and the six-variant candidate set stayed fixed.
- The selected-token pressure oracle is retrospective and should not be treated
  as a generalized selector.
- Hardware counters are still unavailable, so the explanation is based on
  latency and tile/block metadata.

## Next Step

Outer Loop 5 is now warranted. H4.1-H4.9 have produced a coherent vertical
story:

1. sparse Video/VLM retrieval must be lowered below framework loops;
2. full sparse attention fusion and online-softmax scaling are viable;
3. query grouping is necessary;
4. runtime variant selection can use static resource filtering, order stats,
   query pressure, and shape metadata.

The next synthesis should decide whether to:

- deepen with value-dimension held-out shapes (`V=64/256`);
- add a learned or table-driven selector over all H4 data;
- return to literature/hardware co-design and map the selected-token movement
  and online-reduction bottlenecks to accelerator/NPU primitives.

