# H4.5 Protocol: Multi-Query Sparse KV Online Softmax

Date: 2026-05-26

## Hypothesis

H4.5 deepens H4 by moving from one sparse-attention query to a small
multi-query/multi-head fragment. If several query vectors reuse the same
selected K/V set, a compiler-visible grouped-query sparse attention kernel
should amortize selected-token movement, K/V reads, score computation, and
kernel-launch overhead better than treating every query as a separate sparse
attention request.

## Confirmatory Prediction

For one request, `Q=4` query vectors, `32` segments, and `32` selected tokens
per segment (`1024` selected tokens total), a grouped-query two-stage Triton
online-softmax sparse attention implementation will beat flat PyTorch eager
gather+GEMM+softmax+value by at least `2.0x` on at least three of four selected
token orders.

## Secondary Measurements

- Compare grouped-query latency with a simple `Q * H4.4 one-query best` estimate
  for repeated per-query online kernels. This is not a separately measured
  baseline, but it exposes the launch and repeated-K/V-work pressure that a
  compiler would face if it lowered every query independently.
- Report per-query amortization: grouped-query latency divided by `Q`.
- Track selected-token order statistics: mean neighbor span, p95 span, and
  monotonic fraction.
- Report validation error against PyTorch eager for every Triton variant.

## Workload

- One request.
- Shared selected-token indices across all query vectors.
- `tokens_per_segment=256`.
- Default selected-token orders:
  - `random_segment`
  - `random_sorted`
  - `random_shuffled`
  - `clustered_segment`
- Default dimensions: `key_dim=128`, `value_dim=128`, `query_count=4`.
- Inputs: fp16 K/V/Q, fp32 score/softmax/value accumulation.

## Baselines

1. Flat PyTorch eager:
   - materialize `K_selected` and `V_selected`;
   - compute `scores = K_selected @ Q`;
   - softmax over selected tokens for each query;
   - compute `out = softmax(scores).T @ V_selected`.
2. Grouped-query Triton online softmax:
   - partial kernel computes per selected-token block and value block:
     local max, local denominator, and partial value output for all queries in a
     query block;
   - reduce kernel combines selected-token blocks with online-softmax rescaling
     and writes `Q x value_dim` output.

## Variant Sweep

Each variant is `(BLOCK_N, BLOCK_D, BLOCK_V, BLOCK_Q, warps)`.

Default sweep:

- `128,64,64,4,4`
- `128,64,128,4,4`
- `256,64,64,4,4`
- `128,128,64,4,4`
- `128,64,64,4,8`
- `64,64,64,4,4`

## Sanity Checks

- CUDA must be visible; otherwise the run is marked skipped.
- All outputs must be finite.
- Relative max error versus PyTorch eager should stay below `1e-4`.
- Runs are sequential to avoid cross-process GPU interference.

## Decision Rule

- **Supported** if grouped-query Triton beats flat PyTorch by at least `2.0x` in
  at least three order modes and validation passes.
- **Partially supported** if speedup is positive but below threshold, or if only
  specific order/tile regimes benefit.
- **Not supported** if grouped-query Triton is slower than flat PyTorch in most
  valid order modes.

