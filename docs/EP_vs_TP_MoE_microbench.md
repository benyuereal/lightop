# EP vs TP MoE micro-benchmark (GLM-5.3-Flash, gfx938)

Per-rank cost of the routed-expert MoE layer under the two deployment layouts,
measured on a single BW1100 (gfx938, 147440 MiB) with the LightOp W8A8 FP8
per-channel Marlin kernel.

## Setup

- Shapes: hidden 4096, moe_inter 2048, top-k 8, 288 routed experts.
- **EP8**: each rank owns 36 experts (288/8) and runs the full expert width;
  a global batch of B tokens leaves B/8 tokens per rank. Second column adds 96
  redundant experts (`--ep-num-redundant-experts 96`), i.e. 48 local experts.
- **TP8**: each rank keeps all 288 experts and runs 1/8 of each expert GEMM
  (`N1/8 = 512`, `K2/8 = 256`); a global batch of B tokens is B tokens per rank.
- Both layouts therefore move the same per-rank FLOPs per token; the benchmark
  isolates the kernel, not the collectives.
- Timing: 3 warmup + 20 iterations, mean, wall clock with `torch.cuda.synchronize()`.
- Patched LightOp (`moe_align_block_size` sentinel fill) staged via `PYTHONPATH`;
  the stock binary cannot run E<=64 at all.

## Per-rank kernel time (ms)

| Global batch | EP8 E=36 | EP8 E=48 (R=96) | TP8 |
|---:|---:|---:|---:|
| 8 | 0.24 | 0.25 | 0.25 |
| 16 | 0.25 | 0.25 | 0.25 |
| 32 | 0.34 | 0.42 | 0.35 |
| 64 | 0.45 | 0.54 | 0.46 |
| 128 | 0.53 | 0.65 | 0.54 |
| 192 | 0.54 | 0.66 | 0.57 |
| 256 | 0.55 | 0.68 | - |

Observations:

- EP8 (E=36) and TP8 are within ~1-2% of each other across the whole range.
  Both read roughly the same expert weight bytes per rank per layer
  (EP: 36 full experts ~ 0.9 GB; TP: 288 shards ~ 0.9 GB), and both are
  bound by that weight stream, not by arithmetic.
- Redundant experts cost 10-20% at batch >= 32 (48 vs 36 experts read per rank).
  They buy tail-latency smoothing, not throughput.
- Effective ~12 TFLOPS/rank at batch 128 for both layouts. The MoE layer is
  HBM-bound: ~0.9 GB of expert weights per rank per layer at ~1.7 TB/s.
- At batch <= 16 both layouts are launch-latency bound (~0.24-0.25 ms).

## What the kernel numbers do not include

- **TP8 collectives**: two all-reduces of a 4096-wide bf16 activation per layer
  (attention output, MLP output), plus the same for attention itself. These are
  small, latency-dominated messages on 8 ranks.
- **EP8 dispatch/combine**: with `--moe-a2a-backend none` the experts are still
  reached through collectives; only `deepep` makes the traffic sparse. EP is not
  automatically communication-free.
- **Attention side**: DP attention is where TP8 pays the most (KV replicated on
  all 8 ranks, 16 FLOP/byte) versus DPA (64 heads per rank, 1/8 of the KV,
  ~128 FLOP/byte). That difference is not visible in this MoE-only benchmark.

Conclusion: the MoE kernel itself is layout-neutral on this model; EP8's value
is (a) dropping attention tensor parallelism and the 8x KV replication, and
(b) avoiding padded experts when the routing is skewed. Do not expect the MoE
kernel to be faster under EP just because it is EP.

## Known measurement caveats

- Co-tenant load: the runs happened while a separate service was loading weights
  on the same DCU; absolute numbers are conservative (roughly 5-10%).
- `M=64` rows of some Marlin configs carry a pre-existing tuning bug
  (`DELTA=2` with `BLOCK_SIZE_M=16`), which makes those configs numerically
  wrong (mean_abs ~0.107 vs reference). Timings remain representative; do not
  use those rows for correctness claims.
- Verify kernel output is finite and compared against a float32 reference before
  trusting any configuration change; `M=1/32` paths are deterministic, larger
  batches showed run-to-run variation on the stock binary.
