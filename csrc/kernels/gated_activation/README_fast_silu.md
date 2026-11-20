# fuse_silu_mul_fp8_quant_fast.hip

Specialised silu*up + per-row FP8 E4M3 quantization for the MoE prefill shape
used by GLM-5.3-Flash under TP8: `[rows, 2*256] bf16 -> [rows, 256] fp8 + [rows] f32`.

Motivation: the generic `fuse_silu_mul_fp8_quant` launches one 256-thread block
per row, of which only 32 threads are active when the per-rank width is 256, and
runs that path at 342 GB/s (0.492 ms for 131072 rows).

This kernel assigns one warp per row (32 lanes x 8 bf16 via 16-byte vector
loads/stores) and is **bit-identical** to the generic kernel: silu and the
gated product are rounded through bf16, `scale = max|y|/448` clamped to
`1/(448*512)`, FP8 conversion through `__builtin_hcu_cvt_pk_fp8_f32`.

Measured on BW1100 (gfx938, 64 CU), rows=131072, d=256:

| variant | ms | GB/s |
|---|---|---|
| generic `fuse_silu_mul_fp8_quant` | 0.492 | 342 |
| this kernel, 1 row/warp | 0.323 | 522 |
| this kernel, 2 rows/warp | 0.313 | 537 |
| this kernel, 4 rows/warp | 0.309 | 545 |

Build (standalone shared object):

    hipcc -O3 --offload-arch=gfx938 -shared -fPIC \
        -o silu_mul_fp8.so fuse_silu_mul_fp8_quant_fast.hip

Exports: `hcu_silu_mul_fp8_quant` (1 row/warp), `..._r2`, `..._r4`.
