// Fast silu*up + per-row fp8 e4m3 quantization for MoE prefill (d=256/rank).
// Bit-identical to lightop fuse_silu_mul_fp8_quant: silu and the product are
// rounded through bf16, scale = max|y|/448 clamped to 1/(448*512), fp8 via
// cvt_pk_fp8. One warp per row with 16-byte vector loads/stores.
// Measured on BW1100 (gfx938): 0.492ms -> 0.323ms at [131072, 512] (1.52x).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

template <int RPW>
__global__ void silu_mul_fp8_kernel(const __hip_bfloat16* __restrict__ x,
                                    uint8_t* __restrict__ out,
                                    float* __restrict__ scales,
                                    int rows, int d) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int row0 = warp * RPW;
  if (row0 >= rows) return;
  constexpr int VEC = 8;
  const int base = lane * VEC;
  for (int rr = 0; rr < RPW; rr++) {
  const int wr = row0 + rr;
  if (wr >= rows) break;
  const __hip_bfloat16* row = x + (size_t)wr * 2 * d;
  const uint4 g4 = *reinterpret_cast<const uint4*>(row + base);
  const uint4 u4 = *reinterpret_cast<const uint4*>(row + d + base);
  const uint32_t gp[4] = {g4.x, g4.y, g4.z, g4.w};
  const uint32_t up[4] = {u4.x, u4.y, u4.z, u4.w};
  float vals[VEC];
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const __hip_bfloat162 g2 = *reinterpret_cast<const __hip_bfloat162*>(&gp[i]);
    const __hip_bfloat162 u2 = *reinterpret_cast<const __hip_bfloat162*>(&up[i]);
    const float gg[2] = {__bfloat162float(g2.x), __bfloat162float(g2.y)};
    const float uu[2] = {__bfloat162float(u2.x), __bfloat162float(u2.y)};
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const float s = gg[j] * __builtin_amdgcn_rcpf(
          1.0f + exp2f(-gg[j] * 1.4426950408889634f));
      const float sb = __bfloat162float(__float2bfloat16(s));
      const float y = __bfloat162float(__float2bfloat16(sb * uu[j]));
      vals[i * 2 + j] = y;
      m = fmaxf(m, fabsf(y));
    }
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
  const float scale = fmaxf(m / 448.0f, 1.0f / (448.0f * 512.0f));
  if (lane == 0) scales[wr] = scale;
  const float inv = 1.0f / scale;
  int32_t r[2];
  r[0] = __builtin_hcu_cvt_pk_fp8_f32(vals[0] * inv, vals[1] * inv, r[0], false);
  r[0] = __builtin_hcu_cvt_pk_fp8_f32(vals[2] * inv, vals[3] * inv, r[0], true);
  r[1] = __builtin_hcu_cvt_pk_fp8_f32(vals[4] * inv, vals[5] * inv, r[1], false);
  r[1] = __builtin_hcu_cvt_pk_fp8_f32(vals[6] * inv, vals[7] * inv, r[1], true);
  *reinterpret_cast<int2*>(out + (size_t)wr * d + base) = make_int2(r[0], r[1]);
  }
}

extern "C" int hcu_silu_mul_fp8_quant(void* x, void* out, void* scales,
                                      int rows, int d, void* stream) {
  if (d != 256 || rows <= 0) return -1;
  const int threads = 256, wpb = threads / 32;
  const int blocks = (rows + wpb - 1) / wpb;
  silu_mul_fp8_kernel<1><<<blocks, threads, 0, (hipStream_t)stream>>>(
      (const __hip_bfloat16*)x, (uint8_t*)out, (float*)scales, rows, d);
  return 0;
}

extern "C" int hcu_silu_mul_fp8_quant_r2(void* x, void* out, void* scales,
                                         int rows, int d, void* stream) {
  if (d != 256 || rows <= 0) return -1;
  const int threads = 256, wpb = threads / 32;
  const int blocks = (rows + wpb * 2 - 1) / (wpb * 2);
  silu_mul_fp8_kernel<2><<<blocks, threads, 0, (hipStream_t)stream>>>(
      (const __hip_bfloat16*)x, (uint8_t*)out, (float*)scales, rows, d);
  return 0;
}

extern "C" int hcu_silu_mul_fp8_quant_r4(void* x, void* out, void* scales,
                                         int rows, int d, void* stream) {
  if (d != 256 || rows <= 0) return -1;
  const int threads = 256, wpb = threads / 32;
  const int blocks = (rows + wpb * 4 - 1) / (wpb * 4);
  silu_mul_fp8_kernel<4><<<blocks, threads, 0, (hipStream_t)stream>>>(
      (const __hip_bfloat16*)x, (uint8_t*)out, (float*)scales, rows, d);
  return 0;
}
