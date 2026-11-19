// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Grouped paged MQA logits for the decode shapes of GLM-5.3-Flash on gfx938.
//
// Problem (measured, 192 rows x 32K, H=32, D=128, page 64): the default BF16
// kernel moves 1.61 GB of index-K but only reaches 37 TFLOPS / 1.16 TB/s; a
// load-only ablation already hits the 1.74 TB/s read peak, so the KV fetch is
// not the limiter.  The six EAGLE verify tokens of a request address the same
// pages, so grouping rows lets one wave serve several rows from one KV tile:
// KV traffic, LDS reads and page-table work drop by kGroupRows while the MMAC
// count is unchanged.  Measured 1.21x @ 32K and 1.39x @ 128K, bit-identical.
//
// The kernel detects per tile whether the grouped rows really share a physical
// page and falls back to per-row loads when they do not, so arbitrary row
// orders stay correct (they just lose the reuse).

#include "paged_mqa_logits_grouped.h"

#include "paged_mqa_logits_impl.hpp"
#include "paged_mqa_logits_int8_impl.hpp"

#include "device_runtime.h"
#include "utils.h"

#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace lightop {
namespace deep_gemm {

namespace {

struct GroupedMqaConfig {
    int bf16_group_rows = 0;
    int int8_group_rows = 0;
    int min_rows = 24;
    int int8_min_pool_ctx = 16384;
};

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || value < 0 || value > 1024) return fallback;
    return static_cast<int>(value);
}

const GroupedMqaConfig& grouped_mqa_config() {
    static const GroupedMqaConfig cfg = [] {
        GroupedMqaConfig c;
        c.bf16_group_rows = env_int("LIGHTOP_PAGED_MQA_LOGITS_GROUP_BF16", 0);
        c.int8_group_rows = env_int("LIGHTOP_PAGED_MQA_LOGITS_GROUP_INT8", 0);
        c.min_rows = env_int("LIGHTOP_PAGED_MQA_LOGITS_GROUP_MIN_ROWS", 24);
        c.int8_min_pool_ctx =
            env_int("LIGHTOP_PAGED_MQA_LOGITS_GROUP_INT8_MIN_POOL_CTX", 16384);
        return c;
    }();
    return cfg;
}

// gfx938 (64 or 72 CUs) only; the tuning was measured there.
bool grouped_mqa_arch_ok(const torch::Tensor& q) {
    static const bool ok = [] {
        int device = 0;
        if (hipGetDevice(&device) != hipSuccess) return false;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        const std::string arch(prop.gcnArchName);
        const int cus = prop.multiProcessorCount;
        return arch.rfind("gfx938", 0) == 0 && (cus == 64 || cus == 72);
    }();
    (void)q;
    return ok;
}

inline int align_up_i(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// ---------------------------------------------------------------------------
// BF16, page-major cache [pages, 64, 1, 128]
// ---------------------------------------------------------------------------
template <int kGroupRows>
__attribute__((amdgpu_flat_work_group_size(1, 512))) __global__ void
paged_mqa_logits_bf16_grouped_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ kv_block,
    const float* __restrict__ weights,
    const int batch_size,
    const uint64_t kv_cache_stride_bytes,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* __restrict__ context_lens,
    float* __restrict__ logits,
    const int* __restrict__ block_table,
    const int ctas_per_group) {
    constexpr int kNumHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kBlockKv = 64;
    constexpr int kNumWarps = 4;
    constexpr int kKTile = 32;
    constexpr int kQTile = 16;
    constexpr int kKvTile = 16;
    constexpr int kQIter = 2;
    constexpr int kKBlocks = kHeadDim / kKTile;      // 4
    constexpr int kTokensPerCta = 256;
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    const int group = blockIdx.y;
    const int group_first = group * kGroupRows;
    const int valid_rows = min(kGroupRows, batch_size - group_first);
    const int warp_idx = threadIdx.x / 64;
    const int lane_idx = threadIdx.x % 64;

    int row_len[kGroupRows];
    #pragma unroll
    for (int r = 0; r < kGroupRows; ++r) {
        const int raw = r < valid_rows ? context_lens[group_first + r] : 0;
        row_len[r] = raw < 0 ? 0
                             : (raw > static_cast<int>(logits_stride)
                                    ? static_cast<int>(logits_stride)
                                    : raw);
    }

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    auto* smem = reinterpret_cast<at::BFloat16*>(smem_buffer);

    // Q and the head-gate weights stay in VGPRs for every grouped row.
    const int base_q = lane_idx % kQTile * kHeadDim + lane_idx / kQTile * 8;
    f16x8_t reg_q[kGroupRows][kQIter][kKBlocks];
    float reg_w[kGroupRows][kQIter * 4];
    #pragma unroll
    for (int r = 0; r < kGroupRows; ++r) {
        const int rr = min(r, valid_rows - 1);
        const auto* gQ = q + (group_first + rr) * kNumHeads * kHeadDim;
        const auto* gW = weights + (group_first + rr) * kNumHeads;
        #pragma unroll
        for (int i = 0; i < kQIter; ++i) {
            #pragma unroll
            for (int j = 0; j < kKBlocks; ++j) {
                const int pos = base_q + i * kQTile * kHeadDim + j * kKTile;
                reg_q[r][i][j] = *reinterpret_cast<const f16x8_t*>(&gQ[pos]);
            }
        }
        #pragma unroll
        for (int j = 0; j < kQIter; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int head = j * kQTile + k * 4 + lane_idx / kQTile;
                reg_w[r][j * 4 + k] = head < kNumHeads ? gW[head] : 0.0f;
            }
        }
    }
    __syncthreads();

    auto* smem_warp_kv = smem + warp_idx * (kKvTile * kHeadDim);
    const int base_kv = lane_idx % kKvTile * kHeadDim + lane_idx / kKvTile * 8;
    const int chunks =
        static_cast<int>((logits_stride + kTokensPerCta - 1) / kTokensPerCta);

    for (int chunk = blockIdx.x; chunk < chunks; chunk += ctas_per_group) {
        const int chunk_start = chunk * kTokensPerCta;
        const int page_start = chunk_start + warp_idx * kBlockKv;

        // A wave owns one 64-token page; all four 16-token tiles share it.
        int row_page[kGroupRows];
        #pragma unroll
        for (int r = 0; r < kGroupRows; ++r) {
            row_page[r] = (r < valid_rows && page_start < row_len[r])
                              ? block_table[(group_first + r) * block_table_stride +
                                            page_start / kBlockKv]
                              : -1;
        }
        bool same_page = row_page[0] >= 0;
        #pragma unroll
        for (int r = 1; r < kGroupRows; ++r) {
            if (r < valid_rows && row_page[r] != row_page[0]) same_page = false;
        }

        #pragma unroll
        for (int tile = 0; tile < kBlockKv / kKvTile; ++tile) {
            const int token_start = page_start + tile * kKvTile;
            f16x8_t reg_kv[kKBlocks];
            int cached_page = -2;
            #pragma unroll
            for (int r = 0; r < kGroupRows; ++r) {
                if (r >= valid_rows) continue;
                const int page = row_page[r];
                const bool kv_valid = page >= 0;
                if (page != cached_page) {
                    cached_page = page;
                    if (kv_valid) {
                        const auto* src = reinterpret_cast<const at::BFloat16*>(
                            reinterpret_cast<const uint8_t*>(kv_block) +
                            static_cast<int64_t>(page) * kv_cache_stride_bytes);
                        buffer_load_page_tile_lds<at::BFloat16, kHeadDim, kKvTile>(
                            src, token_start % kBlockKv, lane_idx, smem_warp_kv);
                        __builtin_amdgcn_sched_barrier(0);
                        asm volatile("s_waitcnt vmcnt(0)\n\t");
                        __builtin_amdgcn_sched_barrier(0);
                        #pragma unroll
                        for (int kb = 0; kb < kKBlocks; ++kb) {
                            const int lds_addr = static_cast<int>(
                                reinterpret_cast<size_t>(smem_warp_kv + base_kv +
                                                         kb * kKTile));
                            __builtin_amdgcn_sched_barrier(0);
                            asm volatile("ds_read_b128 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                                         : "=v"(reg_kv[kb])
                                         : "v"(lds_addr)
                                         :);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                }
                fp32x4 acc[kQIter];
                CLEAR_ACC(acc, kQIter);
                if (kv_valid) {
                    #pragma unroll
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        #pragma unroll
                        for (int i = 0; i < kQIter; ++i) {
                            builtin_b16_mmac<false>(reg_kv[kb].front,
                                                    reg_q[r][i][kb].front, acc[i]);
                            builtin_b16_mmac<false>(reg_kv[kb].rear,
                                                    reg_q[r][i][kb].rear, acc[i]);
                        }
                    }
                }
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < kQIter; ++j) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        sum += fmaxf(acc[j][k], 0.0f) * reg_w[r][j * 4 + k];
                    }
                }
                #pragma unroll
                for (uint32_t delta = 16; delta <= 32; delta <<= 1) {
                    sum += __shfl_down_sync(0xffffffffffffffffull, sum, delta);
                }
                if (lane_idx < kKvTile) {
                    const int token = token_start + lane_idx;
                    if (token < static_cast<int>(logits_stride)) {
                        logits[static_cast<uint64_t>(group_first + r) * logits_stride +
                               token] =
                            (kv_valid && token < row_len[r]) ? sum : -INFINITY;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 pooled cache [pages, 16, 1, 132] (kpool = 4: 128 int8 + fp32 scale)
// ---------------------------------------------------------------------------
template <int kGroupRows>
__attribute__((amdgpu_flat_work_group_size(1, 512))) __global__ void
paged_mqa_logits_int8_pooled_grouped_kernel(
    const int8_t* __restrict__ q,
    const uint8_t* __restrict__ kv_block,
    const float* __restrict__ kv_block_scales,
    const float* __restrict__ weights,
    const int batch_size,
    const uint64_t kv_cache_stride_bytes,
    const uint64_t logits_stride,
    const uint64_t block_table_stride,
    const int* __restrict__ context_lens,
    float* __restrict__ logits,
    const int* __restrict__ block_table,
    const int ctas_per_group) {
    constexpr int kNumHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kBlockKv = 16;
    constexpr int kKTile = 64;
    constexpr int kKvTile = 16;
    constexpr int kQIter = 2;
    constexpr int kKBlocks = kHeadDim / kKTile;      // 2
    constexpr int kTokensPerCta = 256;
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    const int group = blockIdx.y;
    const int group_first = group * kGroupRows;
    const int valid_rows = min(kGroupRows, batch_size - group_first);
    const int warp_idx = threadIdx.x / 64;
    const int lane_idx = threadIdx.x % 64;

    int row_len[kGroupRows];
    #pragma unroll
    for (int r = 0; r < kGroupRows; ++r) {
        const int raw = r < valid_rows ? context_lens[group_first + r] : 0;
        row_len[r] = raw < 0 ? 0
                             : (raw > static_cast<int>(logits_stride)
                                    ? static_cast<int>(logits_stride)
                                    : raw);
    }

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    auto* smem = smem_buffer;

    const int base_q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
    i8x16_t reg_q[kGroupRows][kQIter][kKBlocks];
    float reg_w[kGroupRows][kQIter * 4];
    #pragma unroll
    for (int r = 0; r < kGroupRows; ++r) {
        const int rr = min(r, valid_rows - 1);
        const auto* gQ = q + (group_first + rr) * kNumHeads * kHeadDim;
        const auto* gW = weights + (group_first + rr) * kNumHeads;
        #pragma unroll
        for (int i = 0; i < kQIter; ++i) {
            #pragma unroll
            for (int j = 0; j < kKBlocks; ++j) {
                reg_q[r][i][j] = *reinterpret_cast<const i8x16_t*>(
                    &gQ[base_q + i * 16 * kHeadDim + j * kKTile]);
            }
        }
        #pragma unroll
        for (int j = 0; j < kQIter; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int head = j * 16 + k * 4 + lane_idx / 16;
                reg_w[r][j * 4 + k] = head < kNumHeads ? gW[head] : 0.0f;
            }
        }
    }
    __syncthreads();

    auto* smem_warp_kv = reinterpret_cast<int8_t*>(smem) +
                         warp_idx * (kKvTile * kHeadDim);
    // LDS holds a linear page copy: row stride kHeadDim, k-block stride kKTile
    const int base_kv = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
    const int scale_stride = static_cast<int>(kv_cache_stride_bytes / sizeof(float));
    const int scale_base = kBlockKv * kHeadDim;   // bytes from page start
    const int chunks =
        static_cast<int>((logits_stride + kTokensPerCta - 1) / kTokensPerCta);

    for (int chunk = blockIdx.x; chunk < chunks; chunk += ctas_per_group) {
        const int chunk_start = chunk * kTokensPerCta;
        const int page_start = chunk_start + warp_idx * 64;
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int token_start = page_start + tile * kKvTile;
            const int page_index = token_start / kBlockKv;
            int row_page[kGroupRows];
            #pragma unroll
            for (int r = 0; r < kGroupRows; ++r) {
                row_page[r] = (r < valid_rows && token_start < row_len[r])
                                  ? block_table[(group_first + r) *
                                                    block_table_stride +
                                                page_index]
                                  : -1;
            }
            bool same_page = row_page[0] >= 0;
            #pragma unroll
            for (int r = 1; r < kGroupRows; ++r) {
                if (r < valid_rows && row_page[r] != row_page[0]) same_page = false;
            }

            i8x16_t reg_kv[kKBlocks];
            float scale = 0.0f;
            int cached_page = -2;
            #pragma unroll
            for (int r = 0; r < kGroupRows; ++r) {
                if (r >= valid_rows) continue;
                const int page = row_page[r];
                const bool kv_valid = page >= 0;
                if (page != cached_page) {
                    cached_page = page;
                    if (kv_valid) {
                        const auto* src = reinterpret_cast<const int8_t*>(
                            kv_block + static_cast<int64_t>(page) *
                                           kv_cache_stride_bytes);
                        buffer_load_page_tile_lds<int8_t, kHeadDim, kKvTile>(
                            src, 0, lane_idx, smem_warp_kv);
                        __builtin_amdgcn_sched_barrier(0);
                        asm volatile("s_waitcnt vmcnt(0)\n\t");
                        __builtin_amdgcn_sched_barrier(0);
                        // token_start is page aligned for the pooled layout
                        scale = reinterpret_cast<const float*>(
                            reinterpret_cast<const uint8_t*>(kv_block_scales) +
                            static_cast<int64_t>(page) * kv_cache_stride_bytes)
                                    [token_start % kBlockKv + (lane_idx % 16)];
                        (void)scale_base;
                        #pragma unroll
                        for (int kb = 0; kb < kKBlocks; ++kb) {
                            const int lds_addr = static_cast<int>(
                                reinterpret_cast<size_t>(smem_warp_kv + base_kv +
                                                         kb * kKTile));
                            __builtin_amdgcn_sched_barrier(0);
                            asm volatile("ds_read_b128 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                                         : "=v"(reg_kv[kb])
                                         : "v"(lds_addr)
                                         :);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                }
                int32x4_t acc[kQIter];
                #pragma unroll
                for (int i = 0; i < kQIter; ++i) acc[i] = {0, 0, 0, 0};
                if (kv_valid) {
                    #pragma unroll
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        #pragma unroll
                        for (int i = 0; i < kQIter; ++i) {
                            builtin_i8_mmac(reg_kv[kb].front, reg_q[r][i][kb].front,
                                            acc[i]);
                            builtin_i8_mmac(reg_kv[kb].rear, reg_q[r][i][kb].rear,
                                            acc[i]);
                        }
                    }
                }
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < kQIter; ++j) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        sum += fmaxf(static_cast<float>(acc[j][k]), 0.0f) *
                               reg_w[r][j * 4 + k];
                    }
                }
                sum *= scale;
                #pragma unroll
                for (uint32_t delta = 16; delta <= 32; delta <<= 1) {
                    sum += __shfl_down_sync(0xffffffffffffffffull, sum, delta);
                }
                if (lane_idx < kKvTile) {
                    const int token = token_start + lane_idx;
                    if (token < static_cast<int>(logits_stride)) {
                        logits[static_cast<uint64_t>(group_first + r) * logits_stride +
                               token] =
                            (kv_valid && token < row_len[r]) ? sum : -INFINITY;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
}  // namespace

std::optional<torch::Tensor> maybe_paged_mqa_logits_grouped(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int64_t max_context_len,
    const bool clean_logits,
    const bool use_schedule_meta) {
    const GroupedMqaConfig& cfg = grouped_mqa_config();
    if (!clean_logits || use_schedule_meta) return std::nullopt;
    if (q.dim() != 4 || q.size(1) != 1 || q.size(2) != 32 || q.size(3) != 128) {
        return std::nullopt;
    }
    if (!q.is_contiguous() || !weights.is_contiguous() ||
        !context_lens.is_contiguous() || block_table.dim() != 2 ||
        !block_table.is_contiguous()) {
        return std::nullopt;
    }
    if (weights.scalar_type() != torch::kFloat ||
        context_lens.scalar_type() != torch::kInt32 ||
        block_table.scalar_type() != torch::kInt32) {
        return std::nullopt;
    }
    const int batch_size = static_cast<int>(q.size(0));
    const int64_t max_block_len = block_table.size(1);
    if (max_context_len <= 0 || max_context_len > max_block_len * 64) {
        return std::nullopt;
    }
    if (!grouped_mqa_arch_ok(q) || batch_size < cfg.min_rows) {
        return std::nullopt;
    }

    const bool bf16_layout =
        q.scalar_type() == torch::kBFloat16 &&
        fused_kv_cache.scalar_type() == torch::kBFloat16 &&
        fused_kv_cache.dim() == 4 && fused_kv_cache.size(1) == 64 &&
        fused_kv_cache.size(2) == 1 && fused_kv_cache.size(3) == 128;
    const bool int8_layout =
        q.scalar_type() == torch::kInt8 &&
        fused_kv_cache.scalar_type() == torch::kInt8 &&
        fused_kv_cache.dim() == 4 && fused_kv_cache.size(1) == 16 &&
        fused_kv_cache.size(2) == 1 && fused_kv_cache.size(3) == 132;

    torch::Tensor logits;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (bf16_layout && cfg.bf16_group_rows >= 2) {
        const int block_kv = 64;
        const int aligned = align_up_i(static_cast<int>(max_context_len), block_kv);
        const int64_t stride_bytes =
            fused_kv_cache.stride(0) * fused_kv_cache.element_size();
        logits = torch::empty({batch_size, aligned},
                              q.options().dtype(torch::kFloat));
        const int chunks = (aligned + 255) / 256;
        const int ctas = std::min(chunks, 64);
        const int groups = (batch_size + cfg.bf16_group_rows - 1) /
                           cfg.bf16_group_rows;
        const int smem = 4 * 16 * 128 * static_cast<int>(sizeof(at::BFloat16));
#define LIGHTOP_LAUNCH_BF16_GROUPED(G)                                        \
    paged_mqa_logits_bf16_grouped_kernel<G>                                   \
        <<<dim3(ctas, groups), dim3(256), smem, stream>>>(                    \
            reinterpret_cast<at::BFloat16*>(q.data_ptr()),                    \
            reinterpret_cast<at::BFloat16*>(fused_kv_cache.data_ptr()),       \
            weights.data_ptr<float>(), batch_size, stride_bytes, aligned,     \
            block_table.stride(0), context_lens.data_ptr<int32_t>(),          \
            logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), ctas)
        switch (cfg.bf16_group_rows) {
            case 2: LIGHTOP_LAUNCH_BF16_GROUPED(2); break;
            case 3: LIGHTOP_LAUNCH_BF16_GROUPED(3); break;
            case 4: LIGHTOP_LAUNCH_BF16_GROUPED(4); break;
            case 6: LIGHTOP_LAUNCH_BF16_GROUPED(6); break;
            default: return std::nullopt;
        }
#undef LIGHTOP_LAUNCH_BF16_GROUPED
        return logits.slice(-1, 0, max_context_len);
    }

    if (int8_layout && cfg.int8_group_rows >= 2) {
        const int block_kv = 16;
        const int aligned = align_up_i(static_cast<int>(max_context_len), block_kv);
        const int64_t stride_bytes =
            fused_kv_cache.stride(0) * fused_kv_cache.element_size();
        if (max_context_len < cfg.int8_min_pool_ctx && batch_size < 96) {
            return std::nullopt;
        }
        logits = torch::empty({batch_size, aligned},
                              q.options().dtype(torch::kFloat));
        const int chunks = (aligned + 255) / 256;
        const int ctas = std::min(chunks, 64);
        const int groups = (batch_size + cfg.int8_group_rows - 1) /
                           cfg.int8_group_rows;
        const int smem = 4 * 16 * 128;
        const auto* kv_bytes =
            reinterpret_cast<const uint8_t*>(fused_kv_cache.data_ptr());
        const auto* scales = reinterpret_cast<const float*>(
            kv_bytes + block_kv * 128);
#define LIGHTOP_LAUNCH_INT8_GROUPED(G)                                        \
    paged_mqa_logits_int8_pooled_grouped_kernel<G>                            \
        <<<dim3(ctas, groups), dim3(256), smem, stream>>>(                    \
            q.data_ptr<int8_t>(), kv_bytes, scales,                           \
            weights.data_ptr<float>(), batch_size, stride_bytes, aligned,     \
            block_table.stride(0), context_lens.data_ptr<int32_t>(),          \
            logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), ctas)
        switch (cfg.int8_group_rows) {
            case 2: LIGHTOP_LAUNCH_INT8_GROUPED(2); break;
            case 3: LIGHTOP_LAUNCH_INT8_GROUPED(3); break;
            case 4: LIGHTOP_LAUNCH_INT8_GROUPED(4); break;
            case 6: LIGHTOP_LAUNCH_INT8_GROUPED(6); break;
            default: return std::nullopt;
        }
#undef LIGHTOP_LAUNCH_INT8_GROUPED
        return logits.slice(-1, 0, max_context_len);
    }

    return std::nullopt;
}

}  // namespace deep_gemm
}  // namespace lightop
