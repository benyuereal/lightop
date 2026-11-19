// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <torch/extension.h>

namespace lightop {
namespace deep_gemm {

// Grouped (multi-row KV-reuse) paged MQA logits.
//
// Consecutive query rows that share a block-table entry (the EAGLE verify
// tokens of one request always do) consume one KV tile per wave instead of one
// per row.  Operand layout and FP32 accumulation order are identical to
// `paged_mqa_logits`, so the logits are bit-identical to the default kernel.
//
// Returns std::nullopt unless the opt-in configuration is enabled and the
// tensors match a supported layout; callers must then fall back to the
// default dispatch.
std::optional<torch::Tensor> maybe_paged_mqa_logits_grouped(
    const torch::Tensor& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const int64_t max_context_len,
    const bool clean_logits,
    const bool use_schedule_meta);

}  // namespace deep_gemm
}  // namespace lightop
