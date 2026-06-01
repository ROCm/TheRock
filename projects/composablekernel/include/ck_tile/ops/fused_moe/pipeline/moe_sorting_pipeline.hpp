// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fused_moe/pipeline/moe_sorting_policy.hpp"
#include <string>
#include <type_traits>

#ifndef TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
#define TOPK_SOFTMAX_USE_RAW_TILE_WINDOW 0
#endif

namespace ck_tile {

// template <typename Problem_, typename Policy_ = MoeSortingPolicy>
// struct MoeSortingPipeline
// {
//     // TODO: this kernel only support warp per row
//     using Problem    = remove_cvref_t<Problem_>;
//     using Policy     = remove_cvref_t<Policy_>;
//     using WeightType = typename Problem::WeightType;

//     template <typename TopkIdWindow, typename WeightWindow>
//     CK_TILE_DEVICE auto operator()(const TopkIdWindow& topk_id_window,
//                                    const WeightWindow& weight_window,
//                                     index_t* p_sorted_token_ids,
//                                     WeightType* p_sorted_weights,
//                                     index_t* p_sorted_expert_ids,
//                                     index_t* p_total_tokens_post_pad,
//                                     const index_t num_experts,
//                                     const index_t unit_size,
//                                     const size_t numel,
//                                     const index_t topk)
//     {
//     }
// };
} // namespace ck_tile
