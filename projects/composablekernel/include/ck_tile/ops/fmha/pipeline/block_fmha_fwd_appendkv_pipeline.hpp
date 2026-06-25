// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaFwdAppendKVPipelineDefaultPolicy>
struct BlockFmhaFwdAppendKVPipeline
{
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using QDataType = typename Problem::QDataType;
    using KDataType = typename Problem::KDataType;
    using VDataType = typename Problem::VDataType;

    using VLayout = typename Problem::VLayout;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0 = Problem::kM0;
    static constexpr index_t kN0 = Problem::kN0;
    static constexpr index_t kK0 = Problem::kK0;
    static constexpr index_t kN1 = Problem::kN1;

    static constexpr auto RotaryEnum = Problem::RotaryEnum;
    static constexpr bool kIsPagedKV = Problem::kIsPagedKV;

    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kK0 <= 32)
            {
                return 2;
            }
            else if constexpr(kK0 <= 64)
            {
                return 3;
            }
            else if constexpr(kK0 <= 128)
            {
                return 2;
            }
            else if constexpr(kK0 <= 256)
            {
                return 1;
            }
        }
    }();

    template <typename QDramBlockWindow,
              typename KDramBlockWindow,
              typename KPageBlockNavigator,
              typename KnewDramBlockWindow,
              typename VDramBlockWindow,
              typename VPageBlockNavigator,
              typename VnewDramBlockWindow,
              typename QElementFunction,
              typename KnewElementFunction,
              typename VnewElementFunction,
              typename QRotaryCosDramBlockWindow,
              typename QRotarySinDramBlockWindow,
              typename KnewRotaryCosDramBlockWindow,
              typename KnewRotarySinDramBlockWindow>
    CK_TILE_HOST_DEVICE auto
    operator()(QDramBlockWindow& q_dram_block_window, // M0*K0 tile
               const QElementFunction& q_element_func,
               KDramBlockWindow& k_dram_block_window, // N0*K0 tile
               index_t i_page_block_k,
               const KPageBlockNavigator& k_page_block_navigator,
               const KnewDramBlockWindow& knew_dram_block_window, // N0*K0 tile
               const KnewElementFunction& knew_element_func,
               VDramBlockWindow& v_dram_block_window, // N1*N0 tile
               index_t i_page_block_v,
               const VPageBlockNavigator& v_page_block_navigator,
               const VnewDramBlockWindow& vnew_dram_block_window, // N1*N0 tile
               const VnewElementFunction& vnew_element_func,
               const QRotaryCosDramBlockWindow q_rotary_cos_dram_block_window,
               const QRotarySinDramBlockWindow q_rotary_sin_dram_block_window,
               const KnewRotaryCosDramBlockWindow knew_rotary_cos_dram_block_window,
               const KnewRotarySinDramBlockWindow knew_rotary_sin_dram_block_window,
               index_t rotary_dim,
               bool skip_rotate_q,
               bool skip_rotate_append_kv) const
    {
        if(!skip_rotate_append_kv)
        {
            // append Knew to K
            auto knew_window = make_tile_window(
                knew_dram_block_window, Policy::template MakeKnewDramTileDistribution<Problem>());

            auto knew_tile = [&]() {
                auto knew = load_tile(knew_window);
                return tile_elementwise_in(knew_element_func, knew);
            }();

            // optionally apply rotary embedding to Knew
            if constexpr(RotaryEnum != RotaryEmbeddingEnum::NONE)
            {
                auto rotary_cos_window =
                    make_tile_window(knew_rotary_cos_dram_block_window,
                                     Policy::template MakeRotaryCosSinTileDistribution<
                                         Problem,
                                         /*IsRotaryCosSinForQ=*/false>());

                auto rotary_sin_window =
                    make_tile_window(knew_rotary_sin_dram_block_window,
                                     Policy::template MakeRotaryCosSinTileDistribution<
                                         Problem,
                                         /*IsRotaryCosSinForQ=*/false>());

                // We assume that each thread owns contiguous elements on head dimention. And we
                // will use the distribution to enable/disable threads in order to override partial
                // knew_tile content
                auto [thread_start, thread_end] =
                    Policy::template GetKnewThreadRangeAlongK<Problem>();
                ignore = thread_start;

                BlockRotaryEmbedding<RotaryEnum>::apply(knew_tile,
                                                        knew_window,
                                                        rotary_cos_window,
                                                        rotary_sin_window,
                                                        rotary_dim,
                                                        thread_end);
            }

            store_tile(k_dram_block_window, knew_tile);

            // write tile to another block if nesscary
            if constexpr(kIsPagedKV)
            {
                if(k_page_block_navigator.is_cross_block(i_page_block_k, k_dram_block_window))
                {
                    k_page_block_navigator.move_to_block(
                        i_page_block_k, k_dram_block_window, i_page_block_k + 1);
                    store_tile(k_dram_block_window, knew_tile);
                }
            }

            // append Vnew to V
            auto vnew_window = make_tile_window(
                vnew_dram_block_window, Policy::template MakeVnewDramTileDistribution<Problem>());

            auto vnew_tile = [&]() {
                auto vnew = load_tile(vnew_window);
                return tile_elementwise_in(vnew_element_func, vnew);
            }();

            store_tile(v_dram_block_window, vnew_tile);

            // write tile to another block if nesscary
            if constexpr(kIsPagedKV)
            {
                if(v_page_block_navigator.is_cross_block(i_page_block_v, v_dram_block_window))
                {
                    v_page_block_navigator.move_to_block(
                        i_page_block_v, v_dram_block_window, i_page_block_v + 1);
                    store_tile(v_dram_block_window, vnew_tile);
                }
            }
        }

        if(!skip_rotate_q)
        {
            // optionally apply rotary embedding to Q
            if constexpr(RotaryEnum != RotaryEmbeddingEnum::NONE)
            {
                auto q_window = make_tile_window(
                    q_dram_block_window, Policy::template MakeQDramTileDistribution<Problem>());

                auto q_tile = [&]() {
                    auto q = load_tile(q_window);
                    return tile_elementwise_in(q_element_func, q);
                }();

                auto rotary_cos_window =
                    make_tile_window(q_rotary_cos_dram_block_window,
                                     Policy::template MakeRotaryCosSinTileDistribution<
                                         Problem,
                                         /*IsRotaryCosSinForQ=*/true>());

                auto rotary_sin_window =
                    make_tile_window(q_rotary_sin_dram_block_window,
                                     Policy::template MakeRotaryCosSinTileDistribution<
                                         Problem,
                                         /*IsRotaryCosSinForQ=*/true>());

                // We assume that each thread owns contiguous elements on head dimention. And we
                // will use the distribution to enable/disable threads in order to override partial
                // q_tile content
                auto [thread_start, thread_end] = Policy::template GetQThreadRangeAlongK<Problem>();
                ignore                          = thread_start;

                BlockRotaryEmbedding<RotaryEnum>::apply(
                    q_tile, q_window, rotary_cos_window, rotary_sin_window, rotary_dim, thread_end);

                store_tile(q_dram_block_window, q_tile);
            }
        }
    }

    template <typename QDramBlockWindow,
              typename KDramBlockWindow,
              typename KPageBlockNavigator,
              typename KnewDramBlockWindow,
              typename VDramBlockWindow,
              typename VPageBlockNavigator,
              typename VnewDramBlockWindow,
              typename QRotaryCosDramBlockWindow,
              typename QRotarySinDramBlockWindow,
              typename KnewRotaryCosDramBlockWindow,
              typename KnewRotarySinDramBlockWindow>
    CK_TILE_HOST_DEVICE auto
    operator()(QDramBlockWindow& q_dram_block_window,
               KDramBlockWindow& k_dram_block_window,
               index_t i_page_block_k,
               const KPageBlockNavigator& k_page_block_navigator,
               const KnewDramBlockWindow& knew_dram_block_window,
               VDramBlockWindow& v_dram_block_window,
               index_t i_page_block_v,
               const VPageBlockNavigator& v_page_block_navigator,
               const VnewDramBlockWindow& vnew_dram_block_window,
               const QRotaryCosDramBlockWindow& q_rotary_cos_dram_block_window,
               const QRotarySinDramBlockWindow& q_rotary_sin_dram_block_window,
               const KnewRotaryCosDramBlockWindow& knew_rotary_cos_dram_block_window,
               const KnewRotarySinDramBlockWindow& knew_rotary_sin_dram_block_window,
               index_t rotary_dim,
               bool skip_rotate_q,
               bool skip_rotate_append_kv) const
    {
        return operator()(q_dram_block_window,
                          identity{},
                          k_dram_block_window,
                          i_page_block_k,
                          k_page_block_navigator,
                          knew_dram_block_window,
                          identity{},
                          v_dram_block_window,
                          i_page_block_v,
                          v_page_block_navigator,
                          vnew_dram_block_window,
                          identity{},
                          q_rotary_cos_dram_block_window,
                          q_rotary_sin_dram_block_window,
                          knew_rotary_cos_dram_block_window,
                          knew_rotary_sin_dram_block_window,
                          rotary_dim,
                          skip_rotate_q,
                          skip_rotate_append_kv);
    }
};

} // namespace ck_tile
