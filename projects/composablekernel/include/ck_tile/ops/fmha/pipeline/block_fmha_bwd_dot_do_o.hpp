// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdOGradDotO
{
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
    using DDataType     = remove_cvref_t<typename Problem::DDataType>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kVHeaddim   = Problem::kVHeaddim;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    template <typename ODramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename DDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void operator()(const ODramBlockWindowTmp& o_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        DDramBlockWindowTmp& d_dram_block_window_tmp,
                                        float p_undrop) const
    {
        static_assert(
            std::is_same_v<ODataType, remove_cvref_t<typename ODramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kBlockSize == ODramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kBlockSize ==
                              OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kBlockSize == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");

        auto o_dram_window =
            make_tile_window(o_dram_block_window_tmp.get_bottom_tensor_view(),
                             o_dram_block_window_tmp.get_window_lengths(),
                             o_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePreODramTileDistribution<Problem>());

        auto o = load_tile(o_dram_window);

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             do_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePreOGradDramTileDistribution<Problem>());

        auto do_ = load_tile(do_dram_window);

        // declare d
        constexpr auto d_dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                o.get_tile_distribution().get_static_tile_distribution_encoding(), sequence<1>{}));

        auto d = make_static_distributed_tensor<DDataType>(d_dstr);

        clear_tile(d); // Initialize D

        constexpr auto o_spans = decltype(o)::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                d(i_idx) +=
                    (type_convert<DDataType>(o[i_j_idx]) * type_convert<DDataType>(do_[i_j_idx]));
            });
        });

        tile_elementwise_inout([&p_undrop](auto& x) { x = x * p_undrop; }, d);

        store_tile(d_dram_block_window_tmp, d);
    }
};

} // namespace ck_tile
