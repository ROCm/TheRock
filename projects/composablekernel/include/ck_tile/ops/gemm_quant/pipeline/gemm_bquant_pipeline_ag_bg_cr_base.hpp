// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmBQuantPipelineAgBgCrImplBase : public GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using Base           = GemmPipelineAgBgCrImplBase<Problem, Policy>;
    using ADataType      = typename Base::ADataType;
    using ALayout        = typename Base::ALayout;
    using BDataType      = typename Base::BDataType;
    using BLayout        = typename Base::BLayout;
    using BlockGemmShape = typename Base::BlockGemmShape;
    using QuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t NPerBlockBQ = NPerBlock / QuantGroupSize::kN;
    static constexpr index_t KPerBlockBQ = KPerBlock / QuantGroupSize::kK;

    static_assert(NPerBlockBQ >= 1, "NPerBlock must be >= QuantGroupSize");
    static_assert(KPerBlockBQ >= 1, "KPerBlock must be >= QuantGroupSize");

    static_assert(NPerBlock % QuantGroupSize::kN == 0,
                  "NPerBlock must be a multiple of QuantGroupSize::kN");
    static_assert(KPerBlock % QuantGroupSize::kK == 0,
                  "KPerBlock must be a multiple of QuantGroupSize::kK");

    // Create DRAM tile window for BQ
    template <typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetBQDramLoadWindow(const BQDramBlockWindowTmp& bq_dram_block_window_tmp) const
    {
        using YPerTile =
            std::conditional_t<std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                               number<NPerBlockBQ>,
                               number<KPerBlockBQ>>;
        using XPerTile =
            std::conditional_t<std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                               number<KPerBlockBQ>,
                               number<NPerBlockBQ>>;

        auto bq_copy_dram_window =
            make_tile_window(bq_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             bq_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBQDramTileDistribution<Problem>());
        return bq_copy_dram_window;
    }
};

} // namespace ck_tile
