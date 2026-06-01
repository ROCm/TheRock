// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmABQuantPipelineAgBgCrImplBase : public GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using AQuantBase = GemmAQuantPipelineAgBgCrImplBase<Problem, Policy>;
    using BQuantBase = GemmBQuantPipelineAgBgCrImplBase<Problem, Policy>;

    template <typename AQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetAQDramLoadWindow(const AQDramBlockWindowTmp& aq_dram_block_window_tmp) const
    {
        return AQuantBase{}.GetAQDramLoadWindow(aq_dram_block_window_tmp);
    }

    template <typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetBQDramLoadWindow(const BQDramBlockWindowTmp& bq_dram_block_window_tmp) const
    {
        return BQuantBase{}.GetBQDramLoadWindow(bq_dram_block_window_tmp);
    }
};

} // namespace ck_tile
