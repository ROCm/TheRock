// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"

#include "../block_level/practice_gemm_block_policy_agmem_bgmem_creg.hpp"
#include "../block_level/practice_gemm_block_pipeline_agmem_bgmem_creg.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename Shape_>
struct PracticeGemmHostProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using CDataType   = CDataType_;
    using AccDataType = AccDataType_;
    using Shape       = remove_cvref_t<Shape_>;
};

struct PracticeGemmHostPolicy
{
    CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
    {
        const auto unmerge = make_merge_transform(make_tuple(N0, M0));

        return [unmerge](index_t block_id) {
            multi_index<2> unmerged;
            unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

            return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPracticeGemmBlockPipeline()
    {
        using PracticeGemmBlockPipelineProblem_ =
            PracticeGemmBlockPipelineProblem<typename Problem::ADataType,
                                             typename Problem::BDataType,
                                             typename Problem::CDataType,
                                             typename Problem::AccDataType,
                                             typename Problem::Shape>;
        return PracticeGemmBlockPipelineAGmemBGmemCreg<PracticeGemmBlockPipelineProblem_>{};
    }
};
} // namespace ck_tile
