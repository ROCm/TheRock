// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/softmax.hpp"
#include "ck_tile/ops/topk.hpp"

namespace ck_tile {

struct TopkSoftmaxWarpPerRowPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        // TODO: Y dim must have one dim that is not reduced
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<Problem::IssuesPerCol,
                               Problem::WarpsPerBlock,
                               Problem::RowsPerWarpPerColIssue>,
                      sequence<Problem::IssuesPerRow, Problem::LanesPerRow, Problem::VectorSize>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Problem::LanesPerRow>, // repeat this one
                                       tuple<sequence<Problem::IssuesPerCol,
                                                      Problem::WarpsPerBlock,
                                                      Problem::RowsPerWarpPerColIssue>,
                                             sequence<1>>, // each row write out single element
                                       tuple<sequence<1>, sequence<1, 0>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSoftmax()
    {
        using softmax_problem = BlockSoftmax2DProblem<typename Problem::WeightType>;
        return BlockSoftmax2D<softmax_problem>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTopk()
    {
        using topk_problem = BlockTopkStream2DProblem<typename Problem::WeightType,
                                                      typename Problem::IndexType,
                                                      Problem::LanesPerRow>;
        // Note: replicate is LanesPerRow
        return BlockTopkStream2D<topk_problem>{};
    }
};
} // namespace ck_tile
