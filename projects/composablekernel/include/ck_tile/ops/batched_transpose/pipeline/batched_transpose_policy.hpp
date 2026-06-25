// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "batched_transpose_common_policy.hpp"

namespace ck_tile {

struct BatchedTransposePolicy : public BatchedTransposeCommonPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::kMPerBlock;
        constexpr index_t NPerBlock   = Problem::kNPerBlock;
        constexpr index_t VecLoadSize = Problem::VectorSizeOutput;

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          MPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          TileAccessPattern>;
        return TileEncodingPattern::make_shuffled_2d_static_tile_distribution();
    }
};
} // namespace ck_tile
