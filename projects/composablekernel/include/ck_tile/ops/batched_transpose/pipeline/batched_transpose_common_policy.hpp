// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct BatchedTransposeCommonPolicy
{
    CK_TILE_DEVICE static constexpr auto TileAccessPattern =
        tile_distribution_pattern::thread_raked;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t kBlockSize         = Problem::kBlockSize;
        constexpr index_t kLeadDimPerBlock   = Problem::kNPerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kMPerBlock;

        constexpr index_t kVectorSize = Problem::VectorSizeInput;
        static_assert((kLeadDimPerBlock * kVectorSize) % kBlockSize == 0, "");
        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<kBlockSize,
                                                                          kSecondDimPerBlock,
                                                                          kLeadDimPerBlock,
                                                                          kVectorSize,
                                                                          TileAccessPattern>;
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }
};

} // namespace ck_tile
