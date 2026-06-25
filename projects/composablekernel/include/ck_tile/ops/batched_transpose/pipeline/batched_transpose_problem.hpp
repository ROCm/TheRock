// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {

template <typename DataType_,
          typename BlockTile, // Sequence<...
          typename WarpLayout,
          bool kPadM_ = false,
          bool kPadN_ = false> // Sequence<...
struct BatchedTransposeProblem
{
    using DataType = remove_cvref_t<DataType_>;

    static constexpr index_t kMPerWarp = WarpLayout::at(number<0>{});
    static constexpr index_t kNPerWarp = WarpLayout::at(number<1>{});

    static constexpr index_t kMPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t kNPerBlock = BlockTile::at(number<1>{});

    static constexpr index_t kBlockSize = kMPerWarp * kNPerWarp * get_warp_size();

    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;

    // 128-bit is the max single-instruction bandwidth for load/store
    static constexpr index_t MaxLoadStoreSize = 16;
    static constexpr index_t VectorSizeInput  = kPadN ? 1 : MaxLoadStoreSize / sizeof(DataType);
    static constexpr index_t VectorSizeOutput = kPadM ? 1 : MaxLoadStoreSize / sizeof(DataType);
};
} // namespace ck_tile
