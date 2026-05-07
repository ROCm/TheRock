// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BlockWarps, typename BlockTile, typename WarpTile, typename ComputeDataType>
struct ElementWiseShape
{
    static constexpr index_t kBlockM = BlockTile::at(number<0>{});

    static constexpr index_t kWarpM = WarpTile::at(number<0>{});

    static constexpr index_t kVectorM =
        min(static_cast<index_t>(16 / sizeof(ComputeDataType)), kWarpM / get_warp_size());

    static constexpr index_t kWarpPerBlockM = BlockWarps::at(number<0>{});

    static constexpr index_t kThreadPerWarpM = get_warp_size();

    static constexpr index_t kRepeatM = kBlockM / (kWarpPerBlockM * kVectorM * kThreadPerWarpM);

    static constexpr index_t kBlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};

} // namespace ck_tile
