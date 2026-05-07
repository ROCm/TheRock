// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
struct ElementWiseDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>, // Replicate
                                       tuple<sequence<S::kRepeatM,
                                                      S::kWarpPerBlockM,
                                                      S::kThreadPerWarpM,
                                                      S::kVectorM>>,    // Hierarchical
                                       tuple<sequence<1>, sequence<1>>, // Parallel
                                       tuple<sequence<1>, sequence<2>>, // Parallel
                                       sequence<1, 1>,                  // Yield
                                       sequence<0, 3>>{}                // Yield
        );
    }
};

} // namespace ck_tile
