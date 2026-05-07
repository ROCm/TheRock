// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename BlockTile_, typename BlockWarps_, typename WarpTile_>
struct TileFlatmmShape
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr auto idxM = number<0>{};
    static constexpr auto idxN = number<1>{};
    static constexpr auto idxK = number<2>{};

    static constexpr index_t NumWarps = reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(idxM);
    static constexpr index_t kN = BlockTile::at(idxN);
    static constexpr index_t kK = BlockTile::at(idxK);

    static constexpr index_t flatNPerWarp  = BlockWarps::at(idxN);
    static constexpr index_t flatKPerWarp  = WarpTile::at(idxK) * WarpTile::at(idxN);
    static constexpr index_t flatKPerBlock = flatKPerWarp * kK / WarpTile::at(idxK);

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "tile_flatmm_shape",
                      concat('x', kM, kN, kK, NumWarps),
                      concat('x', BlockWarps::at(idxM), BlockWarps::at(idxN), BlockWarps::at(idxK)),
                      concat('x', (WarpTile::at(idxM)), WarpTile::at(idxN), WarpTile::at(idxK)));
        // clang-format on
    }
};

} // namespace ck_tile
