// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename ThreadTile> // contiguous pixels(vector size) along seq<M, N>
struct Reduce2dShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t ThreadTile_M = ThreadTile::at(number<0>{});
    static constexpr index_t ThreadTile_N = ThreadTile::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t RepeatInWarp =
        Warp_M * Warp_N / ThreadTile_M / ThreadTile_N / ck_tile::get_warp_size();
    static constexpr index_t RepeatInWarp_M =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? RepeatInWarp : 1;
    static constexpr index_t RepeatInWarp_N =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? 1 : RepeatInWarp;

    static constexpr index_t ThreadPerWarp_M = Warp_M / ThreadTile_M / RepeatInWarp_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / ThreadTile_N / RepeatInWarp_N;

    static constexpr index_t Repeat_M = Block_M * RepeatInWarp_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N * RepeatInWarp_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t BlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};
} // namespace ck_tile
