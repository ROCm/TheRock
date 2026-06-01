// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename ThreadTile> // contiguous pixels(vector size) along seq<M, N>
struct PoolShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t ThreadTile_M = ThreadTile::at(number<0>{});
    static constexpr index_t ThreadTile_N = ThreadTile::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static_assert(Warp_M % ThreadTile_M == 0, "Warp_M must be divisible by ThreadTile_M");
    static_assert(Warp_N % ThreadTile_N == 0, "Warp_N must be divisible by ThreadTile_N");
    static_assert((Warp_M * Warp_N / ThreadTile_M / ThreadTile_N) % ck_tile::get_warp_size() == 0,
                  "Warp_M * Warp_N / ThreadTile_M / ThreadTile_N must be a multiple of warp size");

    // Scale factor to account for warp size
    // WarpSizeScaleFactor = warp tile/ thread tile / warp size
    static constexpr index_t WarpSizeScaleFactor =
        Warp_M * Warp_N / ThreadTile_M / ThreadTile_N / ck_tile::get_warp_size();

    static constexpr index_t WarpSizeScaleFactor_M =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? WarpSizeScaleFactor : 1;
    static constexpr index_t WarpSizeScaleFactor_N =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? 1 : WarpSizeScaleFactor;

    static constexpr index_t ThreadPerWarp_M = Warp_M / ThreadTile_M / WarpSizeScaleFactor_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / ThreadTile_N / WarpSizeScaleFactor_N;

    static_assert((Block_M * WarpSizeScaleFactor_M) % (WarpPerBlock_M * Warp_M) == 0,
                  "Block_M * WarpSizeScaleFactor_M must be divisible by WarpPerBlock_M * Warp_M");
    static_assert((Block_N * WarpSizeScaleFactor_N) % (WarpPerBlock_N * Warp_N) == 0,
                  "Block_N * WarpSizeScaleFactor_N must be divisible by WarpPerBlock_N * Warp_N");

    static constexpr index_t Repeat_M = Block_M * WarpSizeScaleFactor_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N * WarpSizeScaleFactor_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t BlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};
} // namespace ck_tile
