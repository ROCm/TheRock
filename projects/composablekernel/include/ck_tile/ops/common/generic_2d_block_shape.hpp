// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

/*
// clang-format off

4-level descriptor: BlockTile-> WarpPerBlock-> WarpTile-> Vector

                         Block_N (Warp_N * WarpPerBlock_N * Repeat_N )
        +<----------------------< Repeat_N(2)>--------------------->+
        |                                                           |
        +<--    <WarpPerBlock_N(2)>  -->+
            Warp_N
        +--------------+--------------+--------------+--------------+----+----------------+
 Warp_M | wrap_0       | wrap_1       |                             |    ^                ^
        +--------------+--------------+                             |   <WarpPerBlock_M(2)> |
        | wrap_2       | wrap_3       |                             |    v
        +--------------+--------------+--------------+--------------+----+           Block_M
        |                             |                             |
        +                             +                             |
        |                             |                             |                     v
        +--------------+--------------+--------------+--------------+                     +

        each Warp-tile (e.g 16 thrd per row)

         Vector_N (contiguous pixels each thrd holds along N, or vector size)
        +-----------+-----------+-----------+-----------+-----------+
        | thrd_0    | thrd_1    | thrd_2    | thrd_3    | ...         Vector_M
        +-----------+-----------+-----------+-----------+-----------+
        | thrd_16   | thrd_17   | thrd_18   | thrd_19   | ...
        +-----------+-----------+-----------+-----------+-----------+
// clang-format on
*/
template <typename BlockTile_,      // block size, seq<M, N>
          typename ThreadPerBlock_, // num threads along seq<M, N>
          typename Vector_>         // contiguous pixels(vector size) along seq<M, N>)>
struct Generic2dBlockShape
{
    // block size
    static constexpr index_t Block_M          = BlockTile_::at(number<0>{});
    static constexpr index_t Block_N          = BlockTile_::at(number<1>{});
    static constexpr index_t ThreadPerBlock_M = ThreadPerBlock_::at(number<0>{});
    static constexpr index_t ThreadPerBlock_N = ThreadPerBlock_::at(number<1>{});

    // vector size along seq<M, N>
    static constexpr index_t Vector_M = Vector_::at(number<0>{});
    static constexpr index_t Vector_N = Vector_::at(number<1>{});

    // num warps along seq<M, N>, within each block
    template <bool isHostWave32>
    static constexpr index_t GetWarpPerBlock_M()
    {
        constexpr index_t warp_size    = isHostWave32 ? 32 : get_warp_size();
        constexpr bool is_warp_per_row = ThreadPerBlock_N <= warp_size;
        static_assert((ThreadPerBlock_M * ThreadPerBlock_N) % warp_size == 0);
        constexpr index_t total_warps = (ThreadPerBlock_M * ThreadPerBlock_N) / warp_size;

        if constexpr(is_warp_per_row)
        {
            static_assert(warp_size % ThreadPerBlock_N == 0);
            return total_warps * (warp_size / ThreadPerBlock_N);
        }
        else
        {
            // static_assert(ck_tile::get_warp_size() % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N / warp_size);
        }
    };

    // num of warps along n
    template <bool isHostWave32>
    static constexpr index_t GetWarpPerBlock_N()
    {
        constexpr index_t warp_size    = isHostWave32 ? 32 : get_warp_size();
        constexpr bool is_warp_per_row = ThreadPerBlock_N <= warp_size;
        if constexpr(is_warp_per_row)
        {
            static_assert(warp_size % ThreadPerBlock_N == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N % warp_size == 0);
            return ThreadPerBlock_N / warp_size;
        }
    }

    static constexpr index_t WarpPerBlock_M = GetWarpPerBlock_M<false>();
    static constexpr index_t WarpPerBlock_N = GetWarpPerBlock_N<false>();

    // warp size
    static constexpr index_t BlockSize = WarpPerBlock_M * WarpPerBlock_N * get_warp_size();
    static constexpr index_t Warp_M    = ThreadPerBlock_M / WarpPerBlock_M * Vector_M;
    static constexpr index_t Warp_N    = ThreadPerBlock_N / WarpPerBlock_N * Vector_N;
    static_assert(Warp_M % Vector_M == 0);
    static_assert(Warp_N % Vector_N == 0);
    static_assert(Block_M % (WarpPerBlock_M * Warp_M) == 0);
    static_assert(Block_N % (WarpPerBlock_N * Warp_N) == 0);

    // repeat of each thread along seq<M, N>
    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    // num of threads along seq<M, N>, within each warp
    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    template <bool isHostWave32>
    static constexpr index_t GetBlockSize()
    {
        constexpr index_t warp_size = isHostWave32 ? 32 : get_warp_size();
        return GetWarpPerBlock_M<isHostWave32>() * GetWarpPerBlock_N<isHostWave32>() * warp_size;
    }
};

} // namespace ck_tile
