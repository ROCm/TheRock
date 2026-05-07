// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

template <typename BlockWaves, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WaveTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous elements (vector size) along seq<M, N>
struct TileCopyShape
{
    // We split Workgroup waves into two specialized groups.
    // One for reading data from global -> LDS, the other idling
    static constexpr index_t WaveGroups = 2;
    static constexpr index_t MWarps     = BlockWaves::at(number<0>{});
    static constexpr index_t NWarps     = BlockWaves::at(number<1>{});

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WaveTile::at(number<0>{});
    static constexpr index_t Warp_N = WaveTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    // We splitted the waves on M dimension
    static constexpr index_t WarpPerBlock_M = integer_divide_ceil(MWarps, WaveGroups);
    static constexpr index_t WarpPerBlock_N = NWarps;

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t WaveNum = reduce_on_sequence(BlockWaves{}, multiplies{}, number<1>{});

    static constexpr index_t BlockSize     = get_warp_size() * WaveNum;
    static constexpr index_t WaveGroupSize = WaveNum / WaveGroups;
    static_assert(WaveGroupSize == WarpPerBlock_M * WarpPerBlock_N,
                  "Inconsistent wave group size!");
};

template <typename XDataType_, typename BlockShape_, bool AsyncCopy_>
struct TileCopyProblem
{
    using XDataType                 = remove_cvref_t<XDataType_>;
    using BlockShape                = remove_cvref_t<BlockShape_>;
    static constexpr bool AsyncCopy = AsyncCopy_;
};

template <typename Problem_>
struct TileCopy
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;
    static constexpr bool AsyncCopy     = Problem::AsyncCopy;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        using S = typename Problem::BlockShape;

        constexpr index_t warp_size = get_warp_size();
        constexpr index_t X0 = S::ThreadPerWarp_N; // threads needed along N dimension, fastest
                                                   // changing with given vector size.
        constexpr index_t X1 =
            S::Vector_N; // no. of elements along N dimensions to be read by each thread.

        constexpr index_t Y0 =
            S::WaveNum / S::WaveGroups; // number of active warps working in this thread block.
        constexpr index_t Y2 =
            warp_size / X0; // number of threads in a warp needed along M dimension.
        constexpr index_t Y1 =
            S::Warp_M /
            Y2; // number of iterations each warp needs to perform to cover the entire tile window.

        constexpr auto outer_encoding =
            tile_distribution_encoding<sequence<S::WaveGroups>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{};

        return make_static_tile_distribution(outer_encoding);
    }

    CK_TILE_DEVICE void
    operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N, index_t warp_id) const
    {
        using S = typename Problem::BlockShape;

        // LDS buffer
        __shared__ XDataType x_lds[S::Block_M * S::Block_N];

        constexpr auto block_dims    = make_tuple(number<S::Block_M>{}, number<S::Block_N>{});
        constexpr auto block_strides = make_tuple(number<S::Block_N>{}, number<1>{});

        const auto x_lds_desc = make_naive_tensor_descriptor(
            block_dims, block_strides, number<S::Vector_N>{}, number<1>{});

        auto x_lds_view = make_tensor_view<address_space_enum::lds>(x_lds, x_lds_desc);

        auto x_block_lds_write_window = make_tile_window(x_lds_view, block_dims, {0, 0});

        auto x_block_lds_read_window =
            make_tile_window(x_lds_view, block_dims, {0, 0}, MakeDRAMDistribution<Problem>());

        const index_t iM = __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_M);
        // Input tensor
        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        auto x_block_window =
            make_tile_window(x_m_n, block_dims, {iM, 0}, MakeDRAMDistribution<Problem>());

        // Output tensor
        const auto y_m = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        auto y_block_window = make_tile_window(y_m, block_dims, {iM, 0});

        const index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));
        const index_t my_id                    = __builtin_amdgcn_readfirstlane(get_warp_id());
        constexpr index_t async_copy_fence_cnt = 0;
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            if(my_id == warp_id)
            {
                if constexpr(AsyncCopy)
                {
                    async_load_tile(x_block_lds_write_window, x_block_window);
                    // We don't have prefetch here, wait the data back immediately.
                    // Wait all asyncload insts complete.
                    // Wait all waves synced
                    s_waitcnt_barrier<async_copy_fence_cnt>();
                    auto lds_tile = load_tile(x_block_lds_read_window);
                    // store from registers to DRAM
                    store_tile(y_block_window, lds_tile);
                }
                else
                {
                    // load from DRAM to registers
                    auto dram_tile = load_tile(x_block_window);
                    // store in lds
                    store_tile(x_block_lds_write_window, dram_tile);
                    // Wait all lds write insts complete
                    // Wait all waves synced
                    block_sync_lds();
                    // read from lds to registers
                    auto lds_tile = load_tile(x_block_lds_read_window);
                    // store from registers to DRAM
                    store_tile(y_block_window, lds_tile);
                }
            }

            move_tile_window(x_block_window, {0, S::Block_N});
            move_tile_window(y_block_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
