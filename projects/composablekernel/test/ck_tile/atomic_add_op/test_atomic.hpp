// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

template <typename BlockWaves, typename BlockTile, typename WaveTile, typename Vector>
struct AtomicKernelShape
{
    static constexpr index_t MWarps = BlockWaves::at(number<0>{});
    static constexpr index_t NWarps = BlockWaves::at(number<1>{});

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WaveTile::at(number<0>{});
    static constexpr index_t Warp_N = WaveTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = MWarps;
    static constexpr index_t WarpPerBlock_N = NWarps;

    static constexpr index_t RepeatInWarp =
        Warp_M * Warp_N / Vector_M / Vector_N / ck_tile::get_warp_size();
    static constexpr index_t RepeatInWarp_M =
        (Warp_M / Vector_M > Warp_N / Vector_N) ? RepeatInWarp : 1;
    static constexpr index_t RepeatInWarp_N =
        (Warp_M / Vector_M > Warp_N / Vector_N) ? 1 : RepeatInWarp;

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M / RepeatInWarp_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N / RepeatInWarp_N;

    static constexpr index_t Repeat_M = Block_M * RepeatInWarp_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N * RepeatInWarp_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t WaveNum = reduce_on_sequence(BlockWaves{}, multiplies{}, number<1>{});

    static constexpr index_t BlockSize = get_warp_size() * WaveNum;
};

template <typename XDataType_, typename BlockShape_>
struct AtomicKernelProblem
{
    using XDataType  = remove_cvref_t<XDataType_>;
    using BlockShape = remove_cvref_t<BlockShape_>;
};

template <typename Problem_>
struct AtomicKernel
{
    using Problem   = remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;
    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return ck_tile::is_wave32() ? kBlockSize / 2 : kBlockSize;
    }
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        using S = typename Problem::BlockShape;

        constexpr index_t warp_size = get_warp_size();

        constexpr index_t X0 = S::ThreadPerWarp_N;
        constexpr index_t X1 = S::Vector_N;

        constexpr index_t Y0 = S::WaveNum;
        constexpr index_t Y2 = warp_size / X0;
        constexpr index_t Y1 = S::Warp_M / Y2;

        constexpr auto encoding =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{};

        return make_static_tile_distribution(encoding);
    }

    CK_TILE_DEVICE void operator()(XDataType* input, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        constexpr auto block_dims = make_tuple(number<S::Block_M>{}, number<S::Block_N>{});

        const index_t iM = __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_M);

        const auto input_view =
            make_naive_tensor_view<address_space_enum::global, memory_operation_enum::atomic_add>(
                input, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        auto input_window = make_tile_window(input_view, block_dims, {iM, 0});

        const index_t num_iterations =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));
        using tmp_tile =
            decltype(make_static_distributed_tensor<XDataType>(MakeTileDistribution<Problem>()));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_iterations; iN++)
        {
            tmp_tile add_value_tile;
            tile_elementwise_inout([](auto& c) { c = static_cast<XDataType>(1.0f); },
                                   add_value_tile);

            update_tile(input_window, add_value_tile);
            __syncthreads();

            move_tile_window(input_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
