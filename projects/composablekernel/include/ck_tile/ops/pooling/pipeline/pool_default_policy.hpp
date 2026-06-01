// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d.hpp"

namespace ck_tile {

struct PoolDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::ThreadTile_M>,
                    sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::ThreadTile_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2d()
    {
        using P_ = BlockReduce2dProblem<typename Problem::InDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape,
                                        Problem::kOutputIndex>;
        return BlockReduce2d<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dSync()
    {
        using P_ = BlockReduce2dProblem<typename Problem::InDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape,
                                        Problem::kOutputIndex>;
        return BlockReduce2dSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dCrossWarpSync()
    {
        using P_ = BlockReduce2dProblem<typename Problem::InDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape,
                                        Problem::kOutputIndex>;
        return BlockReduce2dCrossWarpSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(Problem::kNeedCrossWarpSync)
        {
            using P_ = BlockReduce2dProblem<typename Problem::InDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::BlockShape,
                                            Problem::kOutputIndex>;

            using block_reduce2d = BlockReduce2d<P_>;
            using x_block_tile =
                decltype(make_static_distributed_tensor<typename Problem::InDataType>(
                    MakeXBlockTileDistribution<Problem>()));
            using y_block_tile = decltype(block_reduce2d::template MakeYBlockTile<x_block_tile>());

            return GetBlockReduce2dCrossWarpSync<Problem>().template GetSmemSize<y_block_tile>();
        }
        else
        {
            return 1; // zero size arrays are an extension
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetIndicesSmemSize()
    {

        using P_ = BlockReduce2dProblem<typename Problem::InDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape,
                                        Problem::kOutputIndex>;

        using block_reduce2d = BlockReduce2d<P_>;
        using x_block_tile = decltype(make_static_distributed_tensor<typename Problem::InDataType>(
            MakeXBlockTileDistribution<Problem>()));
        using y_index_block_tile = decltype(block_reduce2d::template MakeYIndexBlockTile<
                                            x_block_tile,
                                            typename Problem::IndexDataType>());

        return GetBlockReduce2dCrossWarpSync<Problem>()
            .template GetIndicesSmemSize<y_index_block_tile>();
    }
};
} // namespace ck_tile
