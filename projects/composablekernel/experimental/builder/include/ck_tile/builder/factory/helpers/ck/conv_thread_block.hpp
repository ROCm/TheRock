// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory::internal {

/// @brief Data tile dimensions processed by a workgroup.
/// @details This struct defines the M, N, and K dimensions of the data tile
/// that a single workgroup (thread block) is responsible for processing in the
/// underlying GEMM computation.
struct DataTileInfo
{
    int m; ///< M dimension of the tile processed by the workgroup (MPerBlock).
    int n; ///< N dimension of the tile processed by the workgroup (NPerBlock).
    int k; ///< K dimension of the tile processed by the workgroup (KPerBlock).
};

struct ConvBlock
{
    size_t block_size      = 0;
    DataTileInfo per_block = {};
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvBlock SetThreadBlockInfo()
{
    constexpr auto& TB = ALGORITHM.thread_block;
    return ConvBlock{
        .block_size = TB.block_size,
        .per_block  = {.m = TB.tile_size.m, .n = TB.tile_size.n, .k = TB.tile_size.k},
    };
}

} // namespace ck_tile::builder::factory::internal
