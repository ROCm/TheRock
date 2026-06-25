// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory::internal {

// Convenience struct for a tuple of m, n, and k values.
struct TileBlockMNK
{
    int m{};
    int n{};
    int k{};
};

struct TileConvBlock
{
    TileBlockMNK per_block = {};
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr TileConvBlock SetTileThreadBlockInfo()
{
    constexpr auto& TB = ALGORITHM.thread_block;
    return TileConvBlock{
        .per_block = {.m = TB.tile_size.m, .n = TB.tile_size.n, .k = TB.tile_size.k},
    };
}

} // namespace ck_tile::builder::factory::internal
