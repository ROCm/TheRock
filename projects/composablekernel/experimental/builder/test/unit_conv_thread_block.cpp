// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/factory/helpers/ck/conv_thread_block.hpp"

namespace {

using ::ck_tile::builder::factory::internal::ConvBlock;
using ::ck_tile::builder::factory::internal::SetThreadBlockInfo;

TEST(ConvThreadBlock, AssignsThreadBlockAndTileSize)
{
    constexpr struct Algorithm
    {
        struct ThreadBlock
        {
            int block_size = 256;
            struct TileSize
            {
                int m = 128;
                int n = 128;
                int k = 16;
            } tile_size;
        } thread_block;
    } kAlgorithm;
    constexpr ConvBlock block_info = SetThreadBlockInfo<kAlgorithm>();

    EXPECT_EQ(block_info.block_size, 256);
    EXPECT_EQ(block_info.per_block.m, 128);
    EXPECT_EQ(block_info.per_block.n, 128);
    EXPECT_EQ(block_info.per_block.k, 16);
}

} // namespace
