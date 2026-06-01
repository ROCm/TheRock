// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise.hpp"

#include "test_multi_reduce2d_multiblock_impl.hpp"

// Shape parameters for different test configurations
using Shape1_BlockWarps = ck_tile::sequence<4, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 128>;
using Shape1_WarpTile   = ck_tile::sequence<32, 128>;
using Shape1_ThreadTile = ck_tile::sequence<8, 8>;

// Test configurations for different data types and operations
using TestConfig_F16_Add = std::tuple<ck_tile::half_t,
                                      float,
                                      float, // Output and multiblock reducing buffer. Using float
                                             // to avoid too many accumulation errors
                                      ck_tile::tuple<ck_tile::ReduceOp::Add>,
                                      ck_tile::tuple<ck_tile::element_wise::PassThrough>,
                                      ck_tile::tuple<ck_tile::element_wise::PassThrough>,
                                      ck_tile::tuple<ck_tile::ReduceOp::Add>,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;

using TestConfig_F16_Add_MeanSquare = std::tuple<
    ck_tile::half_t,
    float,
    float, // Output and multiblock reducing buffer. Using float to avoid too many accumulation
           // errors
    ck_tile::tuple<ck_tile::ReduceOp::Add, ck_tile::ReduceOp::Add>, // Intra block reductions
    ck_tile::tuple<ck_tile::element_wise::PassThrough,
                   ck_tile::element_wise::UnarySquare>, // Elementwise
                                                        // ops
    ck_tile::tuple<ck_tile::element_wise::PassThrough,
                   ck_tile::element_wise::UnaryDivide>, // Accumulator Elementiwise ops, intra block
    ck_tile::tuple<ck_tile::ReduceOp::Add, ck_tile::ReduceOp::Add>, // Inter block reduction
    Shape1_BlockWarps,
    Shape1_BlockTile,
    Shape1_WarpTile,
    Shape1_ThreadTile>;

using TestTypes = ::testing::Types<TestConfig_F16_Add, TestConfig_F16_Add_MeanSquare>;

TYPED_TEST_SUITE(TestCkTileMultiReduceMultiblock, TestTypes);

// 2D Tests - Keep dim0, reduce dim1
TYPED_TEST(TestCkTileMultiReduceMultiblock, Test2D_KeepDim0_ReduceDim1_64x32)
{
    this->RunTest2D_KeepDim0_ReduceDim1(64, 32);
}

TYPED_TEST(TestCkTileMultiReduceMultiblock, Test2D_KeepDim0_ReduceDim1_1024x512)
{
    this->RunTest2D_KeepDim0_ReduceDim1(1024, 512);
}

// 3D Tests - Keep dim0, reduce dim1,2
TYPED_TEST(TestCkTileMultiReduceMultiblock, Test3D_KeepDim0_ReduceDim12_128x128x1)
{
    this->RunTest3D_KeepDim0_ReduceDim12(128, 128, 8);
}
// 3D Tests - Keep dim0,1, reduce dim1
TYPED_TEST(TestCkTileMultiReduceMultiblock, Test3D_KeepDim01_ReduceDim2_512x1024x16)
{
    this->RunTest3D_KeepDim01_ReduceDim2(512, 1024, 16);
}

// 4D Tests - Keep dim0,1, reduce dim2,3 (NCHW -> NC)
TYPED_TEST(TestCkTileMultiReduceMultiblock, Test4D_KeepDim01_ReduceDim23_32x256x16x16)
{
    this->RunTest4D_KeepDim01_ReduceDim23(32, 256, 16, 16);
}
// 4D Tests - Keep dim0,3, reduce dim1,2 (NHWC -> NC)
TYPED_TEST(TestCkTileMultiReduceMultiblock, Test4D_KeepDim03_ReduceDim12_16x32x32x128)
{
    this->RunTest4D_KeepDim03_ReduceDim12(16, 32, 32, 128);
}
