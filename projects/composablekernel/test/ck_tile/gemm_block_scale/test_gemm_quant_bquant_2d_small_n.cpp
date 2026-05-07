// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Type aliases for readability
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
using Half          = ck_tile::half_t;
using PkInt4        = ck_tile::pk_int4_t;
using BQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>;

// 2d block sizes for BQuant
using GroupSize2D8N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
using GroupSize2D16N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 16, 128>>;

// Type combinations for BQuant tests - 2D Small N (8N and 16N)
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant2DSmallNTypes = ::testing::Types<
    // 2d cases with grouping also on the n axis
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>
>;
// clang-format on

// Test suite for BQuant 2D Small N
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant2DSmallNTypes);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
