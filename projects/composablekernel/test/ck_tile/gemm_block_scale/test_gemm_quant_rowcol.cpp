// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Type aliases for readability
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
using Half        = ck_tile::half_t;
using RowColQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::RowColQuant>;
using GroupSize   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for RowColQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using RowColQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, RowColQuant, GemmConfigBase, GroupSize>
>;
// clang-format on

// Test suite for RowColQuant
TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantTypes);

// RowColQuant tests
TYPED_TEST(TestCkTileGemmRowColQuant, RowColQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
