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
using PkInt4      = ck_tile::pk_int4_t;
using ABQuantGrouped =
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::ABQuantGrouped>;
using GroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for ABQuant padding padding tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantPaddingTypes = ::testing::Types<
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigPadding, GroupSize, GroupSize, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant Padding
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantPaddingTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    this->run_test_with_validation(1024, 832, 832);
}
