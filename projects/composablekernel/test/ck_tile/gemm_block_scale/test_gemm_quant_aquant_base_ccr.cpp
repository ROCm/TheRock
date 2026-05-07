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
using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;
using GroupSize     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for AQuant tests - CCR layout
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantBaseCCRTypes = ::testing::Types<
    // CCR layout (ColumnMajor A, ColumnMajor B, RowMajor C with ColumnMajor AQ) - NEW layout support
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>
>;
// clang-format on

// Test suite for AQuant Base CCR
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantBaseCCRTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
