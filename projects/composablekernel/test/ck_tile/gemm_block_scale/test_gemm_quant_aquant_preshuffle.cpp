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

// Type combinations for AQuant tests - PreshuffleQuant Configurations
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantPreshuffleTypes = ::testing::Types<
    // PreshuffleQuant = true && TransposeC = false (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,

    // PreshuffleQuant = true && TransposeC = true (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>
>;
// clang-format on

// Test suite for AQuant Preshuffle
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantPreshuffleTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
