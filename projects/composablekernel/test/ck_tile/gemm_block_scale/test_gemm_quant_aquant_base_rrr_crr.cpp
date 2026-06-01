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

// Type combinations for AQuant tests - RRR and CRR layouts
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantBaseRRRCRRTypes = ::testing::Types<
    // RRR layout (RowMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // CRR layout (ColumnMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>
>;
// clang-format on

// Test suite for AQuant Base RRR/CRR
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantBaseRRRCRRTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
