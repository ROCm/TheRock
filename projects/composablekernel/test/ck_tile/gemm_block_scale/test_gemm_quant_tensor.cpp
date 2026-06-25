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
using TensorQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>;
using GroupSize   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for TensorQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using TensorQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, TensorQuant, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, TensorQuant, GemmConfigBase, GroupSize>
>;
// clang-format on

// Test suite for TensorQuant
TYPED_TEST_SUITE(TestCkTileGemmTensorQuant, TensorQuantTypes);

// TensorQuant tests
TYPED_TEST(TestCkTileGemmTensorQuant, TensorQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
