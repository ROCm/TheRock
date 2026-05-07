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
using GroupSize     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for BQuant Preshuffle tests - Decode Config 1D
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BPreshuffleDecode1DTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>
>;
// clang-format on

// Test suite for BQuant Preshuffle Decode 1D
TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, BPreshuffleDecode1DTypes);

// BQuant PreshuffleB tests
TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, BQuantPreshuffleTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
