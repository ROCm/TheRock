// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_abquant_util.hpp"

using F16   = ck_tile::half_t;
using F32   = float;
using FP8   = ck_tile::fp8_t;
using BF8   = ck_tile::bf8_t;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// AQuant group size is fixed at 1x1x128
using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
// BQuant group size: 1x1x128
using BQuantGroupSize_1x1x128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// clang-format off
using KernelTypes_ABQuant_1x1x128 = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, AQDataType, BDataType, BQDataType, AccDataType, CDataType, AQuantGroupSize, BQuantGroupSize,       Persistent

    // FP8 variants
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,      False>,
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,       True>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,      False>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,       True>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,      False>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,       True>,

    // BF8 variants
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,      False>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, AQuantGroupSize, BQuantGroupSize_1x1x128,       True>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmABQuant_1x1x128, KernelTypes_ABQuant_1x1x128);

#define TEST_CLASS_NAME TestCkTileGroupedGemmABQuant_1x1x128
#include "test_grouped_gemm_abquant_ut_cases.inc"
#undef TEST_CLASS_NAME
