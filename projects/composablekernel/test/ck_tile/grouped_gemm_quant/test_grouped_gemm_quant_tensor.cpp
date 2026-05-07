// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util_quant.hpp"

using F16         = ck_tile::half_t;
using F32         = float;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
using Row         = ck_tile::tensor_layout::gemm::RowMajor;
using Col         = ck_tile::tensor_layout::gemm::ColumnMajor;
using True        = ck_tile::bool_constant<true>;
using False       = ck_tile::bool_constant<false>;
using TensorQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>;

// clang-format off
using KernelTypes_Tensor = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, AQDataType, BDataType, BQDataType, AccDataType, CDataType,   QuantType, PreshuffleB, Persistent, TransposeC
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,      False,      False>,
    std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,      False,      False>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmQuant_Tensor, KernelTypes_Tensor);

#define TEST_CLASS_NAME TestCkTileGroupedGemmQuant_Tensor
#include "test_grouped_gemm_quant_ut_cases.inc"
#undef TEST_CLASS_NAME
