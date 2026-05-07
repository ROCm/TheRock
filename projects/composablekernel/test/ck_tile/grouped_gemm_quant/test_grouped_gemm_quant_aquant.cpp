// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util_quant.hpp"

using F16    = ck_tile::half_t;
using F32    = float;
using FP8    = ck_tile::fp8_t;
using BF8    = ck_tile::bf8_t;
using Row    = ck_tile::tensor_layout::gemm::RowMajor;
using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;
using True   = ck_tile::bool_constant<true>;
using False  = ck_tile::bool_constant<false>;
using AQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// clang-format off
using KernelTypes_AQuant = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, AQDataType, BDataType, BQDataType, AccDataType, CDataType, QuantType, PreshuffleB, Persistent, TransposeC
    // RCR FP8 (with/without TransposeC)
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,     False>,
    
    // RCR BF8 (with/without TransposeC)
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    AQuant,       False,       True,     False>,
    
    // RCR non-persistent  (with/without TransposeC)
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,      False,      True>,
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,      False,     False>,
     
    // RRR layout (with/without TransposeC)
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,     False>,
    
    // CRR layout (with/without TransposeC)
    // NOT SUPPORTED: std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,     False>,
    
    // CCR layout (with/without TransposeC)
    // NOT SUPPORTED: std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,     False>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmQuant_AQuant, KernelTypes_AQuant);

#define TEST_CLASS_NAME TestCkTileGroupedGemmQuant_AQuant
#include "test_grouped_gemm_quant_ut_cases.inc"
#undef TEST_CLASS_NAME
