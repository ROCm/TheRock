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
using RowColQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::RowColQuant>;
using TensorQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>;
using AQuant      = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;
using BQuant      = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, AQDataType, BDataType, BQDataType, AccDataType, CDataType,   QuantType, PreshuffleB, Persistent, TransposeC
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,

    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Col,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Row,     Row,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,
    std::tuple<    Col,     Row,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, RowColQuant,       False,       True,      False>,

    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Row,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Row,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,

    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Row,     Row,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    std::tuple<    Col,     Row,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16, TensorQuant,       False,       True,      False>,
    
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    AQuant,       False,       True,      False>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    AQuant,       False,       True,      True>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    AQuant,       False,       True,      False>,

    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    BQuant,       False,       True,      False>,
    std::tuple<    Row,     Col,     Row,       FP8,        F32,       FP8,        F32,         F32,       F16,    BQuant,        True,       True,      False>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    BQuant,       False,       True,      False>,
    std::tuple<    Row,     Col,     Row,       BF8,        F32,       BF8,        F32,         F32,       F16,    BQuant,        True,       True,      False>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmQuant, KernelTypes);

#include "test_grouped_gemm_quant_ut_cases.inc"
