// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_d_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;
using F8   = ck_tile::fp8_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    // Has cshuffle epilogue disabled
    //          ALayout, BLayout, CLayout, D0Layout, D1Layout, ADataType, BDataType, D0DataType,  D1DataType, AccDataType, EDataType, CDElementWiseFn,   UseCshuffleEpilog
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F16,     ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F16,     ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      F32,     ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F32,     ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,      BF16,    ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      BF16,    ElementWiseAddAdd, std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           F16,        F16,        F32,      F16,     ElementWiseAddAdd, std::false_type>,

    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F16,        F16,        F32,      F16,     MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F16,     MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      F32,     MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,      F32,     MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,      BF16,    MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,      BF16,    MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           F16,        F16,        F32,      F16,     MultiplyMultiply,  std::false_type>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiD, KernelTypes);

#include "test_gemm_multi_d_ut_cases_default2d.inc"
