// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_abd_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;
using F8   = ck_tile::fp8_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    // Has cshuffle epilogue disabled
    //          A0Layout, A1Layout, B0Layout, B1Layout CLayout, D0Layout, D1Layout, A0DataType, A01DataType B0DataType, B0DataType, D0DataType,  D1DataType, AccDataType, EDataType, AElementWiseFn, BElementWiseFn, CDElementWiseFn, UseCshuffleEpilog
#if !CK_TILE_USE_WMMA || CK_TILE_USE_OCP_FP8
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F8,           F8,         F8,         F8,           BF16,       BF16,       F32,      BF16,        AddScale,       AddScale,    ElementWiseAddAdd,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F8,           F8,         F8,         F8,           BF16,       BF16,       F32,      BF16,        AddScale,       AddScale,    MultiplyMultiply,   std::false_type>,
#endif
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F16,        F16,        F32,      F16,         AddScale,       AddScale,    ElementWiseAddAdd,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F32,        F32,        F32,      F16,         AddScale,       AddScale,    ElementWiseAddAdd,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F32,        F32,        F32,      F32,         AddScale,       AddScale,    ElementWiseAddAdd,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          BF16,       BF16,       F32,      BF16,        AddScale,       AddScale,    ElementWiseAddAdd,  std::false_type>,

    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F16,        F16,        F32,      F16,         AddScale,       AddScale,    MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F32,        F32,        F32,      F16,         AddScale,       AddScale,    MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          F32,        F32,        F32,      F32,         AddScale,       AddScale,    MultiplyMultiply,  std::false_type>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,     Row,      Row,      F16,          F16,        F16,        F16,          BF16,       BF16,       F32,      BF16,        AddScale,       AddScale,    MultiplyMultiply,  std::false_type>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiABD, KernelTypes);

#include "test_gemm_multi_abd_ut_cases_default2d.inc"
