// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using F16  = ck_tile::half_t;
using F32  = float;
using F8   = ck_tile::fp8_t;
using BF8  = ck_tile::bf8_t;
using BF16 = ck_tile::bf16_t;
using I4   = ck_tile::pk_int4_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Default = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                           ck_tile::GemmPipelineScheduler::Default>;

using WeightPreshuffleV2 =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::WeightPreshuffleV2>;

// clang-format off

using KernelTypesWeightPreshuffle = ::testing::Types<
     std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Default,        WeightPreshuffleV2>,
     std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,            Default,        WeightPreshuffleV2>
#if !CK_TILE_USE_WMMA || CK_TILE_USE_OCP_FP8
     ,
     std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,             Default,        WeightPreshuffleV2>,
     std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,             Default,        WeightPreshuffleV2>,
     std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,             Default,        WeightPreshuffleV2>
#endif     
     >;

// clang-format on
