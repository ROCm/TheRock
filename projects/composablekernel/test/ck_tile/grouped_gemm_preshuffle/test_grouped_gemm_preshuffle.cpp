// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_preshuffle_util.hpp"

using F16   = ck_tile::half_t;
using F8    = ck_tile::fp8_t;
using BF16  = ck_tile::bf16_t;
using F32   = float;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using False = std::false_type;
using True  = std::true_type;

// Custom tuple-like structure for kernel configuration
template <typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename Persistent_,
          int M_Tile_val_,
          int N_Tile_val_,
          int K_Tile_val_,
          int BlockPerCu_val_>
struct KernelConfig
{
    using ALayoutType = ALayout_;
    using BLayoutType = BLayout_;
    using CLayoutType = CLayout_;
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;
    using Persistent  = Persistent_;

    static constexpr int M_Tile_     = M_Tile_val_;
    static constexpr int N_Tile_     = N_Tile_val_;
    static constexpr int K_Tile_     = K_Tile_val_;
    static constexpr int BlockPerCu_ = BlockPerCu_val_;
};

// clang-format off
using KernelTypes = ::testing::Types<
    //               ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent ,M_Tile, N_Tile, K_Tile, BlockPerCu
#if !CK_TILE_USE_WMMA || CK_TILE_USE_OCP_FP8
    KernelConfig<    Row,     Col,     Row,       F8,        F8,          F32,       F16,   False,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       F8,        F8,          F32,       F16,   False,   128,    128,    128,         2>,
    KernelConfig<    Row,     Col,     Row,       F8,        F8,          F32,       F16,   True,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       F8,        F8,          F32,       F16,   True,   128,    128,    128,         2>,
#endif
    KernelConfig<    Row,     Col,     Row,       F16,       F16,         F32,       F16,   False,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       F16,       F16,         F32,       F16,   False,  128,    128,    128,         2>,
    KernelConfig<    Row,     Col,     Row,       F16,       F16,         F32,       F16,   True,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       F16,       F16,         F32,       F16,   True,  128,    128,    128,         2>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   False,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   False,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   False,  128,    128,    128,         2>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   False,   128,    128,    128,         2>,

    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   True,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   True,    16,     64,    256,         1>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   True,  128,    128,    128,         2>,
    KernelConfig<    Row,     Col,     Row,       BF16,      BF16,         F32,       BF16,   True,   128,    128,    128,         2>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmPreshuffle, KernelTypes);

#include "test_grouped_gemm_preshuffle_ut_cases.inc"
#include "test_grouped_gemm_preshuffle_prefill_cases.inc"
