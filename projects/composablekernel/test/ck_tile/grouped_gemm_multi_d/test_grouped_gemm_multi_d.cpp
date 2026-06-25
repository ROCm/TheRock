// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_multi_d_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F8   = ck_tile::fp8_t;
using F32  = float;

// Custom tuple-like structure for kernel configuration
template <typename ALayout_,
          typename BLayout_,
          typename ELayout_,
          typename ADataType_,
          typename BDataType_,
          typename D0DataType_,
          typename D1DataType_,
          typename AccDataType_,
          typename EDataType_,
          int M_Tile_val_,
          int N_Tile_val_,
          int K_Tile_val_,
          int M_Warp_val_,
          int N_Warp_val_,
          int K_Warp_val_,
          bool DoubleSmemBuffer_val_,
          ck_tile::GemmPipelineScheduler Scheduler_val_,
          PipelineType Pipeline_val_,
          bool Persistent_val_>
struct KernelConfig
{
    using ALayoutType  = ALayout_;
    using BLayoutType  = BLayout_;
    using ELayoutType  = ELayout_;
    using DsLayoutType = ck_tile::tuple<Row, Row>;
    using ADataType    = ADataType_;
    using BDataType    = BDataType_;
    using D0DataType   = D0DataType_;
    using D1DataType   = D1DataType_;
    using AccDataType  = AccDataType_;
    using EDataType    = EDataType_;
    using DsDataType   = ck_tile::tuple<D0DataType_, D1DataType_>;

    static constexpr int M_Tile_ = M_Tile_val_;
    static constexpr int N_Tile_ = N_Tile_val_;
    static constexpr int K_Tile_ = K_Tile_val_;
    static constexpr int M_Warp_ = M_Warp_val_;
    static constexpr int N_Warp_ = N_Warp_val_;
    static constexpr int K_Warp_ = K_Warp_val_;
#if CK_TILE_USE_WMMA
    static constexpr int M_Warp_Tile_ = 16;
    static constexpr int N_Warp_Tile_ = 16;
    static constexpr int K_Warp_Tile_ = 16;
#else
    static constexpr int M_Warp_Tile_ = 32;
    static constexpr int N_Warp_Tile_ = 32;
    static constexpr int K_Warp_Tile_ = (M_Warp_val_ == 2) ? 16 : 8;
#endif
    static constexpr bool DoubleSmemBuffer_ = DoubleSmemBuffer_val_;
    static constexpr auto Scheduler_        = Scheduler_val_;
    static constexpr PipelineType Pipeline_ = Pipeline_val_;
    static constexpr bool Persistent_       = Persistent_val_;
    static constexpr int BlockPerCu_        = 1;
};

// clang-format off
using KernelTypes = ::testing::Types<
    //             ALayout, BLayout, ELayout, ADataType, BDataType, D0DataType, D1DataType, AccDataType, EDataType, M_N_KTiles,    M_N_K_Warps,     DoubleSmemBuffer, Scheduler, Pipeline, Persistent
    // FP16 A/B/D/E
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 32, 64,    4, 1, 1,       false,           ck_tile::GemmPipelineScheduler::Interwave, PipelineType::Memory, false>, // memory
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 32, 64,    4, 1, 1,       false,           ck_tile::GemmPipelineScheduler::Interwave, PipelineType::Memory, true>, // memory
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 128, 64,   2, 2, 1,       false,           ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV3, false>, // v3
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 128, 64,   2, 2, 1,       false,           ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV3, true>, // v3
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 128, 32,   2, 2, 1,       true,            ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV4, false>, // v4
    KernelConfig<    Row,     Col,     Row,         F16,       F16,        F16,        F16,         F32,       F16,  128, 128, 32,   2, 2, 1,       true,            ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV4, true>, // v4
    // BF16 A/B/D/E
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 32, 64,    4, 1, 1,       false,           ck_tile::GemmPipelineScheduler::Interwave, PipelineType::Memory, false>, // memory
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 32, 64,    4, 1, 1,       false,           ck_tile::GemmPipelineScheduler::Interwave, PipelineType::Memory, true>, // memory
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 128, 64,   2, 2, 1,       false,           ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV3, false>, // v3
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 128, 64,   2, 2, 1,       false,           ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV3, true>, // v3
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 128, 32,   2, 2, 1,       true,            ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV4, false>, // v4
    KernelConfig<    Row,     Col,     Row,        BF16,      BF16,       BF16,       BF16,         F32,      BF16,  128, 128, 32,   2, 2, 1,       true,            ck_tile::GemmPipelineScheduler::Intrawave, PipelineType::CompV4, true> // v4
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmMultiD, KernelTypes);

#include "test_grouped_gemm_multi_d_ut_cases.inc"
