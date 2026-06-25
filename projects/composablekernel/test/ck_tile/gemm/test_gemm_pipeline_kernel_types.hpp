// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "test_gemm_pipeline_prec_types.hpp"

using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Interwave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Interwave>;

using Mem       = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;
using CompV6    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV6>;
using CompAsync = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompAsync>;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I256 = ck_tile::number<256>;

// clang-format off
using KernelTypesMem = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, K_TileSize, Scheduler, PipelineType
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem>
>;

using KernelTypesMemWmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem>
>;

using KernelTypesCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3>
>;

using KernelTypesCompV3Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3>
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4>
>;

// clang-format on
template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncConfig = std::tuple<ALayout,
                                   BLayout,
                                   CLayout,
                                   InputType, // AType
                                   InputType, // BType
                                   F32,       // AccType
                                   F16,       // OutputType
                                   I256,      // MBlockTileSize
                                   I256,      // NBlockTileSize
                                   I32,       // KBlockTileSize
                                   I32,       // MWarpTileSize
                                   I32,       // NWarpTileSize
                                   I16,       // KWarpTileSize
                                   Intrawave,
                                   CompAsync>;

using KernelTypesCompAsync = ::testing::Types<CompAsyncConfig<Row, Row, Row, F16>,
                                              CompAsyncConfig<Row, Col, Row, F16>,
                                              CompAsyncConfig<Col, Row, Row, F16>,
                                              CompAsyncConfig<Col, Col, Row, F16>,
                                              CompAsyncConfig<Row, Row, Row, F8>,
                                              CompAsyncConfig<Row, Col, Row, F8>,
                                              CompAsyncConfig<Col, Row, Row, F8>,
                                              CompAsyncConfig<Col, Col, Row, F8>>;
// clang-format off

using KernelTypesCompV6 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV6>
>;

using KernelTypesCompV4Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4>
>;


using KernelTypesPersistent = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, M_TileSize, K_TileSize, Scheduler,  PipelineType,    Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3, NonPersistent>
>;

using KernelTypesPersistentWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3, NonPersistent>
>;

// clang-format on
