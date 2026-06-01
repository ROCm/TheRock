// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"

struct ConvConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr int kBlockPerCu                = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    static constexpr ck_tile::index_t NumGroupsToMerge = 1;
};

template <typename PrecType>
struct ConvConfigMemoryInterwave : public ConvConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct ConvConfigMemoryIntrawave : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

template <typename PrecType>
struct ConvConfigComputeV3 : public ConvConfigBase
{
    // Compute V3 only support Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct ConvConfigComputeV3_1 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct ConvConfigComputeV3_2 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct ConvConfigComputeV3_WMMA : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct ConvConfigComputeV4 : public ConvConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct ConvConfigComputeV4_1 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct ConvConfigComputeV5 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V5;
    static constexpr ck_tile::index_t NumWaveGroups = 2;
};

template <typename PrecType>
struct ConvConfigComputeV3_merged_groups : public ConvConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr ck_tile::index_t NumGroupsToMerge = 2;
};

template <typename InDataType, typename WeiDataType = InDataType, typename OutDataType = InDataType>
struct ConvTypeConfig;

template <>
struct ConvTypeConfig<ck_tile::half_t>
{
    using InDataType  = ck_tile::half_t;
    using WeiDataType = ck_tile::half_t;
    using AccDataType = float;
    using OutDataType = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct ConvTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using InDataType  = ck_tile::bf16_t;
    using WeiDataType = ck_tile::bf16_t;
    using AccDataType = float;
    using OutDataType = ck_tile::bf16_t;
};

template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV5<PipelineProblem>;
};
