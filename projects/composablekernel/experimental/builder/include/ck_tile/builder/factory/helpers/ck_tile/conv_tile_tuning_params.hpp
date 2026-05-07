// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory::internal {

// Convenience struct for a tuple of m, n, and k values.
struct TileBlockGemmMNK
{
    int m{};
    int n{};
    int k{};
};

struct TileBlockGemmSpec
{
    TileBlockGemmMNK warps     = {};
    TileBlockGemmMNK warp_tile = {};

    bool double_smem_buffer = false;
    int num_wave_groups     = 1;

    ck_tile::GemmPipeline pipeline_version;
    ck_tile::GemmPipelineScheduler scheduler;
};

struct TileOptimizations
{
    int num_groups_to_merge = 1;
    bool split_image        = false;
    bool explicit_gemm      = false;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck_tile::GemmPipelineScheduler SetTileScheduler()
{
    constexpr auto scheduler = ALGORITHM.block_gemm.scheduler;
    using ck_tile_sched      = ck_tile::GemmPipelineScheduler;
    switch(scheduler)
    {
    case PipelineScheduler::DEFAULT: return ck_tile_sched::Default;
    case PipelineScheduler::INTERWAVE: return ck_tile_sched::Interwave;
    case PipelineScheduler::INTRAWAVE: return ck_tile_sched::Intrawave;
    default: throw "Unknown PipelineScheduler";
    }
}

template <ck_tile::GemmPipeline PipelineId>
struct TilePipelineType
{
    static_assert(false, "Unknown PipelineScheduler");
};

template <>
struct TilePipelineType<ck_tile::GemmPipeline::BASIC_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
};

template <>
struct TilePipelineType<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct TilePipelineType<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct TilePipelineType<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct TilePipelineType<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck_tile::GemmPipeline SetTileBlockGemmPipelineVersion()
{
    constexpr auto version = ALGORITHM.block_gemm.pipeline_version;
    using ck_tile_pipeline = ck_tile::GemmPipeline;
    switch(version)
    {
    case PipelineVersion::V1: return ck_tile_pipeline::BASIC_V1;
    case PipelineVersion::V2: return ck_tile_pipeline::MEMORY;
    case PipelineVersion::V3: return ck_tile_pipeline::COMPUTE_V3;
    case PipelineVersion::V4: return ck_tile_pipeline::COMPUTE_V4;
    case PipelineVersion::V5: return ck_tile_pipeline::COMPUTE_V5;
    case PipelineVersion::WEIGHT_ONLY:
        throw "PipelineVersion::WEIGHT_ONLY is not supported for block GEMM pipeline version.";
    default: throw "Unknown block GEMM PipelineVersion";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck_tile::ConvolutionSpecialization SetTileConvSpecialization()
{
    constexpr auto specialization = ALGORITHM.specialization;
    using ck_tile_conv_spec       = ck_tile::ConvolutionSpecialization;
    switch(specialization)
    {
    case TileConvSpecialization::DEFAULT: return ck_tile_conv_spec::Default;
    case TileConvSpecialization::FILTER_1X1_PAD0: return ck_tile_conv_spec::Filter1x1Pad0;
    case TileConvSpecialization::FILTER_1X1_STRIDE1_PAD0:
        return ck_tile_conv_spec::Filter1x1Stride1Pad0;
    case TileConvSpecialization::FILTER_3x3: return ck_tile_conv_spec::Filter3x3;
    default: throw "Unknown ConvFwdSpecialization";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval TileBlockGemmSpec SetTileBlockGemm()
{
    constexpr auto& BG = ALGORITHM.block_gemm;

    constexpr bool double_smem_buffer = BG.double_smem_buffer;
    constexpr int num_wave_groups     = BG.num_wave_groups;

    constexpr ck_tile::GemmPipeline pipeline_version = SetTileBlockGemmPipelineVersion<ALGORITHM>();
    constexpr ck_tile::GemmPipelineScheduler scheduler = SetTileScheduler<ALGORITHM>();

    return TileBlockGemmSpec{
        .warps              = {.m = BG.warps.m, .n = BG.warps.n, .k = BG.warps.k},
        .warp_tile          = {.m = BG.warp_tile.m, .n = BG.warp_tile.n, .k = BG.warp_tile.k},
        .double_smem_buffer = double_smem_buffer,
        .num_wave_groups    = num_wave_groups,
        .pipeline_version   = pipeline_version,
        .scheduler          = scheduler};
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval TileOptimizations SetTileOptimizations()
{
    constexpr auto& OPT = ALGORITHM.optimizations;

    return TileOptimizations{.num_groups_to_merge = OPT.num_groups_to_merge,
                             .split_image         = OPT.split_image,
                             .explicit_gemm       = OPT.explicit_gemm};
}

} // namespace ck_tile::builder::factory::internal
