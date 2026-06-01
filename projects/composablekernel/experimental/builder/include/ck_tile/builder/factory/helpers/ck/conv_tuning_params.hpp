// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory::internal {

// The algorithm specializations for the convolution and GEMM.
template <typename CONV_ENUM>
    requires(
        std::is_same_v<CONV_ENUM, ck::tensor_operation::device::ConvolutionForwardSpecialization>)
struct ConvSpec
{
    CONV_ENUM conv_spec;
    ck::tensor_operation::device::GemmSpecialization gemm_spec;
};

// Deduction guide for ConvSpec to simplify brace initialization.
template <typename CONV_ENUM, typename GEMM_ENUM>
ConvSpec(CONV_ENUM, GEMM_ENUM) -> ConvSpec<CONV_ENUM>;

struct BlockGemmSpec
{
    ck::BlockGemmPipelineVersion pipeline_version;
    ck::BlockGemmPipelineScheduler scheduler;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval BlockGemmSpec SetBlockGemm()
{
    constexpr auto& BG = ALGORITHM.block_gemm;

    ck::BlockGemmPipelineScheduler scheduler;
    ck::BlockGemmPipelineVersion version;

    switch(BG.scheduler)
    {
    case PipelineScheduler::INTRAWAVE: scheduler = ck::BlockGemmPipelineScheduler::Intrawave; break;
    case PipelineScheduler::INTERWAVE: scheduler = ck::BlockGemmPipelineScheduler::Interwave; break;
    case PipelineScheduler::DEFAULT: throw "Block GEMM scheduler must be Intrawave or Interwave.";
    default: throw "Unknown PipelineScheduler";
    }

    switch(BG.pipeline_version)
    {
    case PipelineVersion::V1: version = ck::BlockGemmPipelineVersion::v1; break;
    case PipelineVersion::V2: version = ck::BlockGemmPipelineVersion::v2; break;
    case PipelineVersion::V3: version = ck::BlockGemmPipelineVersion::v3; break;
    case PipelineVersion::V4: version = ck::BlockGemmPipelineVersion::v4; break;
    case PipelineVersion::V5: version = ck::BlockGemmPipelineVersion::v5; break;
    case PipelineVersion::WEIGHT_ONLY:
        throw "PipelineVersion::WEIGHT_ONLY is not supported for block GEMM.";
    default: throw "Unknown PipelineVersion";
    }

    return BlockGemmSpec{.pipeline_version = version, .scheduler = scheduler};
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::LoopScheduler SetLoopScheduler()
{
    constexpr auto loop_scheduler = ALGORITHM.loop_scheduler;
    using ck_loop_sched           = ck::LoopScheduler;
    switch(loop_scheduler)
    {
    case PipelineScheduler::DEFAULT: return ck_loop_sched::Default;
    case PipelineScheduler::INTERWAVE: return ck_loop_sched::Interwave;
    case PipelineScheduler::INTRAWAVE: throw "LoopScheduler must be either DEFAULT or INTERWAVE.";
    default: throw "Unknown PipelineScheduler";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::PipelineVersion SetGridwiseGemmPipelineVersion()
{
    constexpr auto pipeline_version = ALGORITHM.gridwise_gemm.pipeline_version;
    using ck_pipeline               = ck::PipelineVersion;
    switch(pipeline_version)
    {
    case PipelineVersion::V1: return ck_pipeline::v1;
    case PipelineVersion::V2: return ck_pipeline::v2;
    case PipelineVersion::V3: throw "PipelineVersion::V3 is used only for stream-K.";
    case PipelineVersion::V4: return ck_pipeline::v4;
    case PipelineVersion::V5: throw "PipelineVersion::V5 cannot be used for gridwise GEMM.";
    case PipelineVersion::WEIGHT_ONLY: return ck_pipeline::weight_only;
    default: throw "Unknown GridwiseGemmPipelineVersion";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::GemmSpecialization SetGemmSpecialization()
{
    constexpr auto gemm_spec = ALGORITHM.gemm_specialization;
    using ck_gemm_spec       = ck::tensor_operation::device::GemmSpecialization;

    switch(gemm_spec)
    {
    case GemmSpecialization::Default: return ck_gemm_spec::Default;
    case GemmSpecialization::MPadding: return ck_gemm_spec::MPadding;
    case GemmSpecialization::NPadding: return ck_gemm_spec::NPadding;
    case GemmSpecialization::KPadding: return ck_gemm_spec::KPadding;
    case GemmSpecialization::MNPadding: return ck_gemm_spec::MNPadding;
    case GemmSpecialization::MKPadding: return ck_gemm_spec::MKPadding;
    case GemmSpecialization::NKPadding: return ck_gemm_spec::NKPadding;
    case GemmSpecialization::MNKPadding: return ck_gemm_spec::MNKPadding;
    case GemmSpecialization::OPadding: return ck_gemm_spec::OPadding;
    case GemmSpecialization::MOPadding: return ck_gemm_spec::MOPadding;
    case GemmSpecialization::NOPadding: return ck_gemm_spec::NOPadding;
    case GemmSpecialization::KOPadding: return ck_gemm_spec::KOPadding;
    case GemmSpecialization::MNOPadding: return ck_gemm_spec::MNOPadding;
    case GemmSpecialization::MKOPadding: return ck_gemm_spec::MKOPadding;
    case GemmSpecialization::NKOPadding: return ck_gemm_spec::NKOPadding;
    case GemmSpecialization::MNKOPadding: return ck_gemm_spec::MNKOPadding;
    default: throw "Unknown GemmSpecialization";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    constexpr auto version = ALGORITHM.pipeline_version;
    using ck_pipeline      = ck::BlockGemmPipelineVersion;
    switch(version)
    {
    case PipelineVersion::V1: return ck_pipeline::v1;
    case PipelineVersion::V2: return ck_pipeline::v2;
    case PipelineVersion::V3: return ck_pipeline::v3;
    case PipelineVersion::V4: return ck_pipeline::v4;
    case PipelineVersion::V5: return ck_pipeline::v5;
    case PipelineVersion::WEIGHT_ONLY:
        throw "PipelineVersion::WEIGHT_ONLY is not supported for block GEMM pipeline version.";
    default: throw "Unknown block GEMM PipelineVersion";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::ConvolutionForwardSpecialization SetFwdConvSpecialization()
{
    constexpr auto specialization = ALGORITHM.fwd_specialization;
    using ck_conv_spec            = ck::tensor_operation::device::ConvolutionForwardSpecialization;
    switch(specialization)
    {
    case ConvFwdSpecialization::DEFAULT: return ck_conv_spec::Default;
    case ConvFwdSpecialization::FILTER_1X1_PAD0: return ck_conv_spec::Filter1x1Pad0;
    case ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0: return ck_conv_spec::Filter1x1Stride1Pad0;
    case ConvFwdSpecialization::FILTER_3x3: return ck_conv_spec::Filter3x3;
    case ConvFwdSpecialization::ODD_C: return ck_conv_spec::OddC;
    default: throw "Unknown ConvFwdSpecialization";
    }
}

} // namespace ck_tile::builder::factory::internal
