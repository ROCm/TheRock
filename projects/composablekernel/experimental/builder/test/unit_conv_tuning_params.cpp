// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/builder/factory/helpers/ck/conv_tuning_params.hpp"

namespace {

namespace ckb = ::ck_tile::builder;
using namespace ck_tile::builder;
using namespace ck_tile::builder::factory::internal;

TEST(ConvTuningParams, AssignsBlockGemmParams)
{
    constexpr struct Algorithm
    {
        struct BlockGemm
        {
            ckb::PipelineVersion pipeline_version = ckb::PipelineVersion::V3;
            ckb::PipelineScheduler scheduler      = ckb::PipelineScheduler::INTRAWAVE;
        } block_gemm;
    } kAlgorithm;
    constexpr auto block_gemm = SetBlockGemm<kAlgorithm>();

    EXPECT_EQ(block_gemm.pipeline_version, ck::BlockGemmPipelineVersion::v3);
    EXPECT_EQ(block_gemm.scheduler, ck::BlockGemmPipelineScheduler::Intrawave);
}

TEST(ConvTuningParams, AssignsLoopSchedulerParam)
{
    constexpr struct Algorithm
    {
        ckb::PipelineScheduler loop_scheduler = ckb::PipelineScheduler::INTERWAVE;
    } kAlgorithm;
    constexpr auto loop_scheduler = SetLoopScheduler<kAlgorithm>();

    EXPECT_EQ(loop_scheduler, ck::LoopScheduler::Interwave);
}

TEST(ConvTuningParams, AssignsGridwiseGemmPipelineVersion)
{
    constexpr struct Algorithm
    {
        struct GridwiseGemm
        {
            ckb::PipelineVersion pipeline_version = ckb::PipelineVersion::V4;
        } gridwise_gemm;
    } kAlgorithm;
    constexpr auto pipeline_version = SetGridwiseGemmPipelineVersion<kAlgorithm>();

    EXPECT_EQ(pipeline_version, ck::PipelineVersion::v4);
}

TEST(ConvTuningParams, AssignsGemmSpecialization)
{
    constexpr struct Algorithm
    {
        ckb::GemmSpecialization gemm_specialization = ckb::GemmSpecialization::MNKPadding;
    } kAlgorithm;
    constexpr auto gemm_spec = SetGemmSpecialization<kAlgorithm>();

    EXPECT_EQ(gemm_spec, ck::tensor_operation::device::GemmSpecialization::MNKPadding);
}

TEST(ConvTuningParams, AssignsBlockGemmPipelineVersion)
{
    constexpr struct Algorithm
    {
        ckb::PipelineVersion pipeline_version = ckb::PipelineVersion::V2;
    } kAlgorithm;
    constexpr auto pipeline_version = SetBlockGemmPipelineVersion<kAlgorithm>();

    EXPECT_EQ(pipeline_version, ck::BlockGemmPipelineVersion::v2);
}

TEST(ConvTuningParams, AssignsFwdConvSpecialization)
{
    constexpr struct Algorithm
    {
        ckb::ConvFwdSpecialization fwd_specialization =
            ckb::ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0;
    } kAlgorithm;
    constexpr auto conv_spec = SetFwdConvSpecialization<kAlgorithm>();

    EXPECT_EQ(conv_spec,
              ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0);
}

} // namespace
