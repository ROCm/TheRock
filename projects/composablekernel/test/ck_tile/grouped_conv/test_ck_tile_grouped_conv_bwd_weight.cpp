// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"

using namespace ck_tile;

struct TestConvConfig
{
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 8;
    static constexpr index_t VectorSizeC = 8;

    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 128;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V3;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
};

// Helper to build full kernel type
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          index_t NDimSpatial = 2>
struct BuildKernel
{
    using GemmShape = TileGemmShape<
        sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         InLayout,
                                         WeiLayout,
                                         tuple<>,
                                         OutLayout,
                                         ConvConfig::VectorSizeA,
                                         ConvConfig::VectorSizeB,
                                         ConvConfig::VectorSizeC,
                                         ConvConfig::NumGroupsToMerge>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutBwdWeight,
                                typename ConvTraits::BsLayoutBwdWeight,
                                typename ConvTraits::CLayoutBwdWeight,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

    using GemmPipelineProblem =
        GemmPipelineProblem<PrecType, // OutDataType (A in bwd weight)
                            PrecType, // InDataType (B in bwd weight)
                            float,    // AccDataType
                            GemmShape,
                            typename ConvTraits::template GroupedConvImplicitGemmTraitsBwdWeight<
                                ConvConfig::NumWaveGroups>,
                            element_wise::PassThrough,
                            element_wise::PassThrough,
                            PrecType, // WeiDataType (C in bwd weight)
                            ConvTraits::FixedGemmParams::FixedVectorSize,
                            ConvTraits::VectorSizeA,
                            ConvTraits::VectorSizeB>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<PrecType,
                                     PrecType,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     ConvConfig::Scheduler,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     PrecType,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    ConvConfig::M_Warp,
                                                    ConvConfig::N_Warp,
                                                    ConvConfig::M_Warp_Tile,
                                                    ConvConfig::N_Warp_Tile,
                                                    ConvConfig::K_Warp_Tile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    ConvConfig::NumWaveGroups,
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using Epilogue = CShuffleEpilogue<EpilogueProblem>;

    using type =
        GroupedConvolutionBackwardWeightKernel<ConvTraits, TilePartitioner, GemmPipeline, Epilogue>;
};

// Helper to create 2D host args
static GroupedConvBwdWeightHostArgs create_2d_host_args(index_t G,
                                                        index_t N,
                                                        index_t K,
                                                        index_t C,
                                                        index_t Y,
                                                        index_t X,
                                                        index_t Hi,
                                                        index_t Wi,
                                                        index_t stride_y,
                                                        index_t stride_x,
                                                        index_t dilation_y,
                                                        index_t dilation_x,
                                                        index_t left_pad_y,
                                                        index_t left_pad_x,
                                                        index_t right_pad_y,
                                                        index_t right_pad_x,
                                                        index_t k_batch = 1)
{
    auto conv_param = conv::ConvParam{2,
                                      G,
                                      N,
                                      K,
                                      C,
                                      {Y, X},
                                      {Hi, Wi},
                                      {stride_y, stride_x},
                                      {dilation_y, dilation_x},
                                      {left_pad_y, left_pad_x},
                                      {right_pad_y, right_pad_x}};

    return GroupedConvBwdWeightHostArgs{conv_param, nullptr, nullptr, {}, nullptr, k_batch};
}

static GroupedConvBwdWeightHostArgs create_2d_host_args(index_t k_batch)
{
    return create_2d_host_args(2, 2, 8, 8, 3, 3, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, k_batch);
}

static GroupedConvBwdWeightHostArgs create_large_2d_host_args(index_t k_batch)
{
    return create_2d_host_args(2, 2, 8, 8, 3, 3, 70, 70, 1, 1, 1, 1, 1, 1, 1, 1, k_batch);
}

class GroupedConvBwdWeightIsSupportedArgumentTest : public ::testing::Test
{
};

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, ValidKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    auto host_args_kbatch_1 = create_2d_host_args(1);
    auto kargs_1 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_1);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_1));

    auto host_args_kbatch_4 = create_2d_host_args(4);
    auto kargs_4 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_4);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_4));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, InvalidKBatchLessThanOne)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    auto host_args_kbatch_0 = create_2d_host_args(0);
    auto kargs = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_0);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, K0KBatchLimitation)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch = 128 should pass
    auto host_args_kbatch_6 = create_2d_host_args(6);
    auto kargs_6 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_6);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_6));

    // k_batch = 129 should fail for half_t output
    auto host_args_kbatch_7 = create_2d_host_args(7);
    auto kargs_7 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_7);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_7));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, NonFloatDoubleOutputLimitsKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch = 128 should pass
    auto host_args_kbatch_128 = create_large_2d_host_args(128);
    auto kargs_128 =
        typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_128);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_128));

    // k_batch = 129 should fail for half_t output
    auto host_args_kbatch_129 = create_large_2d_host_args(129);
    auto kargs_129 =
        typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_129);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_129));
}
